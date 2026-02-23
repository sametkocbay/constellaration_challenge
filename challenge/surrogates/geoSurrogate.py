"""Surrogate model for stellarator metric prediction.

Predicts TWO outputs per boundary:
  col 0 — edge_rotational_transform_over_n_field_periods  (iota/nfp)
  col 1 — max_elongation

Both are needed by the optimizer:
  - iota/nfp cannot be computed analytically (needs VMEC or surrogate)
  - max_elongation from the surrogate is more VMEC-correlated than the
    boundary-only analytical estimate, helping filter VMEC-failing shapes

Key features:
  1. Lightweight architecture (128→64→32 + ResBlocks) — trains in minutes on CPU
  2. Ensemble of N_ENSEMBLE models with different seeds for uncertainty estimates
  3. Feature engineering: spectral energy, mode amplitudes
  4. Mixup augmentation for better interpolation

API:
  load_surrogate()  → (models, scales)
  predict(models, scales, x_raw) → np.ndarray   shape (n, 2): [iota/nfp, elongation]
  predict_with_uncertainty(models, scales, x_raw) → (mean, std)  each (n, 2)
"""

import random
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset, random_split

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --- CONFIGURATION ---
THIS_DIR = Path(__file__).resolve().parent

MODEL_DIR_CANDIDATES = [THIS_DIR / "models", THIS_DIR / "stellarator_models"]


def _resolve_save_dir() -> Path:
    for candidate in MODEL_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    return MODEL_DIR_CANDIDATES[0]


SAVE_DIR = _resolve_save_dir()
SAVE_DIR.mkdir(exist_ok=True, parents=True)

# Single-model paths (backward compat)
MODEL_PATH = SAVE_DIR / "stellarator_surrogate.pth"
SCALING_PATH = SAVE_DIR / "feature_scales.npy"
METRICS_PATH = SAVE_DIR / "metrics_scales.npy"
TRAIN_LOG_PATH = SAVE_DIR / "training_log.txt"

# --- TRAINING HYPER-PARAMETERS ---
N_ENSEMBLE = 3           # more lightweight models = better coverage
BATCH_SIZE = 512
EPOCHS = 500
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15
NOISE_STD = 0.01         # lighter noise — too much hurts accuracy
MIXUP_ALPHA = 0.2        # mixup augmentation strength
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_METRICS = [
    "metrics.edge_rotational_transform_over_n_field_periods",
    "metrics.max_elongation",
]

# Weights per output — both equally important for the optimizer
TARGET_WEIGHTS = torch.tensor([1.0, 1.0], device=DEVICE)


# =========================================================================
# FEATURE ENGINEERING
# =========================================================================
def compute_extra_features(r_cos: np.ndarray, z_sin: np.ndarray) -> np.ndarray:
    """Compute physics-inspired features from Fourier coefficients.

    Returns a 1D array of extra features:
      - spectral energy per poloidal mode (sum of squares across toroidal modes)
      - total spectral energy ratio (high modes / low modes)
      - dominant mode amplitudes
    """
    extras = []

    # Spectral energy per poloidal row
    for row in r_cos:
        extras.append(np.sum(row ** 2))
    for row in z_sin:
        extras.append(np.sum(row ** 2))

    # Ratio of high-mode to low-mode energy
    r_flat = r_cos.flatten()
    z_flat = z_sin.flatten()
    all_coeffs = np.concatenate([r_flat, z_flat])
    n = len(all_coeffs)
    low_energy = np.sum(all_coeffs[: n // 2] ** 2) + 1e-10
    high_energy = np.sum(all_coeffs[n // 2 :] ** 2) + 1e-10
    extras.append(np.log(high_energy / low_energy))

    # Key individual coefficients (major radius, elongation-related)
    if r_cos.shape[0] > 1 and r_cos.shape[1] > 4:
        extras.append(r_cos[0, 4])   # R_00 (major radius proxy)
        extras.append(r_cos[1, 4])   # R_10 (shaping)
        extras.append(z_sin[1, 4] if z_sin.shape[0] > 1 else 0.0)  # Z_10

    return np.array(extras, dtype=np.float32)


# =========================================================================
# MODEL ARCHITECTURE — lightweight
# =========================================================================
class ResidualBlock(nn.Module):
    """Simple pre-norm residual block."""

    def __init__(self, dim: int, dropout: float = 0.05):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class SurrogateModel(nn.Module):
    """Compact feedforward network for stellarator metrics."""

    def __init__(
        self,
        input_dim: int = 90,
        output_dim: int = 1,
        hidden_dims: list[int] | None = None,
        n_res_blocks: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # Stem
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout),
        ]

        prev = hidden_dims[0]
        for h in hidden_dims[1:]:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            prev = h

        self.stem = nn.Sequential(*layers)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dims[-1], dropout) for _ in range(n_res_blocks)]
        )

        # Per-metric output heads (separate small heads for each metric)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dims[-1]),
                nn.Linear(hidden_dims[-1], 32),
                nn.SiLU(),
                nn.Linear(32, 1),
            )
            for _ in range(output_dim)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.res_blocks(h)
        outputs = [head(h) for head in self.heads]
        return torch.cat(outputs, dim=-1)


# =========================================================================
# DATA PREPARATION
# =========================================================================
def prepare_data(filter_n_field_periods: int = 3):
    """Load and preprocess data from HuggingFace dataset."""
    print("Loading dataset 'proxima-fusion/constellaration'...")
    try:
        ds = load_dataset("proxima-fusion/constellaration", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

    if filter_n_field_periods:
        ds = ds.filter(
            lambda x: x["boundary.n_field_periods"] == filter_n_field_periods
        )
        print(
            f"Filtered to {len(ds)} configs with "
            f"n_field_periods={filter_n_field_periods}"
        )

    X_raw, X_extra, Y = [], [], []
    skipped = 0

    for row_raw in ds:
        row = cast(dict[str, Any], row_raw)
        targets = [row.get(m) for m in TARGET_METRICS]
        if any(t is None for t in targets):
            skipped += 1
            continue

        try:
            r_cos = np.array(
                cast(Sequence[float], row["boundary.r_cos"]), dtype=np.float32
            )
            z_sin = np.array(
                cast(Sequence[float], row["boundary.z_sin"]), dtype=np.float32
            )

            r_flat = r_cos.flatten()
            z_flat = z_sin.flatten()
            if len(r_flat) == 0 or len(z_flat) == 0:
                skipped += 1
                continue

            X_raw.append(np.concatenate([r_flat, z_flat]))

            # Extra engineered features
            if r_cos.ndim == 2 and z_sin.ndim == 2:
                X_extra.append(compute_extra_features(r_cos, z_sin))
            else:
                X_extra.append(compute_extra_features(
                    r_cos.reshape(1, -1), z_sin.reshape(1, -1)
                ))

            Y.append(targets)
        except (ValueError, TypeError):
            skipped += 1
            continue

    print(f"Skipped {skipped} invalid rows")
    print(f"Total valid samples: {len(X_raw)}")
    if len(X_raw) == 0:
        raise ValueError("No valid data samples found!")

    X_raw_np = np.array(X_raw, dtype=np.float32)
    X_extra_np = np.array(X_extra, dtype=np.float32)
    X_np = np.concatenate([X_raw_np, X_extra_np], axis=1)
    Y_np = np.array(Y, dtype=np.float32)
    print(f"Raw features: {X_raw_np.shape[1]}, Extra features: {X_extra_np.shape[1]}")
    print(f"Total input dim: {X_np.shape[1]}, Output dim: {Y_np.shape[1]}")

    # --- Remove outliers (4 sigma) ---
    y_mean = Y_np.mean(axis=0, keepdims=True)
    y_std = Y_np.std(axis=0, keepdims=True)
    y_std[y_std == 0] = 1.0
    z_scores = np.abs((Y_np - y_mean) / y_std)
    keep = (z_scores < 4.0).all(axis=1)
    X_np = X_np[keep]
    Y_np = Y_np[keep]
    print(f"After outlier removal: {len(X_np)} samples")

    # --- Normalization ---
    x_mean = X_np.mean(axis=0, keepdims=True)
    x_std = X_np.std(axis=0, keepdims=True)
    x_std[x_std == 0] = 1.0
    X_norm = (X_np - x_mean) / x_std

    y_mean = Y_np.mean(axis=0, keepdims=True)
    y_std = Y_np.std(axis=0, keepdims=True)
    y_std[y_std == 0] = 1.0
    Y_norm = (Y_np - y_mean) / y_std

    scaling_info = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "target_metrics": TARGET_METRICS,
        "n_raw_features": X_raw_np.shape[1],
        "n_extra_features": X_extra_np.shape[1],
        "output_dim": len(TARGET_METRICS),
    }
    np.save(SCALING_PATH, scaling_info, allow_pickle=True)  # type: ignore
    np.save(METRICS_PATH, y_mean, allow_pickle=True)  # type: ignore

    X_t = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE)
    Y_t = torch.tensor(Y_norm, dtype=torch.float32, device=DEVICE)
    return X_t, Y_t, X_np.shape[1]


# =========================================================================
# TRAINING
# =========================================================================
def _mixup_batch(bx, by, alpha=0.2):
    """Mixup augmentation: blend random pairs of samples."""
    if alpha <= 0:
        return bx, by
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5
    idx = torch.randperm(bx.size(0), device=bx.device)
    bx_mix = lam * bx + (1 - lam) * bx[idx]
    by_mix = lam * by + (1 - lam) * by[idx]
    return bx_mix, by_mix


def _train_single_model(
    X: torch.Tensor,
    Y: torch.Tensor,
    input_dim: int,
    seed: int,
    model_path: Path,
    log_file=None,
) -> SurrogateModel:
    """Train one model of the ensemble."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset = TensorDataset(X, Y)
    train_size = int(len(dataset) * (1 - VAL_SPLIT))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    output_dim = int(Y.shape[1])
    model = SurrogateModel(input_dim=input_dim, output_dim=output_dim).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.SmoothL1Loss(reduction="none", beta=0.5)

    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-5
    )

    best_val = float("inf")
    patience_counter = 0
    patience = 80

    for epoch in range(1, EPOCHS + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            # Mixup augmentation
            bx_aug, by_aug = _mixup_batch(bx, by, MIXUP_ALPHA)
            # Light Gaussian noise
            bx_aug = bx_aug + torch.randn_like(bx_aug) * NOISE_STD

            optimizer.zero_grad()
            pred = model(bx_aug)
            loss = (criterion(pred, by_aug) * TARGET_WEIGHTS).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                pred = model(bx)
                loss = (criterion(pred, by) * TARGET_WEIGHTS).mean()
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            msg = (
                f"  [seed={seed}] Epoch {epoch:3d}/{EPOCHS} | "
                f"train={train_loss:.6f} val={val_loss:.6f} lr={lr_now:.2e}"
            )
            print(msg)
            if log_file:
                log_file.write(msg + "\n")
                log_file.flush()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"  [seed={seed}] Early stop at epoch {epoch} "
                    f"(best val={best_val:.6f})"
                )
                break

    # Reload best checkpoint
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"  [seed={seed}] Best val loss: {best_val:.6f}")
    return model


def train():
    """Train an ensemble of N_ENSEMBLE models."""
    X, Y, input_dim = prepare_data(filter_n_field_periods=3)

    print(f"\nTraining ensemble of {N_ENSEMBLE} models on {X.shape[0]} samples...")
    print(f"Architecture: 128→64→32 + 2 ResBlocks + {Y.shape[1]} heads {TARGET_METRICS}")
    print(f"Device: {DEVICE}\n")

    with open(TRAIN_LOG_PATH, "w") as log_file:
        log_file.write(f"Ensemble training: {N_ENSEMBLE} models\n")
        log_file.write(f"Input dim: {input_dim}, Samples: {X.shape[0]}\n\n")

        for i in range(N_ENSEMBLE):
            seed = 42 + i * 1337  # more spread-out seeds for diversity
            if i == 0:
                mpath = MODEL_PATH
            else:
                mpath = SAVE_DIR / f"stellarator_surrogate_{i}.pth"
            print(f"--- Ensemble member {i+1}/{N_ENSEMBLE} (seed={seed}) ---")
            log_file.write(f"--- Member {i+1}/{N_ENSEMBLE} (seed={seed}) ---\n")
            _train_single_model(X, Y, input_dim, seed, mpath, log_file)
            print()

    print("=" * 60)
    print("Ensemble training complete!")
    print(f"Models saved in: {SAVE_DIR}")
    print(f"Log: {TRAIN_LOG_PATH}")
    print("=" * 60)


# =========================================================================
# INFERENCE
# =========================================================================
def load_surrogate():
    """Load the ensemble (or single model) and scaling factors.

    Returns (models, scales) where models is a list of SurrogateModel.
    """
    if not SCALING_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Surrogate artifacts not found. Train first: python {__file__}"
        )

    scales = np.load(SCALING_PATH, allow_pickle=True).item()
    input_dim = int(scales["x_mean"].shape[1])

    models: list[SurrogateModel] = []

    output_dim = int(scales.get("output_dim", 1))

    # Load primary model
    m = SurrogateModel(input_dim=input_dim, output_dim=output_dim)
    m.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    m.to(DEVICE)
    m.eval()
    models.append(m)

    # Load additional ensemble members
    for i in range(1, 20):
        p = SAVE_DIR / f"stellarator_surrogate_{i}.pth"
        if not p.exists():
            break
        m = SurrogateModel(input_dim=input_dim, output_dim=output_dim)
        m.load_state_dict(
            torch.load(p, map_location=DEVICE, weights_only=True)
        )
        m.to(DEVICE)
        m.eval()
        models.append(m)

    print(f"Loaded ensemble of {len(models)} model(s)")
    return models, scales


def predict(model, scales, x_raw):
    """Predict stellarator metrics using the surrogate (single model or ensemble).

    Args:
        model: SurrogateModel or list[SurrogateModel]
        scales: Scaling factors from training
        x_raw: Raw Fourier coefficients (numpy array, shape (n, n_raw_features))
               Extra features are computed automatically.

    Returns:
        Predictions in original scale (numpy array, shape (n, output_dim)).
        Columns: [iota/nfp, max_elongation]  (if trained with 2 outputs)
                 [iota/nfp]                  (if legacy 1-output model)
    """
    with torch.no_grad():
        # Compute extra features if needed
        n_raw = int(scales.get("n_raw_features", x_raw.shape[1]))
        if x_raw.shape[1] == n_raw and "n_extra_features" in scales:
            # Need to add engineered features
            extras = []
            for row in x_raw:
                half = len(row) // 2
                r_cos = row[:half]
                z_sin = row[half:]
                # Estimate 2D shape for feature computation
                # Default to 5 x (n/5) shape
                n_tor = max(1, len(r_cos) // 5)
                try:
                    r_2d = r_cos.reshape(-1, n_tor)
                    z_2d = z_sin.reshape(-1, n_tor)
                except ValueError:
                    r_2d = r_cos.reshape(1, -1)
                    z_2d = z_sin.reshape(1, -1)
                extras.append(compute_extra_features(r_2d, z_2d))

            extras_np = np.array(extras, dtype=np.float32)
            x_full = np.concatenate([x_raw, extras_np], axis=1)
        else:
            x_full = x_raw

        # Normalize
        x_norm = (x_full - scales["x_mean"]) / scales["x_std"]
        x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=DEVICE)

        if isinstance(model, list):
            preds = []
            for m in model:
                m.eval()
                preds.append(m(x_tensor).cpu().numpy())
            y_norm = np.mean(preds, axis=0)
        else:
            model.eval()
            y_norm = model(x_tensor).cpu().numpy()

        y_pred = y_norm * scales["y_std"] + scales["y_mean"]
    return y_pred


def predict_with_uncertainty(
    model: list[SurrogateModel] | SurrogateModel,
    scales: dict,
    x_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict metrics and return ensemble mean + std for uncertainty estimation.

    This allows callers to reject out-of-distribution inputs where ensemble
    members disagree strongly.

    Args:
        model: list[SurrogateModel] or single SurrogateModel
        scales: Scaling factors from training
        x_raw: Raw Fourier coefficients (numpy array, shape (n, n_raw_features))

    Returns:
        (y_mean, y_std) — each shape (n, output_dim).
        output_dim=2: col 0 = iota/nfp, col 1 = max_elongation
        If only a single model is provided, y_std is zeros.
    """
    with torch.no_grad():
        # Compute extra features if needed
        n_raw = int(scales.get("n_raw_features", x_raw.shape[1]))
        if x_raw.shape[1] == n_raw and "n_extra_features" in scales:
            extras = []
            for row in x_raw:
                half = len(row) // 2
                r_cos = row[:half]
                z_sin = row[half:]
                n_tor = max(1, len(r_cos) // 5)
                try:
                    r_2d = r_cos.reshape(-1, n_tor)
                    z_2d = z_sin.reshape(-1, n_tor)
                except ValueError:
                    r_2d = r_cos.reshape(1, -1)
                    z_2d = z_sin.reshape(1, -1)
                extras.append(compute_extra_features(r_2d, z_2d))
            extras_np = np.array(extras, dtype=np.float32)
            x_full = np.concatenate([x_raw, extras_np], axis=1)
        else:
            x_full = x_raw

        x_norm = (x_full - scales["x_mean"]) / scales["x_std"]
        x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=DEVICE)

        if isinstance(model, list) and len(model) > 1:
            preds = []
            for m in model:
                m.eval()
                p_norm = m(x_tensor).cpu().numpy()
                preds.append(p_norm * scales["y_std"] + scales["y_mean"])
            preds_arr = np.array(preds)  # (n_models, n_samples, 1)
            y_mean = np.mean(preds_arr, axis=0)
            y_std = np.std(preds_arr, axis=0)
        else:
            single = model[0] if isinstance(model, list) else model
            single.eval()
            y_norm = single(x_tensor).cpu().numpy()
            y_mean = y_norm * scales["y_std"] + scales["y_mean"]
            y_std = np.zeros_like(y_mean)

    return y_mean, y_std


if __name__ == "__main__":
    train()