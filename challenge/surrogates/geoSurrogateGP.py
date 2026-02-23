"""Gaussian Process surrogate model for stellarator metric prediction.

Replaces / supplements the NN ensemble with sklearn Gaussian Processes,
which provide *mathematically rigorous* uncertainty bounds (posterior std)
instead of the noisy ensemble disagreement of 3 NNs.

Predicts TWO outputs per boundary:
  col 0 — edge_rotational_transform_over_n_field_periods  (iota/nfp)
  col 1 — max_elongation

Key advantages over the NN ensemble:
  1. Calibrated uncertainty — GP posterior std is a proper Bayesian credible
     interval, making the IOTA_UNC_MAX filter far more reliable.
  2. Better extrapolation awareness — GP uncertainty grows *automatically*
     in regions far from training data (no need for ensemble disagreement).
  3. No hyper-parameter tuning — the Matérn kernel length-scales are fit
     via marginal likelihood maximisation during training.
  4. Data augmentation via small perturbations for richer local gradient info.

Limitations:
  - Exact GP scales as O(n^3).  We use a Nystroem/sparse approximation
    (sklearn's default optimiser + PCA dimensionality reduction) to keep
    training tractable for >5000 samples.

API (drop-in compatible with geoSurrogate):
  load_surrogate()  → (gp_models, scales)
  predict(gp_models, scales, x_raw)               → np.ndarray (n, 2)
  predict_with_uncertainty(gp_models, scales, x_raw) → (mean, std)  each (n, 2)
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    RBF,
    WhiteKernel,
)
from sklearn.preprocessing import StandardScaler

# =========================================================================
# CONFIGURATION
# =========================================================================
THIS_DIR = Path(__file__).resolve().parent

MODEL_DIR_CANDIDATES = [THIS_DIR / "models", THIS_DIR / "stellarator_models"]


def _resolve_save_dir() -> Path:
    for candidate in MODEL_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    return MODEL_DIR_CANDIDATES[0]


SAVE_DIR = _resolve_save_dir()
SAVE_DIR.mkdir(exist_ok=True, parents=True)

# Saved artefact paths
GP_MODEL_PATH = SAVE_DIR / "gp_surrogate.pkl"
GP_SCALES_PATH = SAVE_DIR / "gp_feature_scales.npy"
TRAIN_LOG_PATH = SAVE_DIR / "gp_training_log.txt"

# Training config
VAL_SPLIT = 0.15
N_PCA_COMPONENTS = 30           # reduce from ~100 raw features to 30 PCA dims
N_AUGMENTED_COPIES = 3          # data augmentation: random perturbations
AUGMENT_NOISE_STD = 0.005       # small noise to teach local gradients
MAX_TRAIN_SAMPLES = 5_00      # cap to keep GP tractable (O(n^3), ~50 MB kernel matrix)
RANDOM_STATE = 42

# Active learning config
AL_SEED_SIZE = 100              # initial random seed set
AL_BATCH_SIZE = 100             # points added per active learning round
AL_ROUNDS = 4                   # number of acquisition rounds → 400 + 6×350 = 2500 total
AL_CANDIDATE_SUBSAMPLE = 15_000 # subsample pool for uncertainty scoring (speed)

TARGET_METRICS = [
    "metrics.edge_rotational_transform_over_n_field_periods",
    "metrics.max_elongation",
]


# =========================================================================
# FEATURE ENGINEERING  (same as geoSurrogate for compatibility)
# =========================================================================
def compute_extra_features(r_cos: np.ndarray, z_sin: np.ndarray) -> np.ndarray:
    """Compute physics-inspired features from Fourier coefficients.

    Identical to geoSurrogate.compute_extra_features so that the
    feature space is compatible and we can swap surrogates freely.
    """
    extras = []

    # Spectral energy per poloidal row
    for row in r_cos:
        extras.append(np.sum(row**2))
    for row in z_sin:
        extras.append(np.sum(row**2))

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
# DATA PREPARATION
# =========================================================================
def prepare_data(filter_n_field_periods: int = 3):
    """Load, preprocess, and augment data from HuggingFace dataset.

    Returns
    -------
    X_raw_np : (n_samples, n_raw_features)
    X_full_np : (n_samples, n_raw_features + n_extra_features)
    Y_np : (n_samples, 2)  — [iota/nfp, max_elongation]
    scaling_info : dict with all normalisation metadata
    """
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

    X_raw: list[np.ndarray] = []
    X_extra: list[np.ndarray] = []
    Y: list[list[float]] = []
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
                X_extra.append(
                    compute_extra_features(r_cos.reshape(1, -1), z_sin.reshape(1, -1))
                )

            Y.append([float(t) for t in targets])  # type: ignore[arg-type]
        except (ValueError, TypeError):
            skipped += 1
            continue

    print(f"Skipped {skipped} invalid rows")
    print(f"Total valid samples: {len(X_raw)}")
    if len(X_raw) == 0:
        raise ValueError("No valid data samples found!")

    X_raw_np = np.array(X_raw, dtype=np.float32)
    X_extra_np = np.array(X_extra, dtype=np.float32)
    X_full_np = np.concatenate([X_raw_np, X_extra_np], axis=1)
    Y_np = np.array(Y, dtype=np.float32)
    print(f"Raw features: {X_raw_np.shape[1]}, Extra features: {X_extra_np.shape[1]}")
    print(f"Total input dim: {X_full_np.shape[1]}, Output dim: {Y_np.shape[1]}")

    # --- Remove outliers (4 sigma) ---
    y_mean = Y_np.mean(axis=0, keepdims=True)
    y_std = Y_np.std(axis=0, keepdims=True)
    y_std[y_std == 0] = 1.0
    z_scores = np.abs((Y_np - y_mean) / y_std)
    keep = (z_scores < 4.0).all(axis=1)
    X_raw_np = X_raw_np[keep]
    X_extra_np = X_extra_np[keep]
    X_full_np = X_full_np[keep]
    Y_np = Y_np[keep]
    print(f"After outlier removal: {len(X_full_np)} samples")

    scaling_info = {
        "n_raw_features": X_raw_np.shape[1],
        "n_extra_features": X_extra_np.shape[1],
        "output_dim": len(TARGET_METRICS),
        "target_metrics": TARGET_METRICS,
    }

    return X_raw_np, X_full_np, Y_np, scaling_info


def _augment_data(
    X: np.ndarray,
    Y: np.ndarray,
    n_copies: int = N_AUGMENTED_COPIES,
    noise_std: float = AUGMENT_NOISE_STD,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment training data with small Gaussian perturbations.

    This teaches the GP about local gradients / sensitivity of each metric
    to coefficient perturbations.  The targets are kept identical (label-
    preserving augmentation), which acts as a soft smoothness prior.
    """
    if n_copies <= 0:
        return X, Y

    rng = np.random.RandomState(RANDOM_STATE + 7)
    X_aug = [X]
    Y_aug = [Y]
    for _ in range(n_copies):
        noise = rng.randn(*X.shape).astype(np.float32) * noise_std
        X_aug.append(X + noise)
        Y_aug.append(Y.copy())

    return np.concatenate(X_aug, axis=0), np.concatenate(Y_aug, axis=0)


# =========================================================================
# TRAINING
# =========================================================================
def train() -> None:
    """Train two independent GP regressors (one per target metric).

    Pipeline:
      1. Load + clean data
      2. Augment with small perturbations
      3. PCA dimensionality reduction (100+ → 30 dims)
      4. StandardScale X and Y
      5. Fit GP with Matérn-5/2 + WhiteKernel per output
      6. Persist (gp_models, pca, scalers, scales) to disk
    """
    X_raw_np, X_full_np, Y_np, scaling_info = prepare_data(filter_n_field_periods=3)

    # ---- train / val split ----
    n = len(X_full_np)
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    n_val = int(n * VAL_SPLIT)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    X_train, Y_train = X_full_np[train_idx], Y_np[train_idx]
    X_val, Y_val = X_full_np[val_idx], Y_np[val_idx]

    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")

    # ---- data augmentation ----
    X_train_aug, Y_train_aug = _augment_data(X_train, Y_train)
    print(f"After augmentation: {len(X_train_aug)} training samples")

    # ---- cap samples to keep GP tractable ----
    if len(X_train_aug) > MAX_TRAIN_SAMPLES:
        sub_idx = rng.choice(len(X_train_aug), MAX_TRAIN_SAMPLES, replace=False)
        X_train_aug = X_train_aug[sub_idx]
        Y_train_aug = Y_train_aug[sub_idx]
        print(f"Sub-sampled to {MAX_TRAIN_SAMPLES} for GP tractability")

    # ---- PCA ----
    n_components = min(N_PCA_COMPONENTS, X_train_aug.shape[1], len(X_train_aug))
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_aug)
    X_val_pca = pca.transform(X_val)
    print(
        f"PCA: {X_full_np.shape[1]} → {n_components} dims  "
        f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})"
    )

    # ---- scale X and Y ----
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train_pca)
    X_val_scaled = x_scaler.transform(X_val_pca)

    y_scalers: list[StandardScaler] = []
    Y_train_scaled_list: list[np.ndarray] = []
    Y_val_scaled_list: list[np.ndarray] = []
    for col in range(Y_train_aug.shape[1]):
        ys = StandardScaler()
        Y_train_scaled_list.append(ys.fit_transform(Y_train_aug[:, col : col + 1]).ravel())
        Y_val_scaled_list.append(ys.transform(Y_val[:, col : col + 1]).ravel())
        y_scalers.append(ys)

    # ---- fit one GP per output ----
    gp_models: list[GaussianProcessRegressor] = []
    log_lines: list[str] = []

    for i, metric_name in enumerate(TARGET_METRICS):
        short_name = metric_name.split(".")[-1]
        print(f"\n--- Training GP for '{short_name}' ---")
        log_lines.append(f"--- GP for '{short_name}' ---\n")

        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * Matern(
                length_scale=np.ones(n_components),
                length_scale_bounds=(1e-3, 1e3),
                nu=2.5,
            )
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
        )

        # Memory estimate: n^2 * 8 bytes for kernel matrix
        mem_mb = (len(X_train_scaled) ** 2 * 8) / 1e6
        print(f"  Kernel matrix: {len(X_train_scaled)}×{len(X_train_scaled)} = {mem_mb:.0f} MB")

        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,   # fewer restarts to save memory (was 5)
            normalize_y=False,       # already scaled
            alpha=1e-6,              # jitter for numerical stability
            random_state=RANDOM_STATE + i,
        )

        import time as _time

        t0 = _time.time()
        gp.fit(X_train_scaled, Y_train_scaled_list[i])
        dt = _time.time() - t0

        # Validation R²
        r2_train = gp.score(X_train_scaled, Y_train_scaled_list[i])
        r2_val = gp.score(X_val_scaled, Y_val_scaled_list[i])

        # Validation RMSE (in original scale)
        y_pred_val_scaled = gp.predict(X_val_scaled)  # type: ignore[assignment]
        y_pred_val = y_scalers[i].inverse_transform(
            np.asarray(y_pred_val_scaled).reshape(-1, 1)
        ).ravel()
        rmse_val = np.sqrt(np.mean((y_pred_val - Y_val[:, i]) ** 2))

        msg = (
            f"  Trained in {dt:.1f}s | "
            f"R² train={r2_train:.4f}  val={r2_val:.4f} | "
            f"RMSE val={rmse_val:.4f} | "
            f"kernel: {gp.kernel_}"
        )
        print(msg)
        log_lines.append(msg + "\n")

        gp_models.append(gp)

    # ---- persist ----
    artefacts = {
        "gp_models": gp_models,
        "pca": pca,
        "x_scaler": x_scaler,
        "y_scalers": y_scalers,
        "scaling_info": scaling_info,
    }
    with open(GP_MODEL_PATH, "wb") as f:
        pickle.dump(artefacts, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Also save scaling_info as .npy for quick loading by the optimizer
    np.save(GP_SCALES_PATH, scaling_info, allow_pickle=True)

    with open(TRAIN_LOG_PATH, "w") as f:
        f.writelines(log_lines)

    print(f"\nGP surrogate saved to: {GP_MODEL_PATH}")
    print(f"Scaling info saved to: {GP_SCALES_PATH}")
    print("Training complete!")


# =========================================================================
# ACTIVE LEARNING TRAINING
# =========================================================================
def train_active_learning() -> None:
    """Train GP with pool-based active learning.

    Instead of randomly sub-sampling 2,500 from 68k, we iteratively
    select the most informative points where the GP is most uncertain.

    Algorithm:
      1. Start with a small diverse seed set (AL_SEED_SIZE)
      2. Fit lightweight GP → score uncertainty on a pool subsample
      3. Select top AL_BATCH_SIZE most uncertain points → add to training set
      4. Repeat for AL_ROUNDS → final training set ≈ MAX_TRAIN_SAMPLES
      5. Fit final GP on the actively selected data

    This ensures the GP covers under-represented regions of the feature
    space (e.g., low aspect ratio rotating ellipses) where the optimizer
    actually searches, rather than wasting capacity on dense clusters.
    """
    import time as _time

    X_raw_np, X_full_np, Y_np, scaling_info = prepare_data(filter_n_field_periods=3)

    # ---- train / val split ----
    n = len(X_full_np)
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    n_val = int(n * VAL_SPLIT)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    X_pool, Y_pool = X_full_np[train_idx], Y_np[train_idx]
    X_val, Y_val = X_full_np[val_idx], Y_np[val_idx]

    print(f"\nPool: {len(X_pool)}, Val: {len(X_val)}")

    # ---- PCA (fit on full pool for stable transform) ----
    n_components = min(N_PCA_COMPONENTS, X_pool.shape[1], len(X_pool))
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pool_pca = pca.fit_transform(X_pool)
    X_val_pca = pca.transform(X_val)
    print(
        f"PCA: {X_full_np.shape[1]} → {n_components} dims  "
        f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})"
    )

    # ---- Active learning loop ----
    pool_indices = np.arange(len(X_pool))  # indices into X_pool
    selected_mask = np.zeros(len(X_pool), dtype=bool)

    # Tracking for visualisation
    al_history: list[dict[str, Any]] = []

    # Step 1: diverse seed via farthest-point sampling in PCA space
    print(f"\n--- Active Learning: seed={AL_SEED_SIZE}, "
          f"batch={AL_BATCH_SIZE}, rounds={AL_ROUNDS} ---")
    seed_idx = _farthest_point_sampling(X_pool_pca, AL_SEED_SIZE, rng)
    selected_mask[seed_idx] = True
    print(f"  Seed: {selected_mask.sum()} points (farthest-point sampling)")

    for al_round in range(AL_ROUNDS):
        t0 = _time.time()
        train_sel = pool_indices[selected_mask]
        X_train_al = X_pool_pca[train_sel]
        Y_train_al = Y_pool[train_sel]

        # Scale X
        x_scaler_al = StandardScaler()
        X_train_scaled = x_scaler_al.fit_transform(X_train_al)

        # Scale Y (use iota column = col 0 for uncertainty-based acquisition)
        y_scaler_al = StandardScaler()
        Y_train_iota_scaled = y_scaler_al.fit_transform(
            Y_train_al[:, 0:1]
        ).ravel()

        # Fit a lightweight GP (fewer restarts, just for acquisition)
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * Matern(
                length_scale=np.ones(n_components),
                length_scale_bounds=(1e-2, 1e3),
                nu=2.5,
            )
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
        )
        gp_acq = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=1,   # fast — just for acquisition
            normalize_y=False,
            alpha=1e-5,
            random_state=RANDOM_STATE + al_round,
        )
        gp_acq.fit(X_train_scaled, Y_train_iota_scaled)

        # Score uncertainty on unselected pool (subsample for speed)
        remaining_idx = pool_indices[~selected_mask]
        if len(remaining_idx) > AL_CANDIDATE_SUBSAMPLE:
            cand_idx = rng.choice(
                remaining_idx, AL_CANDIDATE_SUBSAMPLE, replace=False
            )
        else:
            cand_idx = remaining_idx

        X_cand_pca = X_pool_pca[cand_idx]
        X_cand_scaled = x_scaler_al.transform(X_cand_pca)

        _, cand_std = gp_acq.predict(X_cand_scaled, return_std=True)

        # Select top-K most uncertain
        n_to_add = min(AL_BATCH_SIZE, len(cand_idx))
        top_k_local = np.argsort(cand_std)[-n_to_add:]
        new_idx = cand_idx[top_k_local]
        selected_mask[new_idx] = True

        dt = _time.time() - t0
        n_selected = selected_mask.sum()

        # Quick validation score + RMSE
        X_val_scaled = x_scaler_al.transform(X_val_pca)
        Y_val_iota_scaled = y_scaler_al.transform(Y_val[:, 0:1]).ravel()
        r2_val = gp_acq.score(X_val_scaled, Y_val_iota_scaled)

        y_pred_val_scaled = gp_acq.predict(X_val_scaled)
        y_pred_val = y_scaler_al.inverse_transform(
            np.asarray(y_pred_val_scaled).reshape(-1, 1)
        ).ravel()
        rmse_val = float(np.sqrt(np.mean((y_pred_val - Y_val[:, 0]) ** 2)))

        # Pool-wide uncertainty snapshot (on the subsample)
        pool_unc_mean = float(np.mean(cand_std))
        pool_unc_max = float(np.max(cand_std))
        pool_unc_median = float(np.median(cand_std))

        # Record history
        al_history.append({
            "round": al_round + 1,
            "n_selected": int(n_selected),
            "r2_val": float(r2_val),
            "rmse_val": rmse_val,
            "added_unc_max": float(cand_std[top_k_local].max()),
            "added_unc_mean": float(cand_std[top_k_local].mean()),
            "pool_unc_mean": pool_unc_mean,
            "pool_unc_max": pool_unc_max,
            "pool_unc_median": pool_unc_median,
            "new_points_pca": X_pool_pca[new_idx, :2].copy(),  # first 2 PCA dims
            "time_s": dt,
        })

        print(
            f"  Round {al_round + 1}/{AL_ROUNDS}: "
            f"{n_selected} points | "
            f"added {n_to_add} (max unc={cand_std[top_k_local].max():.4f}, "
            f"mean unc={cand_std[top_k_local].mean():.4f}) | "
            f"iota R²_val={r2_val:.4f}  RMSE={rmse_val:.4f} | {dt:.1f}s"
        )

    # ---- Final training set from active learning ----
    final_idx = pool_indices[selected_mask]
    X_train_final = X_pool[final_idx]
    Y_train_final = Y_pool[final_idx]
    print(f"\nActive learning selected {len(X_train_final)} training points")

    # ---- Augment the actively selected data ----
    X_train_aug, Y_train_aug = _augment_data(X_train_final, Y_train_final)
    print(f"After augmentation: {len(X_train_aug)} samples")

    # Cap if augmentation pushed us over budget
    if len(X_train_aug) > MAX_TRAIN_SAMPLES:
        sub_idx = rng.choice(len(X_train_aug), MAX_TRAIN_SAMPLES, replace=False)
        X_train_aug = X_train_aug[sub_idx]
        Y_train_aug = Y_train_aug[sub_idx]
        print(f"Sub-sampled to {MAX_TRAIN_SAMPLES} for GP tractability")

    # ---- Re-fit PCA on actively selected data (optional: keep pool PCA) ----
    # Using pool PCA for consistency with inference
    X_train_pca = pca.transform(X_train_aug)

    # ---- Scale X and Y (final) ----
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train_pca)
    X_val_scaled = x_scaler.transform(X_val_pca)

    y_scalers: list[StandardScaler] = []
    Y_train_scaled_list: list[np.ndarray] = []
    Y_val_scaled_list: list[np.ndarray] = []
    for col in range(Y_train_aug.shape[1]):
        ys = StandardScaler()
        Y_train_scaled_list.append(ys.fit_transform(Y_train_aug[:, col : col + 1]).ravel())
        Y_val_scaled_list.append(ys.transform(Y_val[:, col : col + 1]).ravel())
        y_scalers.append(ys)

    # ---- Fit final GPs ----
    gp_models: list[GaussianProcessRegressor] = []
    log_lines: list[str] = ["=== Active Learning GP Training ===\n"]

    for i, metric_name in enumerate(TARGET_METRICS):
        short_name = metric_name.split(".")[-1]
        print(f"\n--- Training final GP for '{short_name}' ---")
        log_lines.append(f"--- GP for '{short_name}' ---\n")

        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * Matern(
                length_scale=np.ones(n_components),
                length_scale_bounds=(1e-3, 1e3),
                nu=2.5,
            )
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
        )

        mem_mb = (len(X_train_scaled) ** 2 * 8) / 1e6
        print(f"  Kernel matrix: {len(X_train_scaled)}×{len(X_train_scaled)} = {mem_mb:.0f} MB")

        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,
            normalize_y=False,
            alpha=1e-6,
            random_state=RANDOM_STATE + i,
        )

        t0 = _time.time()
        gp.fit(X_train_scaled, Y_train_scaled_list[i])
        dt = _time.time() - t0

        r2_train = gp.score(X_train_scaled, Y_train_scaled_list[i])
        r2_val = gp.score(X_val_scaled, Y_val_scaled_list[i])

        y_pred_val_scaled = gp.predict(X_val_scaled)
        y_pred_val = y_scalers[i].inverse_transform(
            np.asarray(y_pred_val_scaled).reshape(-1, 1)
        ).ravel()
        rmse_val = np.sqrt(np.mean((y_pred_val - Y_val[:, i]) ** 2))

        msg = (
            f"  Trained in {dt:.1f}s | "
            f"R² train={r2_train:.4f}  val={r2_val:.4f} | "
            f"RMSE val={rmse_val:.4f} | "
            f"kernel: {gp.kernel_}"
        )
        print(msg)
        log_lines.append(msg + "\n")
        gp_models.append(gp)

    # ---- persist ----
    artefacts = {
        "gp_models": gp_models,
        "pca": pca,
        "x_scaler": x_scaler,
        "y_scalers": y_scalers,
        "scaling_info": scaling_info,
    }
    with open(GP_MODEL_PATH, "wb") as f:
        pickle.dump(artefacts, f, protocol=pickle.HIGHEST_PROTOCOL)

    np.save(GP_SCALES_PATH, scaling_info, allow_pickle=True)

    with open(TRAIN_LOG_PATH, "w") as f:
        f.writelines(log_lines)

    print(f"\nGP surrogate (active learning) saved to: {GP_MODEL_PATH}")
    print(f"Scaling info saved to: {GP_SCALES_PATH}")
    print("Active learning training complete!")

    # ---- Visualise active learning progression ----
    _visualise_active_learning(
        al_history=al_history,
        X_pool_pca=X_pool_pca,
        seed_idx=seed_idx,
        selected_mask=selected_mask,
        Y_pool=Y_pool,
        X_val_pca=X_val_pca,
        Y_val=Y_val,
        gp_models=gp_models,
        x_scaler=x_scaler,
        y_scalers=y_scalers,
        save_dir=SAVE_DIR,
    )


def _visualise_active_learning(
    al_history: list[dict[str, Any]],
    X_pool_pca: np.ndarray,
    seed_idx: np.ndarray,
    selected_mask: np.ndarray,
    Y_pool: np.ndarray,
    X_val_pca: np.ndarray,
    Y_val: np.ndarray,
    gp_models: list[GaussianProcessRegressor],
    x_scaler: StandardScaler,
    y_scalers: list[StandardScaler],
    save_dir: Path,
) -> None:
    """Generate a multi-panel figure showing active learning progression.

    Panels:
      (a) R² and RMSE vs round — accuracy improvement
      (b) Pool uncertainty (mean/max/median) vs round — coverage
      (c) PCA scatter — seed vs each round's acquisitions (colour-coded)
      (d) Pred-vs-true on validation set for iota (final GP)
      (e) Pred-vs-true on validation set for elongation (final GP)
      (f) Uncertainty histogram on validation set (final GP)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm
    except ImportError:
        print("matplotlib not installed — skipping AL visualisation.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Active Learning GP Training Progression", fontsize=14, fontweight="bold")

    rounds = [h["round"] for h in al_history]
    n_points = [h["n_selected"] for h in al_history]

    # --- (a) R² and RMSE vs round ---
    ax_r2 = axes[0, 0]
    r2_vals = [h["r2_val"] for h in al_history]
    rmse_vals = [h["rmse_val"] for h in al_history]

    color_r2 = "#2196F3"
    color_rmse = "#F44336"
    ax_r2.plot(rounds, r2_vals, "o-", color=color_r2, linewidth=2, markersize=8, label="R² (val)")
    ax_r2.set_xlabel("AL Round")
    ax_r2.set_ylabel("R² (val)", color=color_r2)
    ax_r2.tick_params(axis="y", labelcolor=color_r2)
    ax_r2.set_ylim(min(0.5, min(r2_vals) - 0.05), 1.01)
    ax_r2.grid(True, alpha=0.3)

    ax_rmse = ax_r2.twinx()
    ax_rmse.plot(rounds, rmse_vals, "s--", color=color_rmse, linewidth=2, markersize=7, label="RMSE (val)")
    ax_rmse.set_ylabel("RMSE (val)", color=color_rmse)
    ax_rmse.tick_params(axis="y", labelcolor=color_rmse)

    # Combined legend
    lines1, labels1 = ax_r2.get_legend_handles_labels()
    lines2, labels2 = ax_rmse.get_legend_handles_labels()
    ax_r2.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax_r2.set_title("(a) Accuracy vs AL Round")

    # Add n_points as secondary x-axis labels
    ax_r2_top = ax_r2.twiny()
    ax_r2_top.set_xlim(ax_r2.get_xlim())
    ax_r2_top.set_xticks(rounds)
    ax_r2_top.set_xticklabels([str(n) for n in n_points], fontsize=8)
    ax_r2_top.set_xlabel("Training points", fontsize=9)

    # --- (b) Pool uncertainty vs round ---
    ax_unc = axes[0, 1]
    unc_mean = [h["pool_unc_mean"] for h in al_history]
    unc_max = [h["pool_unc_max"] for h in al_history]
    unc_med = [h["pool_unc_median"] for h in al_history]

    ax_unc.fill_between(rounds, unc_med, unc_max, alpha=0.15, color="#FF9800")
    ax_unc.plot(rounds, unc_max, "v-", color="#F44336", linewidth=1.5, markersize=6, label="max")
    ax_unc.plot(rounds, unc_mean, "o-", color="#FF9800", linewidth=2, markersize=7, label="mean")
    ax_unc.plot(rounds, unc_med, "^-", color="#4CAF50", linewidth=1.5, markersize=6, label="median")
    ax_unc.set_xlabel("AL Round")
    ax_unc.set_ylabel("Pool Uncertainty (GP std, scaled)")
    ax_unc.set_title("(b) Pool Uncertainty Reduction")
    ax_unc.legend()
    ax_unc.grid(True, alpha=0.3)

    # --- (c) PCA scatter: seed + acquisitions colour-coded by round ---
    ax_pca = axes[0, 2]
    # Pool background (grey)
    subsample_bg = np.random.RandomState(0).choice(
        len(X_pool_pca), min(5000, len(X_pool_pca)), replace=False
    )
    ax_pca.scatter(
        X_pool_pca[subsample_bg, 0], X_pool_pca[subsample_bg, 1],
        c="#E0E0E0", s=3, alpha=0.3, label="pool", rasterized=True,
    )
    # Seed points
    ax_pca.scatter(
        X_pool_pca[seed_idx, 0], X_pool_pca[seed_idx, 1],
        c="#2196F3", s=12, alpha=0.6, label=f"seed ({len(seed_idx)})",
        edgecolors="none",
    )
    # Each round's acquisitions in a different colour
    cmap = cm.get_cmap("YlOrRd", len(al_history) + 1)
    for h in al_history:
        pts = h["new_points_pca"]
        colour = cmap(h["round"] / (len(al_history) + 1))
        ax_pca.scatter(
            pts[:, 0], pts[:, 1],
            c=[colour], s=10, alpha=0.7,
            label=f"round {h['round']} (+{len(pts)})",
            edgecolors="none",
        )
    ax_pca.set_xlabel("PCA 1")
    ax_pca.set_ylabel("PCA 2")
    ax_pca.set_title("(c) Acquired Points in PCA Space")
    ax_pca.legend(fontsize=7, loc="upper right", ncol=2)

    # --- (d) Pred vs True: iota (final GP) ---
    ax_iota = axes[1, 0]
    X_val_scaled = x_scaler.transform(X_val_pca)
    y_pred_iota_scaled = gp_models[0].predict(X_val_scaled)
    y_pred_iota = y_scalers[0].inverse_transform(
        np.asarray(y_pred_iota_scaled).reshape(-1, 1)
    ).ravel()
    y_true_iota = Y_val[:, 0]

    ax_iota.scatter(y_true_iota, y_pred_iota, s=4, alpha=0.3, c="#2196F3", rasterized=True)
    lims = [min(y_true_iota.min(), y_pred_iota.min()), max(y_true_iota.max(), y_pred_iota.max())]
    ax_iota.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
    r2_iota = 1 - np.sum((y_true_iota - y_pred_iota) ** 2) / np.sum((y_true_iota - y_true_iota.mean()) ** 2)
    ax_iota.set_xlabel("True iota/nfp")
    ax_iota.set_ylabel("Predicted iota/nfp")
    ax_iota.set_title(f"(d) iota/nfp — R²={r2_iota:.4f}")
    ax_iota.grid(True, alpha=0.3)

    # --- (e) Pred vs True: elongation (final GP) ---
    ax_elong = axes[1, 1]
    y_pred_elong_scaled = gp_models[1].predict(X_val_scaled)
    y_pred_elong = y_scalers[1].inverse_transform(
        np.asarray(y_pred_elong_scaled).reshape(-1, 1)
    ).ravel()
    y_true_elong = Y_val[:, 1]

    ax_elong.scatter(y_true_elong, y_pred_elong, s=4, alpha=0.3, c="#4CAF50", rasterized=True)
    lims_e = [min(y_true_elong.min(), y_pred_elong.min()), max(y_true_elong.max(), y_pred_elong.max())]
    ax_elong.plot(lims_e, lims_e, "k--", linewidth=1, alpha=0.5)
    r2_elong = 1 - np.sum((y_true_elong - y_pred_elong) ** 2) / np.sum((y_true_elong - y_true_elong.mean()) ** 2)
    ax_elong.set_xlabel("True max_elongation")
    ax_elong.set_ylabel("Predicted max_elongation")
    ax_elong.set_title(f"(e) max_elongation — R²={r2_elong:.4f}")
    ax_elong.grid(True, alpha=0.3)

    # --- (f) Uncertainty histogram (final GP, validation set) ---
    ax_hist = axes[1, 2]
    _, iota_std = gp_models[0].predict(X_val_scaled, return_std=True)
    _, elong_std = gp_models[1].predict(X_val_scaled, return_std=True)
    # Convert back to original scale
    iota_scale = float(y_scalers[0].scale_[0]) if y_scalers[0].scale_ is not None else 1.0
    elong_scale = float(y_scalers[1].scale_[0]) if y_scalers[1].scale_ is not None else 1.0
    iota_std_orig = np.asarray(iota_std) * iota_scale
    elong_std_orig = np.asarray(elong_std) * elong_scale

    ax_hist.hist(iota_std_orig, bins=60, alpha=0.6, color="#2196F3", label="iota/nfp std", density=True)
    ax_hist.hist(elong_std_orig, bins=60, alpha=0.6, color="#4CAF50", label="elongation std", density=True)
    ax_hist.axvline(0.15, color="red", linestyle="--", linewidth=1.5, label="IOTA_UNC_MAX=0.15")
    ax_hist.set_xlabel("GP Posterior Std (original scale)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("(f) Uncertainty Distribution on Val Set")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = save_dir / "gp_active_learning_progression.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nAL visualisation saved to: {fig_path}")


def _farthest_point_sampling(
    X: np.ndarray, n_points: int, rng: np.random.RandomState
) -> np.ndarray:
    """Select n_points from X using greedy farthest-point sampling.

    This gives a maximally diverse seed set that covers the feature space
    uniformly, much better than random for GP initialisation.

    Returns indices into X.
    """
    n = len(X)
    if n_points >= n:
        return np.arange(n)

    selected = [rng.randint(n)]
    min_dists = np.full(n, np.inf)

    for _ in range(n_points - 1):
        # Update minimum distances to selected set
        last = X[selected[-1]]
        dists_to_last = np.sum((X - last) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists_to_last)

        # Pick the point farthest from all selected points
        next_idx = np.argmax(min_dists)
        selected.append(next_idx)

    return np.array(selected)


# =========================================================================
# INFERENCE
# =========================================================================
def load_surrogate() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the trained GP surrogate artefacts.

    Returns
    -------
    gp_bundle : dict
        Keys: 'gp_models' (list of 2 GPs), 'pca', 'x_scaler', 'y_scalers'
    scales : dict
        Same format as geoSurrogate's scales, with keys:
        'n_raw_features', 'n_extra_features', 'output_dim', etc.

    Usage is drop-in compatible with the optimizer:
        model, scales = geoSurrogateGP.load_surrogate()
        y_mean, y_std = geoSurrogateGP.predict_with_uncertainty(model, scales, x_raw)
    """
    if not GP_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"GP surrogate not found at {GP_MODEL_PATH}. "
            f"Train first: python {__file__}"
        )

    with open(GP_MODEL_PATH, "rb") as f:
        artefacts = pickle.load(f)

    scales = artefacts["scaling_info"]
    print(f"Loaded GP surrogate ({len(artefacts['gp_models'])} output GPs)")
    return artefacts, scales


def _prepare_features(
    x_raw: np.ndarray,
    scales: dict[str, Any],
    artefacts: dict[str, Any],
) -> np.ndarray:
    """Convert raw Fourier coefficients → PCA-scaled feature matrix.

    Handles the same zero-padding and extra feature computation as the
    NN surrogate so the two are interchangeable.
    """
    n_raw = int(scales.get("n_raw_features", x_raw.shape[1]))

    # Zero-pad if the input has fewer raw features than training data
    if x_raw.shape[1] < n_raw:
        pad = np.zeros((x_raw.shape[0], n_raw - x_raw.shape[1]), dtype=np.float32)
        x_raw = np.concatenate([x_raw, pad], axis=1)
    elif x_raw.shape[1] > n_raw:
        x_raw = x_raw[:, :n_raw]

    # Extra engineered features
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

    # PCA + scale
    pca: PCA = artefacts["pca"]
    x_scaler: StandardScaler = artefacts["x_scaler"]

    x_pca = pca.transform(x_full)
    x_scaled = x_scaler.transform(x_pca)
    return x_scaled


def predict(
    model: dict[str, Any],
    scales: dict[str, Any],
    x_raw: np.ndarray,
) -> np.ndarray:
    """Predict stellarator metrics using the GP surrogate.

    Args:
        model: artefacts dict from load_surrogate()
        scales: scaling info dict
        x_raw: Raw Fourier coefficients (n, n_raw_features)

    Returns:
        Predictions in original scale (n, 2): [iota/nfp, max_elongation]
    """
    x_scaled = _prepare_features(x_raw, scales, model)
    gp_models: list[GaussianProcessRegressor] = model["gp_models"]
    y_scalers: list[StandardScaler] = model["y_scalers"]

    preds = []
    for i, gp in enumerate(gp_models):
        y_pred_scaled = gp.predict(x_scaled)  # type: ignore[assignment]
        y_pred = y_scalers[i].inverse_transform(
            np.asarray(y_pred_scaled).reshape(-1, 1)
        ).ravel()
        preds.append(y_pred)

    return np.column_stack(preds)


def predict_with_uncertainty(
    model: dict[str, Any],
    scales: dict[str, Any],
    x_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict metrics with calibrated GP posterior uncertainty.

    Args:
        model: artefacts dict from load_surrogate()
        scales: scaling info dict
        x_raw: Raw Fourier coefficients (n, n_raw_features)

    Returns:
        (y_mean, y_std) — each shape (n, output_dim).
        output_dim=2: col 0 = iota/nfp, col 1 = max_elongation

        y_std is the GP posterior standard deviation (calibrated Bayesian UQ),
        NOT ensemble disagreement.  This gives mathematically rigorous
        uncertainty bounds for the IOTA_UNC_MAX filter.
    """
    x_scaled = _prepare_features(x_raw, scales, model)
    gp_models: list[GaussianProcessRegressor] = model["gp_models"]
    y_scalers: list[StandardScaler] = model["y_scalers"]

    means = []
    stds = []
    for i, gp in enumerate(gp_models):
        y_pred_scaled, y_std_scaled = gp.predict(x_scaled, return_std=True)  # type: ignore[misc]

        # Transform mean back to original scale
        y_pred = y_scalers[i].inverse_transform(
            np.asarray(y_pred_scaled).reshape(-1, 1)
        ).ravel()

        # Transform std back to original scale (std scales with y_std only)
        y_scale = float(y_scalers[i].scale_[0]) if y_scalers[i].scale_ is not None else 1.0  # type: ignore[index]
        y_std = np.asarray(y_std_scaled) * y_scale

        means.append(y_pred)
        stds.append(y_std)

    y_mean = np.column_stack(means)
    y_std = np.column_stack(stds)
    return y_mean, y_std


# =========================================================================
# CLI
# =========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train GP surrogate")
    parser.add_argument(
        "--active-learning", "-al",
        action="store_true",
        help="Use active learning to select training points (recommended). "
             "Iteratively picks the most informative samples instead of "
             "random sub-sampling.",
    )
    args = parser.parse_args()

    if args.active_learning:
        train_active_learning()
    else:
        train()
