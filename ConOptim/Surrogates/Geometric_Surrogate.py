import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets import load_dataset

# --- KONFIGURATION ---
SAVE_PATH = "stellarator_surrogate.pth"
SCALING_PATH = "feature_scales.npy"
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-4
SCALING_DATA_PATH = "feature_scales.npy"
MODEL_PATH = "stellarator_surrogate.pth"

def load_surrogate():
    scales = np.load(SCALING_DATA_PATH, allow_pickle=True).item()
    model = SurrogateModel()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model, scales

# 1. DATEN LADEN & PREPROCESSING
def prepare_data():
    print("Lade Dataset 'proxima-fusion/constellaration'...")
    ds = load_dataset("proxima-fusion/constellaration", split="train")
    ds = ds.filter(lambda x: x["boundary.n_field_periods"] == 3)

    X, Y = [], []
    for row in ds:
        # Extrahiere Zielgrößen
        targets = [
            row["metrics.max_elongation"],
            row["metrics.aspect_ratio"],
            row["metrics.average_triangularity"],
            row["metrics.edge_rotational_transform_over_n_field_periods"]
        ]

        if any(t is None for t in targets):
            continue

        # Input: Flatten R_cos und Z_sin
        r_cos = np.array(row["boundary.r_cos"]).flatten()
        z_sin = np.array(row["boundary.z_sin"]).flatten()

        X.append(np.concatenate([r_cos, z_sin]))
        Y.append(targets)

    # Konvertierung in reine Float-Arrays vor dem Tensor-Erstellen
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    # Normalisierung
    x_mean, x_std = X.mean(axis=0), X.std(axis=0)
    x_std[x_std == 0] = 1.0
    np.save(SCALING_PATH, {"mean": x_mean, "std": x_std})

    X_norm = (X - x_mean) / x_std

    # Jetzt ist die Konvertierung in Tensors sicher
    return torch.tensor(X_norm), torch.tensor(Y)

# 2. MODELL-ARCHITEKTUR
class SurrogateModel(nn.Module):
    def __init__(self, input_dim=90, output_dim=4):
        super(SurrogateModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# 3. TRAINING-LOOP
def train():
    X, Y = prepare_data()
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SurrogateModel()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    model, _ = load_surrogate()
    print(f"Starte Training auf {X.shape[0]} Beispielen...")
    model.train()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{EPOCHS} | Loss: {total_loss / len(loader):.6f}")

    # Speichern
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nTraining abgeschlossen. Modell gespeichert als: {SAVE_PATH}")
    print(f"Feature-Skalierung gespeichert als: {SCALING_PATH}")


if __name__ == "__main__":
    train()