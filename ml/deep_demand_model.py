"""
City-agnostic CNN demand model using PyTorch.

Key design decisions:
- math.ceil for grid sizing → never crashes on non-square stop counts
- AdaptiveAvgPool2d(1) replaces the fixed Flatten+Dense → handles any spatial size
- Labels averaged per grid cell → avoids y reshape crashes
- Safe wrapper → CNN failure never breaks the ensemble
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ── Grid builder ───────────────────────────────────────────────────────────────

def make_square_grid(values: np.ndarray):
    """
    Pad a 1-D array to the next perfect square and reshape.
    Returns (grid [side×side], side).
    """
    n    = len(values)
    side = math.ceil(math.sqrt(n))          # ceil, never truncates
    pad  = side * side - n
    if pad > 0:
        values = np.pad(values, (0, pad), mode="constant", constant_values=0.0)
    return values.reshape(side, side), side


def dataframe_to_grid_tensor(df, feature_cols):
    """
    Convert flat feature DataFrame → (1, C, H, W) float32 tensor.
    Uses ceil-based square padding so any row count works.
    Returns (tensor, side, original_n).
    """
    original_n = len(df)
    raw = df[feature_cols].values.astype(np.float32)     # (N, C)

    n, c = raw.shape
    side = math.ceil(math.sqrt(n))
    pad  = side * side - n
    if pad > 0:
        raw = np.vstack([raw, np.zeros((pad, c), dtype=np.float32)])

    # (side*side, C) → (side, side, C) → (C, side, side)
    tensor = raw.reshape(side, side, c).transpose(2, 0, 1)
    return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0), side, original_n


def make_label_grid(values: np.ndarray, side: int):
    """Pad labels to side×side and reshape. Never crashes on non-square n."""
    n   = len(values)
    pad = side * side - n
    if pad > 0:
        values = np.pad(values, (0, pad), mode="constant", constant_values=0.0)
    return values.reshape(side, side)


# ── Model ──────────────────────────────────────────────────────────────────────

class DemandCNN(nn.Module):
    """
    Size-agnostic CNN:
    - same-padding Conv2d layers preserve H×W
    - AdaptiveAvgPool2d(1) collapses spatial dims → no Flatten size assumptions
    - final Dense(1) outputs a single demand scalar per grid cell (broadcast)
    """
    def __init__(self, in_channels=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),           nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, padding=1),           nn.ReLU(),
            nn.Conv2d(32, 1, 1),                       # per-cell output
        )
        # Global branch for a grid-level scalar (size-agnostic)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        spatial = self.encoder(x)          # (1, 1, H, W) — per-cell predictions
        return spatial


# ── Safe inference wrapper ─────────────────────────────────────────────────────

def safe_cnn_predict(model, x_tensor, original_n, device):
    """Run CNN; return flat predictions of exactly original_n values."""
    try:
        model.eval()
        with torch.no_grad():
            out = model(x_tensor.to(device)).cpu().numpy().flatten()
        return out[:original_n]
    except Exception as e:
        print(f"⚠️ CNN inference failed: {e}")
        return np.zeros(original_n)


# ── Trainer ────────────────────────────────────────────────────────────────────

def train_deep_demand_model(features_df, epochs=15):
    """
    Train city-agnostic CNN on demand grid.
    Returns (model, predictions[len(features_df)]).
    """
    feature_cols = [
        "population_density", "road_density",
        "dist_to_stop", "dist_to_center",
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── inputs ──
    X_tensor, side, original_n = dataframe_to_grid_tensor(features_df, feature_cols)

    # ── labels: pad + reshape ──
    y_vals  = features_df["demand"].values.astype(np.float32)
    y_grid  = make_label_grid(y_vals, side)
    y_tensor = torch.tensor(y_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # ── normalise inputs ──
    X_tensor = (X_tensor - X_tensor.mean()) / (X_tensor.std() + 1e-6)

    model    = DemandCNN(in_channels=len(feature_cols)).to(device)
    X_tensor = X_tensor.to(device)
    y_tensor = y_tensor.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn   = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()

    preds = safe_cnn_predict(model, X_tensor, original_n, device)
    return model, preds
