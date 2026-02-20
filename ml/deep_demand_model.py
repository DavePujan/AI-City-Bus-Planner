import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ===============================
# CNN MODEL
# ===============================
class DemandCNN(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        return self.net(x)


# ===============================
# GRID RESHAPER
# ===============================
def dataframe_to_grid_tensor(df, feature_cols):
    """
    Converts flat grid -> spatial tensor.
    Assumes grid is roughly square.
    """

    size = int(np.sqrt(len(df)))
    if size * size != len(df):
        # pad to square
        pad_needed = size * size - len(df)
        if pad_needed < 0:
            size += 1
            pad_needed = size * size - len(df)

        pad_df = df.sample(pad_needed, replace=True)
        df = np.concatenate([df[feature_cols].values,
                             pad_df[feature_cols].values])
    else:
        df = df[feature_cols].values

    tensor = df.reshape(size, size, len(feature_cols))
    tensor = np.transpose(tensor, (2, 0, 1))  # C,H,W
    return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0), size


# ===============================
# TRAINER
# ===============================
def train_deep_demand_model(features_df, epochs=15):

    feature_cols = [
        "population_density",
        "road_density",
        "dist_to_stop",
        "dist_to_center"
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_tensor, grid_size = dataframe_to_grid_tensor(features_df, feature_cols)

    y = features_df["demand"].values
    y_tensor = torch.tensor(y.reshape(grid_size, grid_size),
                            dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    model = DemandCNN(in_channels=len(feature_cols)).to(device)
    X_tensor = X_tensor.to(device)
    y_tensor = y_tensor.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_tensor)
        loss = loss_fn(preds, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        pred_grid = model(X_tensor).cpu().numpy().flatten()

    return model, pred_grid[:len(features_df)]
