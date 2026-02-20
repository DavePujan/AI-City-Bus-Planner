import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# ===============================
# GNN MODEL
# ===============================
class DemandGNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        return x.squeeze()


# ===============================
# GRAPH BUILDER
# ===============================
def build_graph(features_df, k=6):
    coords = features_df[["lat", "lon"]].values

    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(coords)
    _, indices = nbrs.kneighbors(coords)

    edges = []
    for i in range(len(indices)):
        for j in indices[i]:
            edges.append([i, j])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


# ===============================
# TRAINER
# ===============================
def train_gnn_demand_model(features_df, epochs=25):

    feature_cols = [
        "population_density",
        "road_density",
        "dist_to_stop",
        "dist_to_center"
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.tensor(
        features_df[feature_cols].values,
        dtype=torch.float32
    ).to(device)

    y = torch.tensor(
        features_df["demand"].values,
        dtype=torch.float32
    ).to(device)

    edge_index = build_graph(features_df).to(device)

    data = Data(x=X, edge_index=edge_index, y=y)

    model = DemandGNN(in_channels=len(feature_cols)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(data.x, data.edge_index)
        loss = loss_fn(preds, data.y)
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        preds = model(data.x, data.edge_index).cpu().numpy()

    return model, preds
