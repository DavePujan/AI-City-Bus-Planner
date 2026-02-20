import numpy as np
from sklearn.neighbors import KDTree


def build_features(grid_df, stops_df):
    features = grid_df.copy()

    # -----------------------------
    # Synthetic population proxy
    # -----------------------------
    features["population_density"] = np.random.randint(1000, 10000, len(features))

    # -----------------------------
    # Road density proxy
    # -----------------------------
    features["road_density"] = np.random.uniform(0.5, 5.0, len(features))

    # -----------------------------
    # FAST nearest stop using KDTree
    # -----------------------------
    stop_coords = stops_df[["lat", "lon"]].values
    grid_coords = features[["lat", "lon"]].values

    tree = KDTree(stop_coords)
    dist, _ = tree.query(grid_coords, k=1)

    features["dist_to_stop"] = dist.flatten()

    # -----------------------------
    # City center distance
    # -----------------------------
    center_lat = stops_df["lat"].mean()
    center_lon = stops_df["lon"].mean()

    features["dist_to_center"] = np.sqrt(
        (features["lat"] - center_lat) ** 2 +
        (features["lon"] - center_lon) ** 2
    )

    return features
