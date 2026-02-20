from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


def cluster_stops(df, n_clusters):
    """
    Safe clustering that does NOT mutate input.
    Enforces a minimum cluster size of 3 stops so tiny
    isolated fragments don't appear as routes.
    """

    if len(df) == 0:
        raise ValueError("No stops provided for clustering.")

    n_clusters = min(n_clusters, max(1, len(df)))

    coords = df[["lat", "lon"]].values

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(coords)

    out = df.copy()
    out["cluster"] = labels

    # Remove tiny clusters (< 3 stops); reassign to nearest valid centroid
    MIN_SIZE = 3
    counts = out["cluster"].value_counts()
    valid_clusters = counts[counts >= MIN_SIZE].index

    if len(valid_clusters) == 0:
        # All clusters tiny — just keep everything as-is
        return out

    if len(valid_clusters) < n_clusters:
        # Reassign orphan stops to nearest valid cluster centroid
        centroids = (
            out[out["cluster"].isin(valid_clusters)]
            .groupby("cluster")[["lat", "lon"]]
            .mean()
        )
        mask = ~out["cluster"].isin(valid_clusters)
        for i in out[mask].index:
            pt = out.loc[i, ["lat", "lon"]].values
            dists = np.linalg.norm(centroids.values - pt, axis=1)
            out.at[i, "cluster"] = centroids.index[np.argmin(dists)]

    return out
