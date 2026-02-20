import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


def detect_demand_corridors(
    demand_df,
    demand_col="final_demand",
    percentile_threshold=80,
    eps_km=1.2,
    min_samples=6,
):
    """
    Detect linear high-demand corridors using DBSCAN + PCA linearity check.

    Returns a DataFrame with one row per detected corridor, including:
        corridor_id, n_cells, linearity, length_km,
        demand_score, is_brt_candidate
    Returns empty DataFrame if no corridors found.
    """
    if demand_df is None or len(demand_df) < min_samples:
        return pd.DataFrame()

    thresh = np.percentile(demand_df[demand_col], percentile_threshold)
    high = demand_df[demand_df[demand_col] >= thresh].copy()

    if len(high) < min_samples:
        return pd.DataFrame()

    coords = high[["lat", "lon"]].values
    eps_deg = eps_km / 111.0

    labels = DBSCAN(eps=eps_deg, min_samples=min_samples).fit_predict(coords)
    high["corridor_id"] = labels
    high = high[high["corridor_id"] >= 0]

    if len(high) == 0:
        return pd.DataFrame()

    rows = []
    for cid, group in high.groupby("corridor_id"):
        if len(group) < min_samples:
            continue

        pts = group[["lat", "lon"]].values
        pca = PCA(n_components=2)
        pca.fit(pts)
        linearity = pca.explained_variance_ratio_[0]

        if linearity < 0.70:   # must be roughly linear
            continue

        length_km = _bbox_diagonal_km(group)
        demand_sum = group[demand_col].sum()

        rows.append({
            "corridor_id":      int(cid),
            "n_cells":          len(group),
            "linearity":        round(linearity, 3),
            "length_km":        round(length_km, 2),
            "demand_score":     round(demand_sum, 1),
            "is_brt_candidate": bool(length_km >= 6 and demand_sum > 500),
        })

    return pd.DataFrame(rows)


def _bbox_diagonal_km(group):
    lat_span = group["lat"].max() - group["lat"].min()
    lon_span = group["lon"].max() - group["lon"].min()
    return np.sqrt(lat_span**2 + lon_span**2) * 111.0
