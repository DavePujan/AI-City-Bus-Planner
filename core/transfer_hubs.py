import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def optimize_transfer_hubs(
    stops_df,
    trunk_routes,
    feeder_routes,
    hub_radius_m=500,
):
    """
    Detect transfer hubs at trunk-stop clusters and snap feeder
    route endpoints to the nearest hub.

    Routes use (lat, lon, name) tuples throughout.

    Returns
    -------
    hubs_df          : DataFrame with hub_id, lat, lon, n_trunk_points
    updated_feeders  : feeder_routes with hub stop prepended where needed
    """
    if not trunk_routes or not feeder_routes:
        return pd.DataFrame(), feeder_routes

    # ── 1. Collect all trunk stop coordinates ─────────────────────
    trunk_points = [
        [lat, lon]
        for route in trunk_routes.values()
        for (lat, lon, *_) in route          # handles (lat, lon, name) tuples
    ]

    if not trunk_points:
        return pd.DataFrame(), feeder_routes

    trunk_arr = np.array(trunk_points)
    eps_deg   = hub_radius_m / 111_000.0

    # ── 2. DBSCAN to form hub clusters ───────────────────────────
    labels = DBSCAN(eps=eps_deg, min_samples=3).fit_predict(trunk_arr)

    hubs = []
    for hid in np.unique(labels):
        if hid < 0:
            continue
        grp = trunk_arr[labels == hid]
        hubs.append({
            "hub_id":          f"HUB_{hid}",
            "lat":             grp[:, 0].mean(),
            "lon":             grp[:, 1].mean(),
            "n_trunk_points":  len(grp),
        })

    hubs_df = pd.DataFrame(hubs)

    if hubs_df.empty:
        return hubs_df, feeder_routes

    hub_coords = hubs_df[["lat", "lon"]].values

    # ── 3. Snap each feeder start to nearest hub ──────────────────
    updated_feeders = {}
    for rid, route in feeder_routes.items():
        if not route:
            updated_feeders[rid] = route
            continue

        start_lat, start_lon = route[0][0], route[0][1]
        dists = np.linalg.norm(hub_coords - [start_lat, start_lon], axis=1)
        nearest_idx = int(np.argmin(dists))

        if dists[nearest_idx] > eps_deg:
            hub_lat  = hub_coords[nearest_idx, 0]
            hub_lon  = hub_coords[nearest_idx, 1]
            hub_name = hubs_df.iloc[nearest_idx]["hub_id"]
            updated_feeders[rid] = [(hub_lat, hub_lon, hub_name)] + list(route)
        else:
            updated_feeders[rid] = route

    return hubs_df, updated_feeders
