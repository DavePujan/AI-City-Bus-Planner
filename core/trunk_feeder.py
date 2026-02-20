import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def generate_trunk_feeder_network(
    stops_df,
    demand_df,
    corridors_df,
    trunk_buffer_km=1.2,
):
    """
    Build a trunk + feeder route structure.

    Trunk stops = those near high-demand corridor spine.
    Feeder stops = all others, clustered locally.

    Returns
    -------
    trunk_routes  : dict  {route_id: [(lat, lon, name), ...]}
    feeder_routes : dict  {route_id: [(lat, lon, name), ...]}
    role_df       : DataFrame with added 'route_role' column
    """
    if corridors_df is None or len(corridors_df) == 0:
        return {}, {}, pd.DataFrame()

    stops = stops_df.copy()
    if "stop_name" not in stops.columns:
        stops["stop_name"] = [f"Stop_{i}" for i in range(len(stops))]

    # High-demand spine points (top 15%)
    high_demand = demand_df[
        demand_df["final_demand"] >= demand_df["final_demand"].quantile(0.85)
    ][["lat", "lon"]].values

    if len(high_demand) == 0:
        return {}, {}, pd.DataFrame()

    buffer_deg = trunk_buffer_km / 111.0

    def _near_trunk(row):
        dists = np.sqrt(
            (high_demand[:, 0] - row["lat"]) ** 2 +
            (high_demand[:, 1] - row["lon"]) ** 2
        )
        return dists.min() <= buffer_deg

    stops["near_trunk"] = stops.apply(_near_trunk, axis=1)

    trunk_stops  = stops[stops["near_trunk"]].copy()
    feeder_stops = stops[~stops["near_trunk"]].copy()

    trunk_routes  = {}
    feeder_routes = {}

    # ── Trunk route ──────────────────────────────────────────────
    if len(trunk_stops) >= 4:
        ordered = trunk_stops.sort_values(["lat", "lon"])
        trunk_routes["TRUNK_0"] = [
            (row.lat, row.lon, row.stop_name)
            for row in ordered.itertuples()
        ]

    # ── Feeder clusters ──────────────────────────────────────────
    if len(feeder_stops) >= 4:
        k = max(2, min(len(feeder_stops) // 8, 10))
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(
            feeder_stops[["lat", "lon"]]
        )
        feeder_stops = feeder_stops.copy()
        feeder_stops["_fcid"] = labels

        for cid, grp in feeder_stops.groupby("_fcid"):
            if len(grp) < 2:
                continue
            ordered = grp.sort_values(["lat", "lon"])
            feeder_routes[f"FEEDER_{cid}"] = [
                (row.lat, row.lon, row.stop_name)
                for row in ordered.itertuples()
            ]

    # ── Role table ────────────────────────────────────────────────
    role_df = stops.copy()
    role_df["route_role"] = np.where(role_df["near_trunk"], "trunk", "feeder")

    return trunk_routes, feeder_routes, role_df
