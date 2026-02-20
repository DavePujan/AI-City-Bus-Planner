import numpy as np
import pandas as pd
import networkx as nx


def optimize_transfer_hubs(
    stops_df,
    trunk_routes,
    feeder_routes,
    max_hubs=6,
    min_spacing_m=1200,
):
    """
    Detect transfer hubs using transit-network betweenness centrality.

    Builds a stop-connectivity graph from all routes, computes betweenness
    centrality, adds a trunk-bonus, then selects well-spaced high-centrality
    nodes as formal interchange hubs.  Feeder start-points are then snapped
    to the nearest hub.

    Parameters
    ----------
    stops_df       : DataFrame with lat/lon columns
    trunk_routes   : {rid: [(lat, lon, name), ...]}
    feeder_routes  : {rid: [(lat, lon, name), ...]}
    max_hubs       : max number of hubs to return
    min_spacing_m  : minimum distance between hubs (metres)

    Returns
    -------
    hubs_df          : DataFrame – hub_id, lat, lon, centrality, n_trunk_stops
    updated_feeders  : feeder_routes with hub stop prepended where needed
    """
    all_routes = {**trunk_routes, **feeder_routes}

    if not all_routes:
        return pd.DataFrame(), feeder_routes

    # ── 1. Build transit graph from route connections ─────────────
    G = nx.Graph()

    def _sid(stop):
        """Stable string node-id from lat/lon rounded to ~1 m."""
        return f"{round(stop[0], 5)}_{round(stop[1], 5)}"

    for route in all_routes.values():
        for stop in route:
            sid = _sid(stop)
            G.add_node(sid, lat=stop[0], lon=stop[1])

        stop_ids = [_sid(s) for s in route]
        for i in range(len(stop_ids) - 1):
            if G.has_edge(stop_ids[i], stop_ids[i + 1]):
                G[stop_ids[i]][stop_ids[i + 1]]["weight"] += 1
            else:
                G.add_edge(stop_ids[i], stop_ids[i + 1], weight=1)

    if len(G.nodes) < 3:
        return pd.DataFrame(), feeder_routes

    # ── 2. Betweenness centrality (approximate for speed) ─────────
    k_sample = min(300, len(G.nodes))
    centrality = nx.betweenness_centrality(
        G, k=k_sample, normalized=True, seed=42
    )

    # ── 3. Trunk-stop set for bonus scoring ───────────────────────
    trunk_sids = {
        _sid(stop)
        for route in trunk_routes.values()
        for stop in route
    }

    # ── 4. Score each node ────────────────────────────────────────
    node_scores = []
    for sid, cent in centrality.items():
        trunk_bonus = 0.35 if sid in trunk_sids else 0.0
        node_scores.append((
            sid,
            cent + trunk_bonus,
            G.nodes[sid]["lat"],
            G.nodes[sid]["lon"],
        ))

    node_scores.sort(key=lambda x: x[1], reverse=True)

    # ── 5. Greedy spatial selection ───────────────────────────────
    eps_deg = min_spacing_m / 111_000.0
    selected = []

    for sid, score, lat, lon in node_scores:
        too_close = any(
            (((lat - hlat) ** 2 + (lon - hlon) ** 2) ** 0.5) < eps_deg
            for _, _, hlat, hlon in selected
        )
        if not too_close:
            selected.append((sid, score, lat, lon))
        if len(selected) >= max_hubs:
            break

    if not selected:
        return pd.DataFrame(), feeder_routes

    # ── 6. Build hubs DataFrame ───────────────────────────────────
    rows = []
    for i, (sid, score, lat, lon) in enumerate(selected):
        n_trunk = sum(
            1 for route in trunk_routes.values()
            for stop in route
            if _sid(stop) == sid
        )
        rows.append({
            "hub_id":        f"HUB_{i}",
            "lat":           lat,
            "lon":           lon,
            "centrality":    round(score, 4),
            "n_trunk_stops": n_trunk,
        })

    hubs_df = pd.DataFrame(rows)
    hub_coords = hubs_df[["lat", "lon"]].values

    # ── 7. Snap feeder start-points to nearest hub ────────────────
    updated_feeders = {}
    for rid, route in feeder_routes.items():
        if not route:
            updated_feeders[rid] = route
            continue

        start = np.array([route[0][0], route[0][1]])
        dists = np.linalg.norm(hub_coords - start, axis=1)
        ni = int(np.argmin(dists))

        if dists[ni] > eps_deg:
            hub_lat  = hub_coords[ni, 0]
            hub_lon  = hub_coords[ni, 1]
            hub_name = hubs_df.iloc[ni]["hub_id"]
            updated_feeders[rid] = [(hub_lat, hub_lon, hub_name)] + list(route)
        else:
            updated_feeders[rid] = route

    return hubs_df, updated_feeders
