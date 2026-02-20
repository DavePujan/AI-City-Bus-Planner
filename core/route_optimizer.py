import numpy as np
from functools import lru_cache

# ~8 km max single hop before we force best available
MAX_HOP = 0.08


def _dist(a, b):
    """
    Penalized distance to discourage long water/sea crossings.
    """
    euclid = np.hypot(a[0] - b[0], a[1] - b[1])

    # soft penalty for very long jumps (~5 km threshold)
    if euclid > 0.05:
        euclid *= 3.0

    return euclid


@lru_cache(maxsize=200000)
def _cached_dist(a_lat, a_lon, b_lat, b_lon):
    return np.hypot(a_lat - b_lat, a_lon - b_lon)


def optimize_routes(df):
    """
    Improved greedy TSP with caching, hop guard, and better start selection.
    """

    routes = {}

    for cid in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == cid]

        if len(cluster_df) < 2:
            continue

        points = cluster_df[["lat", "lon", "stop_name"]].values.tolist()

        # better start: farthest from mean
        centroid = (
            cluster_df["lat"].mean(),
            cluster_df["lon"].mean()
        )

        start_idx = max(
            range(len(points)),
            key=lambda i: _dist(
                centroid,
                (points[i][0], points[i][1])
            )
        )

        route = [points.pop(start_idx)]

        # greedy chain with cached distances + hop guard
        while points:
            last = route[-1]

            # prefer reachable stops within MAX_HOP
            candidates = [
                i for i in range(len(points))
                if _cached_dist(last[0], last[1],
                                points[i][0], points[i][1]) <= MAX_HOP
            ]

            # fallback: use all if no close neighbor
            if not candidates:
                candidates = list(range(len(points)))

            next_idx = min(
                candidates,
                key=lambda i: _cached_dist(
                    last[0], last[1],
                    points[i][0], points[i][1]
                )
            )

            route.append(points.pop(next_idx))

        routes[cid] = route

    return routes
