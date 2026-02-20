import numpy as np


def haversine_m(lat1, lon1, lat2, lon2):
    """
    Distance in meters between two lat/lon points.
    """
    R = 6371000  # Earth radius (m)

    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2 +
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(a))


def optimize_stop_spacing(
    routes,
    demand_df=None,
    min_spacing_m=400,
    high_demand_quantile=0.75,
):
    """
    Remove redundant stops while preserving high-demand ones.
    Enforces minimum spacing of `min_spacing_m` metres between stops.
    """

    optimized_routes = {}

    # build demand lookup: (lat, lon) -> bool (is high demand?)
    demand_lookup = {}
    if demand_df is not None and "final_demand" in demand_df.columns:
        q = demand_df["final_demand"].quantile(high_demand_quantile)
        for _, r in demand_df.iterrows():
            demand_lookup[(round(r["lat"], 5), round(r["lon"], 5))] = (
                r["final_demand"] >= q
            )

    for rid, route in routes.items():
        if len(route) <= 2:
            optimized_routes[rid] = route
            continue

        new_route = [route[0]]  # always keep first
        last_kept = route[0]

        for stop in route[1:-1]:
            lat, lon, name = stop

            # distance from last kept stop
            d = haversine_m(last_kept[0], last_kept[1], lat, lon)

            is_high_demand = demand_lookup.get((round(lat, 5), round(lon, 5)), False)

            # keep if far enough OR high demand
            if d >= min_spacing_m or is_high_demand:
                new_route.append(stop)
                last_kept = stop

        new_route.append(route[-1])  # always keep last
        optimized_routes[rid] = new_route

    return optimized_routes
