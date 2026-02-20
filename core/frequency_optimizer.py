import numpy as np


def optimize_route_frequency(
    routes,
    allocation,
    demand_df,
    operating_hours,
    bus_capacity=60,
    target_load_factor=0.75,
    min_headway_min=5,
    max_headway_min=60,
):
    """
    Demand-driven frequency optimization.

    Converts demand -> required buses -> realistic headways.
    Falls back to existing allocation if no demand data available.
    """

    if demand_df is None or "final_demand" not in demand_df.columns:
        return allocation

    if "cluster" not in demand_df.columns:
        return allocation

    optimized = {}

    for rid, route in routes.items():

        # Route demand estimation
        cluster_mask = demand_df["cluster"] == rid
        route_demand = demand_df.loc[cluster_mask, "final_demand"].sum()

        # passengers per hour (rough normalization)
        passengers_per_hour = route_demand / max(1, operating_hours)

        # Required frequency
        effective_capacity = bus_capacity * target_load_factor

        required_buses = max(
            1,
            int(np.ceil(passengers_per_hour / effective_capacity))
        )

        # Convert to headway
        headway_hours = operating_hours / required_buses
        headway_min = headway_hours * 60

        # clamp to realistic bounds
        headway_min = max(min_headway_min, headway_min)
        headway_min = min(max_headway_min, headway_min)

        optimized[rid] = {
            "bus_id": f"BUS_{rid}",
            "buses_assigned": required_buses,
            "estimated_headway_hours": round(headway_min / 60, 2),
            "estimated_headway_min": round(headway_min, 1),
            "route_demand": round(route_demand, 1),
            "passengers_per_hour": round(passengers_per_hour, 1),
            "capacity_utilization": round(
                min(1.0, passengers_per_hour / max(1, effective_capacity)),
                2,
            ),
        }

    return optimized
