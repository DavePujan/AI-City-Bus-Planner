
def allocate_buses(routes, total_buses, operating_hours, demand_df=None):
    """
    Smart allocation based on route demand.
    Falls back to equal split if no demand data available.
    """

    if not routes:
        return {}

    # fallback: equal split
    if demand_df is None or "final_demand" not in demand_df.columns:
        n_routes = len(routes)
        base = max(1, total_buses // n_routes)
        remainder = total_buses % n_routes

        allocation = {}
        for i, rid in enumerate(routes.keys()):
            buses = base + (1 if i < remainder else 0)
            headway = operating_hours / buses if buses else operating_hours

            allocation[rid] = {
                "bus_id": f"BUS_{rid}",
                "buses_assigned": buses,
                "estimated_headway_hours": round(headway, 2)
            }

        return allocation

    # 🔥 demand-weighted allocation
    route_demands = {}

    for rid in routes.keys():
        if "cluster" in demand_df.columns:
            cluster_demand = demand_df[
                demand_df["cluster"] == rid
            ]["final_demand"].sum()
        else:
            cluster_demand = 1.0

        route_demands[rid] = max(cluster_demand, 1)

    total_demand = sum(route_demands.values())

    allocation = {}

    for rid, d in route_demands.items():
        share = d / total_demand
        buses = max(1, int(round(share * total_buses)))
        headway = operating_hours / buses

        allocation[rid] = {
            "bus_id": f"BUS_{rid}",
            "buses_assigned": buses,
            "estimated_headway_hours": round(headway, 2)
        }

    return allocation
