import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox

from sklearn.cluster import KMeans


# =====================================================
# BUILD ROAD GRAPH
# =====================================================
def build_road_graph(center_lat, center_lon, dist=3000):
    """
    Downloads drivable road network around city.
    """

    G = ox.graph_from_point(
        (center_lat, center_lon),
        dist=dist,
        network_type="drive"
    )

    return G


# =====================================================
# SELECT HIGH DEMAND CELLS
# =====================================================
def select_hotspots(features_df, top_pct=0.25):
    """
    Keep top demand zones.
    """

    threshold = features_df["final_demand"].quantile(1 - top_pct)
    hotspots = features_df[features_df["final_demand"] >= threshold].copy()

    return hotspots


# =====================================================
# CLUSTER INTO BUS ROUTES
# =====================================================
def cluster_hotspots(hotspots, num_buses):
    coords = hotspots[["lat", "lon"]].values

    kmeans = KMeans(n_clusters=num_buses, random_state=42, n_init=10)
    hotspots["cluster"] = kmeans.fit_predict(coords)

    return hotspots


# =====================================================
# SNAP TO ROAD NODES
# =====================================================
def snap_to_graph(G, df):
    node_ids = []

    for _, row in df.iterrows():
        node = ox.nearest_nodes(G, row["lon"], row["lat"])
        node_ids.append(node)

    df = df.copy()
    df["node"] = node_ids
    return df


# =====================================================
# BUILD ROUTES
# =====================================================
def build_routes_from_clusters(G, clustered_df, num_buses):
    routes = []

    for bus_id in range(num_buses):

        group = clustered_df[clustered_df["cluster"] == bus_id]

        if len(group) < 2:
            continue

        nodes = group["node"].unique().tolist()

        # greedy path chaining
        route_nodes = [nodes[0]]

        remaining = set(nodes[1:])

        while remaining:
            last = route_nodes[-1]

            next_node = min(
                remaining,
                key=lambda n: nx.shortest_path_length(G, last, n, weight="length")
            )

            path = nx.shortest_path(G, last, next_node, weight="length")

            route_nodes.extend(path[1:])
            remaining.remove(next_node)

        routes.append({
            "bus_id": bus_id,
            "nodes": route_nodes
        })

    return routes


# =====================================================
# MAIN OPTIMIZER
# =====================================================
def optimize_bus_routes(features_df, num_buses=5):

    print("🧠 Optimizing routes using demand...")

    center_lat = features_df["lat"].mean()
    center_lon = features_df["lon"].mean()

    # 1. road graph
    G = build_road_graph(center_lat, center_lon)

    # 2. hotspots
    hotspots = select_hotspots(features_df)

    # 3. clustering
    clustered = cluster_hotspots(hotspots, num_buses)

    # 4. snap to roads
    snapped = snap_to_graph(G, clustered)

    # 5. build routes
    routes = build_routes_from_clusters(G, snapped, num_buses)

    print(f"✅ Generated {len(routes)} optimized routes")

    return G, routes

def routes_to_latlon(G, routes):

    final_routes = {}

    for r in routes:
        coords = [
            (G.nodes[n]["y"], G.nodes[n]["x"])
            for n in r["nodes"]
        ]

        final_routes[r["bus_id"]] = coords

    return final_routes
