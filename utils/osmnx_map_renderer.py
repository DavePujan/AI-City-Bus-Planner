import os
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def generate_beautiful_map(routes, num_buses, base_lat, base_lon, original_image_data=None, output_path="outputs/beautiful_map.png"):
    """
    Generates a professional transit map using osmnx and networkx.
    Routes are snapped to the real street network.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Configure OSMnx to have a slightly longer timeout and fewer retries to fail fast
        ox.settings.timeout = 20
        ox.settings.max_query_area_size = 50 * 1000 * 50 * 1000 # 50km * 50km
        
        # 1. Download street network using a bounding box around the routes, or use the base lat/lon
        all_lats = []
        all_lons = []
        for route in routes.values():
            for lat, lon, _ in route:
                all_lats.append(lat)
                all_lons.append(lon)
                
        if not all_lats or not all_lons:
            # Fallback to city center box if no routes
            print("No valid route points. Using base lat/lon.")
            G = ox.graph_from_point((base_lat, base_lon), dist=5000, network_type='drive')
        else:
            # Safely get the center of all points
            center_lat = sum(all_lats) / len(all_lats)
            center_lon = sum(all_lons) / len(all_lons)
            
            # Find the max distance from center to any point (in roughly meters)
            # 1 degree lat ~ 111km, 1 degree lon ~ 111km * cos(lat)
            import math
            max_dist_m = 0
            for lat, lon in zip(all_lats, all_lons):
                dlat = (lat - center_lat) * 111000
                dlon = (lon - center_lon) * 111000 * math.cos(math.radians(center_lat))
                dist = math.sqrt(dlat**2 + dlon**2)
                if dist > max_dist_m:
                    max_dist_m = dist
            
            # Add a 1km buffer, but cap the total radius at 50km to prevent crashing overpass
            radius = min(max_dist_m + 1000, 50000)
            
            print(f"Downloading OSM graph with radius {radius}m...")
            try:
                G = ox.graph_from_point((center_lat, center_lon), dist=radius, network_type='drive')
            except Exception as e:
                print(f"Could not download OSM graph: {e}")
                return draw_fallback_map(routes, base_lat, base_lon, output_path)

        # Project graph
        G_proj = ox.project_graph(G)
        
        # 2. Plot base map
        fig, ax = ox.plot_graph(G_proj, show=False, close=False, edge_color='#333333', edge_linewidth=0.5, node_size=0, bgcolor='#111111', figsize=(12, 12))
        
        # 3. Snap route points to network nodes and compute shortest paths
        colors = plt.cm.get_cmap('hsv', len(routes))
        
        for i, (rid, route) in enumerate(routes.items()):
            color = colors(i)
            # Convert RGBA to hex
            hex_color = '#%02x%02x%02x' % (int(color[0]*255), int(color[1]*255), int(color[2]*255))
            
            route_nodes = []
            for lat, lon, _ in route:
                # Find the nearest node for each stop
                node = ox.distance.nearest_nodes(G, X=lon, Y=lat)
                route_nodes.append(node)
                
            if len(route_nodes) < 2:
                continue
                
            # Create full path along the network
            full_path = []
            for j in range(len(route_nodes) - 1):
                try:
                    # Compute shortest path between consecutive stops
                    path_segment = nx.shortest_path(G, route_nodes[j], route_nodes[j+1], weight='length')
                    
                    if not full_path:
                        full_path.extend(path_segment)
                    else:
                        full_path.extend(path_segment[1:]) # Avoid duplicating the connecting node
                except nx.NetworkXNoPath:
                    print(f"Warning: No path found between nodes {route_nodes[j]} and {route_nodes[j+1]} on route {rid}. Skipping segment.")
            
            if full_path:
                # Plot the route on top of the base map
                fig, ax = ox.plot_graph_route(G_proj, full_path, ax=ax, route_color=hex_color, route_linewidth=5, route_alpha=0.8, orig_dest_node_color=hex_color, orig_dest_node_size=50, show=False, close=False)
                
                # Plot stops as circles
                for lat, lon, _ in route:
                    # Project lat/lon to match the graph's CRS
                    # This is slightly tricky, so we plot the actual node we snapped to
                    pass

        # 4. Save figure
        plt.tight_layout(pad=0)
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        
        return output_path

    except Exception as e:
        print(f"❌ Error generating deterministic map: {e}")
        return draw_fallback_map(routes, base_lat, base_lon, output_path)

def draw_fallback_map(routes, base_lat, base_lon, output_path):
    """Fallback plotting just points and lines if OSMnx fails."""
    try:
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='#111111')
        ax.set_facecolor('#111111')
        
        colors = plt.cm.get_cmap('hsv', max(1, len(routes)))
        
        # Plot all paths
        for i, (rid, route) in enumerate(routes.items()):
            if len(route) < 2: continue
            color = colors(i)
            lats = [pt[0] for pt in route]
            lons = [pt[1] for pt in route]
            ax.plot(lons, lats, color=color, linewidth=5, alpha=0.8, marker='o', markersize=8)
            
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout(pad=0)
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return output_path
    except Exception as e:
        print(f"❌ Fallback map also failed: {e}")
        return None
