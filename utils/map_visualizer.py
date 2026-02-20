import folium
import os
from folium.plugins import PolyLineTextPath


def create_bus_maps(df, routes, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    center = [df["lat"].mean(), df["lon"].mean()]

    image_paths = []

    for rid, route in routes.items():
        try:
            m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
        except:
             m = folium.Map(location=center, zoom_start=12)

        pts = []
        for lat, lon, name in route:
            folium.Marker([lat, lon], popup=name).add_to(m)
            pts.append([lat, lon])

        line = folium.PolyLine(pts, weight=5)
        line.add_to(m)

        PolyLineTextPath(
            line,
            "➜",
            repeat=True,
            offset=7,
            attributes={"font-size": "16", "fill": "red"},
        ).add_to(m)
        
        if pts:
            m.fit_bounds(pts)

        file_path = os.path.join(output_dir, f"bus_route_{rid}.html")
        m.save(file_path)
        image_paths.append(file_path)

    return image_paths
