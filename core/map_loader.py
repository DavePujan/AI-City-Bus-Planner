import geopandas as gpd
import pandas as pd


def load_geojson(uploaded_file):
    gdf = gpd.read_file(uploaded_file)

    if len(gdf) == 0:
        raise ValueError("GeoJSON has no features.")

    # Ensure geometry is Point
    if not all(gdf.geometry.type == "Point"):
        raise ValueError("GeoJSON must contain Point features (bus stops).")

    # Stop name fallback
    if "name" in gdf.columns:
        names = gdf["name"]
    else:
        names = [f"Stop_{i}" for i in range(len(gdf))]

    df = pd.DataFrame({
        "stop_name": names,
        "lat": gdf.geometry.y,
        "lon": gdf.geometry.x
    })

    return df
