import numpy as np
import pandas as pd


def create_spatial_grid(df, grid_size=0.01):
    """
    Converts stop coordinates into spatial grid cells.
    grid_size ~ 0.01 ≈ ~1km (rough)
    """

    min_lat, max_lat = df["lat"].min(), df["lat"].max()
    min_lon, max_lon = df["lon"].min(), df["lon"].max()

    lat_bins = np.arange(min_lat, max_lat + grid_size, grid_size)
    lon_bins = np.arange(min_lon, max_lon + grid_size, grid_size)

    grid_points = []

    grid_id = 0
    for lat in lat_bins:
        for lon in lon_bins:
            grid_points.append({
                "grid_id": grid_id,
                "lat": lat,
                "lon": lon
            })
            grid_id += 1

    return pd.DataFrame(grid_points)
