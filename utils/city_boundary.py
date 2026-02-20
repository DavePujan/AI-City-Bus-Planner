import osmnx as ox
from shapely.geometry import Point


def get_city_boundary(city_name=None, lat=None, lon=None):
    """
    Returns city boundary GeoDataFrame (EPSG:4326).
    Accepts city name string OR lat/lon coordinates.
    """
    try:
        if city_name:
            gdf = ox.geocode_to_gdf(city_name)
        elif lat is not None and lon is not None:
            gdf = ox.geocode_to_gdf(f"{lat},{lon}")
        else:
            raise ValueError("Provide city_name or lat/lon")
        return gdf.to_crs(epsg=4326)
    except Exception as e:
        print(f"[city_boundary] Boundary fetch failed: {e}")
        return None


def clip_points_to_boundary(df, boundary_gdf):
    """
    Filter DataFrame rows to only those whose (lat, lon) falls
    inside the city boundary polygon. Returns a copy.
    """
    if boundary_gdf is None or len(boundary_gdf) == 0:
        return df

    boundary = boundary_gdf.geometry.unary_union  # handles MultiPolygon

    mask = df.apply(
        lambda r: boundary.contains(Point(r["lon"], r["lat"])),
        axis=1
    )
    return df[mask].copy()
