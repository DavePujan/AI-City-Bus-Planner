def compute_city_metrics(boundary_gdf):
    """
    Compute physical metrics of the city boundary.
    Returns dict with area_km2, or None if boundary unavailable.
    """
    if boundary_gdf is None or len(boundary_gdf) == 0:
        return None
    try:
        projected = boundary_gdf.to_crs(epsg=3857)
        area_km2 = projected.area.iloc[0] / 1e6
        return {"area_km2": round(area_km2, 1)}
    except Exception:
        return None


def classify_city_scale(area_km2):
    """
    Classify city into one of four scale tiers.
    Thresholds tuned for global major cities.
    """
    if area_km2 >= 800:
        return "mega_metro"
    elif area_km2 >= 300:
        return "large_metro"
    elif area_km2 >= 120:
        return "medium_city"
    else:
        return "small_city"


# Auto-tuned service design parameters per city scale
_PRESETS = {
    "mega_metro": {
        "stop_spacing":       350,
        "target_load":        0.85,
        "peak_boost":         1.8,
        "offpeak_factor":     0.55,
        "cluster_factor":     0.45,
        "demand_multiplier":  1.4,
    },
    "large_metro": {
        "stop_spacing":       400,
        "target_load":        0.80,
        "peak_boost":         1.7,
        "offpeak_factor":     0.60,
        "cluster_factor":     0.50,
        "demand_multiplier":  1.2,
    },
    "medium_city": {
        "stop_spacing":       450,
        "target_load":        0.75,
        "peak_boost":         1.5,
        "offpeak_factor":     0.65,
        "cluster_factor":     0.60,
        "demand_multiplier":  1.0,
    },
    "small_city": {
        "stop_spacing":       500,
        "target_load":        0.70,
        "peak_boost":         1.3,
        "offpeak_factor":     0.70,
        "cluster_factor":     0.70,
        "demand_multiplier":  0.8,
    },
}


def get_auto_parameters(city_scale):
    """Return design preset for the given city scale."""
    return _PRESETS.get(city_scale, _PRESETS["medium_city"])
