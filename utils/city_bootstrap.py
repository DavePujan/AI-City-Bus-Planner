"""
City bootstrap: fetch bus stops from OSM for any city + CSV cache.
Uses boundary-aware adaptive radius so large metros get wider coverage.
"""
import math
import os

import osmnx as ox
import pandas as pd
import streamlit as st

CACHE_DIR = "data/city_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# ── Radius estimation ──────────────────────────────────────────────────────────

def estimate_city_radius_km(city_name=None, lat=None, lon=None):
    """
    Compute a sensible OSM fetch radius (in metres) from the real city boundary.
    Falls back to 15 000 m if geocoding fails.
    """
    try:
        gdf = ox.geocode_to_gdf(city_name) if city_name else ox.geocode_to_gdf((lat, lon))
        area_km2 = gdf.to_crs(epsg=3857).area.iloc[0] / 1e6
        radius_km = math.sqrt(area_km2 / math.pi) * 1.25      # equivalent circle + 25 % buffer
        fetch_km  = max(5.0, min(radius_km, 40.0))             # clamp 5–40 km
        return int(fetch_km * 1000)
    except Exception:
        # Network unavailable — use metro-aware heuristic
        metro_hint = (city_name or "").lower()
        big_metros = ["mumbai", "london", "new york", "delhi", "beijing",
                      "shanghai", "tokyo", "jakarta", "cairo", "mexico"]
        if any(m in metro_hint for m in big_metros):
            print("[city_bootstrap] boundary unavailable → mega-metro fallback (25 km)")
            return 25_000
        print("[city_bootstrap] boundary unavailable → default fallback (15 km)")
        return 15_000


def get_zoom_from_radius(radius_m):
    """Map tile zoom level from fetch radius."""
    km = radius_m / 1000
    if km > 30: return 10
    if km > 15: return 11
    if km > 8:  return 12
    return 13


# ── OSM stop fetch ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=86_400)      # 24 h cache
def fetch_city_bus_stops(city_name, lat, lon, radius_m=15_000):
    """
    Pull bus-stop Points from OSM within `radius_m` metres of (lat, lon).
    Returns a DataFrame with columns [lat, lon, stop_name].
    """
    try:
        tags = {"highway": "bus_stop"}
        # features_from_point is the current OSMnx API (geometries_from_point deprecated)
        stops = ox.features_from_point((lat, lon), tags=tags, dist=radius_m)
        stops = stops[stops.geometry.type == "Point"].copy()
        stops["lat"] = stops.geometry.y
        stops["lon"] = stops.geometry.x
        stops["stop_name"] = (
            stops["name"].fillna("") if "name" in stops.columns
            else ""
        )
        stops["stop_name"] = stops.apply(
            lambda r: r["stop_name"] or f"Stop_{r.name}", axis=1
        )
        return stops[["lat", "lon", "stop_name"]].dropna().reset_index(drop=True)
    except Exception as e:
        print(f"[city_bootstrap] OSM fetch failed: {e}")
        return pd.DataFrame()


# ── Cache manager ──────────────────────────────────────────────────────────────

def get_city_cache_path(city_name):
    safe = city_name.lower().strip().replace(" ", "_")
    return os.path.join(CACHE_DIR, f"{safe}.csv")


def load_or_create_city(city_name, lat, lon):
    """
    Main entry point.
    1. Returns cached CSV if it exists.
    2. Otherwise fetches from OSM, saves CSV, returns DataFrame.

    Returns (df, from_cache: bool)
    """
    path = get_city_cache_path(city_name)

    if os.path.exists(path):
        df = pd.read_csv(path)
        if "stop_name" not in df.columns:
            df["stop_name"] = [f"Stop_{i}" for i in range(len(df))]
        return df, True

    radius_m = estimate_city_radius_km(city_name, lat, lon)
    df = fetch_city_bus_stops(city_name, lat, lon, radius_m=radius_m)

    if len(df) == 0:
        return df, False

    df.to_csv(path, index=False)
    return df, False
