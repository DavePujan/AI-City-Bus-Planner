import pandas as pd
import re

TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")


def _is_valid_time(t):
    return isinstance(t, str) and bool(TIME_RE.match(t))


def validate_and_fix_gtfs(output_dir):
    """
    Validates GTFS files and auto-fixes common issues.
    Returns a list of warning strings (empty = clean feed).
    """

    warnings = []

    stops = pd.read_csv(f"{output_dir}/stops.txt")
    routes = pd.read_csv(f"{output_dir}/routes.txt")
    trips = pd.read_csv(f"{output_dir}/trips.txt")
    stop_times = pd.read_csv(f"{output_dir}/stop_times.txt")

    # 1. Remove duplicate stop_ids
    before = len(stops)
    stops = stops.drop_duplicates(subset=["stop_id"])
    if len(stops) < before:
        warnings.append(f"Removed {before - len(stops)} duplicate stop_id(s).")

    # 2. Remove trips with fewer than 2 stops
    valid_trip_ids = (
        stop_times.groupby("trip_id").size().loc[lambda x: x >= 2].index
    )
    removed = set(trips["trip_id"]) - set(valid_trip_ids)
    if removed:
        warnings.append(f"Removed {len(removed)} trip(s) with < 2 stops.")
    trips = trips[trips["trip_id"].isin(valid_trip_ids)]
    stop_times = stop_times[stop_times["trip_id"].isin(valid_trip_ids)]

    # 3. Ensure stop_sequence is sorted
    stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])

    # 4. Fix invalid time formats
    for col in ["arrival_time", "departure_time"]:
        bad = ~stop_times[col].apply(_is_valid_time)
        if bad.any():
            default = "00:00:00" if col == "arrival_time" else "00:00:30"
            warnings.append(f"Fixed {bad.sum()} invalid {col} value(s).")
            stop_times.loc[bad, col] = default

    # 5. Remove orphan routes (no trips referencing them)
    valid_routes = trips["route_id"].unique()
    before = len(routes)
    routes = routes[routes["route_id"].isin(valid_routes)]
    if len(routes) < before:
        warnings.append(f"Removed {before - len(routes)} orphan route(s).")

    # 6. Clamp lat/lon precision
    stops["stop_lat"] = stops["stop_lat"].round(6)
    stops["stop_lon"] = stops["stop_lon"].round(6)

    # 7. Validate shapes.txt if present
    try:
        shapes = pd.read_csv(f"{output_dir}/shapes.txt")
        shapes = shapes.sort_values(["shape_id", "shape_pt_sequence"])
        shapes["shape_pt_lat"] = shapes["shape_pt_lat"].round(6)
        shapes["shape_pt_lon"] = shapes["shape_pt_lon"].round(6)
        shapes.to_csv(f"{output_dir}/shapes.txt", index=False)
    except FileNotFoundError:
        warnings.append("shapes.txt not found (optional but recommended).")

    # Save fixed files
    stops.to_csv(f"{output_dir}/stops.txt", index=False)
    routes.to_csv(f"{output_dir}/routes.txt", index=False)
    trips.to_csv(f"{output_dir}/trips.txt", index=False)
    stop_times.to_csv(f"{output_dir}/stop_times.txt", index=False)

    # 8. Validate frequencies.txt if present
    try:
        freq = pd.read_csv(f"{output_dir}/frequencies.txt")
        bad = freq["headway_secs"] <= 0
        if bad.any():
            freq.loc[bad, "headway_secs"] = 600
            warnings.append(f"Fixed {bad.sum()} invalid headway_secs value(s).")
        freq.to_csv(f"{output_dir}/frequencies.txt", index=False)
    except FileNotFoundError:
        warnings.append("frequencies.txt not found (optional).")

    return warnings
