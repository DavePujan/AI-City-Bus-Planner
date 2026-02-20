import os
import zipfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def _compute_shape_distances(coords):
    """
    Compute cumulative distance (metres approx) along a list of (lat, lon) points.
    """
    dists = [0.0]
    for i in range(1, len(coords)):
        lat1, lon1 = coords[i - 1]
        lat2, lon2 = coords[i]
        d = np.hypot(lat2 - lat1, lon2 - lon1) * 111000
        dists.append(dists[-1] + d)
    return dists


def _time_to_str(minutes_from_midnight):
    """Convert minutes-from-midnight to HH:MM:SS string."""
    h = int(minutes_from_midnight // 60)
    m = int(minutes_from_midnight % 60)
    return f"{h:02d}:{m:02d}:00"


def export_gtfs(
    stops_df,
    routes,
    allocation,
    output_dir="outputs/gtfs",
    agency_name="BusAI Transit",
):
    """
    Generate a minimal (but valid) GTFS feed including stops, routes,
    trips, stop_times, calendar, shapes, and a combined zip archive.
    """

    os.makedirs(output_dir, exist_ok=True)

    # =====================================================
    # agency.txt
    # =====================================================
    agency = pd.DataFrame([{
        "agency_id": "BUSAI",
        "agency_name": agency_name,
        "agency_url": "https://busai.local",
        "agency_timezone": "Asia/Kolkata",
    }])
    agency.to_csv(f"{output_dir}/agency.txt", index=False)

    # =====================================================
    # stops.txt
    # =====================================================
    stop_rows = []
    stop_id_map = {}  # (lat, lon, name) -> stop_id

    for sid, row in enumerate(stops_df.itertuples(), start=1):
        stop_id = f"S{sid}"
        stop_id_map[(row.lat, row.lon, row.stop_name)] = stop_id
        stop_rows.append({
            "stop_id": stop_id,
            "stop_name": row.stop_name,
            "stop_lat": round(row.lat, 6),
            "stop_lon": round(row.lon, 6),
        })

    stops_txt = pd.DataFrame(stop_rows)
    stops_txt.to_csv(f"{output_dir}/stops.txt", index=False)

    # =====================================================
    # routes.txt
    # =====================================================
    route_rows = []
    for rid in routes.keys():
        route_rows.append({
            "route_id": f"R{rid}",
            "agency_id": "BUSAI",
            "route_short_name": f"R{rid}",
            "route_long_name": f"BusAI Route {rid}",
            "route_type": 3,  # bus
        })
    pd.DataFrame(route_rows).to_csv(f"{output_dir}/routes.txt", index=False)

    # =====================================================
    # calendar.txt
    # =====================================================
    calendar = pd.DataFrame([{
        "service_id": "WEEK",
        "monday": 1, "tuesday": 1, "wednesday": 1,
        "thursday": 1, "friday": 1, "saturday": 1, "sunday": 1,
        "start_date": datetime.now().strftime("%Y%m%d"),
        "end_date": (datetime.now() + timedelta(days=365)).strftime("%Y%m%d"),
    }])
    calendar.to_csv(f"{output_dir}/calendar.txt", index=False)

    # =====================================================
    # trips.txt + stop_times.txt + shapes.txt
    # =====================================================
    trip_rows = []
    stop_time_rows = []
    shape_rows = []

    for rid, route in routes.items():
        if len(route) < 2:
            continue

        trip_id = f"T{rid}"   # MUST match frequencies.txt trip_id
        route_id = f"R{rid}"
        shape_id = f"SH{rid}"

        trip_rows.append({
            "route_id": route_id,
            "service_id": "WEEK",
            "trip_id": trip_id,
            "shape_id": shape_id,
        })

        # stop_times — synthetic 4-min spacing starting at 06:00
        base_time = datetime(2025, 1, 1, 6, 0, 0)
        for seq, (lat, lon, name) in enumerate(route):
            stop_id = stop_id_map.get((lat, lon, name))
            if not stop_id:
                continue
            arr = base_time + timedelta(minutes=seq * 4)
            dep = arr + timedelta(seconds=30)
            stop_time_rows.append({
                "trip_id": trip_id,
                "arrival_time": arr.strftime("%H:%M:%S"),
                "departure_time": dep.strftime("%H:%M:%S"),
                "stop_id": stop_id,
                "stop_sequence": seq + 1,
            })

        # shapes — road geometry via stop coords
        coords = [(lat, lon) for lat, lon, _ in route]
        cum_dists = _compute_shape_distances(coords)
        for seq, ((lat, lon), dist) in enumerate(zip(coords, cum_dists)):
            shape_rows.append({
                "shape_id": shape_id,
                "shape_pt_lat": round(lat, 6),
                "shape_pt_lon": round(lon, 6),
                "shape_pt_sequence": seq + 1,
                "shape_dist_traveled": round(dist, 1),
            })

        pass  # rid-based IDs; no counters needed

    pd.DataFrame(trip_rows).to_csv(f"{output_dir}/trips.txt", index=False)
    pd.DataFrame(stop_time_rows).to_csv(f"{output_dir}/stop_times.txt", index=False)
    pd.DataFrame(shape_rows).to_csv(f"{output_dir}/shapes.txt", index=False)

    # =====================================================
    # frequencies.txt  (HIGH-FREQUENCY / BRT-style service)
    # =====================================================
    freq_rows = []

    for rid in routes.keys():
        if len(routes[rid]) < 2:
            continue

        alloc = allocation.get(rid, {})
        peak_hw = alloc.get("peak_headway_min", 15)
        off_hw  = alloc.get("offpeak_headway_min", 30)

        windows = [
            # (start_min, end_min, headway_secs)
            (7 * 60,  10 * 60, int(peak_hw * 60)),   # morning peak
            (10 * 60, 17 * 60, int(off_hw  * 60)),   # midday off-peak
            (17 * 60, 20 * 60, int(peak_hw * 60)),   # evening peak
        ]

        for start_min, end_min, hw_secs in windows:
            freq_rows.append({
                "trip_id":      f"T{rid}",
                "start_time":   _time_to_str(start_min),
                "end_time":     _time_to_str(end_min),
                "headway_secs": max(60, hw_secs),  # minimum 60 s
                "exact_times":  0,
            })

    pd.DataFrame(freq_rows).to_csv(f"{output_dir}/frequencies.txt", index=False)

    # =====================================================
    # ZIP
    # =====================================================
    zip_path = f"{output_dir}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for fname in os.listdir(output_dir):
            z.write(os.path.join(output_dir, fname), arcname=fname)

    return zip_path
