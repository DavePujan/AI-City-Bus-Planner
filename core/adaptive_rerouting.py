import numpy as np
import pandas as pd


def simulate_adaptive_rerouting(
    load_df,
    allocation,
    bus_capacity=60,
    intervention_threshold=1.05,
):
    """
    Simulate real-time response to overcrowding events.

    When a route-hour exceeds `intervention_threshold` load factor,
    an extra bus is dispatched (+30% capacity for that hour), and the
    adjusted load is recorded.

    Returns
    -------
    events_df : pd.DataFrame
        One row per intervention triggered.
    updated_load_df : pd.DataFrame
        Load data with post-intervention load_factors.
    """
    updated = load_df.copy()
    events = []

    for idx, row in updated.iterrows():
        if row["load_factor"] <= intervention_threshold:
            continue

        rid  = row["route_id"]
        hour = row["hour"]

        alloc    = allocation.get(rid, {})
        peak_hw  = float(alloc.get("peak_headway_min",   15))
        off_hw   = float(alloc.get("offpeak_headway_min", 30))

        headway = peak_hw if (7 <= hour < 10 or 17 <= hour < 20) else off_hw

        # Dispatch extra bus: boost effective capacity by 30 %
        buses_per_hour  = 60 / max(headway, 1)
        boosted_buses   = buses_per_hour * 1.3
        new_capacity    = boosted_buses * bus_capacity
        new_load        = row["passengers"] / max(new_capacity, 1)

        updated.at[idx, "load_factor"]          = round(new_load, 2)
        updated.at[idx, "capacity_per_hour"]    = round(new_capacity, 1)
        updated.at[idx, "overcrowded"]          = new_load > 1.0
        updated.at[idx, "intervention_applied"] = True

        events.append({
            "route_id":  rid,
            "hour":      hour,
            "event":     "EXTRA_BUS_DISPATCHED",
            "old_load":  round(row["load_factor"], 2),
            "new_load":  round(new_load, 2),
            "resolved":  new_load <= 1.0,
        })

    events_df = pd.DataFrame(events) if events else pd.DataFrame(
        columns=["route_id", "hour", "event", "old_load", "new_load", "resolved"]
    )

    if "intervention_applied" not in updated.columns:
        updated["intervention_applied"] = False

    return events_df, updated
