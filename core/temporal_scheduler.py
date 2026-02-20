import numpy as np


def apply_peak_offpeak_scheduling(
    routes,
    allocation,
    demand_df,
    operating_hours,
    peak_multiplier=1.6,
    offpeak_multiplier=0.6,
    peak_share=0.35,
):
    """
    Split service into peak vs off-peak periods.

    peak_share = fraction of day considered peak (e.g., 0.35 ≈ 8 hours of 24)
    Falls back silently to existing allocation if demand data is missing.
    """

    if demand_df is None or "final_demand" not in demand_df.columns:
        return allocation

    scheduled = {}

    peak_hours = operating_hours * peak_share
    offpeak_hours = operating_hours - peak_hours

    for rid, route in routes.items():

        base = allocation.get(rid, {})
        base_buses = max(1, base.get("buses_assigned", 1))

        # Peak scaling
        peak_buses = max(1, int(np.ceil(base_buses * peak_multiplier)))
        offpeak_buses = max(1, int(np.ceil(base_buses * offpeak_multiplier)))

        # Headways
        peak_headway = (peak_hours / peak_buses) * 60 if peak_hours > 0 else 0
        offpeak_headway = (offpeak_hours / offpeak_buses) * 60 if offpeak_hours > 0 else 0

        scheduled[rid] = {
            **base,
            "peak_buses": peak_buses,
            "offpeak_buses": offpeak_buses,
            "peak_headway_min": round(peak_headway, 1),
            "offpeak_headway_min": round(offpeak_headway, 1),
            "service_intensity": round(peak_buses / max(1, offpeak_buses), 2),
        }

    return scheduled
