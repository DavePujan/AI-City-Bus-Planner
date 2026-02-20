import numpy as np
import pandas as pd


def _hourly_profile():
    """
    Demand multiplier array for each hour of the day (0-23).
    Modelled on real urban transit demand patterns.
    """
    profile = np.ones(24)
    profile[7:10]  = 1.8   # morning peak
    profile[11:16] = 1.2   # midday mild
    profile[17:20] = 1.9   # evening peak
    profile[0:5]   = 0.4   # late-night low
    profile[21:24] = 0.6   # night low
    return profile


def simulate_passenger_load(
    routes,
    allocation,
    demand_df,
    operating_hours,
    bus_capacity=60,
    random_seed=42,
):
    """
    Simulate hourly passenger loads for each route using Poisson arrivals
    and the demand model's route-level demand estimate.

    Returns a DataFrame with columns:
        route_id, hour, passengers, capacity_per_hour,
        load_factor, overcrowded
    """
    rng = np.random.default_rng(random_seed)
    hourly_mult = _hourly_profile()

    rows = []

    for rid in routes.keys():
        alloc = allocation.get(rid, {})
        route_demand = float(alloc.get("route_demand", 0))

        # Spread daily demand across operating hours
        base_hourly = route_demand / max(1, operating_hours)

        peak_hw = float(alloc.get("peak_headway_min", 15))
        off_hw  = float(alloc.get("offpeak_headway_min", 30))

        for hour in range(24):
            expected = base_hourly * hourly_mult[hour]
            # per-route stochastic variation so each route looks different
            variation = rng.normal(1.0, 0.25)
            passengers = int(rng.poisson(max(expected * max(variation, 0.1), 0)))

            # Which headway applies this hour?
            if 7 <= hour < 10 or 17 <= hour < 20:
                hw = peak_hw
            else:
                hw = off_hw

            buses_per_hour     = 60 / max(hw, 1)
            capacity_per_hour  = buses_per_hour * bus_capacity
            load_factor        = passengers / capacity_per_hour if capacity_per_hour > 0 else 0

            rows.append({
                "route_id":           rid,
                "hour":               hour,
                "passengers":         passengers,
                "capacity_per_hour":  round(capacity_per_hour, 1),
                "load_factor":        round(load_factor, 2),
                "overcrowded":        load_factor > 1.0,
            })

    return pd.DataFrame(rows)
