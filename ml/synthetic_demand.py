import numpy as np


def generate_synthetic_demand(features_df):
    """
    Realistic urban demand with a strong spatial gradient.
    Uses an exponential distance-from-centre weight so central stops
    get significantly higher demand, creating visible hotspots.
    """

    # Gaussian centre weight: central stops get exponentially higher demand
    center_lat = features_df["lat"].mean()
    center_lon = features_df["lon"].mean()
    dist_center = np.sqrt(
        (features_df["lat"] - center_lat) ** 2 +
        (features_df["lon"] - center_lon) ** 2
    )
    center_weight = np.exp(-dist_center * 8)   # steeper = tighter core

    # Use lognormal population for realistic multi-modal distribution
    pop  = features_df["population_density"] / features_df["population_density"].max()
    road = features_df["road_density"] / features_df["road_density"].max()
    cw   = center_weight / center_weight.max()

    demand = (
        0.50 * pop   +
        0.30 * road  +
        0.20 * cw
    )

    # Hotspot noise: sharp spikes at random locations
    hotspot_noise = np.random.gamma(shape=2.0, scale=0.05, size=len(demand))
    gaussian_noise = np.random.normal(0, 0.025, len(demand))

    features_df["demand"] = (demand + hotspot_noise + gaussian_noise) * 120

    return features_df
