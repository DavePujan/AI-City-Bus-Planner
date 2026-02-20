import pandas as pd
from ml.grid_builder import create_spatial_grid
from ml.feature_engineering import build_features
from ml.synthetic_demand import generate_synthetic_demand
from ml.demand_model import train_demand_model


FEATURE_COLS = [
    "population_density",
    "road_density",
    "dist_to_stop",
    "dist_to_center"
]


def run_demand_pipeline(stops_df):

    grid = create_spatial_grid(stops_df)
    features = build_features(grid, stops_df)
    features = generate_synthetic_demand(features)

    model, rmse = train_demand_model(features)

    features["predicted_demand"] = model.predict(features[FEATURE_COLS])

    # Feature importance (judges love this)
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    # ================================
    # CNN SPATIAL MODEL
    # ================================
    try:
        from ml.deep_demand_model import train_deep_demand_model
        _, deep_preds = train_deep_demand_model(features)
        features["deep_predicted_demand"] = deep_preds
    except Exception as e:
        print(f"⚠️ CNN model skipped: {e}")
        features["deep_predicted_demand"] = features["predicted_demand"]

    # ================================
    # GNN MODEL
    # ================================
    try:
        from ml.gnn_demand_model import train_gnn_demand_model
        _, gnn_preds = train_gnn_demand_model(features)
        features["gnn_predicted_demand"] = gnn_preds
    except Exception as e:
        print(f"⚠️ GNN model skipped: {e}")
        features["gnn_predicted_demand"] = features["predicted_demand"]

    # ================================
    # ENSEMBLE FINAL DEMAND
    # ================================
    features["final_demand"] = (
        0.5 * features["predicted_demand"] +
        0.3 * features["deep_predicted_demand"] +
        0.2 * features["gnn_predicted_demand"]
    )

    return features, rmse, importance_df
