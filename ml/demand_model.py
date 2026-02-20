from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def train_demand_model(df):

    feature_cols = [
        "population_density",
        "road_density",
        "dist_to_stop",
        "dist_to_center"
    ]

    X = df[feature_cols]
    y = df["demand"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.2,
        random_state=42,
        tree_method="hist"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    return model, rmse
