"""Train a product-level daily demand model with chronological validation."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "dataset.xlsx"
MODEL_PATH = PROJECT_DIR / "ml" / "demand_model.joblib"
FEATURE_DATA_PATH = PROJECT_DIR / "data" / "feature_dataset.csv"
METRICS_PATH = PROJECT_DIR / "ml" / "evaluation_metrics.json"

TARGET = "Demand_Units"
CATEGORICAL_FEATURES = ["Product_ID", "Category", "Region", "Supplier"]
NUMERIC_FEATURES = [
    "Unit_Price", "Promotion_Active", "Stock_On_Hand", "Lead_Time_Days",
    "Reorder_Level", "day_of_week", "day_of_month", "month", "day_of_year",
    "is_weekend", "trend_day", "lag_1", "lag_7", "rolling_mean_7",
    "rolling_mean_28",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\nPlace your workbook at data/dataset.xlsx."
        )
    df = pd.read_excel(DATA_PATH, sheet_name="Inventory_Data")
    required = {"Date", "Product_ID", TARGET, "Unit_Price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in Inventory_Data: {sorted(missing)}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Product_ID", TARGET]).copy()
    df["Product_ID"] = df["Product_ID"].astype(str)
    for col in ["Category", "Region", "Supplier"]:
        if col not in df:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype(str)
    for col, default in [
        ("Unit_Price", 0), ("Promotion_Active", 0), ("Stock_On_Hand", 0),
        ("Lead_Time_Days", 0), ("Reorder_Level", 0),
    ]:
        if col not in df:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).sort_values(["Date", "Product_ID"]).reset_index(drop=True)
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create strictly past-looking lags and rolling means within each product."""
    out = df.copy().sort_values(["Product_ID", "Date"]).reset_index(drop=True)
    first_date = out["Date"].min()
    out["day_of_week"] = out["Date"].dt.dayofweek
    out["day_of_month"] = out["Date"].dt.day
    out["month"] = out["Date"].dt.month
    out["day_of_year"] = out["Date"].dt.dayofyear
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["trend_day"] = (out["Date"] - first_date).dt.days
    grouped = out.groupby("Product_ID", sort=False)[TARGET]
    out["lag_1"] = grouped.shift(1)
    out["lag_7"] = grouped.shift(7)
    out["rolling_mean_7"] = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    out["rolling_mean_28"] = grouped.transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
    # At the start of each product's history, fill unavailable lags with its expanding
    # past mean; the first row falls back to the global training median later.
    for col in ["lag_1", "lag_7", "rolling_mean_7", "rolling_mean_28"]:
        out[col] = out[col].fillna(out.groupby("Product_ID")[TARGET].transform(lambda s: s.expanding().mean().shift(1)))
    return out


def main() -> None:
    df = load_data()
    featured = make_features(df)
    featured.to_csv(FEATURE_DATA_PATH, index=False)

    # Hold out the latest 20% of unique dates, avoiding random temporal leakage.
    dates = np.array(sorted(featured["Date"].dropna().unique()))
    if len(dates) < 5:
        raise ValueError("At least five distinct dates are needed for chronological evaluation.")
    split_index = max(1, min(len(dates) - 1, int(len(dates) * 0.8)))
    cutoff = pd.Timestamp(dates[split_index])
    train = featured[featured["Date"] < cutoff].copy()
    test = featured[featured["Date"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Could not create non-empty chronological train/test sets.")

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])
    model = Pipeline([
        ("preprocess", prep),
        ("regressor", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
        )),
    ])
    model.fit(X_train, y_train)
    pred = np.maximum(0, model.predict(X_test))
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)) if len(y_test) > 1 else None,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_start_date": str(train["Date"].min().date()),
        "train_end_date": str(train["Date"].max().date()),
        "test_start_date": str(test["Date"].min().date()),
        "test_end_date": str(test["Date"].max().date()),
        "records": int(len(df)),
        "products": int(df["Product_ID"].nunique()),
        "data_start_date": str(df["Date"].min().date()),
        "data_end_date": str(df["Date"].max().date()),
        "features": FEATURES,
        "model_type": "RandomForestRegressor with lag/rolling and calendar features",
        "note": "Chronological holdout; forecast quality depends on source-data quality and future conditions.",
    }
    artifact = {
        "model": model,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "metrics": metrics,
        "first_date": str(df["Date"].min().date()),
        "last_date": str(df["Date"].max().date()),
    }
    joblib.dump(artifact, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Training completed.")
    print(f"Rows: {len(df):,} | Products: {df['Product_ID'].nunique()} | Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Chronological holdout: {metrics['test_start_date']} to {metrics['test_end_date']}")
    print(f"MAE: {metrics['mae']:.2f} | RMSE: {metrics['rmse']:.2f} | R²: {metrics['r2']}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved features: {FEATURE_DATA_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
