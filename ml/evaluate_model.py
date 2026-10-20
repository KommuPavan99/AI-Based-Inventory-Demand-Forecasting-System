"""Evaluate the trained model on the same chronological holdout definition."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "dataset.xlsx"
MODEL_PATH = PROJECT_DIR / "ml" / "demand_model.joblib"
METRICS_PATH = PROJECT_DIR / "ml" / "evaluation_metrics.json"
FEATURE_DATA_PATH = PROJECT_DIR / "data" / "feature_dataset.csv"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run: python .\\ml\\train_model.py")
    if not FEATURE_DATA_PATH.exists():
        raise FileNotFoundError("Feature dataset not found. Run train_model.py first.")
    df = pd.read_csv(FEATURE_DATA_PATH, parse_dates=["Date"])
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    dates = np.array(sorted(df["Date"].dropna().unique()))
    split_index = max(1, min(len(dates) - 1, int(len(dates) * 0.8)))
    cutoff = pd.Timestamp(dates[split_index])
    test = df[df["Date"] >= cutoff].copy()
    X = test[artifact["features"]]
    y = test["Demand_Units"].astype(float)
    pred = np.maximum(0, model.predict(X))
    metrics = {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else None,
        "train_rows": int((df["Date"] < cutoff).sum()),
        "test_rows": int(len(test)),
        "test_start_date": str(test["Date"].min().date()),
        "test_end_date": str(test["Date"].max().date()),
        "model_type": artifact["metrics"].get("model_type", "Random Forest"),
        "note": "Chronological holdout metrics. These are historical test results, not a guarantee of future accuracy.",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("MODEL EVALUATION (chronological holdout)")
    print("---------------------------------------")
    print(f"Test rows: {metrics['test_rows']}")
    print(f"Test period: {metrics['test_start_date']} to {metrics['test_end_date']}")
    print(f"MAE:  {metrics['mae']:.2f} units")
    print(f"RMSE: {metrics['rmse']:.2f} units")
    print(f"R²:   {metrics['r2'] if metrics['r2'] is not None else 'N/A'}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
