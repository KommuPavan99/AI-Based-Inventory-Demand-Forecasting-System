
"""Inventory Demand Forecasting API."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# PATHS AND APPLICATION
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "dataset.xlsx"
DB_PATH = PROJECT_DIR / "data" / "inventory.db"
MODEL_PATH = PROJECT_DIR / "ml" / "demand_model.joblib"
METRICS_PATH = PROJECT_DIR / "ml" / "evaluation_metrics.json"
FRONTEND_PATH = PROJECT_DIR / "frontend"

# Create the application BEFORE declaring routes.
app = FastAPI(
    title="Inventory Demand Forecasting API",
    description=(
        "Inventory management, demand forecasting, "
        "transactions, replenishment and dataset exploration."
    ),
    version="1.2.0",
)

# Serve optional frontend assets.
app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_PATH), check_dir=False),
    name="static",
)


# ---------------------------------------------------------
# DASHBOARD AND WEB PAGES
# ---------------------------------------------------------

@app.get("/", include_in_schema=False)
def index():
    """Serve the main dashboard."""
    page_path = FRONTEND_PATH / "index.html"

    if not page_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="frontend/index.html was not found.",
        )

    return FileResponse(page_path, media_type="text/html")


@app.get("/dataset", include_in_schema=False)
def dataset_page():
    """Serve the Dataset Explorer page."""
    page_path = FRONTEND_PATH / "dataset.html"

    if not page_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="frontend/dataset.html was not found.",
        )

    return FileResponse(page_path, media_type="text/html")


@app.get("/product-analysis", include_in_schema=False)
def product_analysis_page():
    """Serve the day-to-day product analysis webpage."""
    page_path = FRONTEND_PATH / "product_daily_life_analysis.html"

    if not page_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                "frontend/product_daily_life_analysis.html was not found. "
                "Create the HTML file inside the frontend folder."
            ),
        )

    return FileResponse(page_path, media_type="text/html")


# ---------------------------------------------------------
# DATASET API
# ---------------------------------------------------------

@app.get("/api/dataset")
def get_full_dataset():
    """Return all Excel dataset columns and rows as JSON."""
    if not DATA_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found: {DATA_PATH}",
        )

    try:
        df = pd.read_excel(DATA_PATH)

        # Replace pandas missing values with JSON-safe null values.
        df = df.astype(object).where(pd.notna(df), None)

        # Convert timestamps to ISO strings.
        for column in df.columns:
            df[column] = df[column].map(
                lambda value: (
                    value.isoformat()
                    if isinstance(value, (pd.Timestamp, date))
                    else value
                )
            )

        return {
            "columns": [str(column) for column in df.columns],
            "total_rows": int(len(df)),
            "rows": df.to_dict(orient="records"),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read the Excel dataset: {exc}",
        )


@app.get("/api/dataset/excel")
def download_dataset_excel():
    """Download the original Excel dataset."""
    if not DATA_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found: {DATA_PATH}",
        )

    return FileResponse(
        path=DATA_PATH,
        filename="dataset.xlsx",
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )


# ---------------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------------

def get_db() -> sqlite3.Connection:
    """Open the SQLite database."""
    if not DB_PATH.is_file():
        raise RuntimeError(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def db_product(product_id: str) -> dict:
    """Return one product from SQLite."""
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT * FROM products WHERE product_id = ?",
            (product_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Product not found.")

    return dict(row)


def get_live_products() -> list[dict]:
    """Return current products from SQLite."""
    with closing(get_db()) as conn:
        rows = conn.execute(
            "SELECT * FROM products ORDER BY product_id"
        ).fetchall()

    return [dict(row) for row in rows]


# ---------------------------------------------------------
# HISTORICAL DATA AND MODEL
# ---------------------------------------------------------

def load_inventory() -> pd.DataFrame:
    """Load demand history from the Excel workbook."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_excel(DATA_PATH)

    required = {"Date", "Product_ID", "Demand_Units"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required Excel columns: {sorted(missing)}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Product_ID"] = df["Product_ID"].astype("string")
    df["Demand_Units"] = pd.to_numeric(
        df["Demand_Units"], errors="coerce"
    )

    df = df.dropna(
        subset=["Date", "Product_ID", "Demand_Units"]
    ).copy()

    df["Product_ID"] = df["Product_ID"].astype(str)

    defaults = {
        "Record_ID": "",
        "Product_Name": "Unknown",
        "Category": "Unknown",
        "Supplier": "Unknown",
        "Region": "Unknown",
        "Unit_Price": 0,
        "Promotion_Active": 0,
        "Stock_On_Hand": 0,
        "Lead_Time_Days": 0,
        "Reorder_Level": 0,
        "Projected_Closing_Stock": 0,
        "Stock_Status": "Unknown",
        "Stockout_Risk": "Unknown",
    }

    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default
        df[column] = df[column].fillna(default)

    return df.sort_values(
        ["Date", "Product_ID"]
    ).reset_index(drop=True)


def latest_product_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return the latest historical row for each product."""
    return (
        df.sort_values(["Date", "Product_ID"])
        .groupby("Product_ID", as_index=False)
        .tail(1)
        .sort_values("Product_ID")
        .reset_index(drop=True)
    )


def load_artifact() -> Optional[dict]:
    """Load the saved model if one is available."""
    if not MODEL_PATH.is_file():
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def load_forecast_inventory() -> pd.DataFrame:
    """Combine Excel history, live inventory and recorded sales."""
    df = load_inventory()
    products = get_live_products()

    if products:
        live = pd.DataFrame(products).rename(
            columns={
                "product_id": "Product_ID",
                "product_name": "Product_Name",
                "category": "Category",
                "unit_price": "Unit_Price",
                "stock_on_hand": "Stock_On_Hand",
                "supplier": "Supplier",
                "region": "Region",
                "lead_time_days": "Lead_Time_Days",
                "reorder_level": "Reorder_Level",
                "promotion_active": "Promotion_Active",
            }
        )

        live["Product_ID"] = live["Product_ID"].astype(str)
        live_by_id = live.set_index("Product_ID")

        live_fields = [
            "Product_Name",
            "Category",
            "Unit_Price",
            "Stock_On_Hand",
            "Supplier",
            "Region",
            "Lead_Time_Days",
            "Reorder_Level",
            "Promotion_Active",
        ]

        for field in live_fields:
            if field not in df.columns:
                df[field] = None

        for index, row in df.iterrows():
            product_id = str(row["Product_ID"])

            if product_id in live_by_id.index:
                current = live_by_id.loc[product_id]

                for field in live_fields:
                    df.at[index, field] = current[field]

    # Include sales created through the transaction API.
    with closing(get_db()) as conn:
        sales = conn.execute(
            """
            SELECT record_id, observation_date, product_id,
                   demand_units, promotion_active
            FROM demand_history
            WHERE record_id LIKE 'TXN-%'
            ORDER BY observation_date, record_id
            """
        ).fetchall()

    if sales:
        extra = pd.DataFrame([dict(row) for row in sales]).rename(
            columns={
                "record_id": "Record_ID",
                "observation_date": "Date",
                "product_id": "Product_ID",
                "demand_units": "Demand_Units",
                "promotion_active": "Promotion_Active",
            }
        )

        extra["Date"] = pd.to_datetime(
            extra["Date"], errors="coerce"
        )
        extra["Product_ID"] = extra["Product_ID"].astype(str)
        extra["Demand_Units"] = pd.to_numeric(
            extra["Demand_Units"], errors="coerce"
        )

        extra = extra.dropna(
            subset=["Date", "Product_ID", "Demand_Units"]
        )

        for column in df.columns:
            if column not in extra.columns:
                extra[column] = None

        extra = extra[df.columns]
        df = pd.concat([df, extra], ignore_index=True)

    return df.sort_values(
        ["Date", "Product_ID"]
    ).reset_index(drop=True)


# ---------------------------------------------------------
# FORECASTING HELPERS
# ---------------------------------------------------------

def _features_for_future(
    history: pd.DataFrame,
    product_row: pd.Series,
    forecast_date: pd.Timestamp,
    demand_history: list[float],
    promotion_active: bool,
) -> dict:
    first_date = pd.to_datetime(history["Date"]).min()
    dow = int(forecast_date.dayofweek)

    values = demand_history
    median_demand = float(
        pd.to_numeric(
            history["Demand_Units"], errors="coerce"
        ).median()
    )

    if not math.isfinite(median_demand):
        median_demand = 0.0

    lag1 = values[-1] if values else median_demand
    lag7 = (
        values[-7]
        if len(values) >= 7
        else (float(np.mean(values)) if values else lag1)
    )

    rolling7 = float(np.mean(values[-7:])) if values else lag1
    rolling28 = float(np.mean(values[-28:])) if values else lag1

    def number(value, default=0.0):
        try:
            result = float(value)
            return result if math.isfinite(result) else default
        except (TypeError, ValueError):
            return default

    return {
        "Product_ID": str(product_row["Product_ID"]),
        "Category": str(product_row.get("Category", "Unknown")),
        "Region": str(product_row.get("Region", "Unknown")),
        "Supplier": str(product_row.get("Supplier", "Unknown")),
        "Unit_Price": number(product_row.get("Unit_Price", 0)),
        "Promotion_Active": int(promotion_active),
        "Stock_On_Hand": number(product_row.get("Stock_On_Hand", 0)),
        "Lead_Time_Days": number(product_row.get("Lead_Time_Days", 0)),
        "Reorder_Level": number(product_row.get("Reorder_Level", 0)),
        "day_of_week": dow,
        "day_of_month": int(forecast_date.day),
        "month": int(forecast_date.month),
        "day_of_year": int(forecast_date.dayofyear),
        "is_weekend": int(dow >= 5),
        "trend_day": (
            int((forecast_date - first_date).days)
            if pd.notna(first_date)
            else 0
        ),
        "lag_1": float(lag1),
        "lag_7": float(lag7),
        "rolling_mean_7": rolling7,
        "rolling_mean_28": rolling28,
    }


def build_forecast(
    df: pd.DataFrame,
    artifact: Optional[dict],
    product_id: str,
    horizon: int,
    promotion_active: bool = False,
    start_date: Optional[str] = None,
) -> dict:
    """Generate a recursive daily demand forecast."""
    if horizon not in (7, 14, 30):
        raise ValueError("Horizon must be 7, 14, or 30 days.")

    product_history = df[
        df["Product_ID"].astype(str) == str(product_id)
    ].sort_values("Date").copy()

    if product_history.empty:
        raise ValueError(f"Product '{product_id}' was not found.")

    product_row = product_history.iloc[-1].copy()
    last_date = pd.to_datetime(df["Date"]).max()

    if start_date:
        try:
            start = pd.Timestamp(start_date)
        except (TypeError, ValueError):
            raise ValueError("start_date must be a valid date.")
    else:
        start = max(
            pd.Timestamp(date.today()),
            last_date + pd.Timedelta(days=1),
        )

    if start.normalize() <= last_date.normalize():
        start = last_date.normalize() + pd.Timedelta(days=1)

    demand_history = (
        pd.to_numeric(
            product_history["Demand_Units"], errors="coerce"
        )
        .dropna()
        .astype(float)
        .tolist()
    )

    daily = []
    method = "7-day mean baseline (model not trained)"

    for i in range(horizon):
        forecast_date = start + pd.Timedelta(days=i)

        feature_row = _features_for_future(
            df,
            product_row,
            forecast_date,
            demand_history,
            promotion_active,
        )

        if (
            artifact is not None
            and isinstance(artifact, dict)
            and "model" in artifact
        ):
            feature_cols = artifact.get(
                "features", list(feature_row.keys())
            )

            x = pd.DataFrame([
                {col: feature_row.get(col, 0) for col in feature_cols}
            ])

            try:
                predicted = float(artifact["model"].predict(x)[0])
                method = "Trained ML model with recursive lag features"
            except Exception as exc:
                raise ValueError(
                    f"The saved model could not predict this forecast: {exc}"
                )
        else:
            predicted = (
                float(np.mean(demand_history[-7:]))
                if demand_history
                else float(
                    pd.to_numeric(
                        product_history["Demand_Units"],
                        errors="coerce",
                    ).mean()
                )
            )

        if not math.isfinite(predicted):
            predicted = 0.0

        predicted = max(0.0, predicted)

        daily.append({
            "date": forecast_date.strftime("%Y-%m-%d"),
            "predicted_demand": round(predicted, 2),
            "promotion_active": bool(promotion_active),
        })

        demand_history.append(predicted)

    total = float(sum(item["predicted_demand"] for item in daily))

    def safe_number(value, default=0.0):
        try:
            result = float(value)
            return result if math.isfinite(result) else default
        except (TypeError, ValueError):
            return default

    return {
        "product": {
            "Product_ID": str(product_row["Product_ID"]),
            "Product_Name": str(product_row.get("Product_Name", "Unknown")),
            "Category": str(product_row.get("Category", "Unknown")),
            "Supplier": str(product_row.get("Supplier", "Unknown")),
            "Region": str(product_row.get("Region", "Unknown")),
            "Unit_Price": safe_number(product_row.get("Unit_Price", 0)),
            "Stock_On_Hand": int(
                safe_number(product_row.get("Stock_On_Hand", 0))
            ),
            "Lead_Time_Days": int(
                safe_number(product_row.get("Lead_Time_Days", 0))
            ),
            "Reorder_Level": int(
                safe_number(product_row.get("Reorder_Level", 0))
            ),
            "Projected_Closing_Stock": int(
                safe_number(
                    product_row.get("Projected_Closing_Stock", 0)
                )
            ),
        },
        "horizon_days": horizon,
        "forecast_start": daily[0]["date"],
        "forecast_end": daily[-1]["date"],
        "promotion_scenario": bool(promotion_active),
        "total_forecast_demand": round(total, 2),
        "average_daily_demand": round(total / horizon, 2),
        "daily_forecast": daily,
        "forecast_method": method,
    }


# ---------------------------------------------------------
# REQUEST MODELS
# ---------------------------------------------------------

class ForecastRequest(BaseModel):
    product_id: str
    horizon_days: int = Field(default=7)
    promotion_active: bool = False
    start_date: Optional[str] = None
    stock_on_hand: Optional[float] = Field(default=None, ge=0)
    reserved_stock: float = Field(default=0, ge=0)
    incoming_stock: float = Field(default=0, ge=0)
    safety_stock: Optional[float] = Field(default=None, ge=0)
    review_period_days: int = Field(default=7, ge=1, le=365)
    reorder_level: Optional[float] = Field(default=None, ge=0)


class TransactionRequest(BaseModel):
    product_id: str
    transaction_type: str
    quantity: int = Field(gt=0)
    unit_price: Optional[float] = Field(default=None, ge=0)
    transaction_date: Optional[str] = None
    notes: Optional[str] = None


class ProductUpdateRequest(BaseModel):
    product_name: Optional[str] = None
    category: Optional[str] = None
    unit_price: Optional[float] = Field(default=None, ge=0)
    supplier: Optional[str] = None
    region: Optional[str] = None
    lead_time_days: Optional[int] = Field(default=None, ge=0)
    reorder_level: Optional[int] = Field(default=None, ge=0)
    promotion_active: Optional[bool] = None


# ---------------------------------------------------------
# BASIC ENDPOINTS
# ---------------------------------------------------------

@app.get("/api/health")
def health():
    try:
        with closing(get_db()) as conn:
            table_names = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

        database_ready = {
            "products", "demand_history", "transactions"
        }.issubset(table_names)

    except Exception:
        database_ready = False

    return {
        "status": "ok",
        "dataset_found": DATA_PATH.exists(),
        "database_ready": database_ready,
        "model_ready": (
            MODEL_PATH.exists() and load_artifact() is not None
        ),
        "dataset_path": str(DATA_PATH),
        "database_path": str(DB_PATH),
        "model_path": str(MODEL_PATH),
    }


# ---------------------------------------------------------
# SUMMARY AND PRODUCTS
# ---------------------------------------------------------

@app.get("/api/summary")
def summary():
    try:
        df = load_inventory()
        live_products = get_live_products()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    total_demand = float(
        pd.to_numeric(df["Demand_Units"], errors="coerce").sum()
    )

    product_names = []

    for product in live_products:
        stock = int(product["stock_on_hand"])
        reorder = int(product["reorder_level"])

        product_names.append({
            "Product_ID": product["product_id"],
            "Product_Name": product["product_name"],
            "Category": product["category"],
            "Supplier": product["supplier"],
            "Region": product["region"],
            "Unit_Price": float(product["unit_price"]),
            "Stock_On_Hand": stock,
            "Lead_Time_Days": int(product["lead_time_days"]),
            "Reorder_Level": reorder,
            "Projected_Closing_Stock": stock,
            "Stock_Status": (
                "Out of Stock" if stock == 0
                else "Reorder" if stock <= reorder
                else "Sufficient"
            ),
            "Stockout_Risk": (
                "High" if stock == 0
                else "Medium" if stock <= reorder
                else "Low"
            ),
        })

    demand = pd.to_numeric(df["Demand_Units"], errors="coerce")

    return {
        "records": int(len(df)),
        "products": len(live_products),
        "total_demand": round(total_demand, 2),
        "average_demand": round(float(demand.mean()), 2),
        "date_start": str(df["Date"].min().date()),
        "date_end": str(df["Date"].max().date()),
        "product_names": product_names,
        "low_stock_count": sum(
            p["Stock_On_Hand"] <= p["Reorder_Level"]
            for p in product_names
        ),
        "out_of_stock_count": sum(
            p["Stock_On_Hand"] == 0 for p in product_names
        ),
    }


@app.get("/api/products")
def products():
    return summary()["product_names"]


@app.patch("/api/products/{product_id}")
def update_product(product_id: str, req: ProductUpdateRequest):
    changes = req.model_dump(exclude_unset=True)

    if not changes:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one product field to update.",
        )

    column_map = {
        "product_name": "product_name",
        "category": "category",
        "unit_price": "unit_price",
        "supplier": "supplier",
        "region": "region",
        "lead_time_days": "lead_time_days",
        "reorder_level": "reorder_level",
        "promotion_active": "promotion_active",
    }

    updates = {}

    for key, value in changes.items():
        if key == "promotion_active":
            value = int(value)
        updates[column_map[key]] = value

    assignments = ", ".join(
        f'"{column}" = ?' for column in updates
    )

    conn = get_db()

    try:
        cursor = conn.execute(
            f"UPDATE products SET {assignments} WHERE product_id = ?",
            [*updates.values(), product_id],
        )

        if cursor.rowcount == 0:
            exists = conn.execute(
                "SELECT 1 FROM products WHERE product_id = ?",
                (product_id,),
            ).fetchone()

            if not exists:
                raise HTTPException(
                    status_code=404,
                    detail="Product not found.",
                )

        conn.commit()

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return {
        "message": "Product updated.",
        "product": db_product(product_id),
    }


# ---------------------------------------------------------
# TRANSACTIONS
# ---------------------------------------------------------

@app.get("/api/transactions")
def list_transactions(
    limit: int = Query(default=100, ge=1, le=1000),
):
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT t.*, p.product_name
            FROM transactions AS t
            JOIN products AS p ON p.product_id = t.product_id
            ORDER BY t.transaction_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


@app.post("/api/transactions")
def create_transaction(req: TransactionRequest):
    transaction_type = req.transaction_type.strip().upper()

    if transaction_type not in {"SALE", "PURCHASE"}:
        raise HTTPException(
            status_code=422,
            detail="transaction_type must be SALE or PURCHASE.",
        )

    try:
        transaction_date = (
            date.fromisoformat(req.transaction_date).isoformat()
            if req.transaction_date
            else date.today().isoformat()
        )
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422,
            detail="transaction_date must use YYYY-MM-DD format.",
        )

    conn = get_db()

    try:
        conn.execute("BEGIN IMMEDIATE")

        product = conn.execute(
            "SELECT * FROM products WHERE product_id = ?",
            (req.product_id,),
        ).fetchone()

        if product is None:
            raise HTTPException(status_code=404, detail="Product not found.")

        stock_before = int(product["stock_on_hand"])

        if transaction_type == "SALE":
            if req.quantity > stock_before:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Insufficient stock. Available: {stock_before}; "
                        f"requested: {req.quantity}."
                    ),
                )
            stock_after = stock_before - req.quantity
        else:
            stock_after = stock_before + req.quantity

        unit_price = (
            float(req.unit_price)
            if req.unit_price is not None
            else float(product["unit_price"])
        )

        conn.execute(
            "UPDATE products SET stock_on_hand = ? WHERE product_id = ?",
            (stock_after, req.product_id),
        )

        cursor = conn.execute(
            """
            INSERT INTO transactions (
                product_id, transaction_type, quantity,
                unit_price, transaction_date, notes
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                req.product_id,
                transaction_type,
                req.quantity,
                unit_price,
                transaction_date,
                req.notes,
            ),
        )

        transaction_id = cursor.lastrowid

        if transaction_type == "SALE":
            conn.execute(
                """
                INSERT INTO demand_history (
                    record_id, observation_date, product_id,
                    demand_units, promotion_active
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    f"TXN-{transaction_id}",
                    transaction_date,
                    req.product_id,
                    req.quantity,
                    int(product["promotion_active"]),
                ),
            )

        conn.commit()
        reorder = int(product["reorder_level"])

        return {
            "message": f"{transaction_type.title()} recorded successfully.",
            "transaction_id": transaction_id,
            "product_id": req.product_id,
            "transaction_type": transaction_type,
            "quantity": req.quantity,
            "unit_price": unit_price,
            "stock_before": stock_before,
            "stock_after": stock_after,
            "stock_status": (
                "Out of Stock" if stock_after == 0
                else "Reorder" if stock_after <= reorder
                else "Sufficient"
            ),
            "low_stock_alert": stock_after <= reorder,
        }

    except HTTPException:
        conn.rollback()
        raise
    except sqlite3.Error as exc:
        conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Database transaction failed: {exc}",
        )
    finally:
        conn.close()


# ---------------------------------------------------------
# HISTORICAL DEMAND API
# ---------------------------------------------------------

@app.get("/api/history")
def history(
    product_id: Optional[str] = None,
    days: Optional[int] = Query(default=None, ge=1, le=3650),
):
    """
    Return daily historical demand.

    Optional parameters:
      product_id: filter history for one product
      days: return only the latest N distinct dates in the result

    Note: when product_id is omitted, demand is aggregated across products.
    """
    try:
        df = load_forecast_inventory()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if product_id:
        df = df[df["Product_ID"].astype(str) == str(product_id)]

    if df.empty:
        return []

    # Aggregate multiple records on the same day into daily demand.
    series = (
        df.groupby("Date", as_index=False)["Demand_Units"]
        .sum()
        .sort_values("Date")
    )

    if days is not None:
        series = series.tail(days)

    return [
        {
            "date": row["Date"].strftime("%Y-%m-%d"),
            "demand": round(float(row["Demand_Units"]), 2),
        }
        for _, row in series.iterrows()
    ]


# ---------------------------------------------------------
# MODEL METRICS
# ---------------------------------------------------------

@app.get("/api/metrics")
def metrics():
    if METRICS_PATH.is_file():
        try:
            return {
                "available": True,
                **json.loads(METRICS_PATH.read_text(encoding="utf-8")),
            }
        except Exception:
            pass

    artifact = load_artifact()

    if artifact and artifact.get("metrics"):
        return {"available": True, **artifact["metrics"]}

    return {
        "available": False,
        "message": (
            "Train the model using python .\\ml\\train_model.py "
            "to generate chronological evaluation metrics."
        ),
    }


# ---------------------------------------------------------
# FORECAST AND REPLENISHMENT
# ---------------------------------------------------------

@app.post("/api/predict")
def predict(req: ForecastRequest):
    if req.horizon_days not in (7, 14, 30):
        raise HTTPException(
            status_code=422,
            detail="horizon_days must be 7, 14, or 30.",
        )

    try:
        df = load_forecast_inventory()
        artifact = load_artifact()

        result = build_forecast(
            df,
            artifact,
            req.product_id,
            req.horizon_days,
            req.promotion_active,
            req.start_date,
        )

        product = result["product"]

        stock = float(
            req.stock_on_hand
            if req.stock_on_hand is not None
            else product["Stock_On_Hand"]
        )
        reserved = float(req.reserved_stock)
        incoming = float(req.incoming_stock)

        if reserved > stock:
            raise HTTPException(
                status_code=422,
                detail="Reserved stock cannot exceed stock on hand.",
            )

        net_stock = max(0.0, stock - reserved + incoming)
        lead = max(0, int(product["Lead_Time_Days"]))
        review = int(req.review_period_days)

        daily_values = [
            item["predicted_demand"]
            for item in result["daily_forecast"]
        ]

        def demand_for_days(days: int) -> float:
            covered = min(days, len(daily_values))
            total = sum(daily_values[:covered])

            if days > covered:
                total += (
                    days - covered
                ) * result["average_daily_demand"]

            return float(total)

        safety = float(
            req.safety_stock
            if req.safety_stock is not None
            else max(
                0,
                math.ceil(result["average_daily_demand"] * 2),
            )
        )

        reorder_point = demand_for_days(lead) + safety
        order_up_to = demand_for_days(lead + review) + safety
        order_qty = max(0, math.ceil(order_up_to - net_stock))

        reorder_threshold = float(
            req.reorder_level
            if req.reorder_level is not None
            else product["Reorder_Level"]
        )

        days_cover = (
            net_stock / result["average_daily_demand"]
            if result["average_daily_demand"] > 0
            else None
        )

        if net_stock <= reorder_point or (
            days_cover is not None and days_cover < lead
        ):
            risk = "High"
            advice = (
                "Review replenishment immediately; available stock "
                "may not cover demand until replenishment arrives."
            )
        elif net_stock <= order_up_to or net_stock <= reorder_threshold:
            risk = "Medium"
            advice = (
                "Plan a replenishment order after checking open "
                "purchase orders and supplier constraints."
            )
        else:
            risk = "Low"
            advice = (
                "Stock is above the calculated replenishment threshold; "
                "continue monitoring."
            )

        result["inventory_plan"] = {
            "stock_on_hand": round(stock, 2),
            "reserved_stock": round(reserved, 2),
            "incoming_stock": round(incoming, 2),
            "net_available_stock": round(net_stock, 2),
            "lead_time_days": lead,
            "review_period_days": review,
            "safety_stock_units": round(safety, 2),
            "reorder_point_units": round(reorder_point, 2),
            "order_up_to_units": round(order_up_to, 2),
            "suggested_order_quantity": int(order_qty),
            "reorder_threshold": reorder_threshold,
            "estimated_days_of_cover": (
                round(days_cover, 2)
                if days_cover is not None
                else None
            ),
            "stockout_risk": risk,
            "recommendation": advice,
            "estimated_order_cost": round(
                order_qty * float(product["Unit_Price"]), 2
            ),
        }

        return result

    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast failed: {exc}",
        )


# ---------------------------------------------------------
# REPLENISHMENT CSV REPORT
# ---------------------------------------------------------

@app.get("/api/reports/replenishment.csv")
def replenishment_csv():
    try:
        df = load_forecast_inventory()
        live_products = get_live_products()
        artifact = load_artifact()
        rows = []

        for product in live_products:
            pid = str(product["product_id"])
            result = build_forecast(df, artifact, pid, 7, False)

            avg = result["average_daily_demand"]
            lead = int(product["lead_time_days"])
            safety = max(0, math.ceil(avg * 2))

            reorder_point = avg * lead + safety
            stock = int(product["stock_on_hand"])
            order_up_to = avg * (lead + 7) + safety
            quantity = max(0, math.ceil(order_up_to - stock))

            rows.append({
                "Product_ID": pid,
                "Product_Name": product["product_name"],
                "Category": product["category"],
                "Supplier": product["supplier"],
                "Forecast_7_Day_Demand": result["total_forecast_demand"],
                "Stock_On_Hand": stock,
                "Lead_Time_Days": lead,
                "Safety_Stock_Units": safety,
                "Reorder_Point_Units": round(reorder_point, 2),
                "Suggested_Order_Quantity": quantity,
                "Estimated_Order_Cost": round(
                    quantity * float(product["unit_price"]), 2
                ),
                "Stock_Status": (
                    "Reorder"
                    if stock <= max(
                        reorder_point, int(product["reorder_level"])
                    )
                    else "Sufficient"
                ),
            })

        stream = StringIO()
        pd.DataFrame(rows).to_csv(stream, index=False)
        stream.seek(0)

        return StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                    "attachment; filename=replenishment_recommendations.csv"
            },
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not create replenishment report: {exc}",
        )


# ---------------------------------------------------------
# FORECAST CSV REPORT
# ---------------------------------------------------------

@app.get("/api/reports/forecast.csv")
def forecast_csv(
    product_id: str,
    horizon_days: int = 7,
    promotion_active: bool = False,
):
    if horizon_days not in (7, 14, 30):
        raise HTTPException(
            status_code=422,
            detail="horizon_days must be 7, 14, or 30.",
        )

    try:
        result = build_forecast(
            load_forecast_inventory(),
            load_artifact(),
            product_id,
            horizon_days,
            promotion_active,
        )

        stream = StringIO()
        pd.DataFrame(result["daily_forecast"]).to_csv(
            stream, index=False
        )
        stream.seek(0)

        filename = f"forecast_{product_id}_{horizon_days}d.csv"

        return StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            },
        )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not create forecast report: {exc}",
        )


# ---------------------------------------------------------
# TRANSACTION CSV REPORT
# ---------------------------------------------------------

@app.get("/api/reports/transactions.csv")
def transactions_csv():
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT t.transaction_id, t.transaction_date,
                   t.product_id, p.product_name,
                   t.transaction_type, t.quantity,
                   t.unit_price,
                   t.quantity * t.unit_price AS transaction_value,
                   t.notes
            FROM transactions AS t
            JOIN products AS p ON p.product_id = t.product_id
            ORDER BY t.transaction_id DESC
            """
        ).fetchall()

    stream = StringIO()
    pd.DataFrame([dict(row) for row in rows]).to_csv(
        stream, index=False
    )
    stream.seek(0)

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition":
                "attachment; filename=inventory_transactions.csv"
        },
    )

@app.get("/graphs", include_in_schema=False)
def graphs_page():
    """Serve the Graphs and Analysis page."""
    page_path = FRONTEND_PATH / "graphs.html"

    if not page_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="frontend/graphs.html was not found.",
        )

    return FileResponse(page_path, media_type="text/html")

# ---------------------------------------------------------
# VOICE ASSISTANT
# ---------------------------------------------------------

class VoiceQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=500)


def _voice_dataset() -> pd.DataFrame:
    """Read the source workbook for voice analytics."""
    if not DATA_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found: {DATA_PATH}",
        )
    try:
        data = pd.read_excel(DATA_PATH)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read dataset for voice assistant: {exc}",
        )
    return data


def _voice_col(data: pd.DataFrame, *names: str):
    lookup = {str(c).strip().lower(): c for c in data.columns}
    for name in names:
        if name.strip().lower() in lookup:
            return lookup[name.strip().lower()]
    return None


def _voice_num(data: pd.DataFrame, *names: str) -> pd.Series:
    column = _voice_col(data, *names)
    if column is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(data[column], errors="coerce").dropna()


def _fmt_num(value, decimals: int = 2) -> str:
    try:
        value = float(value)
        if not math.isfinite(value):
            return "not available"
        return f"{value:,.{decimals}f}"
    except (TypeError, ValueError):
        return "not available"


@app.get("/voice-assistant", include_in_schema=False)
def voice_assistant_page():
    """Serve the optional browser-based voice assistant page."""
    page_path = FRONTEND_PATH / "voice_assistant.html"
    if not page_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                "frontend/voice_assistant.html was not found. "
                "Create that file in the frontend folder first."
            ),
        )
    return FileResponse(page_path, media_type="text/html")


@app.get("/api/voice/stats")
def voice_statistics():
    """Return dataset statistics in a compact format for voice answers."""
    data = _voice_dataset()
    demand_col = _voice_col(data, "Demand_Units", "demand", "Units_Sold")
    revenue_col = _voice_col(data, "Estimated_Revenue", "Revenue", "Total_Revenue")
    product_col = _voice_col(data, "Product_ID", "Product", "Product_Name")
    category_col = _voice_col(data, "Category")
    region_col = _voice_col(data, "Region")
    supplier_col = _voice_col(data, "Supplier")
    status_col = _voice_col(data, "Stock_Status")
    risk_col = _voice_col(data, "Stockout_Risk", "Stockout Risk")

    demand = pd.to_numeric(data[demand_col], errors="coerce").dropna() if demand_col else pd.Series(dtype=float)
    revenue = pd.to_numeric(data[revenue_col], errors="coerce").dropna() if revenue_col else pd.Series(dtype=float)

    def counts(column):
        if column is None:
            return {}
        return {str(k): int(v) for k, v in data[column].fillna("Unknown").astype(str).value_counts().items()}

    product_demand = {}
    if product_col and demand_col:
        grouped = data.assign(_demand=pd.to_numeric(data[demand_col], errors="coerce"))
        product_demand = {
            str(k): round(float(v), 2)
            for k, v in grouped.groupby(product_col, dropna=False)["_demand"].sum(min_count=1).dropna().items()
        }

    stats = {
        "available": True,
        "total_rows": int(len(data)),
        "total_columns": int(len(data.columns)),
        "columns": [str(c) for c in data.columns],
        "missing_cells": int(data.isna().sum().sum()),
        "unique_products": int(data[product_col].nunique(dropna=True)) if product_col else None,
        "unique_categories": int(data[category_col].nunique(dropna=True)) if category_col else None,
        "unique_regions": int(data[region_col].nunique(dropna=True)) if region_col else None,
        "unique_suppliers": int(data[supplier_col].nunique(dropna=True)) if supplier_col else None,
        "demand": {
            "column": str(demand_col) if demand_col else None,
            "valid_records": int(demand.count()),
            "total": round(float(demand.sum()), 2) if not demand.empty else None,
            "average": round(float(demand.mean()), 2) if not demand.empty else None,
            "median": round(float(demand.median()), 2) if not demand.empty else None,
            "minimum": round(float(demand.min()), 2) if not demand.empty else None,
            "maximum": round(float(demand.max()), 2) if not demand.empty else None,
            "standard_deviation": round(float(demand.std()), 2) if len(demand) > 1 else (0.0 if len(demand) == 1 else None),
        },
        "revenue": {
            "column": str(revenue_col) if revenue_col else None,
            "total": round(float(revenue.sum()), 2) if not revenue.empty else None,
            "average": round(float(revenue.mean()), 2) if not revenue.empty else None,
        },
        "demand_by_product": product_demand,
        "stock_status_counts": counts(status_col),
        "stockout_risk_counts": counts(risk_col),
        "category_counts": counts(category_col),
        "region_counts": counts(region_col),
        "supplier_counts": counts(supplier_col),
    }
    return stats


def _normalise_id(value: object) -> str:
    """Normalize IDs so 'INV zero zero zero one' and 'INV-0001' match."""
    import re
    raw = str(value or "").strip().upper()
    words = {
        "ZERO": "0", "OH": "0", "ONE": "1", "TWO": "2", "THREE": "3",
        "FOUR": "4", "FIVE": "5", "SIX": "6", "SEVEN": "7",
        "EIGHT": "8", "NINE": "9",
    }
    for word, digit in words.items():
        raw = re.sub(rf"\b{word}\b", digit, raw)
    return re.sub(r"[^A-Z0-9]", "", raw)


def _voice_model_info() -> dict:
    """Describe the saved estimator and evaluation data without guessing."""
    info = {
        "available": False,
        "model_file_exists": MODEL_PATH.is_file(),
        "model_path": str(MODEL_PATH),
        "metrics_file_exists": METRICS_PATH.is_file(),
        "model_type": None,
        "features": [],
        "target": "Demand_Units (forecast target, when configured by the training script)",
        "training_rows": None,
        "training_date_range": None,
        "metrics": None,
        "message": None,
    }
    artifact = load_artifact()
    if not artifact:
        info["message"] = "No readable trained model artifact found. Run: python .\\ml\\train_model.py"
        info["metrics"] = metrics()
        return info

    model = artifact.get("model", artifact.get("estimator", artifact.get("pipeline"))) if isinstance(artifact, dict) else artifact
    info["available"] = model is not None
    if model is not None:
        info["model_type"] = f"{model.__class__.__module__}.{model.__class__.__name__}"
        features = getattr(model, "feature_names_in_", None)
        if features is None and isinstance(artifact, dict):
            features = artifact.get("feature_columns", artifact.get("features", artifact.get("feature_cols")))
        if features is not None:
            try:
                info["features"] = [str(x) for x in list(features)]
            except TypeError:
                pass
        if isinstance(artifact, dict):
            for key in ("training_rows", "n_training_rows", "train_rows", "target"):
                if key in artifact:
                    info["training_rows" if "rows" in key else "target"] = artifact[key]
            if artifact.get("metrics"):
                info["metrics"] = artifact["metrics"]
    if info["metrics"] is None:
        info["metrics"] = metrics()
    try:
        history = load_inventory()
        if not history.empty:
            info["training_date_range"] = {
                "start": str(history["Date"].min().date()),
                "end": str(history["Date"].max().date()),
                "dataset_rows": int(len(history)),
            }
    except Exception:
        pass
    if not info["features"]:
        info["features_note"] = "Feature names are not stored in the saved artifact; inspect ml/train_model.py to verify the training features."
    return info


@app.get("/api/voice/model-info")
def voice_model_info():
    """Expose the trained model's basic metadata and evaluation metrics."""
    return _voice_model_info()


@app.get("/api/voice/dataset")
def voice_dataset_preview(limit: int = Query(default=10, ge=1, le=100)):
    """Return dataset shape, columns, missing-value summary, and a small preview."""
    data = _voice_dataset()
    preview = data.head(limit).copy()
    preview = preview.astype(object).where(pd.notna(preview), None)
    return {
        "rows": int(data.shape[0]),
        "columns_count": int(data.shape[1]),
        "columns": [str(c) for c in data.columns],
        "missing_by_column": {str(k): int(v) for k, v in data.isna().sum().items()},
        "preview_limit": limit,
        "preview": preview.to_dict(orient="records"),
    }



@app.post("/api/voice/query")
def voice_query(req: VoiceQuestion):
    """Answer natural-language questions using the workbook and saved model."""
    import re

    original_question = req.question.strip()
    question = " ".join(original_question.lower().split())

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Please provide a question.",
        )

    def answer(text: str, data=None):
        return {
            "question": original_question,
            "answer": text,
            "speak_text": text,
            "data": data if data is not None else {},
        }

    # Load the actual workbook.
    data = _voice_dataset()

    def get_record_by_id(requested_id):
        """Match typed or spoken record IDs against the workbook."""
        id_col = _voice_col(
            data,
            "Record_ID",
            "Record ID",
            "Inventory_ID",
            "Inventory ID",
            "ID",
        )

        if id_col is None:
            return answer(
                "I understood the record ID, but could not find a "
                "recognized Record_ID column in the dataset.",
                {
                    "found": False,
                    "record_id": requested_id,
                },
            )

        normalized_ids = data[id_col].map(_normalise_id)
        found = data.loc[normalized_ids == requested_id]

        if found.empty:
            return answer(
                f"I couldn't find record {requested_id} in the dataset.",
                {
                    "found": False,
                    "record_id": requested_id,
                    "record_id_column": str(id_col),
                },
            )

        records = (
            found.astype(object)
            .where(pd.notna(found), None)
            .to_dict(orient="records")
        )
        record = records[0]

        details = "; ".join(
            f"{key}: {value}"
            for key, value in record.items()
            if value is not None
        )

        return answer(
            f"I found record {record.get(id_col, requested_id)}. "
            f"{details}",
            {
                "found": True,
                "record": record,
                "matches": len(records),
                "record_id_column": str(id_col),
            },
        )

    # -----------------------------------------------------
    # RECORD LOOKUP — CHECK THIS BEFORE STATISTIC REQUESTS
    # -----------------------------------------------------

    # Examples:
    # Find record INV-0001
    # Show record INV0001
    # Find record ID INV-0001
    # Show record INV zero zero zero one

    record_id_match = re.search(
        r"\bINV[-_\s]?\d{1,8}\b",
        original_question,
        flags=re.IGNORECASE,
    )

    if record_id_match:
        requested_id = _normalise_id(record_id_match.group(0))
        return get_record_by_id(requested_id)

    # Handle speech transcription that writes number words separately.
    spoken_id_match = re.search(
        r"\bINV(?:ENTORY)?"
        r"(?:[\s_-]*(?:ZERO|OH|ONE|TWO|THREE|FOUR|FIVE|"
        r"SIX|SEVEN|EIGHT|NINE|\d)){1,12}\b",
        original_question,
        flags=re.IGNORECASE,
    )

    if spoken_id_match:
        requested_id = _normalise_id(spoken_id_match.group(0))
        return get_record_by_id(requested_id)

    # Also support explicit IDs without the INV prefix:
    # Find record ID ABC-123
    generic_id_match = re.search(
        r"\brecord\s*id\s*"
        r"(?:is|number|no\.?|:)?\s*"
        r"([a-z0-9][a-z0-9_-]*)",
        question,
        flags=re.IGNORECASE,
    )

    if generic_id_match:
        candidate = generic_id_match.group(1).strip(" .?!,:;")

        if candidate.lower() not in {
            "identifies",
            "means",
            "is",
            "the",
            "for",
        }:
            requested_id = _normalise_id(candidate)
            return get_record_by_id(requested_id)

    # Statistics are loaded only after record lookup has been considered.
    stats = voice_statistics()
    demand = stats["demand"]
    revenue = stats["revenue"]

    # -----------------------------------------------------
    # HELP
    # -----------------------------------------------------

    if any(term in question for term in (
        "help",
        "what can you",
        "what questions",
        "commands",
    )):
        return answer(
            "You can ask me to find a record by Record_ID, look up "
            "products, summarize the dataset, show columns, calculate "
            "demand or revenue statistics, check stockout risk, or "
            "explain the trained machine-learning model."
        )

    # -----------------------------------------------------
    # DATASET INFORMATION
    # -----------------------------------------------------

    if any(term in question for term in (
        "load the data",
        "load data",
        "dataset details",
        "dataset information",
        "show dataset",
        "dataset summary",
        "about the dataset",
        "data related to dataset",
        "explain the dataset",
        "explain dataset",
        "describe the dataset",
        "describe dataset",
        "tell me about the dataset",
        "what is in the dataset",
        "what does the dataset contain",
        "dataset overview",
    )):
        preview_data = data.head(5)
        preview = (
            preview_data.astype(object)
            .where(pd.notna(preview_data), None)
            .to_dict(orient="records")
        )

        return answer(
            f"The dataset contains {len(data):,} rows and "
            f"{len(data.columns)} columns, with "
            f"{int(data.isna().sum().sum()):,} missing cells. "
            f"Columns: {', '.join(map(str, data.columns))}.",
            {
                "rows": int(len(data)),
                "columns_count": int(len(data.columns)),
                "columns": list(map(str, data.columns)),
                "missing_cells": int(data.isna().sum().sum()),
                "preview": preview,
            },
        )

    if any(term in question for term in (
        "column names",
        "list columns",
        "what columns",
        "dataset schema",
        "fields in dataset",
    )):
        return answer(
            "Dataset columns: " + ", ".join(map(str, data.columns)),
            {"columns": list(map(str, data.columns))},
        )

    # -----------------------------------------------------
    # IDENTIFIER EXPLANATIONS
    # -----------------------------------------------------

    if any(term in question for term in (
        "what is record id",
        "what does record id",
        "record id means",
        "record id identifies",
        "explain record id",
        "meaning of record id",
    )):
        record_col = _voice_col(data, "Record_ID", "Record ID")

        if record_col is not None:
            product_col = _voice_col(data, "Product_ID", "Product ID")

            return answer(
                "Record_ID identifies an individual dataset record. "
                "Product_ID identifies the product and can appear in "
                "multiple records.",
                {
                    "record_id_column": str(record_col),
                    "product_id_column": (
                        str(product_col) if product_col is not None else None
                    ),
                },
            )

        return answer(
            "I couldn't find a Record_ID column in the workbook."
        )

    if any(term in question for term in (
        "what is product id",
        "what does product id",
        "product id means",
        "product id identifies",
        "explain product id",
        "meaning of product id",
    )):
        product_col = _voice_col(data, "Product_ID", "Product ID")

        if product_col is not None:
            record_col = _voice_col(data, "Record_ID", "Record ID")

            return answer(
                "Product_ID identifies a product. Multiple records may "
                "refer to the same product. Record_ID identifies an "
                "individual record.",
                {
                    "product_id_column": str(product_col),
                    "record_id_column": (
                        str(record_col) if record_col is not None else None
                    ),
                },
            )

        return answer(
            "I couldn't find a Product_ID column in the workbook."
        )

    # -----------------------------------------------------
    # PRODUCT LOOKUP
    # -----------------------------------------------------

    product_id_col = _voice_col(data, "Product_ID", "Product ID")

    product_request = any(term in question for term in (
        "product id",
        "product details",
        "details of product",
        "details for product",
        "information about product",
        "information for product",
        "tell me about product",
        "find product",
        "look up product",
    ))

    if product_id_col is not None and product_request:
        patterns = (
            r"product\s*id\s*(?:is|number|no\.?|:)?\s*"
            r"([a-z0-9][a-z0-9_-]*)",
            r"(?:details|information)\s+(?:for|of|about)\s+"
            r"product\s+([a-z0-9][a-z0-9_-]*)",
            r"(?:find|look up)\s+product\s+"
            r"([a-z0-9][a-z0-9_-]*)",
        )

        candidate = None

        for pattern in patterns:
            found_match = re.search(
                pattern,
                question,
                flags=re.IGNORECASE,
            )

            if found_match:
                token = found_match.group(1).strip(" .?!,:;")

                if token.lower() not in {
                    "the",
                    "is",
                    "are",
                    "details",
                    "related",
                    "please",
                }:
                    candidate = token
                    break

        if candidate:
            wanted = _normalise_id(candidate)
            normalized_products = data[product_id_col].map(_normalise_id)
            product_rows = data.loc[normalized_products == wanted]

            if product_rows.empty:
                return answer(
                    f"I couldn't find Product_ID {candidate} in the dataset.",
                    {
                        "found": False,
                        "product_id": candidate,
                    },
                )

            date_col = _voice_col(data, "Date")

            if date_col is not None:
                product_rows = product_rows.assign(
                    _voice_date=pd.to_datetime(
                        product_rows[date_col],
                        errors="coerce",
                    )
                ).sort_values("_voice_date")

            record = (
                product_rows.iloc[-1]
                .drop(labels=["_voice_date"], errors="ignore")
                .astype(object)
                .where(
                    pd.notna(
                        product_rows.iloc[-1].drop(
                            labels=["_voice_date"],
                            errors="ignore",
                        )
                    ),
                    None,
                )
                .to_dict()
            )

            details = "; ".join(
                f"{key}: {value}"
                for key, value in record.items()
                if value is not None
            )

            return answer(
                f"I found Product_ID {record.get(product_id_col, candidate)}. "
                f"Latest matching record: {details}",
                {
                    "found": True,
                    "product_id": candidate,
                    "matches": int(len(product_rows)),
                    "record": record,
                },
            )

    # -----------------------------------------------------
    # REORDER / LOW-STOCK QUESTIONS (LIVE INVENTORY DATABASE)
    # -----------------------------------------------------

    reorder_question = any(term in question for term in (
        "need reorder",
        "needs reorder",
        "need to reorder",
        "products to reorder",
        "which products need",
        "which products should",
        "what products need",
        "what needs reordering",
        "reorder products",
        "products for reorder",
        "low stock",
        "low-stock",
        "restock",
        "replenishment",
        "purchase suggestions",
        "purchase recommendation",
        "stock running low",
    ))

    if reorder_question:
        try:
            live_products = get_live_products()
        except Exception as exc:
            return answer(
                f"I could not read the live inventory database: {exc}",
                {"error": str(exc)},
            )

        reorder_items = []
        for product in live_products:
            try:
                stock = float(product.get("stock_on_hand") or 0)
                threshold = float(product.get("reorder_level") or 0)
                price = float(product.get("unit_price") or 0)
            except (TypeError, ValueError):
                continue

            if stock <= threshold:
                reorder_items.append({
                    "product_id": product.get("product_id"),
                    "product_name": product.get("product_name"),
                    "category": product.get("category"),
                    "stock_on_hand": int(stock),
                    "reorder_level": int(threshold),
                    "supplier": product.get("supplier"),
                    "unit_price": price,
                    "estimated_cost_to_reorder_to_threshold": round(
                        max(0, threshold - stock) * price, 2
                    ),
                    "status": "Out of Stock" if stock <= 0 else "Reorder",
                })

        reorder_items.sort(key=lambda item: (item["stock_on_hand"], str(item["product_id"])))
        if not reorder_items:
            return answer(
                f"No products in the live inventory are at or below their reorder level. "
                f"I checked {len(live_products)} products in the inventory database.",
                {"count": 0, "checked_products": len(live_products), "products": []},
            )

        descriptions = [
            f"{item['product_name']} (ID {item['product_id']}): "
            f"{item['stock_on_hand']} units in stock, reorder level "
            f"{item['reorder_level']} units, supplier {item.get('supplier') or 'unknown'}"
            for item in reorder_items
        ]
        return answer(
            f"{len(reorder_items)} product(s) need reordering: "
            + ". ".join(descriptions)
            + ". These results use current stock and reorder levels from the live inventory database.",
            {"count": len(reorder_items), "checked_products": len(live_products), "products": reorder_items},
        )

    # -----------------------------------------------------
    # DEMAND FORECAST EXPLANATION / PRODUCT FORECAST
    # -----------------------------------------------------

    forecast_question = any(term in question for term in (
        "forecast",
        "future demand",
        "predict demand",
        "demand prediction",
        "explain the demand",
        "how does forecasting work",
        "how is demand predicted",
    ))

    if forecast_question:
        # If the question names a product ID, run an actual forecast for it.
        forecast_product_match = re.search(
            r"\bproduct\s*(?:id\s*)?(?:is\s*)?"
            r"([A-Z]{1,5}[-_ ]?\d{1,8}|[A-Z])\b",
            original_question,
            flags=re.IGNORECASE,
        )
        horizon_match = re.search(r"\b(7|14|30)\s*(?:day|days|d)\b", question)
        horizon = int(horizon_match.group(1)) if horizon_match else 7

        # Generic explanation requests should explain the current API behavior,
        # not accidentally treat the word "forecast" as a product identifier.
        wants_specific = any(term in question for term in (
            "forecast for", "predict for", "product ", "product id", "for product"
        ))
        if wants_specific and forecast_product_match:
            product_id = forecast_product_match.group(1).replace(" ", "").replace("_", "-")
            try:
                forecast_df = load_forecast_inventory()
                artifact = load_artifact()
                forecast_result = build_forecast(
                    forecast_df, artifact, product_id, horizon, False
                )
                return answer(
                    f"The {horizon}-day demand forecast for "
                    f"{forecast_result['product']['Product_Name']} "
                    f"(product {product_id}) is "
                    f"{_fmt_num(forecast_result['total_forecast_demand'])} units total, "
                    f"averaging {_fmt_num(forecast_result['average_daily_demand'])} units per day. "
                    f"Forecast method: {forecast_result['forecast_method']}.",
                    forecast_result,
                )
            except Exception as exc:
                return answer(
                    f"I couldn't generate a forecast for product {product_id}: {exc}",
                    {"product_id": product_id, "horizon_days": horizon, "error": str(exc)},
                )

        artifact = load_artifact()
        model_info = _voice_model_info()
        method = (
            "the saved trained machine-learning model"
            if artifact is not None
            else "a fallback baseline based on recent average demand, because no readable trained model artifact is available"
        )
        return answer(
            "Demand forecasting estimates future units required from historical demand. "
            "This API supports 7-day, 14-day, and 30-day forecasts. It builds daily features "
            "including calendar information, promotion status, recent demand lags, and rolling "
            f"averages. It currently uses {method}. Forecast values are estimates, not guarantees. "
            "You can ask, for example, 'forecast for product A for 14 days'.",
            {"supported_horizons": [7, 14, 30], "model_info": model_info},
        )

    # -----------------------------------------------------
    # TRAINED MODEL INFORMATION
    # -----------------------------------------------------

    if any(term in question for term in (
        "model info",
        "model information",
        "trained model",
        "training information",
        "which model",
        "model type",
        "features used",
        "what model",
        "ml model",
    )):
        info = _voice_model_info()

        if not info.get("available"):
            return answer(
                "I couldn't find a readable trained model artifact. "
                "Run python .\\ml\\train_model.py. Model status and "
                "available evaluation metrics are included in the response.",
                info,
            )

        message = (
            f"The saved model is {info.get('model_type') or 'available'}."
        )

        if info.get("features"):
            message += (
                " Stored input features: "
                + ", ".join(info["features"])
                + "."
            )

        if info.get("training_rows") is not None:
            message += f" Training rows: {info['training_rows']}."

        if info.get("training_date_range"):
            date_range = info["training_date_range"]
            message += (
                f" The dataset spans {date_range['start']} to "
                f"{date_range['end']} and contains "
                f"{date_range['dataset_rows']} rows."
            )

        return answer(message, info)

    if any(term in question for term in (
        "model",
        "mae",
        "r squared",
        "r2",
        "accuracy",
        "performance",
        "evaluation metrics",
        "rmse",
        "mape",
    )):
        result = _voice_model_info()
        metric_data = result.get("metrics") or {}

        if not result.get("available") and not metric_data.get("available"):
            return answer(
                "No trained model or evaluation metrics are available. "
                "Run python .\\ml\\train_model.py, then evaluate the model.",
                result,
            )

        return answer(
            "Model status and available evaluation metrics are included "
            "in the response. Regression metrics should not be described "
            "as classification accuracy unless the evaluation defines "
            "that metric.",
            result,
        )

    # -----------------------------------------------------
    # DATASET STATISTICS
    # -----------------------------------------------------

    if any(term in question for term in (
        "total records",
        "number of records",
        "how many rows",
        "dataset size",
        "how many records",
        "rows",
    )):
        return answer(
            f"The dataset contains {stats['total_rows']:,} records and "
            f"{stats['total_columns']} columns.",
            {
                "total_rows": stats["total_rows"],
                "total_columns": stats["total_columns"],
            },
        )

    if any(term in question for term in (
        "missing",
        "null",
        "empty cells",
    )):
        return answer(
            f"The dataset has {stats['missing_cells']:,} missing cells.",
            {"missing_cells": stats["missing_cells"]},
        )

    if any(term in question for term in (
        "average revenue",
        "average sales value",
        "average income",
    )):
        value = revenue.get("average")
        message = (
            f"Average estimated revenue per record is {_fmt_num(value)}."
            if value is not None
            else "A revenue column was not found in the dataset."
        )
        return answer(message, revenue)

    if any(term in question for term in (
        "total revenue",
        "revenue total",
        "overall revenue",
        "total sales value",
    )):
        value = revenue.get("total")
        message = (
            f"Total estimated revenue is {_fmt_num(value)}."
            if value is not None
            else "A revenue column was not found in the dataset."
        )
        return answer(message, revenue)

    if any(term in question for term in (
        "average demand",
        "mean demand",
        "average sales",
        "mean sales",
    )):
        value = demand.get("average")
        message = (
            f"Average demand is {_fmt_num(value)} units per record."
            if value is not None
            else "A demand column was not found in the dataset."
        )
        return answer(message, demand)

    if "median" in question and "demand" in question:
        value = demand.get("median")
        message = (
            f"Median demand is {_fmt_num(value)} units."
            if value is not None
            else "A demand column was not found in the dataset."
        )
        return answer(message, demand)

    if any(term in question for term in (
        "maximum demand",
        "highest demand",
        "max demand",
        "most demand",
    )):
        value = demand.get("maximum")
        message = (
            f"The highest demand value is {_fmt_num(value)} units."
            if value is not None
            else "A demand column was not found in the dataset."
        )
        return answer(message, demand)

    if any(term in question for term in (
        "minimum demand",
        "lowest demand",
        "min demand",
        "least demand",
    )):
        value = demand.get("minimum")
        message = (
            f"The lowest demand value is {_fmt_num(value)} units."
            if value is not None
            else "A demand column was not found in the dataset."
        )
        return answer(message, demand)

    if "standard deviation" in question or "spread of demand" in question:
        value = demand.get("standard_deviation")
        message = (
            f"Demand standard deviation is {_fmt_num(value)} units."
            if value is not None
            else "Demand standard deviation is not available."
        )
        return answer(message, demand)

    if any(term in question for term in (
        "how many products",
        "number of products",
        "unique products",
        "product count",
    )):
        value = stats.get("unique_products")
        message = (
            f"There are {value} unique products in the dataset."
            if value is not None
            else "A product column was not found."
        )
        return answer(message, {"unique_products": value})

    if "category" in question or "categories" in question:
        counts = stats["category_counts"]
        details = "; ".join(
            f"{key}: {value}" for key, value in counts.items()
        )
        return answer(
            f"There are {stats['unique_categories']} unique categories. "
            f"{details}",
            {"category_counts": counts},
        )

    if "region" in question or "regions" in question:
        counts = stats["region_counts"]
        details = "; ".join(
            f"{key}: {value}" for key, value in counts.items()
        )
        return answer(
            f"There are {stats['unique_regions']} unique regions. {details}",
            {"region_counts": counts},
        )

    if "supplier" in question or "suppliers" in question:
        counts = stats["supplier_counts"]
        details = "; ".join(
            f"{key}: {value}" for key, value in counts.items()
        )
        return answer(
            f"There are {stats['unique_suppliers']} unique suppliers. "
            f"{details}",
            {"supplier_counts": counts},
        )

    if any(term in question for term in (
        "stockout",
        "stock out",
        "risk",
    )):
        counts = stats["stockout_risk_counts"]

        if counts:
            details = "; ".join(
                f"{key}: {value}" for key, value in counts.items()
            )
            return answer(
                f"Stockout risk counts: {details}.",
                {"stockout_risk_counts": counts},
            )

        return answer(
            "The dataset does not contain a recognized stockout risk column."
        )

    if any(term in question for term in (
        "stock status",
        "inventory status",
        "out of stock",
        "low stock",
    )):
        counts = stats["stock_status_counts"]

        if counts:
            details = "; ".join(
                f"{key}: {value}" for key, value in counts.items()
            )
            return answer(
                f"Stock status counts: {details}.",
                {"stock_status_counts": counts},
            )

        return answer(
            "The dataset does not contain a recognized stock status column."
        )

    if "demand" in question and any(term in question for term in (
        "product",
        "each",
        "by product",
    )):
        values = stats["demand_by_product"]

        if values:
            ordered = sorted(
                values.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            details = "; ".join(
                f"{name}: {_fmt_num(value)} units"
                for name, value in ordered
            )
            return answer(
                "Total recorded demand by product is " + details + ".",
                {"demand_by_product": values},
            )

    return answer(
        "I couldn't identify that request. Try asking: "
        "'Find record INV-0001', 'load dataset details', "
        "'what is average demand?', 'tell me the trained model information', "
        "or 'show stockout risk'."
    )

@app.get("/api/voice/help")
def voice_help():
    """Return example questions supported by the voice assistant."""
    return {
        "examples": [
            "How many records are in the dataset?",
            "What is the average demand?",
            "What is the highest demand?",
            "What is the median demand?",
            "What is the total estimated revenue?",
            "How many products are there?",
            "Show demand by product.",
            "What is the stock status?",
            "What is the stockout risk?",
            "Tell me the model performance metrics.",
            "How many missing cells are there?",
            "Load the dataset and show its details.",
            "Explain the dataset.",
            "Which products need reorder?",
            "Explain the demand forecast.",
            "Forecast for product A for 14 days.",
            "Show record INV zero zero zero one.",
            "Tell me the trained model information and features.",
        ]
    }

