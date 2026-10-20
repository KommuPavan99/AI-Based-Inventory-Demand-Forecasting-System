
import sqlite3
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / "data" / "dataset.xlsx"
DB_PATH = BASE_DIR / "data" / "inventory.db"

def initialize_database():
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Product_ID", "Date"])

    # Keep all dated demand observations for forecasting.
    history = df[
        ["Record_ID", "Date", "Product_ID", "Demand_Units",
         "Promotion_Active"]
    ].copy()

    # One latest record per product determines initial stock.
    latest = (
        df.sort_values(["Date", "Record_ID"])
        .drop_duplicates(subset=["Product_ID"], keep="last")
        .copy()
    )

    connection = sqlite3.connect(DB_PATH)

    try:
        connection.execute("PRAGMA foreign_keys = ON")

        existing = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='products'"
        ).fetchone()

        if existing:
            count = connection.execute(
                "SELECT COUNT(*) FROM products"
            ).fetchone()[0]

            if count > 0:
                print(
                    f"Database already initialized with {count} products. "
                    "No data was changed."
                )
                return

        connection.executescript("""
            CREATE TABLE IF NOT EXISTS products (
                product_id TEXT PRIMARY KEY,
                product_name TEXT NOT NULL,
                category TEXT,
                unit_price REAL NOT NULL DEFAULT 0,
                stock_on_hand INTEGER NOT NULL DEFAULT 0
                    CHECK (stock_on_hand >= 0),
                supplier TEXT,
                region TEXT,
                lead_time_days INTEGER NOT NULL DEFAULT 0,
                reorder_level INTEGER NOT NULL DEFAULT 0,
                promotion_active INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS demand_history (
                record_id TEXT PRIMARY KEY,
                observation_date TEXT NOT NULL,
                product_id TEXT NOT NULL,
                demand_units INTEGER NOT NULL DEFAULT 0,
                promotion_active INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (product_id)
                    REFERENCES products(product_id)
            );

            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,
                transaction_type TEXT NOT NULL
                    CHECK (transaction_type IN ('SALE', 'PURCHASE')),
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                unit_price REAL NOT NULL DEFAULT 0,
                transaction_date TEXT NOT NULL,
                notes TEXT,
                FOREIGN KEY (product_id)
                    REFERENCES products(product_id)
            );

            CREATE INDEX IF NOT EXISTS idx_history_product_date
            ON demand_history(product_id, observation_date);

            CREATE INDEX IF NOT EXISTS idx_transactions_product_date
            ON transactions(product_id, transaction_date);
        """)

        products = latest.rename(columns={
            "Product_ID": "product_id",
            "Product_Name": "product_name",
            "Category": "category",
            "Unit_Price": "unit_price",
            "Stock_On_Hand": "stock_on_hand",
            "Supplier": "supplier",
            "Region": "region",
            "Lead_Time_Days": "lead_time_days",
            "Reorder_Level": "reorder_level",
            "Promotion_Active": "promotion_active",
        })[
            [
                "product_id", "product_name", "category",
                "unit_price", "stock_on_hand", "supplier",
                "region", "lead_time_days", "reorder_level",
                "promotion_active"
            ]
        ]

        history = history.rename(columns={
            "Record_ID": "record_id",
            "Date": "observation_date",
            "Product_ID": "product_id",
            "Demand_Units": "demand_units",
            "Promotion_Active": "promotion_active",
        })

        history["observation_date"] = (
            history["observation_date"].dt.strftime("%Y-%m-%d")
        )

        with connection:
            products.to_sql(
                "products", connection,
                if_exists="append", index=False
            )
            history.to_sql(
                "demand_history", connection,
                if_exists="append", index=False
            )

        print("Database initialized successfully.")
        print("Products imported:", len(products))
        print("Demand history records imported:", len(history))
        print("Database location:", DB_PATH)

    finally:
        connection.close()


if __name__ == "__main__":
    initialize_database()
