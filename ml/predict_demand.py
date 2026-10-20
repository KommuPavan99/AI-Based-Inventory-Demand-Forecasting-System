"""Command-line multi-day demand forecast for one product."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from backend.app import build_forecast, load_inventory, load_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast daily demand for a product.")
    parser.add_argument("--product", default="A", help="Product_ID, e.g. A")
    parser.add_argument("--days", type=int, choices=[7, 14, 30], default=7)
    parser.add_argument("--promotion", action="store_true", help="Assume promotion is active for the forecast horizon.")
    args = parser.parse_args()
    inventory = load_inventory()
    artifact = load_artifact()
    products = inventory["Product_ID"].astype(str).unique().tolist()
    if args.product not in products:
        parser.error(f"Unknown product '{args.product}'. Available: {', '.join(products)}")
    result = build_forecast(inventory, artifact, args.product, args.days, args.promotion)
    out = pd.DataFrame(result["daily_forecast"])
    print(f"Product: {result['product']['Product_Name']} ({args.product})")
    print(f"Horizon: {args.days} days | Promotion scenario: {'on' if args.promotion else 'off'}")
    print(f"Total forecast demand: {result['total_forecast_demand']:.1f} units")
    print(out.to_string(index=False))
    output_path = PROJECT_DIR / "data" / f"forecast_{args.product}_{args.days}d.csv"
    out.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
