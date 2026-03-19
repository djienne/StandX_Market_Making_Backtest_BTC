"""
Parquet File Reader - Display samples in a friendly, readable format.
Usage: python read_parquet.py [--orderbook] [--trades] [--rows N] [--all]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def format_price(price):
    """Format price with commas and 2 decimals."""
    if price is None or pd.isna(price):
        return "N/A"
    return f"{price:,.2f}"


def format_qty(qty):
    """Format quantity with 4 decimals."""
    if qty is None or pd.isna(qty):
        return "N/A"
    return f"{qty:.4f}"


def format_timestamp(ts):
    """Format timestamp for display."""
    if ts is None or pd.isna(ts):
        return "N/A"
    if isinstance(ts, str):
        return ts[:23]
    return str(ts)[:23]


def display_orderbook_sample(df: pd.DataFrame, num_rows: int = 5):
    """Display orderbook samples in a friendly format."""
    print("\n" + "=" * 80)
    print("ORDERBOOK DATA")
    print("=" * 80)
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()

    # Show sample rows
    sample = df.tail(num_rows)

    for idx, row in sample.iterrows():
        print("-" * 80)
        print(f"Record #{idx} | {row['symbol']} | {format_timestamp(row['timestamp'])}")
        print("-" * 80)

        # Summary line
        print(f"  Best Bid: {format_price(row['best_bid'])}  |  "
              f"Best Ask: {format_price(row['best_ask'])}  |  "
              f"Spread: {format_price(row['spread'])}  |  "
              f"Mid: {format_price(row['mid_price'])}")
        print()

        # Orderbook display
        bid_prices = row['bid_prices']
        bid_qtys = row['bid_quantities']
        ask_prices = row['ask_prices']
        ask_qtys = row['ask_quantities']

        print(f"  {'BIDS':<30} | {'ASKS':<30}")
        print(f"  {'Price':<14} {'Qty':<14} | {'Price':<14} {'Qty':<14}")
        print(f"  {'-'*28} | {'-'*28}")

        # Show top 10 levels
        num_levels = min(10, len(bid_prices), len(ask_prices))
        for i in range(num_levels):
            bid_p = format_price(bid_prices[i]) if i < len(bid_prices) else ""
            bid_q = format_qty(bid_qtys[i]) if i < len(bid_qtys) else ""
            ask_p = format_price(ask_prices[i]) if i < len(ask_prices) else ""
            ask_q = format_qty(ask_qtys[i]) if i < len(ask_prices) else ""

            print(f"  {bid_p:<14} {bid_q:<14} | {ask_p:<14} {ask_q:<14}")

        if len(bid_prices) > 10:
            print(f"  ... ({len(bid_prices) - 10} more levels)")

        print()


def display_trades_sample(df: pd.DataFrame, num_rows: int = 10):
    """Display trades samples in a friendly format."""
    print("\n" + "=" * 80)
    print("TRADES DATA")
    print("=" * 80)
    print(f"Total records: {len(df)}")
    if len(df) > 0:
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Calculate stats
        buy_count = (df['side'] == 'BUY').sum()
        sell_count = (df['side'] == 'SELL').sum()
        total_volume = df['quantity'].sum()
        avg_price = df['price'].mean()

        print(f"Buy trades: {buy_count} | Sell trades: {sell_count}")
        print(f"Total volume: {format_qty(total_volume)} | Avg price: {format_price(avg_price)}")
    print()

    if len(df) == 0:
        print("No trades recorded yet.")
        return

    # Show sample rows
    sample = df.tail(num_rows)

    print(f"{'Timestamp':<24} {'Side':<6} {'Price':<14} {'Quantity':<12} {'Value':<14}")
    print("-" * 80)

    for idx, row in sample.iterrows():
        side_color = "\033[92m" if row['side'] == 'BUY' else "\033[91m"
        reset = "\033[0m"

        ts = format_timestamp(row['timestamp'])
        side = f"{side_color}{row['side']:<6}{reset}"
        price = format_price(row['price'])
        qty = format_qty(row['quantity'])
        value = format_price(row.get('quote_quantity')) if row.get('quote_quantity') else "N/A"

        print(f"{ts:<24} {side} {price:<14} {qty:<12} {value:<14}")

    print()


def display_raw_sample(df: pd.DataFrame, num_rows: int = 3):
    """Display raw DataFrame sample."""
    print("\n" + "=" * 80)
    print("RAW DATA SAMPLE (first few columns)")
    print("=" * 80)

    # For display, limit columns if there are list columns
    display_cols = []
    for col in df.columns:
        if col not in ['bid_prices', 'bid_quantities', 'ask_prices', 'ask_quantities']:
            display_cols.append(col)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)

    print(df[display_cols].tail(num_rows).to_string())
    print()

    # Show column info
    print("\nColumn types:")
    for col in df.columns:
        dtype = df[col].dtype
        sample = df[col].iloc[0] if len(df) > 0 else None
        if isinstance(sample, list):
            print(f"  {col}: list (length {len(sample)})")
        else:
            print(f"  {col}: {dtype}")


def find_latest_files(data_dir: Path):
    """Find the latest orderbook and trades files."""
    orderbook_files = sorted(data_dir.glob("orderbook_*.parquet"), reverse=True)
    trades_files = sorted(data_dir.glob("trades_*.parquet"), reverse=True)

    return (
        orderbook_files[0] if orderbook_files else None,
        trades_files[0] if trades_files else None
    )


def main():
    parser = argparse.ArgumentParser(
        description="Read and display Parquet files in a friendly format"
    )
    parser.add_argument(
        "--orderbook", "-o",
        action="store_true",
        help="Show orderbook data"
    )
    parser.add_argument(
        "--trades", "-t",
        action="store_true",
        help="Show trades data"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Show all data (orderbook + trades)"
    )
    parser.add_argument(
        "--rows", "-r",
        type=int,
        default=5,
        help="Number of rows to display (default: 5)"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Also show raw DataFrame format"
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="./data",
        help="Data directory (default: ./data)"
    )
    parser.add_argument(
        "--file", "-f",
        help="Specific parquet file to read"
    )

    args = parser.parse_args()

    # Default to showing all if nothing specified
    if not args.orderbook and not args.trades and not args.all and not args.file:
        args.all = True

    data_dir = Path(args.data_dir)

    if args.file:
        # Read specific file
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return 1

        df = pd.read_parquet(file_path)
        print(f"\nReading: {file_path}")

        # Detect file type
        if 'bid_prices' in df.columns:
            display_orderbook_sample(df, args.rows)
        elif 'side' in df.columns:
            display_trades_sample(df, args.rows)
        else:
            display_raw_sample(df, args.rows)

        if args.raw:
            display_raw_sample(df, args.rows)

        return 0

    # Find latest files
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Run data_collector.py first to collect some data.")
        return 1

    orderbook_file, trades_file = find_latest_files(data_dir)

    if not orderbook_file and not trades_file:
        print(f"No parquet files found in {data_dir}")
        print("Run data_collector.py first to collect some data.")
        return 1

    # Display requested data
    if args.all or args.orderbook:
        if orderbook_file:
            print(f"\nReading: {orderbook_file}")
            df = pd.read_parquet(orderbook_file)
            display_orderbook_sample(df, args.rows)
            if args.raw:
                display_raw_sample(df, args.rows)
        else:
            print("\nNo orderbook files found")

    if args.all or args.trades:
        if trades_file:
            print(f"\nReading: {trades_file}")
            df = pd.read_parquet(trades_file)
            display_trades_sample(df, args.rows)
            if args.raw:
                display_raw_sample(df, args.rows)
        else:
            print("\nNo trades files found")

    return 0


if __name__ == "__main__":
    exit(main())
