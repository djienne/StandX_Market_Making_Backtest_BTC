from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


class DataFormat(Enum):
    STANDX = "standx"
    ORDERBOOK = "orderbook"

ROOT = Path(__file__).resolve().parent

_diff_path = ROOT / "hftbacktest" / "py-hftbacktest" / "hftbacktest" / "data" / "utils" / "difforderbooksnapshot.py"
_spec = importlib.util.spec_from_file_location("difforderbooksnapshot", _diff_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Unable to load difforderbooksnapshot from {_diff_path}")
_diff = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_diff)
DiffOrderBookSnapshot = _diff.DiffOrderBookSnapshot
CHANGED = _diff.CHANGED
INSERTED = _diff.INSERTED

DEPTH_EVENT = 1
TRADE_EVENT = 2
DEPTH_CLEAR_EVENT = 3
EXCH_EVENT = 1 << 31
LOCAL_EVENT = 1 << 30
BUY_EVENT = 1 << 29
SELL_EVENT = 1 << 28

event_dtype = np.dtype(
    [
        ("ev", "u8"),
        ("exch_ts", "i8"),
        ("local_ts", "i8"),
        ("px", "f8"),
        ("qty", "f8"),
        ("order_id", "u8"),
        ("ival", "i8"),
        ("fval", "f8"),
    ],
    align=True,
)


def read_price_files(paths: List[Path]) -> pd.DataFrame:
    bid_price_cols = [f"bid_price_{i}" for i in range(10)]
    bid_size_cols = [f"bid_size_{i}" for i in range(10)]
    ask_price_cols = [f"ask_price_{i}" for i in range(10)]
    ask_size_cols = [f"ask_size_{i}" for i in range(10)]
    cols = ["timestamp", "gap_detected"] + bid_price_cols + bid_size_cols + ask_price_cols + ask_size_cols

    frames: List[pd.DataFrame] = []
    for path in paths:
        frames.append(pd.read_parquet(path, columns=cols))

    if not frames:
        raise RuntimeError("No price snapshots were loaded")

    df = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    mask = ts.notna()
    df = df.loc[mask].copy()
    df["ts_ns"] = ts[mask].astype("int64").to_numpy()
    return df.sort_values("ts_ns", kind="mergesort").reset_index(drop=True)


def read_trade_files(paths: List[Path]) -> pd.DataFrame:
    cols = ["timestamp", "price", "size", "side", "trade_id"]
    frames: List[pd.DataFrame] = []
    for path in paths:
        frames.append(pd.read_parquet(path, columns=cols))

    if not frames:
        raise RuntimeError("No trades were loaded")

    df = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    mask = ts.notna()
    df = df.loc[mask].copy()
    df["ts_ns"] = ts[mask].astype("int64").to_numpy()
    df["side"] = df["side"].astype(str).str.lower()
    df = df.loc[df["side"].isin(["buy", "sell"])]
    return df.sort_values("ts_ns", kind="mergesort").reset_index(drop=True)


def detect_data_format(
    data_dir: Path, symbol: Optional[str]
) -> Tuple[DataFormat, List[Path], List[Path]]:
    """Detect data format based on file naming patterns.

    Returns (format, orderbook_paths, trade_paths).
    """
    # Check for new orderbook format first
    if symbol:
        orderbook_paths = sorted(data_dir.glob(f"orderbook_{symbol}_*.parquet"))
        trade_paths = sorted(data_dir.glob(f"trades_{symbol}_*.parquet"))
    else:
        orderbook_paths = sorted(data_dir.glob("orderbook_*.parquet"))
        trade_paths = sorted(data_dir.glob("trades_*.parquet"))

    if orderbook_paths:
        return DataFormat.ORDERBOOK, orderbook_paths, trade_paths

    # Fall back to original standx format
    if symbol:
        price_paths = sorted(data_dir.glob(f"prices_{symbol}_*.parquet"))
        trade_paths = sorted(data_dir.glob(f"trades_{symbol}_*.parquet"))
    else:
        all_price = sorted(data_dir.glob("prices_*.parquet"))
        if all_price:
            first_name = all_price[0].stem
            parts = first_name.split("_")
            if len(parts) >= 2:
                detected_symbol = parts[1]
                price_paths = sorted(data_dir.glob(f"prices_{detected_symbol}_*.parquet"))
                trade_paths = sorted(data_dir.glob(f"trades_{detected_symbol}_*.parquet"))
            else:
                price_paths = all_price
                trade_paths = sorted(data_dir.glob("trades_*.parquet"))
        else:
            price_paths = []
            trade_paths = []

    return DataFormat.STANDX, price_paths, trade_paths


def read_orderbook_files(paths: List[Path], max_levels: int = 10) -> pd.DataFrame:
    """Read orderbook parquet files with array-based columns.

    Flattens bid_prices/bid_quantities/ask_prices/ask_quantities arrays
    into flat columns compatible with the original format.
    """
    frames: List[pd.DataFrame] = []
    for path in paths:
        df = pd.read_parquet(path)
        frames.append(df)

    if not frames:
        raise RuntimeError("No orderbook files were loaded")

    df = pd.concat(frames, ignore_index=True)

    # Convert timestamp
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    mask = ts.notna()
    df = df.loc[mask].copy()
    df["ts_ns"] = ts[mask].astype("int64").to_numpy()

    # Convert is_valid to gap_detected (inverted logic)
    if "is_valid" in df.columns:
        df["gap_detected"] = ~df["is_valid"].astype(bool)
    else:
        df["gap_detected"] = False

    # Vectorized extraction of array columns
    bid_px_arr = np.vstack(df["bid_prices"].values)
    bid_qty_arr = np.vstack(df["bid_quantities"].values)
    ask_px_arr = np.vstack(df["ask_prices"].values)
    ask_qty_arr = np.vstack(df["ask_quantities"].values)

    # Limit to max_levels
    actual_levels = min(max_levels, bid_px_arr.shape[1])
    for i in range(actual_levels):
        df[f"bid_price_{i}"] = bid_px_arr[:, i]
        df[f"bid_size_{i}"] = bid_qty_arr[:, i]
        df[f"ask_price_{i}"] = ask_px_arr[:, i]
        df[f"ask_size_{i}"] = ask_qty_arr[:, i]

    # Drop original columns
    cols_to_drop = [
        "bid_prices", "bid_quantities", "ask_prices", "ask_quantities",
        "is_valid", "received_at", "symbol", "best_bid", "best_ask",
        "spread", "mid_price", "timestamp",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    return df.sort_values("ts_ns", kind="mergesort").reset_index(drop=True)


def read_trade_files_new_format(paths: List[Path]) -> pd.DataFrame:
    """Read trade parquet files with new format.

    Handles: quantity->size, BUY/SELL->buy/sell, generates trade_id.
    """
    frames: List[pd.DataFrame] = []
    for path in paths:
        df = pd.read_parquet(path)
        frames.append(df)

    if not frames:
        raise RuntimeError("No trades were loaded")

    df = pd.concat(frames, ignore_index=True)

    # Convert timestamp
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    mask = ts.notna()
    df = df.loc[mask].copy()
    df["ts_ns"] = ts[mask].astype("int64").to_numpy()

    # Rename quantity to size
    if "quantity" in df.columns:
        df["size"] = df["quantity"]
    elif "size" not in df.columns:
        raise RuntimeError("No size/quantity column found in trade data")

    # Normalize side to lowercase
    df["side"] = df["side"].astype(str).str.lower()

    # If side data is unreliable (all same value), infer from is_buyer_taker
    unique_sides = df["side"].unique()
    if len(unique_sides) == 1 and "is_buyer_taker" in df.columns:
        df["side"] = np.where(df["is_buyer_taker"], "buy", "sell")
        print("INFO: Trade side inferred from is_buyer_taker field")

    # Generate trade_id if not present
    if "trade_id" not in df.columns:
        df["trade_id"] = np.arange(len(df))

    df = df.loc[df["side"].isin(["buy", "sell"])]

    return df.sort_values("ts_ns", kind="mergesort").reset_index(drop=True)


def _min_positive(values: np.ndarray) -> Optional[float]:
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return None
    return float(values.min())


def infer_tick_size(price_df: pd.DataFrame, max_levels: int = 10) -> float:
    bid_cols = [f"bid_price_{i}" for i in range(max_levels)]
    ask_cols = [f"ask_price_{i}" for i in range(max_levels)]
    # Filter to existing columns only
    bid_cols = [c for c in bid_cols if c in price_df.columns]
    ask_cols = [c for c in ask_cols if c in price_df.columns]
    if not bid_cols or not ask_cols:
        raise RuntimeError("No price columns found in orderbook data")
    bid = price_df[bid_cols].to_numpy()
    ask = price_df[ask_cols].to_numpy()
    bid[bid <= 0] = np.nan
    ask[ask <= 0] = np.nan
    bid_diffs = bid[:, :-1] - bid[:, 1:]
    ask_diffs = ask[:, 1:] - ask[:, :-1]
    diffs = np.concatenate([bid_diffs, ask_diffs], axis=1)
    diffs[diffs <= 0] = np.nan
    if np.isnan(diffs).all():
        raise RuntimeError("Unable to infer tick size from order book data")
    tick = float(np.nanmin(diffs))
    if not math.isfinite(tick) or tick <= 0:
        raise RuntimeError("Invalid tick size inferred")
    return tick


def infer_lot_size(price_df: pd.DataFrame, trade_df: pd.DataFrame, max_levels: int = 10) -> float:
    size_cols = [f"bid_size_{i}" for i in range(max_levels)] + [f"ask_size_{i}" for i in range(max_levels)]
    # Filter to existing columns only
    size_cols = [c for c in size_cols if c in price_df.columns]
    if size_cols:
        sizes = price_df[size_cols].to_numpy()
        sizes[sizes <= 0] = np.nan
        min_book = _min_positive(sizes)
    else:
        min_book = None

    trade_sizes = trade_df["size"].to_numpy(dtype=np.float64)
    trade_sizes[trade_sizes <= 0] = np.nan
    min_trade = _min_positive(trade_sizes)

    candidates = [v for v in [min_book, min_trade] if v is not None]
    if not candidates:
        raise RuntimeError("Unable to infer lot size from data")
    return float(min(candidates))


def align_time_window(
    price_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    max_rows: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    price_df = price_df.sort_values("ts_ns", kind="mergesort")
    trade_df = trade_df.sort_values("ts_ns", kind="mergesort")
    window_start = max(int(price_df["ts_ns"].min()), int(trade_df["ts_ns"].min()))
    window_end = min(int(price_df["ts_ns"].max()), int(trade_df["ts_ns"].max()))
    price_df = price_df[(price_df["ts_ns"] >= window_start) & (price_df["ts_ns"] <= window_end)]
    trade_df = trade_df[(trade_df["ts_ns"] >= window_start) & (trade_df["ts_ns"] <= window_end)]

    if max_rows is not None:
        price_df = price_df.head(max_rows)
        if not price_df.empty:
            window_start = int(price_df["ts_ns"].iloc[0])
            window_end = int(price_df["ts_ns"].iloc[-1])
            trade_df = trade_df[(trade_df["ts_ns"] >= window_start) & (trade_df["ts_ns"] <= window_end)]

    if price_df.empty or trade_df.empty:
        raise RuntimeError("No overlapping data between prices and trades after filtering.")

    return price_df.reset_index(drop=True), trade_df.reset_index(drop=True)


def build_event_array(
    price_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    tick_size: float,
    lot_size: float,
    latency_ns: int,
    max_levels: int = 10,
) -> np.ndarray:
    bid_price_cols = [f"bid_price_{i}" for i in range(max_levels)]
    bid_size_cols = [f"bid_size_{i}" for i in range(max_levels)]
    ask_price_cols = [f"ask_price_{i}" for i in range(max_levels)]
    ask_size_cols = [f"ask_size_{i}" for i in range(max_levels)]

    # Filter to existing columns only
    bid_price_cols = [c for c in bid_price_cols if c in price_df.columns]
    bid_size_cols = [c for c in bid_size_cols if c in price_df.columns]
    ask_price_cols = [c for c in ask_price_cols if c in price_df.columns]
    ask_size_cols = [c for c in ask_size_cols if c in price_df.columns]
    actual_levels = len(bid_price_cols)

    bid_px = price_df[bid_price_cols].to_numpy(dtype=np.float64)
    bid_qty = price_df[bid_size_cols].to_numpy(dtype=np.float64)
    ask_px = price_df[ask_price_cols].to_numpy(dtype=np.float64)
    ask_qty = price_df[ask_size_cols].to_numpy(dtype=np.float64)
    price_ts = price_df["ts_ns"].to_numpy(dtype=np.int64)
    gap_detected = None
    if "gap_detected" in price_df.columns:
        gap_detected = price_df["gap_detected"].to_numpy(dtype=bool)

    trade_ts = trade_df["ts_ns"].to_numpy(dtype=np.int64)
    trade_px = trade_df["price"].to_numpy(dtype=np.float64)
    trade_qty = trade_df["size"].to_numpy(dtype=np.float64)
    trade_side = trade_df["side"].to_numpy()

    events: List[Tuple[int, int, int, float, float, int, int, float]] = []

    diff = DiffOrderBookSnapshot(actual_levels, tick_size, lot_size)
    depth_ev_base = DEPTH_EVENT

    for i in range(len(price_ts)):
        ts = int(price_ts[i])
        exch_ts = ts
        local_ts = ts + latency_ns
        if gap_detected is not None and gap_detected[i]:
            events.append((DEPTH_CLEAR_EVENT, exch_ts, local_ts, 0.0, 0.0, 0, 0, 0.0))
            diff = DiffOrderBookSnapshot(actual_levels, tick_size, lot_size)

        curr_bids, curr_asks, bid_delete, ask_delete = diff.snapshot(
            bid_px[i], bid_qty[i], ask_px[i], ask_qty[i]
        )

        for j in range(curr_bids.shape[0]):
            flag = int(curr_bids[j, 2])
            if flag != CHANGED and flag != INSERTED:
                continue
            px = float(curr_bids[j, 0])
            qty = float(curr_bids[j, 1])
            ev = depth_ev_base | BUY_EVENT
            events.append((ev, exch_ts, local_ts, px, qty, 0, 0, 0.0))

        for j in range(curr_asks.shape[0]):
            flag = int(curr_asks[j, 2])
            if flag != CHANGED and flag != INSERTED:
                continue
            px = float(curr_asks[j, 0])
            qty = float(curr_asks[j, 1])
            ev = depth_ev_base | SELL_EVENT
            events.append((ev, exch_ts, local_ts, px, qty, 0, 0, 0.0))

        for j in range(bid_delete.shape[0]):
            px = float(bid_delete[j, 0])
            ev = depth_ev_base | BUY_EVENT
            events.append((ev, exch_ts, local_ts, px, 0.0, 0, 0, 0.0))

        for j in range(ask_delete.shape[0]):
            px = float(ask_delete[j, 0])
            ev = depth_ev_base | SELL_EVENT
            events.append((ev, exch_ts, local_ts, px, 0.0, 0, 0, 0.0))

    # Build depth events array
    depth_arr = np.array(events, dtype=event_dtype)

    # Vectorized trade events
    buy_mask = trade_side == "buy"
    sell_mask = trade_side == "sell"
    valid_mask = buy_mask | sell_mask
    n_trades = int(np.sum(valid_mask))
    if n_trades > 0:
        trade_arr = np.zeros(n_trades, dtype=event_dtype)
        trade_arr["ev"] = np.where(buy_mask[valid_mask], TRADE_EVENT | BUY_EVENT, TRADE_EVENT | SELL_EVENT)
        trade_arr["exch_ts"] = trade_ts[valid_mask]
        trade_arr["local_ts"] = trade_ts[valid_mask] + latency_ns
        trade_arr["px"] = trade_px[valid_mask]
        trade_arr["qty"] = trade_qty[valid_mask]
        arr = np.concatenate([depth_arr, trade_arr])
    else:
        arr = depth_arr
    # With constant latency, exch_ts and local_ts have identical relative ordering.
    # Single lexsort is sufficient; apply both EXCH_EVENT and LOCAL_EVENT flags directly.
    order = np.lexsort((arr["ev"], arr["px"], arr["exch_ts"]))
    arr = arr[order]
    arr["ev"] |= EXCH_EVENT | LOCAL_EVENT
    return arr


def convert_parquet_to_npz(
    data_dir: Path,
    out_path: Path,
    max_rows: Optional[int],
    latency_ns: int,
    symbol: Optional[str] = None,
    max_levels: int = 10,
) -> Tuple[float, float, int, bool]:
    # Detect format and get file paths
    data_format, price_paths, trade_paths = detect_data_format(data_dir, symbol)

    if not price_paths:
        raise FileNotFoundError(f"No price/orderbook parquet files found under {data_dir}")
    if not trade_paths:
        raise FileNotFoundError(f"No trade parquet files found under {data_dir}")

    print(f"detected format={data_format.value} orderbook_files={len(price_paths)} trade_files={len(trade_paths)}")

    fingerprint = _fingerprint_inputs(price_paths + trade_paths)
    meta_path = _meta_path_for(out_path)
    cached = _load_meta(meta_path)
    if (
        out_path.exists()
        and cached is not None
        and cached.get("input_fingerprint") == fingerprint
        and cached.get("max_rows") == max_rows
        and cached.get("latency_ns") == latency_ns
        and cached.get("max_levels") == max_levels
    ):
        return (
            float(cached["tick_size"]),
            float(cached["lot_size"]),
            int(cached["event_count"]),
            False,
        )

    # Read data using appropriate reader based on format
    if data_format == DataFormat.ORDERBOOK:
        price_df = read_orderbook_files(price_paths, max_levels)
        trade_df = read_trade_files_new_format(trade_paths)
    else:
        price_df = read_price_files(price_paths)
        trade_df = read_trade_files(trade_paths)

    price_df, trade_df = align_time_window(price_df, trade_df, max_rows)

    price_rows = int(len(price_df))
    trade_rows = int(len(trade_df))
    price_min_ns = int(price_df["ts_ns"].min())
    price_max_ns = int(price_df["ts_ns"].max())
    trade_min_ns = int(trade_df["ts_ns"].min())
    trade_max_ns = int(trade_df["ts_ns"].max())
    overlap_start_ns = max(price_min_ns, trade_min_ns)
    overlap_end_ns = min(price_max_ns, trade_max_ns)

    tick_size = infer_tick_size(price_df, max_levels)
    lot_size = infer_lot_size(price_df, trade_df, max_levels)
    events = build_event_array(price_df, trade_df, tick_size, lot_size, latency_ns, max_levels)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, data=events)
    _save_meta(
        meta_path,
        {
            "version": 3,
            "data_format": data_format.value,
            "data_dir": str(data_dir),
            "max_rows": max_rows,
            "latency_ns": latency_ns,
            "max_levels": max_levels,
            "input_fingerprint": fingerprint,
            "tick_size": tick_size,
            "lot_size": lot_size,
            "event_count": len(events),
            "price_files": len(price_paths),
            "trade_files": len(trade_paths),
            "price_rows": price_rows,
            "trade_rows": trade_rows,
            "price_min_ns": price_min_ns,
            "price_max_ns": price_max_ns,
            "trade_min_ns": trade_min_ns,
            "trade_max_ns": trade_max_ns,
            "overlap_start_ns": overlap_start_ns,
            "overlap_end_ns": overlap_end_ns,
        },
    )
    return tick_size, lot_size, len(events), True


def _fingerprint_inputs(paths: List[Path]) -> str:
    hasher = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p)):
        stat = path.stat()
        hasher.update(str(path).encode("utf-8"))
        hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _meta_path_for(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".meta.json")


def _load_meta(meta_path: Path) -> Optional[dict]:
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_meta(meta_path: Path, payload: dict) -> None:
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert parquet data into hftbacktest npz.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="data/btc_hft.npz")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--latency-ns", type=int, default=1_000_000)
    parser.add_argument("--symbol", type=str, default=None, help="Symbol to filter (e.g. BTC-USD). Auto-detected if not specified.")
    parser.add_argument("--max-levels", type=int, default=10, help="Number of orderbook levels to use (default: 10).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    tick_size, lot_size, count, converted = convert_parquet_to_npz(
        data_dir, out_path, args.max_rows, args.latency_ns, args.symbol, args.max_levels
    )
    print(f"tick_size={tick_size} lot_size={lot_size}")
    if converted:
        print(f"wrote {count} events to {out_path}")
    else:
        print(f"conversion skipped; {count} events already in {out_path}")


if __name__ == "__main__":
    main()
