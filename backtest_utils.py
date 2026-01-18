"""Shared utility functions for backtest scripts."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


def load_meta(out_path: Path) -> Optional[dict]:
    """Load metadata JSON file associated with an npz file."""
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def format_ns(ts_ns: Optional[int]) -> str:
    """Convert nanosecond timestamp to ISO format string."""
    if ts_ns is None:
        return "n/a"
    try:
        dt = datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=timezone.utc)
    except Exception:
        return str(ts_ns)
    return dt.isoformat().replace("+00:00", "Z")


def detect_time_gaps(data: np.ndarray, gap_threshold_ns: int) -> tuple[np.ndarray, np.ndarray]:
    """Detect gaps larger than threshold in local timestamps."""
    if gap_threshold_ns <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    ts = data["local_ts"].astype(np.int64)
    if ts.size < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if np.any(ts[1:] < ts[:-1]):
        ts = np.sort(ts)
    diffs = np.diff(ts)
    idx = np.where(diffs > gap_threshold_ns)[0]
    if idx.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    gap_starts = ts[idx] + 1
    gap_ends = ts[idx + 1]
    return gap_starts.astype(np.int64), gap_ends.astype(np.int64)


def build_gap_context(
    data: np.ndarray,
    gap_threshold_minutes: float,
    base_ts_ns: int | None = None,
) -> tuple[int, np.ndarray, np.ndarray, str | None]:
    """Build gap detection context and optional log line."""
    if base_ts_ns is None:
        base_ts_ns = int(data["local_ts"].min()) if len(data) > 0 else 0
    gap_threshold_ns = int(max(0.0, gap_threshold_minutes) * 60 * 1_000_000_000)
    gap_starts_ns, gap_ends_ns = detect_time_gaps(data, gap_threshold_ns)
    log_line = None
    if gap_starts_ns.size > 0:
        max_gap_ns = int(np.max(gap_ends_ns - gap_starts_ns))
        log_line = (
            "gap_filter: "
            f"threshold_min={gap_threshold_minutes} "
            f"gaps={len(gap_starts_ns)} "
            f"max_gap_s={max_gap_ns / 1_000_000_000:.1f}"
        )
    return base_ts_ns, gap_starts_ns, gap_ends_ns, log_line


def print_meta_summary(meta: dict, out_path: Path) -> None:
    """Print summary of conversion metadata."""
    price_files = meta.get("price_files")
    trade_files = meta.get("trade_files")
    price_rows = meta.get("price_rows")
    trade_rows = meta.get("trade_rows")
    overlap_start = meta.get("overlap_start_ns")
    overlap_end = meta.get("overlap_end_ns")

    if price_files is not None or trade_files is not None:
        print(f"price_files={price_files} trade_files={trade_files}")
    if price_rows is not None or trade_rows is not None:
        print(f"price_rows={price_rows} trade_rows={trade_rows}")
    if overlap_start is not None or overlap_end is not None:
        print(
            "overlap_window=",
            format_ns(overlap_start),
            "to",
            format_ns(overlap_end),
        )
    if out_path.exists():
        print(f"npz_path={out_path}")


def save_plots(
    records: np.ndarray,
    plots_dir: Path,
    stem: str,
    include_trading_value: bool = True,
    title: str | None = None,
) -> None:
    """Save equity, position, and trading value plots.

    Args:
        title: Optional title/subtitle to display on plots (e.g. parameter values)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except Exception as exc:
        print(f"plotting skipped: {exc}")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    timestamps = np.asarray(records["timestamp"], dtype="datetime64[ns]")
    balance = records["balance"].astype(np.float64)
    position = records["position"].astype(np.float64)
    price = records["price"].astype(np.float64)
    fee = records["fee"].astype(np.float64)
    equity = balance + position * price - fee

    has_trading_value = (
        include_trading_value
        and "trading_value" in records.dtype.names
    )
    if has_trading_value:
        trading_value = records["trading_value"].astype(np.float64)

    if len(timestamps) == 0:
        print("plotting skipped: no timestamps in records")
        return

    # Balance and equity plot
    fig, ax = plt.subplots()
    if title:
        fig.suptitle(title, fontsize=10)
    ax.plot(timestamps, balance, label="balance (quote)")
    ax.plot(timestamps, equity, label="equity (quote)")
    ax.set_xlabel("time")
    ax.set_ylabel("value (quote)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plots_dir / f"{stem}_balance_equity.png", dpi=150)
    plt.close(fig)

    # Position and equity dual-axis plot
    fig, ax1 = plt.subplots()
    if title:
        fig.suptitle(title, fontsize=10)
    ax1.plot(timestamps, equity, color="tab:blue", label="equity (quote)")
    ax1.set_xlabel("time")
    ax1.set_ylabel("equity (quote)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(timestamps, position, color="tab:orange", label="position (base)")
    ax2.set_ylabel("position (base)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plots_dir / f"{stem}_position_equity.png", dpi=150)
    plt.close(fig)

    # Trading value plot (optional)
    if has_trading_value and np.any(np.isfinite(trading_value)):
        fig, ax = plt.subplots()
        if title:
            fig.suptitle(title, fontsize=10)
        ax.plot(timestamps, trading_value, label="trading_value")
        ax.set_xlabel("time")
        ax.set_ylabel("cumulative trading value (quote)")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(plots_dir / f"{stem}_trading_value.png", dpi=150)
        plt.close(fig)

    print(f"plots_saved_to={plots_dir}")


def float_grid(start: float, stop: float, step: float, decimals: int = 6) -> list[float]:
    """Generate a grid of float values from start to stop (inclusive) with given step."""
    if step <= 0:
        return [start]
    values = []
    current = start
    while current <= stop + step * 0.01:
        values.append(round(current, decimals))
        current += step
    return values


def fmt_float(value: float, precision: int = 6) -> str:
    """Format float to given precision or 'nan' if not finite."""
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{precision}g}"


def progress_str(done: int, total: int) -> str:
    """Format progress as 'done/total (pct%)'."""
    if total <= 0:
        return "0/0 (0.0%)"
    pct = done * 100.0 / total
    return f"{done}/{total} ({pct:.1f}%)"


def load_threads_from_config(config_path: Path, default_threads: int = 4) -> int:
    """Load thread count from config file, with fallback to default."""
    if not config_path.exists():
        return default_threads
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        value = cfg.get("threads")
        if value is not None and int(value) > 0:
            return int(value)
    except Exception:
        pass
    return default_threads


def load_symbol_from_config(
    config_path: Path,
    default_symbol: str | None = "BTC-USD",
) -> str | None:
    """Load symbol from config file, with fallback to default."""
    if not config_path.exists():
        return default_symbol
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        symbol = cfg.get("symbol")
        if symbol:
            return str(symbol)
    except Exception:
        pass
    return default_symbol


def compute_study_fingerprint(config: dict, npz_meta_path: Path) -> str:
    """Compute a fingerprint of study config + data for cache invalidation.

    Returns a hash that changes when:
    - NPZ data changes (via meta.json fingerprint)
    - Search space parameters change
    - Key backtest parameters change
    """
    import hashlib

    hasher = hashlib.sha256()

    # Include NPZ data fingerprint from meta.json
    if npz_meta_path.exists():
        try:
            meta = json.loads(npz_meta_path.read_text(encoding="utf-8"))
            hasher.update(f"npz:{meta.get('input_fingerprint', '')}".encode())
            hasher.update(f"max_rows:{meta.get('max_rows')}".encode())
        except Exception:
            hasher.update(b"npz:unknown")
    else:
        hasher.update(b"npz:missing")

    # Include search space (sorted for consistency)
    search_space = config.get("search_space", {})
    hasher.update(json.dumps(search_space, sort_keys=True).encode())

    # Include key backtest parameters that affect results
    bt_config = config.get("backtest", {})
    for key in ["latency_ns", "step_ns", "order_qty_dollar", "max_position_dollar",
                "grid_num", "window_steps", "update_interval_steps", "gap_threshold_minutes"]:
        hasher.update(f"{key}:{bt_config.get(key)}".encode())

    # Include optimization parameters
    opt_config = config.get("optimization", {})
    hasher.update(f"objective:{opt_config.get('objective_metric')}".encode())
    hasher.update(f"min_trades:{opt_config.get('min_trades')}".encode())

    return hasher.hexdigest()[:16]  # Short hash for readability


def load_fees_from_config(
    config_path: Path,
    default_maker_fee: float = 0.0001,
    default_taker_fee: float = 0.0004,
) -> tuple[float, float]:
    """Load maker and taker fees from config file, with fallback to defaults.

    Returns:
        Tuple of (maker_fee, taker_fee)
    """
    maker_fee = default_maker_fee
    taker_fee = default_taker_fee
    if not config_path.exists():
        return maker_fee, taker_fee
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if "maker_fee" in cfg:
            maker_fee = float(cfg["maker_fee"])
        if "taker_fee" in cfg:
            taker_fee = float(cfg["taker_fee"])
    except Exception:
        pass
    return maker_fee, taker_fee


def extract_backtest_results(records: np.ndarray) -> Optional[dict]:
    """Extract summary statistics from backtest records.

    Returns None if no valid records, otherwise dict with:
    - timestamp, price, position, balance, fee
    - equity_wo_fee, equity, num_trades, max_abs_position
    """
    if len(records) == 0:
        return None

    valid_mask = np.isfinite(records["price"])
    if not np.any(valid_mask):
        return None

    last = records[np.where(valid_mask)[0][-1]]
    equity_wo_fee = float(last["balance"] + last["position"] * last["price"])
    equity = equity_wo_fee - float(last["fee"])
    max_pos = float(np.nanmax(np.abs(records["position"])))

    return {
        "timestamp": int(last["timestamp"]),
        "price": float(last["price"]),
        "position": float(last["position"]),
        "balance": float(last["balance"]),
        "fee": float(last["fee"]),
        "equity_wo_fee": equity_wo_fee,
        "equity": equity,
        "num_trades": int(last["num_trades"]),
        "max_abs_position": max_pos,
        "records_valid": records[valid_mask],
    }


def print_backtest_summary(results: dict) -> None:
    """Print formatted backtest summary."""
    print(
        "backtest summary:",
        f"timestamp={results['timestamp']}",
        f"price={results['price']}",
        f"position={results['position']}",
        f"balance={results['balance']}",
        f"fee={results['fee']}",
        f"equity_wo_fee={results['equity_wo_fee']}",
        f"equity={results['equity']}",
        f"num_trades={results['num_trades']}",
        f"max_abs_position={results['max_abs_position']}",
    )
