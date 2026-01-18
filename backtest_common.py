"""Common backtest infrastructure: numba compatibility, constants, and setup helpers."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Type

import numpy as np

# Numba compatibility layer
try:
    from numba import njit, uint64, float64
    from numba.typed import Dict as NumbaDict
    NUMBA_AVAILABLE = True
except Exception:
    def njit(func=None, **kwargs):
        """Fallback decorator when numba is not available."""
        if func is None:
            return lambda f: f
        return func

    NumbaDict = None
    uint64 = None
    float64 = None
    NUMBA_AVAILABLE = False


# Order side constants (will be overwritten by hftbacktest imports)
BUY = 1
SELL = -1
GTX = 1
LIMIT = 0

# Event flags
BUY_EVENT = 1 << 29


def init_hft_constants() -> tuple:
    """Import and return hftbacktest constants, updating module globals.

    Returns (BUY, SELL, GTX, LIMIT) tuple.
    """
    global BUY, SELL, GTX, LIMIT
    try:
        from hftbacktest import BUY as HBUY, SELL as HSELL, GTX as HGTX, LIMIT as HLIMIT
        BUY, SELL, GTX, LIMIT = HBUY, HSELL, HGTX, HLIMIT
        return HBUY, HSELL, HGTX, HLIMIT
    except ImportError:
        return BUY, SELL, GTX, LIMIT


@dataclass
class BacktestAPI:
    """Container for hftbacktest API classes."""
    asset_cls: Type[Any]
    backtest_cls: Type[Any]
    recorder_cls: Type[Any]


def get_hashmap_api() -> BacktestAPI:
    """Get API classes for HashMap-based backtesting."""
    from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest
    from hftbacktest.recorder import Recorder
    return BacktestAPI(
        asset_cls=BacktestAsset,
        backtest_cls=HashMapMarketDepthBacktest,
        recorder_cls=Recorder,
    )


def get_roi_vector_api() -> BacktestAPI:
    """Get API classes for ROI Vector-based backtesting."""
    from hftbacktest import BacktestAsset, ROIVectorMarketDepthBacktest
    from hftbacktest.recorder import Recorder
    return BacktestAPI(
        asset_cls=BacktestAsset,
        backtest_cls=ROIVectorMarketDepthBacktest,
        recorder_cls=Recorder,
    )


def build_asset(
    api: BacktestAPI,
    npz_path: Path,
    tick_size: float,
    lot_size: float,
    latency_ns: int,
    maker_fee: float = 0.00002,
    taker_fee: float = 0.0002,
    roi_lb: Optional[float] = None,
    roi_ub: Optional[float] = None,
    last_trades_capacity: int = 10000,
):
    """Build a BacktestAsset with common configuration.

    Args:
        api: BacktestAPI with asset class
        npz_path: Path to data file
        tick_size: Price tick size
        lot_size: Order lot size
        latency_ns: Order latency in nanoseconds
        maker_fee: Maker fee rate
        taker_fee: Taker fee rate
        roi_lb: ROI lower bound (for ROIVector only)
        roi_ub: ROI upper bound (for ROIVector only)
        last_trades_capacity: Capacity for last trades buffer

    Returns:
        Configured BacktestAsset
    """
    asset = (
        api.asset_cls()
        .data([str(npz_path)])
        .linear_asset(1.0)
        .constant_order_latency(latency_ns, latency_ns)
        .risk_adverse_queue_model()
        .no_partial_fill_exchange()
        .trading_value_fee_model(maker_fee, taker_fee)
        .tick_size(tick_size)
        .lot_size(lot_size)
        .last_trades_capacity(last_trades_capacity)
    )

    if roi_lb is not None:
        asset = asset.roi_lb(roi_lb)
    if roi_ub is not None:
        asset = asset.roi_ub(roi_ub)

    return asset


def add_common_args(
    parser: argparse.ArgumentParser,
    default_symbol: Optional[str] = None,
) -> None:
    """Add common arguments with optional default symbol."""
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--latency-ns", type=int, default=1_000_000)
    parser.add_argument(
        "--symbol",
        type=str,
        default=default_symbol,
        help="Symbol to filter (e.g. CRV). Defaults to config.json symbol if set.",
    )


def add_backtest_args(
    parser: argparse.ArgumentParser,
    record_every_default: int = 10,
    step_ns_default: int = 100_000_000,
    plots_dir_default: str = "plots",
    gap_threshold_minutes_default: float = 10.0,
    include_plots_dir: bool = True,
) -> None:
    """Add arguments for running backtests."""
    parser.add_argument("--record-every", type=int, default=record_every_default)
    parser.add_argument("--step-ns", type=int, default=step_ns_default)
    parser.add_argument(
        "--gap-threshold-minutes",
        type=float,
        default=gap_threshold_minutes_default,
        help="Cancel orders and pause trading during data gaps longer than this.",
    )
    if include_plots_dir:
        parser.add_argument("--plots-dir", default=plots_dir_default)


def add_run_backtest_args(parser: argparse.ArgumentParser) -> None:
    """Add --run-backtest and --no-run-backtest arguments."""
    parser.add_argument(
        "--run-backtest",
        action="store_true",
        default=True,
        help="Run the backtest after conversion (default: True).",
    )
    parser.add_argument(
        "--no-run-backtest",
        dest="run_backtest",
        action="store_false",
        help="Skip running the backtest after conversion.",
    )


def compute_backtest_params(
    data: np.ndarray,
    step_ns: int,
    record_every: int,
) -> dict:
    """Compute common backtest parameters from data.

    Returns dict with:
    - duration: Duration in nanoseconds
    - max_steps: Maximum simulation steps
    - estimated: Estimated number of records
    - estimated_records: Estimated number of records (legacy key)
    - base_ts_ns: Base timestamp
    - vol_scale: Volatility scaling factor
    """
    duration = int(data["local_ts"].max() - data["local_ts"].min())
    base_ts_ns = int(data["local_ts"].min())
    max_steps = max(10_000, int(duration / step_ns) + 10_000)

    if record_every <= 0:
        record_every = 1
    estimated = max(10_000, int(max_steps / record_every) + 10_000)

    vol_scale = np.sqrt(1_000_000_000 / step_ns)

    return {
        "duration": duration,
        "max_steps": max_steps,
        "estimated": estimated,
        "estimated_records": estimated,
        "base_ts_ns": base_ts_ns,
        "vol_scale": vol_scale,
    }


def infer_roi_bounds(data: np.ndarray, pad_frac: float = 0.02) -> tuple[float, float]:
    """Infer ROI bounds from price data with padding.

    Args:
        data: Event data array with 'px' field
        pad_frac: Fraction to pad bounds (default 2%)

    Returns:
        (roi_lb, roi_ub) tuple
    """
    prices = data["px"].astype(np.float64)
    mask = np.isfinite(prices) & (prices > 0)
    if not np.any(mask):
        raise RuntimeError("Unable to infer ROI bounds; no positive prices found")

    min_px = float(np.min(prices[mask]))
    max_px = float(np.max(prices[mask]))
    pad = max(0.0, float(pad_frac))

    roi_lb = min_px * (1.0 - pad)
    roi_ub = max_px * (1.0 + pad)

    if roi_lb <= 0:
        roi_lb = min_px
    if roi_ub <= roi_lb:
        roi_ub = max_px

    return roi_lb, roi_ub
