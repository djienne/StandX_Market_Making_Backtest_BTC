from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path

import numpy as np
from hftbacktest import BacktestAsset, ROIVectorMarketDepthBacktest

from backtest_standx_OBI import (
    _format_ns,
    _infer_roi_bounds,
    _resolve_price_param,
    _resolve_scalar_param,
)

CACHE_VERSION = 1
CACHE_META_KEYS = [
    "version",
    "npz_path",
    "npz_mtime_ns",
    "npz_size",
    "step_ns",
    "window_steps",
    "update_interval_steps",
    "latency_ns",
    "tick_size",
    "lot_size",
    "looking_depth",
    "roi_lb",
    "roi_ub",
    "roi_pad",
]
CACHE_STATE_KEYS = [
    "last_mid_price_tick",
    "last_best_bid_tick",
    "last_best_ask_tick",
    "alpha",
    "imbalance",
    "volatility",
    "total_steps",
    "last_update_step",
    "base_ts_ns",
    "latest_ts_ns",
    "roi_lb",
    "roi_ub",
]
DEFAULT_STEP_NS = 100_000_000
DEFAULT_WINDOW_STEPS = 6_000
DEFAULT_UPDATE_INTERVAL_STEPS = 50
DEFAULT_LATENCY_NS = 1_000_000
DEFAULT_LOOKING_DEPTH = 0.025
DEFAULT_VOL_TO_HALF_SPREAD = 5.0
DEFAULT_HALF_SPREAD = None
DEFAULT_HALF_SPREAD_BPS = 0.0
DEFAULT_HALF_SPREAD_TICKS = None
DEFAULT_SKEW = 20.0
DEFAULT_SKEW_TICKS = None
DEFAULT_C1 = None
DEFAULT_C1_TICKS = 160
DEFAULT_GRID_INTERVAL = None
DEFAULT_GRID_INTERVAL_TICKS = 1
DEFAULT_GRID_NUM = 1
DEFAULT_POSITION = 0.0
DEFAULT_MAX_POSITION_DOLLAR = 500.0
DEFAULT_ROI_LB = None
DEFAULT_ROI_UB = None
DEFAULT_ROI_PAD = 0.02
DEFAULT_QUOTE = "USDC"
PAIR_SKIP_TOKENS = {"glft", "hft", "obi", "full", "small", "sample"}


def _cache_path_for(npz_path: Path) -> Path:
    return npz_path.with_suffix(".latest_cache.npz")


def _build_cache_meta(
    npz_path: Path,
    step_ns: int,
    window_steps: int,
    update_interval_steps: int,
    latency_ns: int,
    tick_size: float,
    lot_size: float,
    looking_depth: float,
    roi_lb: float | None,
    roi_ub: float | None,
    roi_pad: float,
) -> dict:
    stat = npz_path.stat()
    return {
        "version": CACHE_VERSION,
        "npz_path": str(npz_path),
        "npz_mtime_ns": int(stat.st_mtime_ns),
        "npz_size": int(stat.st_size),
        "step_ns": int(step_ns),
        "window_steps": int(window_steps),
        "update_interval_steps": int(update_interval_steps),
        "latency_ns": int(latency_ns),
        "tick_size": float(tick_size),
        "lot_size": float(lot_size),
        "looking_depth": float(looking_depth),
        "roi_lb": float(roi_lb) if roi_lb is not None else None,
        "roi_ub": float(roi_ub) if roi_ub is not None else None,
        "roi_pad": float(roi_pad),
    }


def _load_cache(
    cache_path: Path,
    expected_meta: dict,
    required_state_keys: list[str],
) -> dict | None:
    if not cache_path.exists():
        return None
    try:
        cached = np.load(cache_path, allow_pickle=False)
    except Exception:
        return None

    try:
        meta = json.loads(cached["meta"].item())
    except Exception:
        return None

    for key in CACHE_META_KEYS:
        expected_value = expected_meta.get(key)
        if expected_value is None:
            continue
        if meta.get(key) != expected_value:
            return None

    try:
        state = json.loads(cached["state"].item())
    except Exception:
        return None

    if not isinstance(state, dict):
        return None
    for key in required_state_keys:
        if key not in state:
            return None
    return state


def _save_cache(
    cache_path: Path,
    meta: dict,
    state: dict,
) -> None:
    try:
        np.savez_compressed(
            cache_path,
            meta=json.dumps(meta),
            state=json.dumps(state),
        )
    except Exception:
        return


@dataclass(frozen=True)
class LatestObiState:
    half_spread_tick: float
    skew: float
    alpha: float
    imbalance: float
    volatility: float
    c1: float
    mid_price_tick: float
    best_bid_tick: float
    best_ask_tick: float
    tick_size: float
    total_steps: int
    last_update_step: int | None
    base_ts_ns: int | None
    latest_ts_ns: int | None
    roi_lb: float | None
    roi_ub: float | None


def _read_npz_timestamps(npz_path: Path) -> tuple[int | None, int | None]:
    try:
        with np.load(npz_path, allow_pickle=False, mmap_mode="r") as npz:
            if "data" not in npz:
                return None, None
            data = npz["data"]
            if data.size == 0 or "local_ts" not in data.dtype.names:
                return None, None
            local_ts = data["local_ts"]
            if local_ts.size == 0:
                return None, None
            return int(local_ts.min()), int(local_ts.max())
    except Exception:
        return None, None


def _ring_window(values: np.ndarray, idx: int) -> np.ndarray:
    return np.roll(values, -(idx + 1))


def _mid_price_from_tick(mid_price_tick: float, tick_size: float) -> float:
    if not np.isfinite(mid_price_tick) or not np.isfinite(tick_size) or tick_size <= 0:
        return float("nan")
    return mid_price_tick * tick_size


def _delta_bps(delta: float, mid_price: float) -> float:
    if not np.isfinite(delta) or not np.isfinite(mid_price) or mid_price <= 0:
        return float("nan")
    return delta / mid_price * 10000.0


def _grid_prices_obi(
    mid_price_tick: float,
    best_bid_tick: float,
    best_ask_tick: float,
    tick_size: float,
    half_spread_tick: float,
    skew: float,
    c1: float,
    alpha: float,
    position: float,
    max_position_dollar: float,
    min_grid_step: float,
    grid_num: int,
) -> tuple[list[float], list[float], float]:
    bid_prices: list[float] = []
    ask_prices: list[float] = []
    if grid_num <= 0:
        return bid_prices, ask_prices, float("nan")
    if (
        not np.isfinite(mid_price_tick)
        or not np.isfinite(best_bid_tick)
        or not np.isfinite(best_ask_tick)
    ):
        return bid_prices, ask_prices, float("nan")
    if not np.isfinite(tick_size) or tick_size <= 0:
        return bid_prices, ask_prices, float("nan")
    if not np.isfinite(half_spread_tick) or half_spread_tick <= 0:
        return bid_prices, ask_prices, float("nan")
    if not np.isfinite(skew) or not np.isfinite(c1) or not np.isfinite(alpha):
        return bid_prices, ask_prices, float("nan")

    mid_price = _mid_price_from_tick(mid_price_tick, tick_size)
    if not np.isfinite(mid_price) or mid_price <= 0:
        return bid_prices, ask_prices, float("nan")

    best_bid = best_bid_tick * tick_size
    best_ask = best_ask_tick * tick_size
    if not np.isfinite(best_bid) or not np.isfinite(best_ask):
        return bid_prices, ask_prices, float("nan")

    fair_price = mid_price + c1 * alpha
    if not np.isfinite(fair_price):
        return bid_prices, ask_prices, float("nan")

    normalized_position = 0.0
    if (
        np.isfinite(position)
        and np.isfinite(max_position_dollar)
        and max_position_dollar > 0
    ):
        normalized_position = (position * mid_price) / max_position_dollar
        if not np.isfinite(normalized_position):
            normalized_position = 0.0

    bid_depth_tick = half_spread_tick * (1.0 + skew * normalized_position)
    ask_depth_tick = half_spread_tick * (1.0 - skew * normalized_position)
    if bid_depth_tick < 0:
        bid_depth_tick = 0.0
    if ask_depth_tick < 0:
        ask_depth_tick = 0.0

    bid_price = min(fair_price - bid_depth_tick * tick_size, best_bid)
    ask_price = max(fair_price + ask_depth_tick * tick_size, best_ask)
    if not np.isfinite(bid_price) or not np.isfinite(ask_price):
        return bid_prices, ask_prices, float("nan")

    bid_price = np.floor(bid_price / tick_size) * tick_size
    ask_price = np.ceil(ask_price / tick_size) * tick_size

    if not np.isfinite(min_grid_step) or min_grid_step <= 0:
        return bid_prices, ask_prices, float("nan")

    grid_interval = max(
        np.round(half_spread_tick * tick_size / min_grid_step) * min_grid_step,
        min_grid_step,
    )
    if not np.isfinite(grid_interval) or grid_interval <= 0:
        return bid_prices, ask_prices, float("nan")

    bid_price = np.floor(bid_price / grid_interval) * grid_interval
    ask_price = np.ceil(ask_price / grid_interval) * grid_interval

    if normalized_position < 1.0 and np.isfinite(bid_price):
        for _ in range(grid_num):
            bid_prices.append(float(bid_price))
            bid_price -= grid_interval
    if normalized_position > -1.0 and np.isfinite(ask_price):
        for _ in range(grid_num):
            ask_prices.append(float(ask_price))
            ask_price += grid_interval
    return bid_prices, ask_prices, float(grid_interval)


def _grid_deltas(
    mid_price: float,
    bid_prices: list[float],
    ask_prices: list[float],
) -> list[dict]:
    rows: list[dict] = []
    levels = max(len(bid_prices), len(ask_prices))
    for level in range(levels):
        bid_price = bid_prices[level] if level < len(bid_prices) else float("nan")
        ask_price = ask_prices[level] if level < len(ask_prices) else float("nan")
        bid_delta = bid_price - mid_price
        ask_delta = ask_price - mid_price
        rows.append(
            {
                "level": level + 1,
                "delta_minus": bid_delta,
                "delta_minus_bps": _delta_bps(bid_delta, mid_price),
                "delta_plus": ask_delta,
                "delta_plus_bps": _delta_bps(ask_delta, mid_price),
            }
        )
    return rows


def _format_float(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def _format_ts_fields(ts_ns: int | None) -> tuple[str, str]:
    if ts_ns is None:
        return "n/a", "n/a"
    return _format_ns(ts_ns), str(int(ts_ns))


def _timestamp_payload(ts_ns: int | None) -> dict:
    if ts_ns is None:
        return {"iso": None, "ts_ns": None}
    return {"iso": _format_ns(ts_ns), "ts_ns": int(ts_ns)}


def _calc_param_ts(
    base_ts_ns: int | None, last_update_step: int | None, step_ns: int
) -> int | None:
    if base_ts_ns is None or last_update_step is None:
        return None
    return int(base_ts_ns + (last_update_step + 1) * step_ns)


def _sanitize_token(value: str) -> str:
    if not value:
        return "unknown"
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    cleaned = cleaned.strip("_")
    return cleaned or "unknown"


def _infer_pair_info(npz_path: Path, default_quote: str = DEFAULT_QUOTE) -> dict:
    stem = npz_path.stem
    parts = [part for part in stem.split("_") if part]
    base_raw = parts[0] if parts else ""
    quote_raw = ""
    source = "filename+default_quote"
    if len(parts) > 1 and parts[1].lower() not in PAIR_SKIP_TOKENS:
        quote_raw = parts[1]
        source = "filename"
    elif default_quote:
        quote_raw = default_quote
    base = base_raw.upper() if base_raw else "UNKNOWN"
    quote = quote_raw.upper() if quote_raw else "UNKNOWN"
    display = f"{base}/{quote}" if quote_raw else base
    code = base_raw.lower() if base_raw else "unknown"
    file_token = f"{base}_{quote}" if quote_raw else base
    return {
        "base": base,
        "quote": quote,
        "display": display,
        "code": code,
        "file_token": file_token,
        "source": source,
    }


def _resolve_spread_params(
    tick_size: float,
    vol_to_half_spread: float,
    half_spread: float | None,
    half_spread_bps: float,
    half_spread_ticks: int | None,
) -> tuple[float, float, float, str]:
    half_spread_value = 0.0
    effective_vol_to_half_spread = float(vol_to_half_spread)
    effective_half_spread_bps = float(half_spread_bps)
    spread_mode = "fixed"
    if half_spread is not None or half_spread_ticks is not None:
        half_spread_value = _resolve_price_param(
            half_spread,
            half_spread_ticks,
            tick_size,
        )
        effective_half_spread_bps = 0.0
        effective_vol_to_half_spread = 0.0
    elif effective_half_spread_bps > 0:
        spread_mode = "bps"
        effective_vol_to_half_spread = 0.0
    elif effective_vol_to_half_spread > 0:
        spread_mode = "volatility"
    return (
        float(half_spread_value),
        float(effective_vol_to_half_spread),
        float(effective_half_spread_bps),
        spread_mode,
    )


def _resolve_grid_interval(
    tick_size: float,
    grid_interval: float | None,
    grid_interval_ticks: int | None,
) -> float:
    min_grid_step = _resolve_price_param(grid_interval, grid_interval_ticks, tick_size)
    if not np.isfinite(min_grid_step) or min_grid_step <= 0:
        min_grid_step = tick_size
    return float(max(min_grid_step, tick_size))


def _latest_obi_state(
    npz_path: str,
    step_ns: int = DEFAULT_STEP_NS,
    window_steps: int = DEFAULT_WINDOW_STEPS,
    update_interval_steps: int = DEFAULT_UPDATE_INTERVAL_STEPS,
    latency_ns: int = DEFAULT_LATENCY_NS,
    looking_depth: float = DEFAULT_LOOKING_DEPTH,
    vol_to_half_spread: float = DEFAULT_VOL_TO_HALF_SPREAD,
    half_spread: float | None = DEFAULT_HALF_SPREAD,
    half_spread_bps: float = DEFAULT_HALF_SPREAD_BPS,
    half_spread_ticks: int | None = DEFAULT_HALF_SPREAD_TICKS,
    skew: float | None = DEFAULT_SKEW,
    skew_ticks: float | None = DEFAULT_SKEW_TICKS,
    c1: float | None = DEFAULT_C1,
    c1_ticks: int | None = DEFAULT_C1_TICKS,
    roi_lb: float | None = DEFAULT_ROI_LB,
    roi_ub: float | None = DEFAULT_ROI_UB,
    roi_pad: float = DEFAULT_ROI_PAD,
) -> LatestObiState:
    npz_path = Path(npz_path).resolve()
    meta = json.loads(
        npz_path.with_suffix(npz_path.suffix + ".meta.json").read_text(
            encoding="utf-8"
        )
    )
    tick_size = float(meta["tick_size"])
    lot_size = float(meta["lot_size"])

    step_ns = int(step_ns)
    if step_ns <= 0:
        raise ValueError("step_ns must be > 0")
    window_steps = max(1, int(window_steps))
    update_interval_steps = max(1, int(update_interval_steps))

    roi_lb_value = roi_lb
    roi_ub_value = roi_ub
    if roi_lb_value is None or roi_ub_value is None:
        roi_lb_value = None
        roi_ub_value = None

    cache_path = _cache_path_for(npz_path)
    expected_meta = _build_cache_meta(
        npz_path,
        step_ns,
        window_steps,
        update_interval_steps,
        latency_ns,
        tick_size,
        lot_size,
        looking_depth,
        roi_lb_value,
        roi_ub_value,
        roi_pad,
    )
    cached_state = _load_cache(cache_path, expected_meta, CACHE_STATE_KEYS)
    if cached_state is None:
        if roi_lb_value is None or roi_ub_value is None:
            with np.load(npz_path, allow_pickle=False, mmap_mode="r") as npz:
                if "data" not in npz:
                    raise RuntimeError("npz missing data array")
                data = npz["data"]
                if data.size == 0:
                    raise RuntimeError("npz data array is empty")
                roi_lb_value, roi_ub_value = _infer_roi_bounds(data, roi_pad)

        base_ts_ns, latest_ts_ns = _read_npz_timestamps(npz_path)

        roi_lb_tick = int(round(roi_lb_value / tick_size))
        roi_ub_tick = int(round(roi_ub_value / tick_size))
        if roi_lb_tick > roi_ub_tick:
            roi_lb_tick, roi_ub_tick = roi_ub_tick, roi_lb_tick

        asset = (
            BacktestAsset()
            .data([str(npz_path)])
            .linear_asset(1.0)
            .constant_order_latency(latency_ns, latency_ns)
            .risk_adverse_queue_model()
            .no_partial_fill_exchange()
            .trading_value_fee_model(0.0, 0.0)
            .tick_size(tick_size)
            .lot_size(lot_size)
            .roi_lb(float(roi_lb_value))
            .roi_ub(float(roi_ub_value))
            .last_trades_capacity(10000)
        )
        hbt = ROIVectorMarketDepthBacktest([asset])

        imbalance_series = np.full(window_steps, np.nan, np.float64)
        mid_price_chg = np.full(window_steps, np.nan, np.float64)
        idx = 0
        t = 0
        prev_mid_price_tick = np.nan
        last_mid_price_tick = np.nan
        last_best_bid_tick = np.nan
        last_best_ask_tick = np.nan
        last_alpha = 0.0
        last_imbalance = np.nan
        volatility = np.nan
        last_update_step: int | None = None
        vol_scale = np.sqrt(1_000_000_000 / step_ns)

        while hbt.elapse(step_ns) == 0:
            depth = hbt.depth(0)

            best_bid = depth.best_bid
            best_ask = depth.best_ask
            if not np.isfinite(best_bid) or not np.isfinite(best_ask):
                imbalance_series[idx] = np.nan
                mid_price_chg[idx] = np.nan
                idx = (idx + 1) % window_steps
                t += 1
                continue

            mid_price = (best_bid + best_ask) / 2.0
            mid_price_tick = mid_price / tick_size
            if np.isfinite(prev_mid_price_tick):
                mid_price_chg[idx] = mid_price_tick - prev_mid_price_tick
            else:
                mid_price_chg[idx] = np.nan
            prev_mid_price_tick = mid_price_tick

            last_mid_price_tick = mid_price_tick
            last_best_bid_tick = depth.best_bid_tick
            last_best_ask_tick = depth.best_ask_tick

            sum_ask_qty = 0.0
            from_tick = max(depth.best_ask_tick, roi_lb_tick)
            upto_tick = min(
                int(np.floor(mid_price * (1.0 + looking_depth) / tick_size)),
                roi_ub_tick,
            )
            for price_tick in range(from_tick, upto_tick):
                sum_ask_qty += depth.ask_depth[price_tick - roi_lb_tick]

            sum_bid_qty = 0.0
            from_tick = min(depth.best_bid_tick, roi_ub_tick)
            upto_tick = max(
                int(np.ceil(mid_price * (1.0 - looking_depth) / tick_size)),
                roi_lb_tick,
            )
            for price_tick in range(from_tick, upto_tick, -1):
                sum_bid_qty += depth.bid_depth[price_tick - roi_lb_tick]

            last_imbalance = sum_bid_qty - sum_ask_qty
            imbalance_series[idx] = last_imbalance

            if update_interval_steps > 0 and t % update_interval_steps == 0:
                if t >= window_steps - 1:
                    imbalance_window = _ring_window(imbalance_series, idx)
                    m = np.nanmean(imbalance_window)
                    s = np.nanstd(imbalance_window)
                    if np.isfinite(m) and np.isfinite(s) and s > 0:
                        last_alpha = (last_imbalance - m) / s
                    else:
                        last_alpha = 0.0
                    mid_window = _ring_window(mid_price_chg, idx)
                    volatility = np.nanstd(mid_window) * vol_scale
                    last_update_step = t
                else:
                    last_alpha = 0.0
                    volatility = np.nan

            idx = (idx + 1) % window_steps
            t += 1

        hbt.close()

        cached_state = {
            "last_mid_price_tick": float(last_mid_price_tick),
            "last_best_bid_tick": float(last_best_bid_tick),
            "last_best_ask_tick": float(last_best_ask_tick),
            "alpha": float(last_alpha),
            "imbalance": float(last_imbalance),
            "volatility": float(volatility),
            "total_steps": int(t),
            "last_update_step": int(last_update_step)
            if last_update_step is not None
            else None,
            "base_ts_ns": int(base_ts_ns) if base_ts_ns is not None else None,
            "latest_ts_ns": int(latest_ts_ns) if latest_ts_ns is not None else None,
            "roi_lb": float(roi_lb_value),
            "roi_ub": float(roi_ub_value),
        }
        expected_meta = _build_cache_meta(
            npz_path,
            step_ns,
            window_steps,
            update_interval_steps,
            latency_ns,
            tick_size,
            lot_size,
            looking_depth,
            roi_lb_value,
            roi_ub_value,
            roi_pad,
        )
        _save_cache(cache_path, expected_meta, cached_state)

    roi_lb_value = cached_state.get("roi_lb")
    roi_ub_value = cached_state.get("roi_ub")

    alpha = float(cached_state["alpha"])
    imbalance = float(cached_state["imbalance"])
    volatility = float(cached_state["volatility"])
    skew_value = _resolve_scalar_param(skew, skew_ticks)
    c1_value = _resolve_price_param(c1, c1_ticks, tick_size)
    (
        half_spread_value,
        effective_vol_to_half_spread,
        effective_half_spread_bps,
        _,
    ) = _resolve_spread_params(
        tick_size,
        vol_to_half_spread,
        half_spread,
        half_spread_bps,
        half_spread_ticks,
    )

    mid_price_tick = float(cached_state["last_mid_price_tick"])
    mid_price = _mid_price_from_tick(mid_price_tick, tick_size)
    half_spread_tick = float("nan")
    if effective_vol_to_half_spread > 0 and np.isfinite(volatility):
        half_spread_tick = volatility * effective_vol_to_half_spread
    elif effective_half_spread_bps > 0:
        if np.isfinite(mid_price) and mid_price > 0:
            half_spread_tick = (
                mid_price * (effective_half_spread_bps / 10000.0) / tick_size
            )
    elif half_spread_value > 0:
        if np.isfinite(tick_size) and tick_size > 0:
            half_spread_tick = half_spread_value / tick_size

    return LatestObiState(
        half_spread_tick=half_spread_tick,
        skew=float(skew_value),
        alpha=alpha,
        imbalance=imbalance,
        volatility=volatility,
        c1=float(c1_value),
        mid_price_tick=mid_price_tick,
        best_bid_tick=float(cached_state["last_best_bid_tick"]),
        best_ask_tick=float(cached_state["last_best_ask_tick"]),
        tick_size=float(tick_size),
        total_steps=int(cached_state["total_steps"]),
        last_update_step=cached_state["last_update_step"],
        base_ts_ns=cached_state["base_ts_ns"],
        latest_ts_ns=cached_state["latest_ts_ns"],
        roi_lb=float(roi_lb_value) if roi_lb_value is not None else None,
        roi_ub=float(roi_ub_value) if roi_ub_value is not None else None,
    )


def latest_obi_spread(
    npz_path: str,
    step_ns: int = DEFAULT_STEP_NS,
    window_steps: int = DEFAULT_WINDOW_STEPS,
    update_interval_steps: int = DEFAULT_UPDATE_INTERVAL_STEPS,
    latency_ns: int = DEFAULT_LATENCY_NS,
    looking_depth: float = DEFAULT_LOOKING_DEPTH,
    vol_to_half_spread: float = DEFAULT_VOL_TO_HALF_SPREAD,
    half_spread: float | None = DEFAULT_HALF_SPREAD,
    half_spread_bps: float = DEFAULT_HALF_SPREAD_BPS,
    half_spread_ticks: int | None = DEFAULT_HALF_SPREAD_TICKS,
    skew: float | None = DEFAULT_SKEW,
    skew_ticks: float | None = DEFAULT_SKEW_TICKS,
    c1: float | None = DEFAULT_C1,
    c1_ticks: int | None = DEFAULT_C1_TICKS,
    roi_lb: float | None = DEFAULT_ROI_LB,
    roi_ub: float | None = DEFAULT_ROI_UB,
    roi_pad: float = DEFAULT_ROI_PAD,
) -> tuple[float, float, float, float, float]:
    state = _latest_obi_state(
        npz_path,
        step_ns,
        window_steps,
        update_interval_steps,
        latency_ns,
        looking_depth,
        vol_to_half_spread,
        half_spread,
        half_spread_bps,
        half_spread_ticks,
        skew,
        skew_ticks,
        c1,
        c1_ticks,
        roi_lb,
        roi_ub,
        roi_pad,
    )
    return (
        state.half_spread_tick,
        state.skew,
        state.volatility,
        state.alpha,
        state.c1,
    )


def latest_obi_quotes(
    npz_path: str,
    step_ns: int = DEFAULT_STEP_NS,
    window_steps: int = DEFAULT_WINDOW_STEPS,
    update_interval_steps: int = DEFAULT_UPDATE_INTERVAL_STEPS,
    latency_ns: int = DEFAULT_LATENCY_NS,
    looking_depth: float = DEFAULT_LOOKING_DEPTH,
    vol_to_half_spread: float = DEFAULT_VOL_TO_HALF_SPREAD,
    half_spread: float | None = DEFAULT_HALF_SPREAD,
    half_spread_bps: float = DEFAULT_HALF_SPREAD_BPS,
    half_spread_ticks: int | None = DEFAULT_HALF_SPREAD_TICKS,
    skew: float | None = DEFAULT_SKEW,
    skew_ticks: float | None = DEFAULT_SKEW_TICKS,
    c1: float | None = DEFAULT_C1,
    c1_ticks: int | None = DEFAULT_C1_TICKS,
    position: float = DEFAULT_POSITION,
    max_position_dollar: float = DEFAULT_MAX_POSITION_DOLLAR,
    grid_interval: float | None = DEFAULT_GRID_INTERVAL,
    grid_interval_ticks: int | None = DEFAULT_GRID_INTERVAL_TICKS,
    roi_lb: float | None = DEFAULT_ROI_LB,
    roi_ub: float | None = DEFAULT_ROI_UB,
    roi_pad: float = DEFAULT_ROI_PAD,
) -> tuple[float, float, float, float, float, float, float, float]:
    state = _latest_obi_state(
        npz_path,
        step_ns,
        window_steps,
        update_interval_steps,
        latency_ns,
        looking_depth,
        vol_to_half_spread,
        half_spread,
        half_spread_bps,
        half_spread_ticks,
        skew,
        skew_ticks,
        c1,
        c1_ticks,
        roi_lb,
        roi_ub,
        roi_pad,
    )
    mid_price = _mid_price_from_tick(state.mid_price_tick, state.tick_size)
    if not np.isfinite(max_position_dollar) or max_position_dollar <= 0:
        raise ValueError("max_position_dollar must be > 0")
    if not np.isfinite(mid_price):
        return (
            state.half_spread_tick,
            state.skew,
            state.volatility,
            state.alpha,
            state.c1,
            np.nan,
            np.nan,
            np.nan,
        )
    min_grid_step = _resolve_grid_interval(
        state.tick_size, grid_interval, grid_interval_ticks
    )
    bid_prices, ask_prices, _ = _grid_prices_obi(
        state.mid_price_tick,
        state.best_bid_tick,
        state.best_ask_tick,
        state.tick_size,
        state.half_spread_tick,
        state.skew,
        state.c1,
        state.alpha,
        position,
        max_position_dollar,
        min_grid_step,
        1,
    )
    if not bid_prices or not ask_prices:
        return (
            state.half_spread_tick,
            state.skew,
            state.volatility,
            state.alpha,
            state.c1,
            mid_price,
            np.nan,
            np.nan,
        )
    bid_delta = bid_prices[0] - mid_price
    ask_delta = ask_prices[0] - mid_price
    bid_bps = _delta_bps(bid_delta, mid_price)
    ask_bps = _delta_bps(ask_delta, mid_price)
    return (
        state.half_spread_tick,
        state.skew,
        state.volatility,
        state.alpha,
        state.c1,
        mid_price,
        bid_bps,
        ask_bps,
    )


def latest_obi_grid_deltas(
    npz_path: str,
    step_ns: int = DEFAULT_STEP_NS,
    window_steps: int = DEFAULT_WINDOW_STEPS,
    update_interval_steps: int = DEFAULT_UPDATE_INTERVAL_STEPS,
    latency_ns: int = DEFAULT_LATENCY_NS,
    looking_depth: float = DEFAULT_LOOKING_DEPTH,
    vol_to_half_spread: float = DEFAULT_VOL_TO_HALF_SPREAD,
    half_spread: float | None = DEFAULT_HALF_SPREAD,
    half_spread_bps: float = DEFAULT_HALF_SPREAD_BPS,
    half_spread_ticks: int | None = DEFAULT_HALF_SPREAD_TICKS,
    skew: float | None = DEFAULT_SKEW,
    skew_ticks: float | None = DEFAULT_SKEW_TICKS,
    c1: float | None = DEFAULT_C1,
    c1_ticks: int | None = DEFAULT_C1_TICKS,
    position: float = DEFAULT_POSITION,
    max_position_dollar: float = DEFAULT_MAX_POSITION_DOLLAR,
    grid_interval: float | None = DEFAULT_GRID_INTERVAL,
    grid_interval_ticks: int | None = DEFAULT_GRID_INTERVAL_TICKS,
    grid_num: int = DEFAULT_GRID_NUM,
    roi_lb: float | None = DEFAULT_ROI_LB,
    roi_ub: float | None = DEFAULT_ROI_UB,
    roi_pad: float = DEFAULT_ROI_PAD,
) -> tuple[LatestObiState, float, list[dict]]:
    state = _latest_obi_state(
        npz_path,
        step_ns,
        window_steps,
        update_interval_steps,
        latency_ns,
        looking_depth,
        vol_to_half_spread,
        half_spread,
        half_spread_bps,
        half_spread_ticks,
        skew,
        skew_ticks,
        c1,
        c1_ticks,
        roi_lb,
        roi_ub,
        roi_pad,
    )
    if not np.isfinite(max_position_dollar) or max_position_dollar <= 0:
        raise ValueError("max_position_dollar must be > 0")
    min_grid_step = _resolve_grid_interval(
        state.tick_size, grid_interval, grid_interval_ticks
    )
    mid_price = _mid_price_from_tick(state.mid_price_tick, state.tick_size)
    bid_prices, ask_prices, grid_interval_value = _grid_prices_obi(
        state.mid_price_tick,
        state.best_bid_tick,
        state.best_ask_tick,
        state.tick_size,
        state.half_spread_tick,
        state.skew,
        state.c1,
        state.alpha,
        position,
        max_position_dollar,
        min_grid_step,
        grid_num,
    )
    rows = _grid_deltas(mid_price, bid_prices, ask_prices)
    return state, grid_interval_value, rows


def _build_output_payload(
    npz_path: Path,
    pair_info: dict,
    model: str,
    state: LatestObiState,
    grid_interval: float,
    rows: list[dict],
    config: dict,
    runtime_seconds: float,
    params_ts_ns: int | None,
) -> dict:
    return {
        "pair": pair_info["display"],
        "pair_base": pair_info["base"],
        "pair_quote": pair_info["quote"],
        "pair_code": pair_info["code"],
        "pair_source": pair_info["source"],
        "model": model,
        "npz_path": str(npz_path),
        "data_latest_time": _timestamp_payload(state.latest_ts_ns),
        "params_time": {
            **_timestamp_payload(params_ts_ns),
            "step": state.last_update_step,
        },
        "config": config,
        "state": {
            "half_spread_tick": state.half_spread_tick,
            "skew": state.skew,
            "alpha": state.alpha,
            "imbalance": state.imbalance,
            "volatility": state.volatility,
            "c1": state.c1,
            "tick_size": state.tick_size,
            "grid_interval": grid_interval,
            "grid_num": config.get("grid_num"),
            "position": config.get("position"),
            "max_position_dollar": config.get("max_position_dollar"),
            "total_steps": state.total_steps,
        },
        "grid_deltas": rows,
        "runtime_seconds": runtime_seconds,
    }


def _write_output_json(
    out_dir: Path,
    pair_token: str,
    model: str,
    params_ts_ns: int | None,
    data_ts_ns: int | None,
    payload: dict,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_pair = _sanitize_token(pair_token)
    suffix = None
    if params_ts_ns is not None:
        suffix = str(int(params_ts_ns))
    elif data_ts_ns is not None:
        suffix = str(int(data_ts_ns))
    else:
        suffix = "latest"
    out_path = out_dir / f"{safe_pair}_{model}_{suffix}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    start = time.perf_counter()
    npz_path = "data/btc_hft_obi.npz"
    step_ns = DEFAULT_STEP_NS
    window_steps = DEFAULT_WINDOW_STEPS
    update_interval_steps = DEFAULT_UPDATE_INTERVAL_STEPS
    latency_ns = DEFAULT_LATENCY_NS
    looking_depth = DEFAULT_LOOKING_DEPTH
    vol_to_half_spread = DEFAULT_VOL_TO_HALF_SPREAD
    half_spread = DEFAULT_HALF_SPREAD
    half_spread_bps = DEFAULT_HALF_SPREAD_BPS
    half_spread_ticks = DEFAULT_HALF_SPREAD_TICKS
    skew = DEFAULT_SKEW
    skew_ticks = DEFAULT_SKEW_TICKS
    c1 = DEFAULT_C1
    c1_ticks = DEFAULT_C1_TICKS
    grid_interval = DEFAULT_GRID_INTERVAL
    grid_interval_ticks = DEFAULT_GRID_INTERVAL_TICKS
    grid_num = DEFAULT_GRID_NUM
    position = DEFAULT_POSITION
    max_position_dollar = DEFAULT_MAX_POSITION_DOLLAR
    roi_lb = DEFAULT_ROI_LB
    roi_ub = DEFAULT_ROI_UB
    roi_pad = DEFAULT_ROI_PAD
    model = "OBI"
    pair_info = _infer_pair_info(Path(npz_path))

    state, grid_interval_value, rows = latest_obi_grid_deltas(
        npz_path,
        step_ns=step_ns,
        window_steps=window_steps,
        update_interval_steps=update_interval_steps,
        latency_ns=latency_ns,
        looking_depth=looking_depth,
        vol_to_half_spread=vol_to_half_spread,
        half_spread=half_spread,
        half_spread_bps=half_spread_bps,
        half_spread_ticks=half_spread_ticks,
        skew=skew,
        skew_ticks=skew_ticks,
        c1=c1,
        c1_ticks=c1_ticks,
        position=position,
        max_position_dollar=max_position_dollar,
        grid_interval=grid_interval,
        grid_interval_ticks=grid_interval_ticks,
        grid_num=grid_num,
        roi_lb=roi_lb,
        roi_ub=roi_ub,
        roi_pad=roi_pad,
    )
    elapsed = time.perf_counter() - start
    data_time_iso, data_time_ns = _format_ts_fields(state.latest_ts_ns)
    params_ts_ns = _calc_param_ts(state.base_ts_ns, state.last_update_step, step_ns)
    params_time_iso, params_time_ns = _format_ts_fields(params_ts_ns)
    params_step = state.last_update_step if state.last_update_step is not None else "n/a"
    (
        half_spread_value,
        effective_vol_to_half_spread,
        effective_half_spread_bps,
        spread_mode,
    ) = _resolve_spread_params(
        state.tick_size,
        vol_to_half_spread,
        half_spread,
        half_spread_bps,
        half_spread_ticks,
    )
    min_grid_step = _resolve_grid_interval(
        state.tick_size,
        grid_interval,
        grid_interval_ticks,
    )
    config = {
        "step_ns": step_ns,
        "window_steps": window_steps,
        "update_interval_steps": update_interval_steps,
        "latency_ns": latency_ns,
        "looking_depth": looking_depth,
        "roi_lb": state.roi_lb,
        "roi_ub": state.roi_ub,
        "roi_pad": roi_pad,
        "vol_to_half_spread": effective_vol_to_half_spread,
        "half_spread": half_spread_value,
        "half_spread_bps": effective_half_spread_bps,
        "half_spread_ticks": half_spread_ticks,
        "spread_mode": spread_mode,
        "skew": skew,
        "skew_ticks": skew_ticks,
        "c1": c1,
        "c1_ticks": c1_ticks,
        "grid_interval": grid_interval,
        "grid_interval_ticks": grid_interval_ticks,
        "min_grid_step": min_grid_step,
        "grid_num": grid_num,
        "position": position,
        "max_position_dollar": max_position_dollar,
    }
    payload = _build_output_payload(
        Path(npz_path),
        pair_info,
        model,
        state,
        grid_interval_value,
        rows,
        config,
        elapsed,
        params_ts_ns,
    )
    json_path = _write_output_json(
        Path("calculated_quotes"),
        pair_info["file_token"],
        model,
        params_ts_ns,
        state.latest_ts_ns,
        payload,
    )

    print(
        "pair_info:",
        f"pair={pair_info['display']}",
        f"base={pair_info['base']}",
        f"quote={pair_info['quote']}",
        f"source={pair_info['source']}",
    )
    print("obi_latest:")
    print(f"data_latest_time={data_time_iso} data_latest_ts_ns={data_time_ns}")
    print(
        f"params_time={params_time_iso} params_ts_ns={params_time_ns} params_step={params_step}"
    )
    print(
        "obi_config:",
        f"model={model}",
        f"step_ns={step_ns}",
        f"window_steps={window_steps}",
        f"update_interval_steps={update_interval_steps}",
        f"latency_ns={latency_ns}",
        f"looking_depth={looking_depth}",
        f"roi_lb={state.roi_lb}",
        f"roi_ub={state.roi_ub}",
        f"roi_pad={roi_pad}",
        f"vol_to_half_spread={effective_vol_to_half_spread}",
        f"half_spread={half_spread_value}",
        f"half_spread_bps={effective_half_spread_bps}",
        f"half_spread_ticks={half_spread_ticks}",
        f"spread_mode={spread_mode}",
        f"skew={skew}",
        f"skew_ticks={skew_ticks}",
        f"c1={c1}",
        f"c1_ticks={c1_ticks}",
        f"grid_interval={grid_interval}",
        f"grid_interval_ticks={grid_interval_ticks}",
        f"min_grid_step={_format_float(min_grid_step)}",
    )
    print(
        "obi_state:",
        f"half_spread_tick={_format_float(state.half_spread_tick)}",
        f"skew={_format_float(state.skew)}",
        f"alpha={_format_float(state.alpha)}",
        f"imbalance={_format_float(state.imbalance)}",
        f"volatility={_format_float(state.volatility)}",
        f"c1={_format_float(state.c1)}",
        f"tick_size={_format_float(state.tick_size)}",
        f"grid_interval={_format_float(grid_interval_value)}",
        f"grid_num={grid_num}",
        f"position={position}",
        f"max_position_dollar={max_position_dollar}",
        f"total_steps={state.total_steps}",
    )
    print(f"grid_deltas_rows={len(rows)}")
    if rows:
        print("level,delta_minus,delta_minus_bps,delta_plus,delta_plus_bps")
        for row in rows:
            print(
                f"{row['level']},"
                f"{_format_float(row['delta_minus'])},"
                f"{_format_float(row['delta_minus_bps'])},"
                f"{_format_float(row['delta_plus'])},"
                f"{_format_float(row['delta_plus_bps'])}"
            )
    print(f"json_output={json_path}")
    print(f"runtime_seconds={elapsed}")
