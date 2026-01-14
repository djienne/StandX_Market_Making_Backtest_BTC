from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path

import numpy as np
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest

from backtest_standx_GLFT import (
    BUY_EVENT,
    _format_ns,
    compute_coeff,
    linear_regression,
    measure_trading_intensity,
)

CACHE_VERSION = 2
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
]
CACHE_STATE_KEYS = [
    "last_mid_price_tick",
    "last_best_bid_tick",
    "last_best_ask_tick",
    "A",
    "k",
    "volatility",
    "total_steps",
    "last_update_step",
    "base_ts_ns",
    "latest_ts_ns",
]
DEFAULT_STEP_NS = 100_000_000
DEFAULT_WINDOW_STEPS = 6_000
DEFAULT_UPDATE_INTERVAL_STEPS = 50
DEFAULT_GAMMA = 0.05
DEFAULT_DELTA = 4.0
DEFAULT_ADJ1 = 1.0
DEFAULT_ADJ2 = 0.05 / 5
DEFAULT_LATENCY_NS = 1_000_000
DEFAULT_GRID_NUM = 5
DEFAULT_QUOTE = "USDC"
PAIR_SKIP_TOKENS = {"glft", "hft", "obi", "full", "small", "sample"}


def _get_trade_field(trade, name: str):
    if hasattr(trade, name):
        return getattr(trade, name)
    try:
        return trade[name]
    except Exception:
        return None


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
        if meta.get(key) != expected_meta.get(key):
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
class LatestGlftState:
    half_spread_tick: float
    skew: float
    volatility: float
    A: float
    k: float
    mid_price_tick: float
    best_bid_tick: float
    best_ask_tick: float
    tick_size: float
    total_steps: int
    last_update_step: int | None
    base_ts_ns: int | None
    latest_ts_ns: int | None


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


def _grid_prices(
    mid_price_tick: float,
    best_bid_tick: float,
    best_ask_tick: float,
    tick_size: float,
    half_spread_tick: float,
    skew: float,
    position: float,
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
    mid_price = _mid_price_from_tick(mid_price_tick, tick_size)
    if not np.isfinite(mid_price) or mid_price <= 0:
        return bid_prices, ask_prices, float("nan")

    reservation_price_tick = mid_price_tick - skew * position
    bid_price_tick = np.minimum(
        np.round(reservation_price_tick - half_spread_tick), best_bid_tick
    )
    ask_price_tick = np.maximum(
        np.round(reservation_price_tick + half_spread_tick), best_ask_tick
    )
    bid_price = bid_price_tick * tick_size
    ask_price = ask_price_tick * tick_size
    if not np.isfinite(bid_price) or not np.isfinite(ask_price):
        return bid_prices, ask_prices, float("nan")

    grid_interval = max(np.round(half_spread_tick) * tick_size, tick_size)
    if not np.isfinite(grid_interval) or grid_interval <= 0:
        return bid_prices, ask_prices, float("nan")

    bid_price = np.floor(bid_price / grid_interval) * grid_interval
    ask_price = np.ceil(ask_price / grid_interval) * grid_interval

    for _ in range(grid_num):
        bid_prices.append(float(bid_price))
        bid_price -= grid_interval
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


def _calc_param_ts(base_ts_ns: int | None, last_update_step: int | None, step_ns: int) -> int | None:
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


def _build_output_payload(
    npz_path: Path,
    pair_info: dict,
    model: str,
    state: LatestGlftState,
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
            "volatility": state.volatility,
            "A": state.A,
            "k": state.k,
            "tick_size": state.tick_size,
            "grid_interval": grid_interval,
            "grid_num": config.get("grid_num"),
            "position": config.get("position"),
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


def _latest_glft_state(
    npz_path: str,
    step_ns: int = DEFAULT_STEP_NS,
    window_steps: int = DEFAULT_WINDOW_STEPS,
    update_interval_steps: int = DEFAULT_UPDATE_INTERVAL_STEPS,
    gamma: float = DEFAULT_GAMMA,
    delta: float = DEFAULT_DELTA,
    adj1: float = DEFAULT_ADJ1,
    adj2: float = DEFAULT_ADJ2,
    latency_ns: int = DEFAULT_LATENCY_NS,
) -> LatestGlftState:
    npz_path = Path(npz_path).resolve()
    meta = json.loads(npz_path.with_suffix(npz_path.suffix + ".meta.json").read_text(encoding="utf-8"))
    tick_size = float(meta["tick_size"])
    lot_size = float(meta["lot_size"])

    cache_path = _cache_path_for(npz_path)
    expected_meta = _build_cache_meta(
        npz_path,
        step_ns,
        window_steps,
        update_interval_steps,
        latency_ns,
        tick_size,
        lot_size,
    )
    cached_state = _load_cache(cache_path, expected_meta, CACHE_STATE_KEYS)
    if cached_state is None:
        base_ts_ns, latest_ts_ns = _read_npz_timestamps(npz_path)

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
            .last_trades_capacity(10000)
        )
        hbt = HashMapMarketDepthBacktest([asset])

        arrival_depth = np.full(window_steps, np.nan, np.float64)
        mid_price_chg = np.full(window_steps, np.nan, np.float64)
        idx = 0
        t = 0
        mid_price_tick = np.nan
        last_mid_price_tick = np.nan
        last_best_bid_tick = np.nan
        last_best_ask_tick = np.nan

        tmp = np.zeros(500, np.float64)
        ticks = np.arange(len(tmp)) + 0.5
        window_seconds = window_steps * step_ns / 1_000_000_000
        vol_scale = np.sqrt(1_000_000_000 / step_ns)

        A = np.nan
        k = np.nan
        volatility = np.nan
        last_update_step: int | None = None

        while hbt.elapse(step_ns) == 0:
            if np.isfinite(mid_price_tick):
                depth = -np.inf
                for last_trade in hbt.last_trades(0):
                    trade_px = _get_trade_field(last_trade, "px")
                    trade_ev = _get_trade_field(last_trade, "ev")
                    if trade_px is None or trade_ev is None:
                        continue
                    trade_tick = trade_px / tick_size
                    if trade_ev & BUY_EVENT == BUY_EVENT:
                        dist = trade_tick - mid_price_tick
                    else:
                        dist = mid_price_tick - trade_tick
                    if dist > depth:
                        depth = dist
                arrival_depth[idx] = depth
            else:
                arrival_depth[idx] = np.nan
            hbt.clear_last_trades(0)

            depth = hbt.depth(0)
            if not np.isfinite(depth.best_bid) or not np.isfinite(depth.best_ask):
                mid_price_chg[idx] = np.nan
                idx = (idx + 1) % window_steps
                t += 1
                continue

            prev_mid_price_tick = mid_price_tick
            best_bid_tick = depth.best_bid_tick
            best_ask_tick = depth.best_ask_tick
            mid_price_tick = (best_bid_tick + best_ask_tick) / 2.0
            last_mid_price_tick = mid_price_tick
            last_best_bid_tick = best_bid_tick
            last_best_ask_tick = best_ask_tick
            mid_price_chg[idx] = mid_price_tick - prev_mid_price_tick

            if update_interval_steps > 0 and t % update_interval_steps == 0:
                if t >= window_steps - 1:
                    arrival_window = _ring_window(arrival_depth, idx)
                    mid_price_window = _ring_window(mid_price_chg, idx)
                    tmp[:] = 0
                    lambda_ = measure_trading_intensity(arrival_window, tmp)
                    if len(lambda_) > 2:
                        lambda_ = lambda_[:70] / window_seconds
                        x = ticks[: len(lambda_)]
                        y = np.log(lambda_)
                        k_, logA = linear_regression(x, y)
                        A = np.exp(logA)
                        k = -k_
                    volatility = np.nanstd(mid_price_window) * vol_scale
                    last_update_step = t

            idx = (idx + 1) % window_steps
            t += 1

        hbt.close()

        cached_state = {
            "last_mid_price_tick": float(last_mid_price_tick),
            "last_best_bid_tick": float(last_best_bid_tick),
            "last_best_ask_tick": float(last_best_ask_tick),
            "A": float(A),
            "k": float(k),
            "volatility": float(volatility),
            "total_steps": int(t),
            "last_update_step": int(last_update_step)
            if last_update_step is not None
            else None,
            "base_ts_ns": int(base_ts_ns) if base_ts_ns is not None else None,
            "latest_ts_ns": int(latest_ts_ns) if latest_ts_ns is not None else None,
        }
        _save_cache(cache_path, expected_meta, cached_state)

    A = float(cached_state["A"])
    k = float(cached_state["k"])
    volatility = float(cached_state["volatility"])
    c1, c2 = compute_coeff(gamma, gamma, delta, A, k)
    half_spread_tick = (c1 + delta / 2 * c2 * volatility) * adj1
    skew = c2 * volatility * adj2
    return LatestGlftState(
        half_spread_tick=half_spread_tick,
        skew=skew,
        volatility=volatility,
        A=A,
        k=k,
        mid_price_tick=float(cached_state["last_mid_price_tick"]),
        best_bid_tick=float(cached_state["last_best_bid_tick"]),
        best_ask_tick=float(cached_state["last_best_ask_tick"]),
        tick_size=float(tick_size),
        total_steps=int(cached_state["total_steps"]),
        last_update_step=cached_state["last_update_step"],
        base_ts_ns=cached_state["base_ts_ns"],
        latest_ts_ns=cached_state["latest_ts_ns"],
    )


def latest_glft_spread(
    npz_path: str,
    step_ns: int = DEFAULT_STEP_NS,
    window_steps: int = DEFAULT_WINDOW_STEPS,
    gamma: float = DEFAULT_GAMMA,
    delta: float = DEFAULT_DELTA,
    adj1: float = DEFAULT_ADJ1,
    adj2: float = DEFAULT_ADJ2,
    latency_ns: int = DEFAULT_LATENCY_NS,
    update_interval_steps: int = DEFAULT_UPDATE_INTERVAL_STEPS,
) -> tuple[float, float, float, float, float]:
    state = _latest_glft_state(
        npz_path,
        step_ns,
        window_steps,
        update_interval_steps,
        gamma,
        delta,
        adj1,
        adj2,
        latency_ns,
    )
    return (
        state.half_spread_tick,
        state.skew,
        state.volatility,
        state.A,
        state.k,
    )


def latest_glft_quotes(
    npz_path: str,
    step_ns: int = DEFAULT_STEP_NS,
    window_steps: int = DEFAULT_WINDOW_STEPS,
    gamma: float = DEFAULT_GAMMA,
    delta: float = DEFAULT_DELTA,
    adj1: float = DEFAULT_ADJ1,
    adj2: float = DEFAULT_ADJ2,
    latency_ns: int = DEFAULT_LATENCY_NS,
    position: float = 0.0,
    update_interval_steps: int = DEFAULT_UPDATE_INTERVAL_STEPS,
) -> tuple[float, float, float, float, float, float, float, float]:
    state = _latest_glft_state(
        npz_path,
        step_ns,
        window_steps,
        update_interval_steps,
        gamma,
        delta,
        adj1,
        adj2,
        latency_ns,
    )
    mid_price = _mid_price_from_tick(state.mid_price_tick, state.tick_size)
    if not np.isfinite(mid_price):
        return (
            state.half_spread_tick,
            state.skew,
            state.volatility,
            state.A,
            state.k,
            np.nan,
            np.nan,
            np.nan,
        )
    bid_prices, ask_prices, _ = _grid_prices(
        state.mid_price_tick,
        state.best_bid_tick,
        state.best_ask_tick,
        state.tick_size,
        state.half_spread_tick,
        state.skew,
        position,
        1,
    )
    if not bid_prices or not ask_prices:
        return (
            state.half_spread_tick,
            state.skew,
            state.volatility,
            state.A,
            state.k,
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
        state.A,
        state.k,
        mid_price,
        bid_bps,
        ask_bps,
    )


def latest_glft_grid_deltas(
    npz_path: str,
    step_ns: int = DEFAULT_STEP_NS,
    window_steps: int = DEFAULT_WINDOW_STEPS,
    update_interval_steps: int = DEFAULT_UPDATE_INTERVAL_STEPS,
    gamma: float = DEFAULT_GAMMA,
    delta: float = DEFAULT_DELTA,
    adj1: float = DEFAULT_ADJ1,
    adj2: float = DEFAULT_ADJ2,
    latency_ns: int = DEFAULT_LATENCY_NS,
    position: float = 0.0,
    grid_num: int = DEFAULT_GRID_NUM,
) -> tuple[LatestGlftState, float, list[dict]]:
    state = _latest_glft_state(
        npz_path,
        step_ns,
        window_steps,
        update_interval_steps,
        gamma,
        delta,
        adj1,
        adj2,
        latency_ns,
    )
    mid_price = _mid_price_from_tick(state.mid_price_tick, state.tick_size)
    bid_prices, ask_prices, grid_interval = _grid_prices(
        state.mid_price_tick,
        state.best_bid_tick,
        state.best_ask_tick,
        state.tick_size,
        state.half_spread_tick,
        state.skew,
        position,
        grid_num,
    )
    rows = _grid_deltas(mid_price, bid_prices, ask_prices)
    return state, grid_interval, rows


if __name__ == "__main__":
    start = time.perf_counter()
    npz_path = "data/btc_hft_glft.npz"
    step_ns = DEFAULT_STEP_NS
    window_steps = DEFAULT_WINDOW_STEPS
    update_interval_steps = DEFAULT_UPDATE_INTERVAL_STEPS
    gamma = DEFAULT_GAMMA
    delta = DEFAULT_DELTA
    adj1 = DEFAULT_ADJ1
    adj2 = DEFAULT_ADJ2
    latency_ns = DEFAULT_LATENCY_NS
    position = 0.0
    grid_num = DEFAULT_GRID_NUM
    model = "GLFT"
    pair_info = _infer_pair_info(Path(npz_path))

    state, grid_interval, rows = latest_glft_grid_deltas(
        npz_path,
        step_ns=step_ns,
        window_steps=window_steps,
        update_interval_steps=update_interval_steps,
        gamma=gamma,
        delta=delta,
        adj1=adj1,
        adj2=adj2,
        latency_ns=latency_ns,
        position=position,
        grid_num=grid_num,
    )
    elapsed = time.perf_counter() - start
    data_time_iso, data_time_ns = _format_ts_fields(state.latest_ts_ns)
    params_ts_ns = _calc_param_ts(state.base_ts_ns, state.last_update_step, step_ns)
    params_time_iso, params_time_ns = _format_ts_fields(params_ts_ns)
    params_step = state.last_update_step if state.last_update_step is not None else "n/a"
    config = {
        "step_ns": step_ns,
        "window_steps": window_steps,
        "update_interval_steps": update_interval_steps,
        "gamma": gamma,
        "delta": delta,
        "adj1": adj1,
        "adj2": adj2,
        "latency_ns": latency_ns,
        "grid_num": grid_num,
        "position": position,
    }
    payload = _build_output_payload(
        Path(npz_path),
        pair_info,
        model,
        state,
        grid_interval,
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
    print("glft_latest:")
    print(f"data_latest_time={data_time_iso} data_latest_ts_ns={data_time_ns}")
    print(
        f"params_time={params_time_iso} params_ts_ns={params_time_ns} params_step={params_step}"
    )
    print(
        "glft_config:",
        f"model={model}",
        f"step_ns={step_ns}",
        f"window_steps={window_steps}",
        f"update_interval_steps={update_interval_steps}",
        f"gamma={gamma}",
        f"delta={delta}",
        f"adj1={adj1}",
        f"adj2={adj2}",
        f"latency_ns={latency_ns}",
    )
    print(
        "glft_state:",
        f"half_spread_tick={_format_float(state.half_spread_tick)}",
        f"skew={_format_float(state.skew)}",
        f"volatility={_format_float(state.volatility)}",
        f"A={_format_float(state.A)}",
        f"k={_format_float(state.k)}",
        f"tick_size={_format_float(state.tick_size)}",
        f"grid_interval={_format_float(grid_interval)}",
        f"grid_num={grid_num}",
        f"position={position}",
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
