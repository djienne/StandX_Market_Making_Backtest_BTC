from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from convert_standx import convert_parquet_to_npz
from backtest_utils import (
    load_meta,
    format_ns,
    print_meta_summary,
    save_plots,
    load_fees_from_config,
)
from backtest_common import (
    njit,
    NumbaDict as Dict,
    uint64,
    float64,
    BUY,
    SELL,
    GTX,
    LIMIT,
    BUY_EVENT,
)

out_dtype = np.dtype(
    [
        ("half_spread_tick", "f8"),
        ("skew", "f8"),
        ("volatility", "f8"),
        ("A", "f8"),
        ("k", "f8"),
    ]
)


@njit
def measure_trading_intensity(order_arrival_depth, out):
    max_tick = 0
    for depth in order_arrival_depth:
        if not np.isfinite(depth):
            continue

        # Sets the tick index to 0 for the nearest possible best price.
        tick = int(np.round(depth / 0.5)) - 1
        if tick < 0 or tick >= len(out):
            continue

        # Avoid slice updates with dynamic endpoints for Numba stability.
        for idx in range(tick):
            out[idx] += 1.0
        max_tick = max(max_tick, tick)
    return out[:max_tick]


@njit
def linear_regression(x, y):
    sx = np.sum(x)
    sy = np.sum(y)
    sx2 = np.sum(x ** 2)
    sxy = np.sum(x * y)
    w = len(x)
    slope = (w * sxy - sx * sy) / (w * sx2 - sx ** 2)
    intercept = (sy - slope * sx) / w
    return slope, intercept


@njit
def compute_coeff(xi, gamma, delta, A, k):
    inv_k = np.divide(1, k)
    c1 = 1 / (xi * delta) * np.log(1 + xi * delta * inv_k)
    c2 = np.sqrt(np.divide(gamma, 2 * A * delta * k) * ((1 + xi * delta * inv_k) ** (k / (xi * delta) + 1)))
    return c1, c2


@njit
def gridtrading_glft_mm(
    hbt,
    recorder,
    record_every,
    step_ns,
    max_steps,
    order_qty_dollar,
    max_position_dollar,
    grid_num,
    gamma,
    delta,
    adj1,
    adj2,
    update_interval_steps,
    window_steps,
    window_seconds,
    vol_scale,
):
    asset_no = 0
    tick_size = hbt.depth(asset_no).tick_size
    lot_size = hbt.depth(asset_no).lot_size
    order_qty = lot_size

    arrival_depth = np.full(max_steps, np.nan, np.float64)
    mid_price_chg = np.full(max_steps, np.nan, np.float64)
    out = np.zeros(max_steps, out_dtype)

    t = 0
    prev_mid_price_tick = np.nan
    mid_price_tick = np.nan

    tmp = np.zeros(500, np.float64)
    ticks = np.arange(len(tmp)) + 0.5

    A = np.nan
    k = np.nan
    volatility = np.nan

    record_counter = 0

    while hbt.elapse(step_ns) == 0:
        if not np.isnan(mid_price_tick):
            depth = -np.inf
            for last_trade in hbt.last_trades(asset_no):
                trade_price_tick = last_trade.px / tick_size
                if last_trade.ev & BUY_EVENT == BUY_EVENT:
                    dist = trade_price_tick - mid_price_tick
                else:
                    dist = mid_price_tick - trade_price_tick
                if dist > depth:
                    depth = dist
            arrival_depth[t] = depth

        hbt.clear_last_trades(asset_no)
        hbt.clear_inactive_orders(asset_no)

        depth = hbt.depth(asset_no)
        position = hbt.position(asset_no)
        orders = hbt.orders(asset_no)

        if record_every > 0 and record_counter % record_every == 0:
            recorder.record(hbt)
        record_counter += 1

        if not np.isfinite(depth.best_bid) or not np.isfinite(depth.best_ask):
            t += 1
            if t >= max_steps:
                break
            continue

        best_bid_tick = depth.best_bid_tick
        best_ask_tick = depth.best_ask_tick

        prev_mid_price_tick = mid_price_tick
        mid_price_tick = (best_bid_tick + best_ask_tick) / 2.0

        mid_price_chg[t] = mid_price_tick - prev_mid_price_tick

        mid_price = mid_price_tick * tick_size
        if not np.isfinite(mid_price) or mid_price <= 0:
            t += 1
            if t >= max_steps:
                break
            continue

        order_qty = np.round((order_qty_dollar / mid_price) / lot_size) * lot_size
        if order_qty < lot_size:
            order_qty = lot_size
        max_position = max_position_dollar / mid_price

        if t % update_interval_steps == 0:
            if t >= window_steps - 1:
                tmp[:] = 0
                lambda_ = measure_trading_intensity(arrival_depth[t + 1 - window_steps:t + 1], tmp)
                if len(lambda_) > 2:
                    lambda_ = lambda_[:70] / window_seconds
                    x = ticks[:len(lambda_)]
                    y = np.log(lambda_)
                    k_, logA = linear_regression(x, y)
                    A = np.exp(logA)
                    k = -k_

                volatility = np.nanstd(mid_price_chg[t + 1 - window_steps:t + 1]) * vol_scale

        c1, c2 = compute_coeff(gamma, gamma, delta, A, k)

        half_spread_tick = (c1 + delta / 2 * c2 * volatility) * adj1
        skew = c2 * volatility * adj2

        reservation_price_tick = mid_price_tick - skew * position

        bid_price_tick = np.minimum(np.round(reservation_price_tick - half_spread_tick), best_bid_tick)
        ask_price_tick = np.maximum(np.round(reservation_price_tick + half_spread_tick), best_ask_tick)

        bid_price = bid_price_tick * tick_size
        ask_price = ask_price_tick * tick_size

        if not np.isfinite(bid_price) or not np.isfinite(ask_price):
            t += 1
            if t >= max_steps:
                break
            continue

        grid_interval = max(np.round(half_spread_tick) * tick_size, tick_size)
        if not np.isfinite(grid_interval) or grid_interval <= 0:
            t += 1
            if t >= max_steps:
                break
            continue

        bid_price = np.floor(bid_price / grid_interval) * grid_interval
        ask_price = np.ceil(ask_price / grid_interval) * grid_interval

        new_bid_orders = Dict.empty(uint64, float64)
        if position < max_position and np.isfinite(bid_price):
            for _ in range(grid_num):
                bid_tick = round(bid_price / tick_size)
                new_bid_orders[uint64(bid_tick)] = bid_price
                bid_price -= grid_interval

        new_ask_orders = Dict.empty(uint64, float64)
        if position > -max_position and np.isfinite(ask_price):
            for _ in range(grid_num):
                ask_tick = round(ask_price / tick_size)
                new_ask_orders[uint64(ask_tick)] = ask_price
                ask_price += grid_interval

        order_values = orders.values()
        while order_values.has_next():
            order = order_values.get()
            if order.cancellable:
                if (
                    (order.side == BUY and order.order_id not in new_bid_orders)
                    or (order.side == SELL and order.order_id not in new_ask_orders)
                ):
                    hbt.cancel(asset_no, order.order_id, False)

        for order_id, order_price in new_bid_orders.items():
            if order_id not in orders:
                hbt.submit_buy_order(asset_no, order_id, order_price, order_qty, GTX, LIMIT, False)

        for order_id, order_price in new_ask_orders.items():
            if order_id not in orders:
                hbt.submit_sell_order(asset_no, order_id, order_price, order_qty, GTX, LIMIT, False)

        out[t].half_spread_tick = half_spread_tick
        out[t].skew = skew
        out[t].volatility = volatility
        out[t].A = A
        out[t].k = k

        t += 1
        if t >= max_steps:
            break

    return out[:t]


def run_backtest(
    npz_path: Path,
    tick_size: float,
    lot_size: float,
    latency_ns: int,
    record_every: int,
    step_ns: int,
    window_steps: int,
    update_interval_steps: int,
    order_qty_dollar: float,
    max_position_dollar: float,
    grid_num: int,
    gamma: float,
    delta: float,
    adj1: float,
    adj2: float,
    plots_dir: Path | None,
) -> None:
    try:
        from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest, BUY as HBUY, SELL as HSELL, GTX as HGTX, LIMIT as HLIMIT
        from hftbacktest.recorder import Recorder
    except Exception as exc:  # pragma: no cover - requires compiled extension
        raise RuntimeError(
            "hftbacktest extension is not available. Build/install py-hftbacktest before running the backtest."
        ) from exc

    global BUY, SELL, GTX, LIMIT
    BUY, SELL, GTX, LIMIT = HBUY, HSELL, HGTX, HLIMIT

    data = np.load(npz_path)["data"]
    if len(data) == 0:
        print("no events in npz; skipping backtest")
        return

    duration = int(data["local_ts"].max() - data["local_ts"].min())
    base_ts_ns = int(data["local_ts"].min())
    max_steps = max(10_000, int(duration / step_ns) + 10_000)

    window_seconds = window_steps * step_ns / 1_000_000_000
    vol_scale = np.sqrt(1_000_000_000 / step_ns)

    if record_every <= 0:
        record_every = 1
    estimated = max(10_000, int(max_steps / record_every) + 10_000)

    maker_fee, taker_fee = load_fees_from_config(Path("config.json"))
    if not np.isfinite(order_qty_dollar) or order_qty_dollar <= 0:
        raise ValueError("order_qty_dollar must be > 0")
    if not np.isfinite(max_position_dollar) or max_position_dollar <= 0:
        raise ValueError("max_position_dollar must be > 0")
    print(
        "backtest_config:",
        f"step_ns={step_ns}",
        f"window_steps={window_steps}",
        f"update_interval_steps={update_interval_steps}",
        f"record_every={record_every}",
        f"order_qty_dollar={order_qty_dollar}",
        f"max_position_dollar={max_position_dollar}",
        f"grid_num={grid_num}",
        f"gamma={gamma}",
        f"delta={delta}",
        f"adj1={adj1}",
        f"adj2={adj2}",
        f"maker_fee={maker_fee}",
        f"taker_fee={taker_fee}",
    )

    asset = (
        BacktestAsset()
        .data([str(npz_path)])
        .linear_asset(1.0)
        .constant_order_latency(latency_ns, latency_ns)
        .risk_adverse_queue_model()
        .no_partial_fill_exchange()
        .trading_value_fee_model(maker_fee, taker_fee)
        .tick_size(tick_size)
        .lot_size(lot_size)
        .last_trades_capacity(10000)
    )

    hbt = HashMapMarketDepthBacktest([asset])
    recorder = Recorder(1, estimated)

    algo_out = gridtrading_glft_mm(
        hbt,
        recorder.recorder,
        record_every,
        step_ns,
        max_steps,
        order_qty_dollar,
        max_position_dollar,
        grid_num,
        gamma,
        delta,
        adj1,
        adj2,
        update_interval_steps,
        window_steps,
        window_seconds,
        vol_scale,
    )

    hbt.close()

    records = recorder.get(0)
    if len(records) == 0:
        print("no records captured; backtest finished without emitting stats")
        return

    valid_mask = np.isfinite(records["price"])
    if np.any(valid_mask):
        last = records[np.where(valid_mask)[0][-1]]
        equity_wo_fee = float(last["balance"] + last["position"] * last["price"])
        equity = equity_wo_fee - float(last["fee"])
        max_pos = float(np.nanmax(np.abs(records["position"])))
        print(
            "backtest summary:",
            f"timestamp={int(last['timestamp'])}",
            f"price={float(last['price'])}",
            f"position={float(last['position'])}",
            f"balance={float(last['balance'])}",
            f"fee={float(last['fee'])}",
            f"equity_wo_fee={equity_wo_fee}",
            f"equity={equity}",
            f"num_trades={int(last['num_trades'])}",
            f"max_abs_position={max_pos}",
        )
        _print_calibration_summary(algo_out, window_steps, step_ns)
        _print_calibration_table(algo_out, update_interval_steps, step_ns, base_ts_ns)
        _save_calibration_csv(algo_out, update_interval_steps, step_ns, base_ts_ns, plots_dir, npz_path)
        if plots_dir is not None:
            save_plots(records[valid_mask], plots_dir, npz_path.stem)
    else:
        print("backtest summary: no finite price records found")




def _print_calibration_summary(algo_out: np.ndarray, window_steps: int, step_ns: int) -> None:
    if algo_out is None or len(algo_out) == 0:
        print("calibration: no output available")
        return

    valid = np.isfinite(algo_out["k"]) & np.isfinite(algo_out["volatility"])
    if not np.any(valid):
        window_seconds = window_steps * step_ns / 1_000_000_000
        print(f"calibration: not enough data yet (need ~{window_seconds:.1f}s window)")
        return

    idx = np.where(valid)[0][-1]
    last = algo_out[idx]
    sigma = float(last["volatility"])
    A = float(last["A"])
    k = float(last["k"])
    half_spread = float(last["half_spread_tick"])
    skew = float(last["skew"])
    print(
        "calibration_last:",
        f"step={int(idx)}",
        f"sigma={sigma}",
        f"A={A}",
        f"k={k}",
        f"half_spread_tick={half_spread}",
        f"skew={skew}",
    )


def _collect_calibration_rows(
    algo_out: np.ndarray,
    update_interval_steps: int,
    step_ns: int,
    base_ts_ns: int,
) -> list[dict]:
    if algo_out is None or len(algo_out) == 0:
        return []

    valid = np.isfinite(algo_out["k"]) & np.isfinite(algo_out["volatility"]) & np.isfinite(algo_out["A"])
    if not np.any(valid):
        return []

    indices = np.where(valid)[0]
    if update_interval_steps > 0:
        indices = indices[indices % update_interval_steps == 0]

    rows: list[dict] = []
    for idx in indices:
        entry = algo_out[idx]
        ts_ns = base_ts_ns + (idx + 1) * step_ns
        rows.append(
            {
                "step": int(idx),
                "timestamp_ns": int(ts_ns),
                "timestamp_iso": format_ns(int(ts_ns)),
                "sigma": float(entry["volatility"]),
                "A": float(entry["A"]),
                "k": float(entry["k"]),
                "half_spread_tick": float(entry["half_spread_tick"]),
                "skew": float(entry["skew"]),
            }
        )
    return rows


def _print_calibration_table(
    algo_out: np.ndarray,
    update_interval_steps: int,
    step_ns: int,
    base_ts_ns: int,
) -> None:
    interval_seconds = update_interval_steps * step_ns / 1_000_000_000
    rows = _collect_calibration_rows(algo_out, update_interval_steps, step_ns, base_ts_ns)
    if not rows:
        print(f"calibration_table_every_s={interval_seconds} rows=0")
        return

    print(f"calibration_table_every_s={interval_seconds} rows={len(rows)}")
    header = [
        "step",
        "timestamp_iso",
        "timestamp_ns",
        "sigma",
        "A",
        "k",
        "half_spread_tick",
        "skew",
    ]
    print(",".join(header))
    for row in rows:
        print(
            f"{row['step']},"
            f"{row['timestamp_iso']},"
            f"{row['timestamp_ns']},"
            f"{row['sigma']:.6g},"
            f"{row['A']:.6g},"
            f"{row['k']:.6g},"
            f"{row['half_spread_tick']:.6g},"
            f"{row['skew']:.6g}"
        )


def _save_calibration_csv(
    algo_out: np.ndarray,
    update_interval_steps: int,
    step_ns: int,
    base_ts_ns: int,
    plots_dir: Path | None,
    npz_path: Path,
) -> None:
    rows = _collect_calibration_rows(algo_out, update_interval_steps, step_ns, base_ts_ns)
    if not rows:
        print("calibration_csv: no rows to write")
        return

    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)
        out_path = plots_dir / f"{npz_path.stem}_calibration.csv"
    else:
        out_path = npz_path.with_name(npz_path.stem + "_calibration.csv")

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"calibration_csv={out_path}")




def main() -> None:
    parser = argparse.ArgumentParser(description="Run GLFT grid strategy on parquet data.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="data/btc_hft_glft.npz")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--latency-ns", type=int, default=1_000_000)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--step-ns", type=int, default=100_000_000)
    parser.add_argument("--window-steps", type=int, default=6_000)
    parser.add_argument("--update-interval-steps", type=int, default=50)
    parser.add_argument(
        "--order-qty",
        "--order-qty-dollar",
        dest="order_qty_dollar",
        type=float,
        default=20.0,
        help="Order size in quote asset (e.g., USD).",
    )
    parser.add_argument(
        "--max-position",
        "--max-position-dollar",
        dest="max_position_dollar",
        type=float,
        default=400.0,
        help="Max position in quote asset (e.g., USD).",
    )
    parser.add_argument("--grid-num", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=10.0)
    parser.add_argument("--adj1", type=float, default=1.0)
    parser.add_argument("--adj2", type=float, default=0.06)
    parser.add_argument("--plots-dir", default="plots")
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
    parser.add_argument("--symbol", type=str, default=None, help="Symbol to filter (e.g. CRV). Auto-detected if not specified.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    print(f"data_dir={data_dir} out={out_path}")
    tick_size, lot_size, count, converted = convert_parquet_to_npz(
        data_dir, out_path, args.max_rows, args.latency_ns, args.symbol
    )
    if converted:
        print(f"conversion=created events={count}")
    else:
        print(f"conversion=skipped events={count}")
    print(f"tick_size={tick_size} lot_size={lot_size} latency_ns={args.latency_ns} max_rows={args.max_rows}")
    meta = load_meta(out_path)
    if meta:
        print_meta_summary(meta, out_path)

    if args.run_backtest:
        plots_dir = Path(args.plots_dir) if args.plots_dir else None
        print("running_backtest=1")
        run_backtest(
            out_path,
            tick_size,
            lot_size,
            args.latency_ns,
            args.record_every,
            args.step_ns,
            args.window_steps,
            args.update_interval_steps,
            args.order_qty_dollar,
            args.max_position_dollar,
            args.grid_num,
            args.gamma,
            args.delta,
            args.adj1,
            args.adj2,
            plots_dir,
        )


if __name__ == "__main__":
    main()
