from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from convert_standx import convert_parquet_to_npz
from backtest_utils import (
    load_meta,
    print_meta_summary,
    save_plots,
    extract_backtest_results,
    print_backtest_summary,
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
    init_hft_constants,
    infer_roi_bounds,
)


@njit
def obi_mm(
    hbt,
    recorder,
    record_every,
    step_ns,
    max_steps,
    vol_to_half_spread,
    min_grid_step,
    half_spread,
    half_spread_bps,
    skew,
    c1,
    looking_depth,
    window_steps,
    update_interval_steps,
    order_qty_dollar,
    max_position_dollar,
    grid_num,
    roi_lb,
    roi_ub,
):
    asset_no = 0
    imbalance_timeseries = np.full(max_steps, np.nan, np.float64)
    mid_price_chg = np.full(max_steps, np.nan, np.float64)

    tick_size = hbt.depth(asset_no).tick_size
    lot_size = hbt.depth(asset_no).lot_size

    t = 0
    record_counter = 0
    last_alpha = 0.0
    prev_mid_price_tick = np.nan
    volatility = np.nan
    last_half_spread_tick = np.nan
    vol_scale = np.sqrt(1_000_000_000 / step_ns)
    roi_lb_tick = int(round(roi_lb / tick_size))
    roi_ub_tick = int(round(roi_ub / tick_size))
    if roi_lb_tick > roi_ub_tick:
        tmp = roi_lb_tick
        roi_lb_tick = roi_ub_tick
        roi_ub_tick = tmp

    while hbt.elapse(step_ns) == 0:
        hbt.clear_inactive_orders(asset_no)

        depth = hbt.depth(asset_no)
        position = hbt.position(asset_no)
        orders = hbt.orders(asset_no)

        if record_every > 0 and record_counter % record_every == 0:
            recorder.record(hbt)
        record_counter += 1

        best_bid = depth.best_bid
        best_ask = depth.best_ask
        if not np.isfinite(best_bid) or not np.isfinite(best_ask):
            t += 1
            if t >= max_steps:
                break
            continue

        mid_price = (best_bid + best_ask) / 2.0
        mid_price_tick = mid_price / tick_size
        if np.isfinite(prev_mid_price_tick):
            mid_price_chg[t] = mid_price_tick - prev_mid_price_tick
        prev_mid_price_tick = mid_price_tick

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

        imbalance_timeseries[t] = sum_bid_qty - sum_ask_qty

        if update_interval_steps > 0 and t % update_interval_steps == 0:
            if t >= window_steps - 1:
                window_start = t + 1 - window_steps
                m = np.nanmean(imbalance_timeseries[window_start : t + 1])
                s = np.nanstd(imbalance_timeseries[window_start : t + 1])
                if np.isfinite(m) and np.isfinite(s) and s > 0:
                    last_alpha = (imbalance_timeseries[t] - m) / s
                else:
                    last_alpha = 0.0
                volatility = (
                    np.nanstd(mid_price_chg[window_start : t + 1]) * vol_scale
                )
            else:
                last_alpha = 0.0
                volatility = np.nan

        order_qty = max(
            round((order_qty_dollar / mid_price) / lot_size) * lot_size,
            lot_size,
        )
        fair_price = mid_price + c1 * last_alpha

        notional_position = position * mid_price
        normalized_position = notional_position / max_position_dollar

        half_spread_tick = last_half_spread_tick
        if vol_to_half_spread > 0 and np.isfinite(volatility):
            half_spread_tick = volatility * vol_to_half_spread
        elif half_spread_bps > 0:
            half_spread_tick = (
                mid_price * (half_spread_bps / 10000.0) / tick_size
            )
        elif half_spread > 0:
            half_spread_tick = half_spread / tick_size
        last_half_spread_tick = half_spread_tick

        if not np.isfinite(half_spread_tick) or half_spread_tick <= 0:
            t += 1
            if t >= max_steps:
                break
            continue

        bid_depth_tick = half_spread_tick * (1.0 + skew * normalized_position)
        ask_depth_tick = half_spread_tick * (1.0 - skew * normalized_position)
        if bid_depth_tick < 0:
            bid_depth_tick = 0.0
        if ask_depth_tick < 0:
            ask_depth_tick = 0.0

        bid_price = min(
            fair_price - bid_depth_tick * tick_size,
            best_bid,
        )
        ask_price = max(
            fair_price + ask_depth_tick * tick_size,
            best_ask,
        )

        bid_price = np.floor(bid_price / tick_size) * tick_size
        ask_price = np.ceil(ask_price / tick_size) * tick_size
        grid_interval = max(
            np.round(half_spread_tick * tick_size / min_grid_step) * min_grid_step,
            min_grid_step,
        )
        if not np.isfinite(grid_interval) or grid_interval <= 0:
            t += 1
            if t >= max_steps:
                break
            continue

        bid_price = np.floor(bid_price / grid_interval) * grid_interval
        ask_price = np.ceil(ask_price / grid_interval) * grid_interval

        new_bid_orders = Dict.empty(uint64, float64)
        if normalized_position < 1.0 and np.isfinite(bid_price):
            for _ in range(grid_num):
                bid_price_tick = round(bid_price / tick_size)
                new_bid_orders[uint64(bid_price_tick)] = bid_price
                bid_price -= grid_interval

        new_ask_orders = Dict.empty(uint64, float64)
        if normalized_position > -1.0 and np.isfinite(ask_price):
            for _ in range(grid_num):
                ask_price_tick = round(ask_price / tick_size)
                new_ask_orders[uint64(ask_price_tick)] = ask_price
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
                hbt.submit_buy_order(
                    asset_no, order_id, order_price, order_qty, GTX, LIMIT, False
                )

        for order_id, order_price in new_ask_orders.items():
            if order_id not in orders:
                hbt.submit_sell_order(
                    asset_no, order_id, order_price, order_qty, GTX, LIMIT, False
                )

        t += 1
        if t >= max_steps:
            break


def _resolve_price_param(value: float | None, ticks: int | None, tick_size: float) -> float:
    if value is not None:
        return float(value)
    if ticks is None:
        raise ValueError("missing ticks for price param")
    return float(ticks) * tick_size


def _resolve_scalar_param(value: float | None, fallback: float | None) -> float:
    if value is not None:
        return float(value)
    if fallback is None:
        raise ValueError("missing scalar param")
    return float(fallback)




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
    max_position_dollar: float | None,
    max_position_multiplier: float,
    grid_num: int,
    vol_to_half_spread: float,
    half_spread: float | None,
    half_spread_bps: float,
    half_spread_ticks: int | None,
    skew: float | None,
    skew_ticks: float | None,
    c1: float | None,
    c1_ticks: int | None,
    grid_interval: float | None,
    grid_interval_ticks: int | None,
    looking_depth: float,
    roi_lb: float | None,
    roi_ub: float | None,
    roi_pad: float,
    plots_dir: Path | None,
) -> None:
    try:
        from hftbacktest import (
            BacktestAsset,
            ROIVectorMarketDepthBacktest,
            BUY as HBUY,
            SELL as HSELL,
            GTX as HGTX,
            LIMIT as HLIMIT,
        )
        from hftbacktest.recorder import Recorder
    except Exception as exc:  # pragma: no cover - requires compiled extension
        raise RuntimeError(
            "hftbacktest extension is not available. Build/install py-hftbacktest before running the backtest."
        ) from exc

    if Dict is None:
        raise RuntimeError("numba is required to run the OBI strategy")

    global BUY, SELL, GTX, LIMIT
    BUY, SELL, GTX, LIMIT = HBUY, HSELL, HGTX, HLIMIT

    data = np.load(npz_path)["data"]
    if len(data) == 0:
        print("no events in npz; skipping backtest")
        return

    if roi_lb is None or roi_ub is None:
        inferred_lb, inferred_ub = infer_roi_bounds(data, roi_pad)
        if roi_lb is None:
            roi_lb = inferred_lb
        if roi_ub is None:
            roi_ub = inferred_ub
    prices = data["px"].astype(np.float64)
    price_mask = np.isfinite(prices) & (prices > 0)
    sample_mid = float(np.nanmedian(prices[price_mask])) if np.any(price_mask) else np.nan

    roi_lb = float(np.floor(roi_lb / tick_size) * tick_size)
    roi_ub = float(np.ceil(roi_ub / tick_size) * tick_size)
    if roi_ub <= roi_lb:
        raise ValueError("roi bounds are invalid or too narrow")

    half_spread_value = 0.0
    if half_spread is not None or half_spread_ticks is not None:
        half_spread_value = _resolve_price_param(
            half_spread,
            half_spread_ticks,
            tick_size,
        )
        half_spread_bps = 0.0
        vol_to_half_spread = 0.0
    if half_spread_bps > 0:
        vol_to_half_spread = 0.0
    spread_mode = "fixed"
    sample_half_spread = half_spread_value
    if half_spread_bps > 0:
        spread_mode = "bps"
        if np.isfinite(sample_mid):
            sample_half_spread = sample_mid * (half_spread_bps / 10000.0)
        else:
            sample_half_spread = np.nan
    elif vol_to_half_spread > 0:
        spread_mode = "volatility"
        sample_half_spread = np.nan
    skew = _resolve_scalar_param(skew, skew_ticks)
    c1 = _resolve_price_param(c1, c1_ticks, tick_size)
    min_grid_step = _resolve_price_param(
        grid_interval,
        grid_interval_ticks,
        tick_size,
    )
    if min_grid_step <= 0:
        min_grid_step = tick_size
    min_grid_step = max(min_grid_step, tick_size)

    if order_qty_dollar <= 0:
        raise ValueError("order_qty_dollar must be > 0")
    if max_position_dollar is None or max_position_dollar <= 0:
        max_position_dollar = order_qty_dollar * max_position_multiplier

    step_ns = int(step_ns)
    if step_ns <= 0:
        raise ValueError("step_ns must be > 0")
    window_steps = max(1, int(window_steps))
    update_interval_steps = max(1, int(update_interval_steps))
    window_seconds = window_steps * step_ns / 1_000_000_000

    duration = int(data["local_ts"].max() - data["local_ts"].min())
    max_steps = max(10_000, int(duration / step_ns) + 10_000)

    if record_every <= 0:
        record_every = 1
    estimated = max(10_000, int(max_steps / record_every) + 10_000)

    maker_fee = 0.00002
    taker_fee = 0.0002
    print(
        "backtest_config:",
        f"step_ns={step_ns}",
        f"window_seconds={window_seconds}",
        f"window_steps={window_steps}",
        f"update_interval_steps={update_interval_steps}",
        f"record_every={record_every}",
        f"order_qty_dollar={order_qty_dollar}",
        f"max_position_dollar={max_position_dollar}",
        f"grid_num={grid_num}",
        f"half_spread={half_spread_value}",
        f"half_spread_bps={half_spread_bps}",
        f"spread_mode={spread_mode}",
        f"vol_to_half_spread={vol_to_half_spread}",
        f"min_grid_step={min_grid_step}",
        f"sample_mid={sample_mid}",
        f"sample_half_spread={sample_half_spread}",
        f"skew={skew}",
        f"c1={c1}",
        f"looking_depth={looking_depth}",
        "grid_interval=dynamic",
        f"roi_lb={roi_lb}",
        f"roi_ub={roi_ub}",
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
        .roi_lb(roi_lb)
        .roi_ub(roi_ub)
        .last_trades_capacity(10000)
    )

    hbt = ROIVectorMarketDepthBacktest([asset])
    recorder = Recorder(1, estimated)

    obi_mm(
        hbt,
        recorder.recorder,
        record_every,
        step_ns,
        max_steps,
        vol_to_half_spread,
        min_grid_step,
        half_spread_value,
        half_spread_bps,
        skew,
        c1,
        looking_depth,
        window_steps,
        update_interval_steps,
        order_qty_dollar,
        max_position_dollar,
        grid_num,
        roi_lb,
        roi_ub,
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
        if plots_dir is not None:
            save_plots(records[valid_mask], plots_dir, npz_path.stem)
    else:
        print("backtest summary: no finite price records found")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an order book imbalance market-making backtest on parquet data."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="data/btc_hft_obi.npz")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--latency-ns", type=int, default=1_000_000)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--step-ns", type=int, default=100_000_000)
    parser.add_argument("--window-steps", type=int, default=6000)
    parser.add_argument("--update-interval-steps", type=int, default=50)
    parser.add_argument("--order-qty-dollar", type=float, default=20.0)
    parser.add_argument("--max-position-dollar", type=float, default=500.0)
    parser.add_argument("--max-position-multiplier", type=float, default=50.0)
    parser.add_argument("--grid-num", type=int, default=1)
    parser.add_argument("--vol-to-half-spread", type=float, default=9.0)
    parser.add_argument("--half-spread", type=float, default=None)
    parser.add_argument("--half-spread-bps", type=float, default=0.0)
    parser.add_argument("--half-spread-ticks", type=int, default=None)
    parser.add_argument("--skew", type=float, default=30.0)
    parser.add_argument("--skew-ticks", type=float, default=None)
    parser.add_argument("--c1", type=float, default=None)
    parser.add_argument("--c1-ticks", type=int, default=160)
    parser.add_argument("--grid-interval", type=float, default=None)
    parser.add_argument("--grid-interval-ticks", type=int, default=1)
    parser.add_argument("--looking-depth", type=float, default=0.025)
    parser.add_argument("--roi-lb", type=float, default=None)
    parser.add_argument("--roi-ub", type=float, default=None)
    parser.add_argument("--roi-pad", type=float, default=0.02)
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
    print(
        f"tick_size={tick_size} lot_size={lot_size} latency_ns={args.latency_ns} max_rows={args.max_rows}"
    )
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
            args.max_position_multiplier,
            args.grid_num,
            args.vol_to_half_spread,
            args.half_spread,
            args.half_spread_bps,
            args.half_spread_ticks,
            args.skew,
            args.skew_ticks,
            args.c1,
            args.c1_ticks,
            args.grid_interval,
            args.grid_interval_ticks,
            args.looking_depth,
            args.roi_lb,
            args.roi_ub,
            args.roi_pad,
            plots_dir,
        )


if __name__ == "__main__":
    main()
