from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from convert_standx import convert_parquet_to_npz
from backtest_utils import extract_backtest_results, print_backtest_summary, load_symbol_from_config
from backtest_common import njit, BacktestAPI, build_asset




def run_backtest(
    npz_path: Path,
    tick_size: float,
    lot_size: float,
    latency_ns: int,
    record_every: int,
) -> None:
    try:
        from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest, BUY, SELL, GTX, LIMIT
        from hftbacktest.recorder import Recorder
        # stats requires polars; we keep metrics lightweight here to avoid extra deps.
    except Exception as exc:  # pragma: no cover - requires compiled extension
        raise RuntimeError(
            "hftbacktest extension is not available. Build/install py-hftbacktest before running the backtest."
        ) from exc

    @njit
    def market_making_algo(hbt, recorder, record_every):
        asset_no = 0
        tick_size = hbt.depth(asset_no).tick_size
        lot_size = hbt.depth(asset_no).lot_size
        record_counter = 0

        while hbt.elapse(10_000_000) == 0:
            hbt.clear_inactive_orders(asset_no)

            a = 1
            b = 1
            c = 1
            hs = 1

            forecast = 0
            volatility = 0
            position = hbt.position(asset_no)
            risk = (c + volatility) * position

            max_notional_position = 1000
            notional_qty = 100

            depth = hbt.depth(asset_no)
            if record_every > 0 and record_counter % record_every == 0:
                recorder.record(hbt)
            record_counter += 1
            if not np.isfinite(depth.best_bid) or not np.isfinite(depth.best_ask):
                continue

            mid_price = (depth.best_bid + depth.best_ask) / 2.0

            base_spread = max(tick_size * 2.0, mid_price * 0.0005)
            half_spread = (c + volatility) * hs * base_spread

            reservation_price = mid_price + a * forecast - b * risk
            new_bid = reservation_price - half_spread
            new_ask = reservation_price + half_spread

            new_bid_tick = min(np.round(new_bid / tick_size), depth.best_bid_tick)
            new_ask_tick = max(np.round(new_ask / tick_size), depth.best_ask_tick)

            order_qty = np.round(notional_qty / mid_price / lot_size) * lot_size

            if hbt.elapse(1_000_000) != 0:
                return False

            last_order_id = -1
            update_bid = True
            update_ask = True
            buy_limit_exceeded = position * mid_price > max_notional_position
            sell_limit_exceeded = position * mid_price < -max_notional_position
            orders = hbt.orders(asset_no)
            order_values = orders.values()
            while order_values.has_next():
                order = order_values.get()
                if order.side == BUY:
                    if order.price_tick == new_bid_tick or buy_limit_exceeded:
                        update_bid = False
                    if order.cancellable and (update_bid or buy_limit_exceeded):
                        hbt.cancel(asset_no, order.order_id, False)
                        last_order_id = order.order_id
                elif order.side == SELL:
                    if order.price_tick == new_ask_tick or sell_limit_exceeded:
                        update_ask = False
                    if order.cancellable and (update_ask or sell_limit_exceeded):
                        hbt.cancel(asset_no, order.order_id, False)
                        last_order_id = order.order_id

            if update_bid:
                order_id = new_bid_tick
                hbt.submit_buy_order(asset_no, order_id, new_bid_tick * tick_size, order_qty, GTX, LIMIT, False)
                last_order_id = order_id
            if update_ask:
                order_id = new_ask_tick
                hbt.submit_sell_order(asset_no, order_id, new_ask_tick * tick_size, order_qty, GTX, LIMIT, False)
                last_order_id = order_id

            if last_order_id >= 0:
                timeout = 5_000_000_000
                if hbt.wait_order_response(asset_no, last_order_id, timeout) != 0:
                    return False

        return True

    api = BacktestAPI(BacktestAsset, HashMapMarketDepthBacktest, Recorder)
    asset = build_asset(
        api,
        npz_path,
        tick_size,
        lot_size,
        latency_ns,
        maker_fee=0.0,
        taker_fee=0.0,
    )
    if record_every <= 0:
        record_every = 1
    data = np.load(npz_path)["data"]
    if len(data) == 0:
        print("no events in npz; skipping backtest")
        return
    duration = int(data["local_ts"].max() - data["local_ts"].min())
    estimated = max(10_000, int(duration / (10_000_000 * record_every)) + 10_000)

    hbt = api.backtest_cls([asset])
    recorder = api.recorder_cls(1, estimated)
    market_making_algo(hbt, recorder.recorder, record_every)
    hbt.close()

    records = recorder.get(0)
    if len(records) == 0:
        print("no records captured; backtest finished without emitting stats")
        return

    results = extract_backtest_results(records)
    print_backtest_summary(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple hftbacktest on parquet data.")
    default_symbol = load_symbol_from_config(Path("config.json"))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="data/btc_hft.npz")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--latency-ns", type=int, default=1_000_000)
    parser.add_argument("--record-every", type=int, default=100)
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        default=False,
        help="Skip running the backtest after conversion.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=default_symbol,
        help="Symbol to filter (e.g. CRV). Defaults to config.json symbol if set.",
    )
    args = parser.parse_args()
    args.run_backtest = not args.skip_backtest

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    tick_size, lot_size, count, converted = convert_parquet_to_npz(
        data_dir, out_path, args.max_rows, args.latency_ns, args.symbol
    )
    print(f"tick_size={tick_size} lot_size={lot_size}")
    if converted:
        print(f"wrote {count} events to {out_path}")
    else:
        print(f"conversion skipped; {count} events already in {out_path}")

    if args.run_backtest:
        run_backtest(out_path, tick_size, lot_size, args.latency_ns, args.record_every)


if __name__ == "__main__":
    main()
