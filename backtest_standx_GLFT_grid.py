from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import backtest_standx_GLFT as glft
from convert_standx import convert_parquet_to_npz
from backtest_utils import (
    float_grid,
    fmt_float,
    progress_str,
    load_threads_from_config,
    load_fees_from_config,
)
from backtest_common import BacktestAPI


@dataclass(frozen=True)
class BacktestContext:
    npz_path: Path
    tick_size: float
    lot_size: float
    latency_ns: int
    record_every: int
    step_ns: int
    max_steps: int
    order_qty_dollar: float
    max_position_dollar: float
    grid_num: int
    adj1: float
    update_interval_steps: int
    window_steps: int
    window_seconds: float
    vol_scale: float
    estimated: int
    maker_fee: float
    taker_fee: float


def _load_backtest_api() -> BacktestAPI:
    try:
        from hftbacktest import (
            BacktestAsset,
            HashMapMarketDepthBacktest,
            BUY as HBUY,
            SELL as HSELL,
            GTX as HGTX,
            LIMIT as HLIMIT,
        )
        from hftbacktest.recorder import Recorder
    except Exception as exc:  # pragma: no cover - requires compiled extension
        raise RuntimeError(
            "hftbacktest extension is not available. Build/install py-hftbacktest "
            "before running the backtest."
        ) from exc

    glft.BUY, glft.SELL, glft.GTX, glft.LIMIT = HBUY, HSELL, HGTX, HLIMIT
    return BacktestAPI(BacktestAsset, HashMapMarketDepthBacktest, Recorder)


def _run_single_backtest(
    ctx: BacktestContext,
    api: BacktestAPI,
    delta: float,
    adj2: float,
    gamma: float,
) -> tuple[float | None, float | None, int | None]:
    asset = (
        api.asset_cls()
        .data([str(ctx.npz_path)])
        .linear_asset(1.0)
        .constant_order_latency(ctx.latency_ns, ctx.latency_ns)
        .risk_adverse_queue_model()
        .no_partial_fill_exchange()
        .trading_value_fee_model(ctx.maker_fee, ctx.taker_fee)
        .tick_size(ctx.tick_size)
        .lot_size(ctx.lot_size)
        .last_trades_capacity(10000)
    )

    hbt = api.backtest_cls([asset])
    recorder = api.recorder_cls(1, ctx.estimated)
    try:
        glft.gridtrading_glft_mm(
            hbt,
            recorder.recorder,
            ctx.record_every,
            ctx.step_ns,
            ctx.max_steps,
            ctx.order_qty_dollar,
            ctx.max_position_dollar,
            ctx.grid_num,
            gamma,
            delta,
            ctx.adj1,
            adj2,
            ctx.update_interval_steps,
            ctx.window_steps,
            ctx.window_seconds,
            ctx.vol_scale,
        )
    finally:
        hbt.close()

    records = recorder.get(0)
    if len(records) == 0:
        return None, None, None

    valid_mask = np.isfinite(records["price"])
    if not np.any(valid_mask):
        return None, None, None

    last = records[np.where(valid_mask)[0][-1]]
    equity = float(last["balance"] + last["position"] * last["price"] - last["fee"])
    trading_volume = float(last["trading_volume"])
    num_trades = int(last["num_trades"])
    if not np.isfinite(equity) or not np.isfinite(trading_volume):
        return None, None, None
    return equity, trading_volume, num_trades




def _prepare_context(args: argparse.Namespace) -> BacktestContext | None:
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    print(f"data_dir={data_dir} out={out_path}")
    tick_size, lot_size, count, converted = convert_parquet_to_npz(
        data_dir, out_path, args.max_rows, args.latency_ns, getattr(args, 'symbol', None)
    )
    if converted:
        print(f"conversion=created events={count}")
    else:
        print(f"conversion=skipped events={count}")
    print(f"tick_size={tick_size} lot_size={lot_size} latency_ns={args.latency_ns}")

    data = np.load(out_path)["data"]
    if len(data) == 0:
        print("no events in npz; skipping backtest")
        return None

    duration = int(data["local_ts"].max() - data["local_ts"].min())
    max_steps = max(10_000, int(duration / args.step_ns) + 10_000)
    record_every = max(1, int(args.record_every))
    estimated = max(10_000, int(max_steps / record_every) + 10_000)
    window_seconds = args.window_steps * args.step_ns / 1_000_000_000
    vol_scale = np.sqrt(1_000_000_000 / args.step_ns)

    maker_fee, taker_fee = load_fees_from_config(Path("config.json"))
    print(f"maker_fee={maker_fee:.4%} taker_fee={taker_fee:.4%}")

    return BacktestContext(
        npz_path=out_path,
        tick_size=tick_size,
        lot_size=lot_size,
        latency_ns=args.latency_ns,
        record_every=record_every,
        step_ns=args.step_ns,
        max_steps=max_steps,
        order_qty_dollar=args.order_qty_dollar,
        max_position_dollar=args.max_position_dollar,
        grid_num=args.grid_num,
        adj1=args.adj1,
        update_interval_steps=args.update_interval_steps,
        window_steps=args.window_steps,
        window_seconds=window_seconds,
        vol_scale=vol_scale,
        estimated=estimated,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
    )




def _run_single_backtest_worker(
    ctx: BacktestContext,
    delta: float,
    adj2: float,
    gamma: float,
) -> tuple[float | None, float | None, int | None]:
    api = _load_backtest_api()
    return _run_single_backtest(ctx, api, delta, adj2, gamma)


def _run_single_backtest_worker_args(
    payload: tuple[BacktestContext, float, float, float],
) -> tuple[float, float, float, float | None, float | None, int | None]:
    ctx, gamma, delta, adj2 = payload
    equity, trading_volume, num_trades = _run_single_backtest_worker(
        ctx, delta, adj2, gamma
    )
    return float(gamma), float(delta), float(adj2), equity, trading_volume, num_trades


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Grid search over delta and adj2 for backtest_standx_GLFT, summarizing"
            "final equity and trading volume."
        )
    )
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
    parser.add_argument("--grid-num", type=int, default=5)
    parser.add_argument(
        "--gamma",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1],
        help="Gamma value(s) to test.",
    )
    parser.add_argument("--adj1", type=float, default=1.0)
    parser.add_argument("--delta-min", type=float, default=1.0)
    parser.add_argument("--delta-max", type=float, default=10.0)
    parser.add_argument("--delta-step", type=float, default=1.0)
    parser.add_argument("--adj2-min", type=float, default=0.01)
    parser.add_argument("--adj2-max", type=float, default=0.1)
    parser.add_argument("--adj2-step", type=float, default=0.01)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-run progress logging.",
    )
    parser.add_argument("--symbol", type=str, default=None, help="Symbol to filter (e.g. CRV). Auto-detected if not specified.")
    args = parser.parse_args()

    if not np.isfinite(args.order_qty_dollar) or args.order_qty_dollar <= 0:
        raise ValueError("order_qty_dollar must be > 0")
    if not np.isfinite(args.max_position_dollar) or args.max_position_dollar <= 0:
        raise ValueError("max_position_dollar must be > 0")
    if not args.gamma or any(not np.isfinite(gamma) for gamma in args.gamma):
        raise ValueError("gamma values must be finite")

    ctx = _prepare_context(args)
    if ctx is None:
        return

    gamma_values = [float(gamma) for gamma in args.gamma]
    delta_values = float_grid(args.delta_min, args.delta_max, args.delta_step, 6)
    adj2_values = float_grid(args.adj2_min, args.adj2_max, args.adj2_step, 6)

    threads = load_threads_from_config(Path("config.json"), default_threads=6)
    print(f"threads={threads} cpu(s)")

    results: list[dict] = []
    total = len(gamma_values) * len(delta_values) * len(adj2_values)
    run_idx = 0
    if threads <= 1:
        api = _load_backtest_api()
        try:
            for gamma in gamma_values:
                for delta in delta_values:
                    for adj2 in adj2_values:
                        run_idx += 1
                        if not args.quiet:
                            print(
                                "run"
                                f" {progress_str(run_idx, total)}:"
                                f" gamma={gamma}"
                                f" delta={delta}"
                                f" adj2={adj2}"
                                " status=running"
                            )
                        equity, trading_volume, num_trades = _run_single_backtest(
                            ctx,
                            api,
                            float(delta),
                            float(adj2),
                            float(gamma),
                        )
                        results.append(
                            {
                                "gamma": float(gamma),
                                "delta": float(delta),
                                "adj2": float(adj2),
                                "equity": equity,
                                "trading_volume": trading_volume,
                                "num_trades": num_trades,
                            }
                        )
        except KeyboardInterrupt:
            print("interrupted: stopping remaining runs")
            return
    else:
        payloads: list[tuple[BacktestContext, float, float, float]] = []
        for gamma in gamma_values:
            for delta in delta_values:
                for adj2 in adj2_values:
                    run_idx += 1
                    if not args.quiet:
                        print(
                            "run"
                            f" {progress_str(run_idx, total)}:"
                            f" gamma={gamma}"
                            f" delta={delta}"
                            f" adj2={adj2}"
                            " status=queued"
                        )
                    payloads.append(
                        (ctx, float(gamma), float(delta), float(adj2))
                    )

        ctx_mp = mp.get_context("spawn")
        done = 0
        with ctx_mp.Pool(processes=threads) as pool:
            try:
                for (
                    gamma,
                    delta,
                    adj2,
                    equity,
                    trading_volume,
                    num_trades,
                ) in pool.imap_unordered(
                    _run_single_backtest_worker_args,
                    payloads,
                ):
                    done += 1
                    if not args.quiet:
                        print(
                            "done:"
                            f" {progress_str(done, total)}"
                            f" gamma={gamma}"
                            f" delta={delta}"
                            f" adj2={adj2}"
                        )
                    results.append(
                        {
                            "gamma": float(gamma),
                            "delta": float(delta),
                            "adj2": float(adj2),
                            "equity": equity,
                            "trading_volume": trading_volume,
                            "num_trades": num_trades,
                        }
                    )
            except KeyboardInterrupt:
                print("interrupted: terminating worker pool")
                pool.terminate()
                pool.join()
                return

    results.sort(
        key=lambda row: (
            row["equity"] is None,
            -(row["equity"] if row["equity"] is not None else 0.0),
        )
    )

    print(f"grid_summary rows={len(results)} sorted_by=equity desc")
    print("gamma,delta,adj2,equity,trading_volume,num_trades")
    for row in results:
        gamma_str = fmt_float(row["gamma"])
        delta_str = fmt_float(row["delta"])
        adj2_str = fmt_float(row["adj2"])
        equity_str = fmt_float(row["equity"])
        volume_str = fmt_float(row["trading_volume"])
        trades_str = "nan" if row["num_trades"] is None else str(row["num_trades"])
        print(
            f"{gamma_str},{delta_str},{adj2_str},{equity_str},{volume_str},{trades_str}"
        )


if __name__ == "__main__":
    main()
