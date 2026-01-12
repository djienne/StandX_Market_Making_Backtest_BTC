from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import backtest_standx_OBI as obi
from convert_standx import convert_parquet_to_npz
from backtest_utils import (
    float_grid,
    fmt_float,
    progress_str,
    load_threads_from_config,
    load_fees_from_config,
)
from backtest_common import infer_roi_bounds, BacktestAPI


@dataclass(frozen=True)
class BacktestContext:
    npz_path: Path
    tick_size: float
    lot_size: float
    latency_ns: int
    record_every: int
    step_ns: int
    window_steps: int
    update_interval_steps: int
    order_qty_dollar: float
    max_position_dollar: float
    skew: float
    min_grid_step: float
    looking_depth: float
    roi_lb: float
    roi_ub: float
    max_steps: int
    estimated: int
    maker_fee: float
    taker_fee: float


def _load_backtest_api() -> BacktestAPI:
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
            "hftbacktest extension is not available. Build/install py-hftbacktest "
            "before running the backtest."
        ) from exc

    if obi.Dict is None:
        raise RuntimeError("numba is required to run the OBI strategy")

    obi.BUY, obi.SELL, obi.GTX, obi.LIMIT = HBUY, HSELL, HGTX, HLIMIT
    return BacktestAPI(BacktestAsset, ROIVectorMarketDepthBacktest, Recorder)


def _run_single_backtest(
    ctx: BacktestContext,
    api: BacktestAPI,
    vol_to_half_spread: float,
    skew: float,
    c1: float,
    grid_num: int,
) -> tuple[float | None, int | None]:
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
        .roi_lb(ctx.roi_lb)
        .roi_ub(ctx.roi_ub)
        .last_trades_capacity(10000)
    )

    hbt = api.backtest_cls([asset])
    recorder = api.recorder_cls(1, ctx.estimated)
    try:
        obi.obi_mm(
            hbt,
            recorder.recorder,
            ctx.record_every,
            ctx.step_ns,
            ctx.max_steps,
            float(vol_to_half_spread),
            ctx.min_grid_step,
            0.0,
            0.0,
            skew,
            c1,
            ctx.looking_depth,
            ctx.window_steps,
            ctx.update_interval_steps,
            ctx.order_qty_dollar,
            ctx.max_position_dollar,
            grid_num,
            ctx.roi_lb,
            ctx.roi_ub,
        )
    finally:
        hbt.close()

    records = recorder.get(0)
    if len(records) == 0:
        return None, None

    valid_mask = np.isfinite(records["price"])
    if not np.any(valid_mask):
        return None, None

    last = records[np.where(valid_mask)[0][-1]]
    equity = float(last["balance"] + last["position"] * last["price"] - last["fee"])
    num_trades = int(last["num_trades"])
    if not np.isfinite(equity):
        return None, None
    return equity, num_trades


def _parse_float_list(values: str) -> list[float]:
    items = [item.strip() for item in values.split(",") if item.strip()]
    if not items:
        raise ValueError("skew-values must contain at least one number")
    parsed: list[float] = []
    for item in items:
        try:
            parsed.append(float(item))
        except ValueError as exc:
            raise ValueError(f"invalid skew value: {item}") from exc
    return parsed


def _parse_int_list(values: str, label: str) -> list[int]:
    items = [item.strip() for item in values.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{label} must contain at least one integer")
    parsed: list[int] = []
    for item in items:
        try:
            parsed.append(int(item))
        except ValueError as exc:
            raise ValueError(f"invalid {label} value: {item}") from exc
    return parsed




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

    roi_lb = args.roi_lb
    roi_ub = args.roi_ub
    if roi_lb is None or roi_ub is None:
        inferred_lb, inferred_ub = infer_roi_bounds(data, args.roi_pad)
        if roi_lb is None:
            roi_lb = inferred_lb
        if roi_ub is None:
            roi_ub = inferred_ub

    roi_lb = float(np.floor(roi_lb / tick_size) * tick_size)
    roi_ub = float(np.ceil(roi_ub / tick_size) * tick_size)
    if roi_ub <= roi_lb:
        raise ValueError("roi bounds are invalid or too narrow")

    skew = obi._resolve_scalar_param(args.skew, args.skew_ticks)
    min_grid_step = obi._resolve_price_param(
        args.grid_interval,
        args.grid_interval_ticks,
        tick_size,
    )
    if min_grid_step <= 0:
        min_grid_step = tick_size
    min_grid_step = max(min_grid_step, tick_size)

    max_position_dollar = args.max_position_dollar
    if max_position_dollar is None or max_position_dollar <= 0:
        max_position_dollar = args.order_qty_dollar * args.max_position_multiplier

    step_ns = int(args.step_ns)
    if step_ns <= 0:
        raise ValueError("step_ns must be > 0")
    window_steps = max(1, int(args.window_steps))
    update_interval_steps = max(1, int(args.update_interval_steps))
    record_every = max(1, int(args.record_every))

    duration = int(data["local_ts"].max() - data["local_ts"].min())
    max_steps = max(10_000, int(duration / step_ns) + 10_000)
    estimated = max(10_000, int(max_steps / record_every) + 10_000)

    maker_fee, taker_fee = load_fees_from_config(Path("config.json"))
    print(f"maker_fee={maker_fee:.4%} taker_fee={taker_fee:.4%}")

    return BacktestContext(
        npz_path=out_path,
        tick_size=tick_size,
        lot_size=lot_size,
        latency_ns=args.latency_ns,
        record_every=record_every,
        step_ns=step_ns,
        window_steps=window_steps,
        update_interval_steps=update_interval_steps,
        order_qty_dollar=args.order_qty_dollar,
        max_position_dollar=max_position_dollar,
        skew=skew,
        min_grid_step=min_grid_step,
        looking_depth=args.looking_depth,
        roi_lb=roi_lb,
        roi_ub=roi_ub,
        max_steps=max_steps,
        estimated=estimated,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
    )




def _warmup_jit(ctx: BacktestContext) -> None:
    """Pre-compile numba JIT functions before spawning workers."""
    print("warming up numba JIT compilation...")
    api = _load_backtest_api()
    _run_single_backtest(ctx, api, 5.0, 10.0, ctx.tick_size * 80, 1)
    print("JIT warmup complete")


def _run_single_backtest_worker(
    ctx: BacktestContext,
    vol_to_half_spread: float,
    skew: float,
    c1: float,
    grid_num: int,
) -> tuple[float | None, int | None]:
    api = _load_backtest_api()
    return _run_single_backtest(ctx, api, vol_to_half_spread, skew, c1, grid_num)


def _run_single_backtest_worker_args(
    payload: tuple[BacktestContext, float, float, float, int, int],
) -> tuple[float, float, int, int, float | None, int | None]:
    ctx, vol_to_half_spread, skew, c1, c1_ticks, grid_num = payload
    equity, num_trades = _run_single_backtest_worker(
        ctx, vol_to_half_spread, skew, c1, grid_num
    )
    return float(vol_to_half_spread), float(skew), int(c1_ticks), int(grid_num), equity, num_trades


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Grid search over vol_to_half_spread, skew, and c1_ticks for "
            "backtest_standx_OBI, summarizing final equity."
        )
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
    parser.add_argument("--grid-num", type=int, default=None, help="Single grid_num value (overrides --grid-num-values)")
    parser.add_argument(
        "--grid-num-values",
        type=str,
        default="1,2,3",
        help="Comma-separated grid_num values for grid search.",
    )
    parser.add_argument("--vol-to-half-spread-min", type=float, default=2.0)
    parser.add_argument("--vol-to-half-spread-max", type=float, default=10.0)
    parser.add_argument("--vol-to-half-spread-step", type=float, default=1.0)
    parser.add_argument("--skew", type=float, default=20.0)
    parser.add_argument("--skew-ticks", type=float, default=None)
    parser.add_argument(
        "--skew-values",
        default="0.1,1,5,10,20,30,40",
        help="Comma-separated skew values for grid search.",
    )
    parser.add_argument("--c1", type=float, default=None)
    parser.add_argument(
        "--c1-ticks",
        type=str,
        default="80,160,240",
        help="Comma-separated c1 tick values for grid search.",
    )
    parser.add_argument("--grid-interval", type=float, default=None)
    parser.add_argument("--grid-interval-ticks", type=int, default=1)
    parser.add_argument("--looking-depth", type=float, default=0.025)
    parser.add_argument("--roi-lb", type=float, default=None)
    parser.add_argument("--roi-ub", type=float, default=None)
    parser.add_argument("--roi-pad", type=float, default=0.02)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-run progress logging.",
    )
    parser.add_argument("--symbol", type=str, default=None, help="Symbol to filter (e.g. CRV). Auto-detected if not specified.")
    args = parser.parse_args()

    if not np.isfinite(args.order_qty_dollar) or args.order_qty_dollar <= 0:
        raise ValueError("order_qty_dollar must be > 0")
    if args.max_position_dollar is not None and args.max_position_dollar <= 0:
        raise ValueError("max_position_dollar must be > 0")

    ctx = _prepare_context(args)
    if ctx is None:
        return
    vol_values = float_grid(
        args.vol_to_half_spread_min,
        args.vol_to_half_spread_max,
        args.vol_to_half_spread_step,
        6,
    )
    skew_values = _parse_float_list(args.skew_values)
    if args.c1 is not None:
        c1_value = float(args.c1)
        c1_ticks = int(round(c1_value / ctx.tick_size))
        c1_values = [(c1_ticks, c1_value)]
    else:
        c1_ticks_values = _parse_int_list(args.c1_ticks, "c1-ticks")
        c1_values = [
            (c1_ticks, obi._resolve_price_param(None, c1_ticks, ctx.tick_size))
            for c1_ticks in c1_ticks_values
        ]
    if args.grid_num is not None:
        grid_num_values = [int(args.grid_num)]
    else:
        grid_num_values = _parse_int_list(args.grid_num_values, "grid-num-values")

    threads = load_threads_from_config(Path("config.json"), default_threads=6)
    print(f"threads={threads} cpu(s)")

    results: list[dict] = []
    total = len(vol_values) * len(skew_values) * len(c1_values) * len(grid_num_values)
    run_idx = 0
    if threads <= 1:
        api = _load_backtest_api()
        try:
            for grid_num in grid_num_values:
                for c1_ticks, c1_value in c1_values:
                    for skew in skew_values:
                        for vol_to_half_spread in vol_values:
                            run_idx += 1
                            if not args.quiet:
                                print(
                                    "run"
                                    f" {progress_str(run_idx, total)}:"
                                    f" vol_to_half_spread={vol_to_half_spread}"
                                    f" skew={skew}"
                                    f" c1_ticks={c1_ticks}"
                                    f" grid_num={grid_num}"
                                    " status=running"
                                )
                            equity, num_trades = _run_single_backtest(
                                ctx, api, vol_to_half_spread, skew, c1_value, grid_num
                            )
                            results.append(
                                {
                                    "vol_to_half_spread": float(vol_to_half_spread),
                                    "skew": float(skew),
                                    "c1_ticks": int(c1_ticks),
                                    "grid_num": int(grid_num),
                                    "equity": equity,
                                    "num_trades": num_trades,
                                }
                            )
        except KeyboardInterrupt:
            print("interrupted: stopping remaining runs")
            return
    else:
        _warmup_jit(ctx)
        payloads: list[tuple[BacktestContext, float, float, float, int, int]] = []
        for grid_num in grid_num_values:
            for c1_ticks, c1_value in c1_values:
                for skew in skew_values:
                    for vol_to_half_spread in vol_values:
                        run_idx += 1
                        if not args.quiet:
                            print(
                                "run"
                                f" {progress_str(run_idx, total)}:"
                                f" vol_to_half_spread={vol_to_half_spread}"
                                f" skew={skew}"
                                f" c1_ticks={c1_ticks}"
                                f" grid_num={grid_num}"
                                " status=queued"
                            )
                        payloads.append(
                            (
                                ctx,
                                float(vol_to_half_spread),
                                float(skew),
                                float(c1_value),
                                int(c1_ticks),
                                int(grid_num),
                            )
                        )

        ctx_mp = mp.get_context("spawn")
        done = 0
        with ctx_mp.Pool(processes=threads) as pool:
            try:
                for (
                    vol_to_half_spread,
                    skew,
                    c1_ticks,
                    grid_num,
                    equity,
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
                            f" vol_to_half_spread={vol_to_half_spread}"
                            f" skew={skew}"
                            f" c1_ticks={c1_ticks}"
                            f" grid_num={grid_num}"
                        )
                    results.append(
                        {
                            "vol_to_half_spread": float(vol_to_half_spread),
                            "skew": float(skew),
                            "c1_ticks": int(c1_ticks),
                            "grid_num": int(grid_num),
                            "equity": equity,
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
    print("vol_to_half_spread,skew,c1_ticks,grid_num,equity,num_trades")
    for row in results:
        vol_str = fmt_float(row["vol_to_half_spread"])
        skew_str = fmt_float(row["skew"])
        c1_ticks_str = fmt_float(row["c1_ticks"])
        grid_num_str = str(row["grid_num"])
        equity_str = fmt_float(row["equity"])
        trades_str = "nan" if row["num_trades"] is None else str(row["num_trades"])
        print(f"{vol_str},{skew_str},{c1_ticks_str},{grid_num_str},{equity_str},{trades_str}")


if __name__ == "__main__":
    main()
