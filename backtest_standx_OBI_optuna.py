"""
Optuna-based Bayesian optimization for OBI market-making strategy parameters.

Uses TPE (Tree-structured Parzen Estimator) sampler for intelligent parameter search.
Supports resumable runs via SQLite storage.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.samplers import TPESampler

import backtest_standx_OBI as obi
from convert_standx import convert_parquet_to_npz
from backtest_utils import (
    load_threads_from_config,
    load_fees_from_config,
    save_plots,
    format_ns,
    compute_study_fingerprint,
    build_gap_context,
    load_symbol_from_config,
    compute_sharpe,
)
from backtest_common import infer_roi_bounds, BacktestAPI, build_asset, compute_backtest_params


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
    min_grid_step: float
    looking_depth: float
    roi_lb: float
    roi_ub: float
    max_steps: int
    estimated: int
    grid_num: int
    maker_fee: float
    taker_fee: float
    base_ts_ns: int
    gap_starts_ns: np.ndarray
    gap_ends_ns: np.ndarray


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
    except Exception as exc:
        raise RuntimeError(
            "hftbacktest extension is not available. Build/install py-hftbacktest "
            "before running the backtest."
        ) from exc

    if obi.Dict is None:
        raise RuntimeError("numba is required to run the OBI strategy")

    obi.BUY, obi.SELL, obi.GTX, obi.LIMIT = HBUY, HSELL, HGTX, HLIMIT
    return BacktestAPI(BacktestAsset, ROIVectorMarketDepthBacktest, Recorder)


@dataclass
class BacktestResult:
    equity: float | None
    num_trades: int | None
    sharpe: float  # 0.0 if insufficient data
    median_spread_bps: float | None


def _run_single_backtest(
    ctx: BacktestContext,
    api: BacktestAPI,
    vol_to_half_spread: float,
    skew: float,
    c1: float,
    min_half_spread_bps: float = 1.0,
) -> BacktestResult:
    asset = build_asset(
        api,
        ctx.npz_path,
        ctx.tick_size,
        ctx.lot_size,
        ctx.latency_ns,
        ctx.maker_fee,
        ctx.taker_fee,
        roi_lb=ctx.roi_lb,
        roi_ub=ctx.roi_ub,
    )

    hbt = api.backtest_cls([asset])
    recorder = api.recorder_cls(1, ctx.estimated)
    try:
        median_spread_bps = obi.obi_mm(
            hbt,
            recorder.recorder,
            ctx.record_every,
            ctx.step_ns,
            ctx.max_steps,
            float(vol_to_half_spread),
            ctx.min_grid_step,
            0.0,  # half_spread
            0.0,  # half_spread_bps
            skew,
            c1,
            ctx.looking_depth,
            ctx.window_steps,
            ctx.update_interval_steps,
            ctx.order_qty_dollar,
            ctx.max_position_dollar,
            ctx.grid_num,
            ctx.roi_lb,
            ctx.roi_ub,
            ctx.base_ts_ns,
            ctx.gap_starts_ns,
            ctx.gap_ends_ns,
            min_half_spread_bps,
        )
    finally:
        hbt.close()

    records = recorder.get(0)
    if len(records) == 0:
        return BacktestResult(None, None, 0.0, None)

    valid_mask = np.isfinite(records["price"])
    if not np.any(valid_mask):
        return BacktestResult(None, None, 0.0, None)

    last = records[np.where(valid_mask)[0][-1]]
    equity = float(last["balance"] + last["position"] * last["price"] - last["fee"])
    num_trades = int(last["num_trades"])
    if not np.isfinite(equity):
        return BacktestResult(None, None, 0.0, None)

    sharpe = compute_sharpe(records)
    return BacktestResult(equity, num_trades, sharpe, median_spread_bps)


def _run_backtest_with_records(
    ctx: BacktestContext,
    api: BacktestAPI,
    vol_to_half_spread: float,
    skew: float,
    c1: float,
    min_half_spread_bps: float = 1.0,
) -> np.ndarray | None:
    """Run backtest and return records for plotting."""
    asset = build_asset(
        api,
        ctx.npz_path,
        ctx.tick_size,
        ctx.lot_size,
        ctx.latency_ns,
        ctx.maker_fee,
        ctx.taker_fee,
        roi_lb=ctx.roi_lb,
        roi_ub=ctx.roi_ub,
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
            0.0,  # half_spread
            0.0,  # half_spread_bps
            skew,
            c1,
            ctx.looking_depth,
            ctx.window_steps,
            ctx.update_interval_steps,
            ctx.order_qty_dollar,
            ctx.max_position_dollar,
            ctx.grid_num,
            ctx.roi_lb,
            ctx.roi_ub,
            ctx.base_ts_ns,
            ctx.gap_starts_ns,
            ctx.gap_ends_ns,
            min_half_spread_bps,
        )
    finally:
        hbt.close()

    records = recorder.get(0)
    if len(records) == 0:
        return None

    valid_mask = np.isfinite(records["price"])
    if not np.any(valid_mask):
        return None

    return records[valid_mask]


def load_config(config_path: Path) -> dict[str, Any]:
    """Load Optuna configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def _format_duration(ns: int) -> str:
    """Format duration in nanoseconds to human-readable string."""
    seconds = ns / 1_000_000_000
    days = seconds / 86400
    hours = (seconds % 86400) / 3600
    if days >= 1:
        return f"{days:.1f} days ({hours:.0f}h remaining)"
    return f"{hours:.1f} hours"


def _prepare_context(config: dict[str, Any], quiet: bool = False) -> BacktestContext | None:
    """Prepare backtest context from config."""
    bt_config = config["backtest"]
    data_dir = Path(bt_config["data_dir"])
    max_rows = bt_config.get("max_rows")
    max_days_val = bt_config.get("max_days", 7.0)
    max_days = max_days_val if max_days_val is not None and max_days_val > 0 else None
    latency_ns = bt_config.get("latency_ns", 1_000_000)
    symbol = bt_config.get("symbol") or load_symbol_from_config(Path("config.json"))

    # Generate output path based on symbol (convert XAU-USD -> xau_hft_obi.npz)
    symbol_prefix = symbol.split("-")[0].lower() if symbol else "data"
    out_path = Path(data_dir) / f"{symbol_prefix}_hft_obi.npz"

    if not quiet:
        print("\n" + "=" * 60)
        print("DATA LOADING")
        print("=" * 60)
        print(f"  Symbol:          {symbol}")
        print(f"  Data directory:  {data_dir}")
        max_days_str = f"{max_days} days" if max_days else "unlimited"
        print(f"  Max duration:    {max_days_str}")

    tick_size, lot_size, count, converted = convert_parquet_to_npz(
        data_dir, out_path, max_rows, latency_ns, symbol, max_days=max_days, quiet=True
    )

    if not quiet:
        status = "created" if converted else "cached"
        print(f"  Conversion:      {status} ({count:,} events)")
        print(f"  Output file:     {out_path}")

    data = np.load(out_path)["data"]
    if len(data) == 0:
        print("  ERROR: No events in data file")
        return None

    roi_pad = bt_config.get("roi_pad", 0.02)
    roi_lb, roi_ub = infer_roi_bounds(data, roi_pad)
    roi_lb = float(np.floor(roi_lb / tick_size) * tick_size)
    roi_ub = float(np.ceil(roi_ub / tick_size) * tick_size)
    if roi_ub <= roi_lb:
        raise ValueError("roi bounds are invalid or too narrow")

    min_grid_step = tick_size
    step_ns = int(bt_config.get("step_ns", 100_000_000))
    window_steps = max(1, int(bt_config.get("window_steps", 6000)))
    update_interval_steps = max(1, int(bt_config.get("update_interval_steps", 50)))
    record_every = max(1, int(bt_config.get("record_every", 10)))
    gap_threshold_minutes = float(bt_config.get("gap_threshold_minutes", 10.0))

    start_ts = int(data["local_ts"].min())
    end_ts = int(data["local_ts"].max())
    duration_ns = end_ts - start_ts

    params = compute_backtest_params(data, step_ns, record_every)
    max_steps = int(params["max_steps"])
    estimated = int(params["estimated"])
    base_ts_ns = int(params["base_ts_ns"])

    base_ts_ns, gap_starts_ns, gap_ends_ns, gap_log = build_gap_context(
        data, gap_threshold_minutes, base_ts_ns
    )

    # Load fees from config.json
    maker_fee, taker_fee = load_fees_from_config(Path("config.json"))
    order_qty_dollar = bt_config.get("order_qty_dollar", 20.0)
    max_position_dollar = bt_config.get("max_position_dollar", 500.0)

    if not quiet:
        print("\n" + "=" * 60)
        print("BACKTEST CONFIGURATION")
        print("=" * 60)
        print(f"  Period:          {format_ns(start_ts)} to {format_ns(end_ts)}")
        print(f"  Duration:        {_format_duration(duration_ns)}")
        print(f"  Tick size:       {tick_size}")
        print(f"  Lot size:        {lot_size}")
        print(f"  Latency:         {latency_ns / 1_000_000:.1f} ms")
        print(f"  Maker fee:       {maker_fee:.4%}")
        print(f"  Taker fee:       {taker_fee:.4%}")
        print(f"  Order size:      ${order_qty_dollar:.0f}")
        print(f"  Max position:    ${max_position_dollar:.0f}")
        print(f"  Price range:     {roi_lb:.2f} - {roi_ub:.2f}")
        n_gaps = len(gap_starts_ns)
        if n_gaps > 0:
            print(f"  Data gaps:       {n_gaps} detected (>{gap_threshold_minutes:.0f}min)")

    return BacktestContext(
        npz_path=out_path,
        tick_size=tick_size,
        lot_size=lot_size,
        latency_ns=latency_ns,
        record_every=record_every,
        step_ns=step_ns,
        window_steps=window_steps,
        update_interval_steps=update_interval_steps,
        order_qty_dollar=order_qty_dollar,
        max_position_dollar=max_position_dollar,
        min_grid_step=min_grid_step,
        looking_depth=bt_config.get("looking_depth", 0.025),
        roi_lb=roi_lb,
        roi_ub=roi_ub,
        max_steps=max_steps,
        estimated=estimated,
        grid_num=bt_config.get("grid_num", 1),
        maker_fee=maker_fee,
        taker_fee=taker_fee,
        base_ts_ns=base_ts_ns,
        gap_starts_ns=gap_starts_ns,
        gap_ends_ns=gap_ends_ns,
    )


def create_objective(ctx: BacktestContext, config: dict[str, Any], quiet: bool = False):
    """Create Optuna objective function with backtest context."""
    search_space = config["search_space"]
    opt_config = config["optimization"]
    min_trades = opt_config.get("min_trades", 10)
    objective_metric = opt_config.get("objective_metric", "equity")  # "equity" or "sharpe"

    def objective(trial: optuna.Trial) -> float:
        # Suggest parameters from search space
        vol_config = search_space["vol_to_half_spread"]
        vol = trial.suggest_float(
            "vol_to_half_spread",
            vol_config["min"],
            vol_config["max"],
            log=vol_config.get("log", False),
        )

        skew_config = search_space["skew"]
        skew = trial.suggest_float(
            "skew",
            skew_config["min"],
            skew_config["max"],
            log=skew_config.get("log", False),
        )

        c1_ticks_config = search_space["c1_ticks"]
        if "choices" in c1_ticks_config:
            # Discrete choices mode
            c1_ticks = trial.suggest_categorical("c1_ticks", c1_ticks_config["choices"])
        elif "step" in c1_ticks_config:
            # Float range mode with step
            c1_ticks = trial.suggest_float(
                "c1_ticks",
                c1_ticks_config["min"],
                c1_ticks_config["max"],
                step=c1_ticks_config["step"],
            )
        else:
            # Continuous float mode (default)
            c1_ticks = trial.suggest_float(
                "c1_ticks",
                c1_ticks_config["min"],
                c1_ticks_config["max"],
                log=c1_ticks_config.get("log", False),
            )
        c1_value = c1_ticks * ctx.tick_size

        # Optional min_half_spread_bps parameter
        min_half_spread_bps = 1.0
        if "min_half_spread_bps" in search_space:
            min_bps_config = search_space["min_half_spread_bps"]
            min_half_spread_bps = trial.suggest_float(
                "min_half_spread_bps",
                min_bps_config["min"],
                min_bps_config["max"],
                log=min_bps_config.get("log", False),
            )

        if not quiet:
            print(
                f"trial {trial.number}: vol_to_half_spread={vol:.3f} "
                f"skew={skew:.3f} c1_ticks={c1_ticks} min_half_spread_bps={min_half_spread_bps:.2f}"
            )

        # Run backtest
        api = _load_backtest_api()
        result = _run_single_backtest(ctx, api, vol, skew, c1_value, min_half_spread_bps)

        # Store metrics as user attributes for later analysis
        trial.set_user_attr("num_trades", result.num_trades)
        trial.set_user_attr("equity", result.equity)
        trial.set_user_attr("sharpe", result.sharpe)
        trial.set_user_attr("median_spread_bps", result.median_spread_bps)

        # Handle invalid results - prune trial so TPE learns to avoid this region
        if result.equity is None or not math.isfinite(result.equity):
            raise optuna.TrialPruned("Invalid equity (NaN or None)")

        # Prune trials with no trades - these parameter combinations are infeasible
        if result.num_trades is None or result.num_trades == 0:
            raise optuna.TrialPruned("No trades executed")

        # Penalize insufficient trades (but still valid trials)
        if result.num_trades < min_trades:
            raise optuna.TrialPruned(f"Insufficient trades: {result.num_trades} < {min_trades}")

        # Select metric to optimize
        if objective_metric == "sharpe":
            if result.sharpe is None or not math.isfinite(result.sharpe):
                raise optuna.TrialPruned("Invalid sharpe ratio")
            metric_value = result.sharpe
        else:
            metric_value = result.equity

        if not quiet:
            sharpe_str = f"{result.sharpe:.4f}" if result.sharpe is not None else "N/A"
            print(f"  -> equity={result.equity:.4f} sharpe={sharpe_str} num_trades={result.num_trades}")

        return metric_value

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Optuna-based Bayesian optimization for OBI market-making strategy. "
            "Uses TPE sampler for intelligent parameter search."
        )
    )
    parser.add_argument(
        "--config",
        default="optuna_obi_config.json",
        help="Path to Optuna configuration JSON file.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override n_trials from config.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Override n_jobs (uses config.json threads if not specified).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh study instead of resuming.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing database and start completely fresh.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-trial logging.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for TPE sampler.",
    )
    args = parser.parse_args()

    # Load Optuna config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_config(config_path)

    opt_config = config["optimization"]
    bt_config = config["backtest"]
    n_trials = args.n_trials or opt_config.get("n_trials", 200)

    # Get n_jobs from: CLI arg > config.json threads
    if args.n_jobs is not None:
        n_jobs = args.n_jobs
    else:
        n_jobs = load_threads_from_config(Path("config.json"), default_threads=2)

    # Get symbol and construct symbol-specific study name
    symbol = bt_config.get("symbol") or load_symbol_from_config(Path("config.json"))
    symbol_prefix = symbol.split("-")[0].lower() if symbol else "data"
    base_study_name = opt_config.get("study_name", "obi_mm_opt")
    study_name = f"{symbol_prefix}_{base_study_name}"
    storage = f"sqlite:///optuna_{symbol_prefix}_obi.db"
    load_if_exists = not args.no_resume

    # Delete existing database if --fresh is specified
    if args.fresh:
        db_path = Path(storage.replace("sqlite:///", ""))
        if db_path.exists():
            db_path.unlink()
            print(f"  Deleted: {db_path}")
        journal_path = db_path.with_suffix(".log")
        if journal_path.exists():
            journal_path.unlink()
            print(f"  Deleted: {journal_path}")
        load_if_exists = False

    objective_metric = opt_config.get("objective_metric", "equity")

    print("=" * 60)
    print(f"OBI MARKET-MAKING OPTIMIZATION - {symbol}")
    print("=" * 60)
    print(f"  Study:           {study_name}")
    print(f"  Trials:          {n_trials}")
    print(f"  Parallel jobs:   {n_jobs}")
    print(f"  Objective:       {objective_metric}")
    print(f"  Resume:          {load_if_exists}")

    # Prepare backtest context
    ctx = _prepare_context(config, quiet=args.quiet)
    if ctx is None:
        return

    # Print search space
    search_space = config["search_space"]
    if not args.quiet:
        print("\n" + "=" * 60)
        print("SEARCH SPACE")
        print("=" * 60)
        for param, cfg in search_space.items():
            if "choices" in cfg:
                print(f"  {param}: {cfg['choices']}")
            else:
                log_str = " (log)" if cfg.get("log") else ""
                print(f"  {param}: [{cfg['min']}, {cfg['max']}]{log_str}")

    # Compute fingerprint for data/config change detection
    npz_meta_path = ctx.npz_path.with_suffix(ctx.npz_path.suffix + ".meta.json")
    current_fingerprint = compute_study_fingerprint(config, npz_meta_path)

    # Create or load study
    # Use JournalStorage for parallel runs (n_jobs > 1) to avoid SQLite locking issues
    sampler = TPESampler(seed=args.seed)
    actual_storage_path = storage  # Track for final message
    use_journal = n_jobs > 1 and storage.startswith("sqlite:///")

    if use_journal:
        journal_path = f"optuna_{symbol_prefix}_obi_journal.log"
        actual_storage_path = journal_path
    else:
        journal_path = None

    def create_study_instance(load_existing: bool):
        if use_journal:
            storage_obj = optuna.storages.JournalStorage(
                optuna.storages.JournalFileStorage(journal_path)
            )
            return optuna.create_study(
                direction="maximize",
                study_name=study_name,
                storage=storage_obj,
                load_if_exists=load_existing,
                sampler=sampler,
            )
        else:
            return optuna.create_study(
                direction="maximize",
                study_name=study_name,
                storage=storage,
                load_if_exists=load_existing,
                sampler=sampler,
            )

    # Check if existing study has matching fingerprint
    study = create_study_instance(load_if_exists)
    existing_trials = len(study.trials)

    if existing_trials > 0 and load_if_exists:
        stored_fingerprint = study.user_attrs.get("fingerprint")
        if stored_fingerprint != current_fingerprint:
            reason = "data/config changed" if stored_fingerprint else "legacy study"
            if not args.quiet:
                print(f"\n  Note: {reason}, starting fresh study")
            # Delete storage and recreate
            if use_journal:
                journal_file = Path(journal_path)
                lock_file = Path(journal_path + ".lock")
                if journal_file.exists():
                    journal_file.unlink()
                if lock_file.exists():
                    lock_file.unlink()
            elif storage.startswith("sqlite:///"):
                db_path = Path(storage.replace("sqlite:///", ""))
                if db_path.exists():
                    db_path.unlink()
            study = create_study_instance(False)
            existing_trials = 0

    # Store/update fingerprint
    study.set_user_attr("fingerprint", current_fingerprint)

    if not args.quiet:
        print("\n" + "=" * 60)
        print("STARTING OPTIMIZATION")
        print("=" * 60)
        if existing_trials > 0:
            print(f"  Resuming from {existing_trials} existing trials...")
        else:
            print(f"  Starting fresh optimization...")
        print()

    # Create objective function
    objective = create_objective(ctx, config, quiet=args.quiet)

    # Suppress Optuna logs if quiet
    if args.quiet:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Run optimization
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=not args.quiet)
    except KeyboardInterrupt:
        print("\ninterrupted: optimization stopped")

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)

    # Count trial states
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"  Total trials:    {len(study.trials)}")
    print(f"  Completed:       {len(completed_trials)}")
    print(f"  Pruned:          {len(pruned_trials)}")

    if len(completed_trials) == 0:
        print("\n  ERROR: No completed trials (all pruned - try adjusting search space)")
        return

    # Best trial
    objective_metric = opt_config.get("objective_metric", "equity")
    best = study.best_trial
    print(f"\n  Best Trial: #{best.number}")
    print(f"  -----------")
    print(f"  vol_to_half_spread: {best.params['vol_to_half_spread']:.4f}")
    print(f"  skew:               {best.params['skew']:.4f}")
    print(f"  c1_ticks:           {best.params['c1_ticks']:.1f}")
    if "min_half_spread_bps" in best.params:
        print(f"  min_half_spread_bps: {best.params['min_half_spread_bps']:.2f}")

    print(f"\n  Performance:")
    if "equity" in best.user_attrs and best.user_attrs['equity'] is not None:
        print(f"  Equity:          ${best.user_attrs['equity']:.2f}")
    if "sharpe" in best.user_attrs and best.user_attrs['sharpe'] is not None:
        print(f"  Sharpe ratio:    {best.user_attrs['sharpe']:.4f}")
    if "num_trades" in best.user_attrs:
        print(f"  Trades:          {best.user_attrs['num_trades']:,}")

    # Parameter importance
    try:
        importances = optuna.importance.get_param_importances(study)
        print("\nparam_importance:")
        for param, imp in importances.items():
            print(f"  {param}: {imp:.4f}")
    except Exception:
        print("\nparam_importance: not enough trials for analysis")

    # Top trials (completed_trials already defined above)
    completed_trials.sort(key=lambda t: t.value if t.value is not None else -1e18, reverse=True)

    print(f"\ntop_20_trials (of {len(completed_trials)} completed):")
    print("trial,vol_to_half_spread,skew,c1_ticks,min_half_spread_bps,equity,sharpe,num_trades")
    for t in completed_trials[:20]:
        num_trades = t.user_attrs.get("num_trades", "nan")
        equity = t.user_attrs.get("equity")
        sharpe = t.user_attrs.get("sharpe")
        equity_str = f"{equity:.6f}" if equity is not None else "nan"
        sharpe_str = f"{sharpe:.6f}" if sharpe is not None else "nan"
        min_bps = t.params.get("min_half_spread_bps", 1.0)
        print(
            f"{t.number},"
            f"{t.params['vol_to_half_spread']:.4f},"
            f"{t.params['skew']:.4f},"
            f"{t.params['c1_ticks']},"
            f"{min_bps:.2f},"
            f"{equity_str},"
            f"{sharpe_str},"
            f"{num_trades}"
        )

    print(f"\nresults saved to: {actual_storage_path}")

    # Generate plots for best parameters
    plots_dir = Path(config.get("backtest", {}).get("plots_dir", "plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nrunning final backtest with best parameters for plots...")
    best_vol = best.params["vol_to_half_spread"]
    best_skew = best.params["skew"]
    best_c1_ticks = best.params["c1_ticks"]
    best_c1 = best_c1_ticks * ctx.tick_size
    best_min_half_spread_bps = best.params.get("min_half_spread_bps", 1.0)

    api = _load_backtest_api()
    records = _run_backtest_with_records(ctx, api, best_vol, best_skew, best_c1, best_min_half_spread_bps)

    if records is not None and len(records) > 0:
        plot_name = f"obi_optuna_best_{study_name}"
        plot_title = (
            f"OBI Best: vol_to_half_spread={best_vol:.2f}, "
            f"skew={best_skew:.2f}, c1_ticks={best_c1_ticks:.1f}"
        )
        save_plots(records, plots_dir, plot_name, title=plot_title)
        print(f"plots saved to: {plots_dir}/{plot_name}_balance_equity.png")
    else:
        print("no valid records for plotting")


if __name__ == "__main__":
    main()
