"""
Optuna-based Bayesian optimization for GLFT market-making strategy parameters.

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

import backtest_standx_GLFT as glft
from convert_standx import convert_parquet_to_npz
from backtest_utils import load_threads_from_config
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
    maker_fee: float = 0.00002
    taker_fee: float = 0.0002


@dataclass
class BacktestResult:
    equity: float | None
    trading_volume: float | None
    num_trades: int | None
    sharpe: float | None


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
    except Exception as exc:
        raise RuntimeError(
            "hftbacktest extension is not available. Build/install py-hftbacktest "
            "before running the backtest."
        ) from exc

    glft.BUY, glft.SELL, glft.GTX, glft.LIMIT = HBUY, HSELL, HGTX, HLIMIT
    return BacktestAPI(BacktestAsset, HashMapMarketDepthBacktest, Recorder)


def _compute_sharpe(records: np.ndarray, annualization: float = 252.0) -> float | None:
    """Compute Sharpe ratio from backtest records."""
    valid_mask = np.isfinite(records["price"])
    if np.sum(valid_mask) < 10:
        return None

    valid_records = records[valid_mask]
    equity_curve = (
        valid_records["balance"]
        + valid_records["position"] * valid_records["price"]
        - valid_records["fee"]
    )

    if len(equity_curve) < 2:
        return None

    returns = np.diff(equity_curve)
    if len(returns) == 0 or np.std(returns) == 0:
        return None

    mean_return = np.mean(returns)
    std_return = np.std(returns)
    if std_return == 0:
        return None

    sharpe = (mean_return / std_return) * np.sqrt(annualization)
    if not np.isfinite(sharpe):
        return None
    return float(sharpe)


def _run_single_backtest(
    ctx: BacktestContext,
    api: BacktestAPI,
    gamma: float,
    delta: float,
    adj2: float,
) -> BacktestResult:
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
        return BacktestResult(None, None, None, None)

    valid_mask = np.isfinite(records["price"])
    if not np.any(valid_mask):
        return BacktestResult(None, None, None, None)

    last = records[np.where(valid_mask)[0][-1]]
    equity = float(last["balance"] + last["position"] * last["price"] - last["fee"])
    trading_volume = float(last["trading_volume"])
    num_trades = int(last["num_trades"])

    if not np.isfinite(equity) or not np.isfinite(trading_volume):
        return BacktestResult(None, None, None, None)

    sharpe = _compute_sharpe(records)
    return BacktestResult(equity, trading_volume, num_trades, sharpe)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load Optuna configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def _prepare_context(config: dict[str, Any]) -> BacktestContext | None:
    """Prepare backtest context from config."""
    bt_config = config["backtest"]
    data_dir = Path(bt_config["data_dir"])
    out_path = Path(bt_config["out"])
    max_rows = bt_config.get("max_rows")
    latency_ns = bt_config.get("latency_ns", 1_000_000)
    symbol = bt_config.get("symbol")

    print(f"data_dir={data_dir} out={out_path}")
    tick_size, lot_size, count, converted = convert_parquet_to_npz(
        data_dir, out_path, max_rows, latency_ns, symbol
    )
    if converted:
        print(f"conversion=created events={count}")
    else:
        print(f"conversion=skipped events={count}")
    print(f"tick_size={tick_size} lot_size={lot_size} latency_ns={latency_ns}")

    data = np.load(out_path)["data"]
    if len(data) == 0:
        print("no events in npz; skipping backtest")
        return None

    step_ns = int(bt_config.get("step_ns", 100_000_000))
    window_steps = max(1, int(bt_config.get("window_steps", 6000)))
    update_interval_steps = max(1, int(bt_config.get("update_interval_steps", 50)))
    record_every = max(1, int(bt_config.get("record_every", 10)))

    duration = int(data["local_ts"].max() - data["local_ts"].min())
    max_steps = max(10_000, int(duration / step_ns) + 10_000)
    estimated = max(10_000, int(max_steps / record_every) + 10_000)
    window_seconds = window_steps * step_ns / 1_000_000_000
    vol_scale = np.sqrt(1_000_000_000 / step_ns)

    return BacktestContext(
        npz_path=out_path,
        tick_size=tick_size,
        lot_size=lot_size,
        latency_ns=latency_ns,
        record_every=record_every,
        step_ns=step_ns,
        max_steps=max_steps,
        order_qty_dollar=bt_config.get("order_qty_dollar", 20.0),
        max_position_dollar=bt_config.get("max_position_dollar", 400.0),
        grid_num=bt_config.get("grid_num", 5),
        adj1=bt_config.get("adj1", 1.0),
        update_interval_steps=update_interval_steps,
        window_steps=window_steps,
        window_seconds=window_seconds,
        vol_scale=vol_scale,
        estimated=estimated,
    )


def create_objective(ctx: BacktestContext, config: dict[str, Any], quiet: bool = False):
    """Create Optuna objective function with backtest context."""
    search_space = config["search_space"]
    opt_config = config["optimization"]
    min_trades = opt_config.get("min_trades", 10)
    objective_metric = opt_config.get("objective_metric", "equity")

    def objective(trial: optuna.Trial) -> float:
        # Suggest parameters from search space
        gamma_config = search_space["gamma"]
        gamma = trial.suggest_float(
            "gamma",
            gamma_config["min"],
            gamma_config["max"],
            log=gamma_config.get("log", False),
        )

        delta_config = search_space["delta"]
        delta = trial.suggest_float(
            "delta",
            delta_config["min"],
            delta_config["max"],
            log=delta_config.get("log", False),
        )

        adj2_config = search_space["adj2"]
        adj2 = trial.suggest_float(
            "adj2",
            adj2_config["min"],
            adj2_config["max"],
            log=adj2_config.get("log", False),
        )

        if not quiet:
            print(
                f"trial {trial.number}: gamma={gamma:.4f} "
                f"delta={delta:.3f} adj2={adj2:.4f}"
            )

        # Run backtest
        api = _load_backtest_api()
        result = _run_single_backtest(ctx, api, gamma, delta, adj2)

        # Store metrics as user attributes for later analysis
        trial.set_user_attr("num_trades", result.num_trades)
        trial.set_user_attr("equity", result.equity)
        trial.set_user_attr("trading_volume", result.trading_volume)
        trial.set_user_attr("sharpe", result.sharpe)

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
            "Optuna-based Bayesian optimization for GLFT market-making strategy. "
            "Uses TPE sampler for intelligent parameter search."
        )
    )
    parser.add_argument(
        "--config",
        default="optuna_glft_config.json",
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
    n_trials = args.n_trials or opt_config.get("n_trials", 200)

    # Get n_jobs from: CLI arg > config.json threads
    if args.n_jobs is not None:
        n_jobs = args.n_jobs
    else:
        n_jobs = load_threads_from_config(Path("config.json"), default_threads=2)

    study_name = opt_config.get("study_name", "glft_mm_opt")
    storage = opt_config.get("storage", "sqlite:///optuna_glft.db")
    load_if_exists = not args.no_resume

    # Delete existing database if --fresh is specified
    if args.fresh:
        if storage.startswith("sqlite:///"):
            db_path = Path(storage.replace("sqlite:///", ""))
            if db_path.exists():
                db_path.unlink()
                print(f"deleted existing database: {db_path}")
        load_if_exists = False

    objective_metric = opt_config.get("objective_metric", "equity")
    print(f"study_name={study_name} storage={storage}")
    print(f"n_trials={n_trials} n_jobs={n_jobs} resume={load_if_exists}")
    print(f"objective_metric={objective_metric}")

    # Prepare backtest context
    ctx = _prepare_context(config)
    if ctx is None:
        return

    print(f"grid_num={ctx.grid_num} adj1={ctx.adj1}")

    # Create or load study
    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
        sampler=sampler,
    )

    existing_trials = len(study.trials)
    if existing_trials > 0:
        print(f"resuming from {existing_trials} existing trials")

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
    print(f"\ntotal_trials={len(study.trials)} completed={len(completed_trials)} pruned={len(pruned_trials)}")

    if len(completed_trials) == 0:
        print("no completed trials (all pruned - try adjusting search space)")
        return

    # Best trial
    objective_metric = opt_config.get("objective_metric", "equity")
    best = study.best_trial
    print(f"\nobjective_metric={objective_metric}")
    print(f"best_value={best.value:.6f} best_trial={best.number}")
    print(f"best_params: gamma={best.params['gamma']:.6f} "
          f"delta={best.params['delta']:.4f} adj2={best.params['adj2']:.6f}")
    if "equity" in best.user_attrs:
        equity_val = best.user_attrs['equity']
        print(f"best_equity={equity_val:.6f}" if equity_val else "best_equity=N/A")
    if "sharpe" in best.user_attrs:
        sharpe_val = best.user_attrs['sharpe']
        print(f"best_sharpe={sharpe_val:.6f}" if sharpe_val else "best_sharpe=N/A")
    if "num_trades" in best.user_attrs:
        print(f"best_num_trades={best.user_attrs['num_trades']}")
    if "trading_volume" in best.user_attrs:
        vol_val = best.user_attrs['trading_volume']
        print(f"best_trading_volume={vol_val:.2f}" if vol_val else "best_trading_volume=N/A")

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
    print("trial,gamma,delta,adj2,equity,sharpe,trading_volume,num_trades")
    for t in completed_trials[:20]:
        num_trades = t.user_attrs.get("num_trades", "nan")
        equity = t.user_attrs.get("equity")
        sharpe = t.user_attrs.get("sharpe")
        trading_volume = t.user_attrs.get("trading_volume")
        equity_str = f"{equity:.6f}" if equity is not None else "nan"
        sharpe_str = f"{sharpe:.6f}" if sharpe is not None else "nan"
        volume_str = f"{trading_volume:.2f}" if trading_volume is not None else "nan"
        print(
            f"{t.number},"
            f"{t.params['gamma']:.6f},"
            f"{t.params['delta']:.4f},"
            f"{t.params['adj2']:.6f},"
            f"{equity_str},"
            f"{sharpe_str},"
            f"{volume_str},"
            f"{num_trades}"
        )

    print(f"\nresults saved to: {storage}")


if __name__ == "__main__":
    main()
