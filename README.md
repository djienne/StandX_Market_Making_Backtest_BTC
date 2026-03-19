# StandX Market Making

> **Support this project:** Sign up on StandX using [this referral link](https://standx.com/referral?code=FREQTRADEFR)

Market-making research pipeline for StandX: **collect** live orderbook/trade data, **convert** to backtesting format, and **backtest** the OBI (Order Book Imbalance) strategy with parameter optimization.

## Pipeline Overview

```
data_collector/            convert_standx.py         backtest_standx_OBI*.py
 (WebSocket → Parquet)  →   (Parquet → NPZ)      →   (NPZ → backtest results)
```

## Project Layout

### Data Collection (`data_collector/`)
- `data_collector/data_collector.py` — Main collector: WebSocket → buffer → hourly Parquet files
- `data_collector/standx_common.py` — WebSocket base class, reconnection logic, parquet I/O, config loading
- `data_collector/test_data_collector.py` — Data validation test suite
- `data_collector/read_parquet.py` — Parquet file viewer
- `data_collector/docker-compose.yml` — Docker Compose for data collection (`docker compose up`)
- `data_collector/Dockerfile` — Container build for data collector
- `data_collector/requirements.txt` — Minimal Python dependencies for collection

### Data Conversion
- `convert_standx.py` — Convert parquet snapshots/trades into hftbacktest event array (`.npz`)

### Backtesting (OBI Strategy)
- `backtest_standx_OBI.py` — OBI (Order Book Imbalance) market-making strategy
- `backtest_standx_OBI_grid.py` — Grid search over parameters
- `backtest_standx_OBI_optuna.py` — Bayesian optimization (Optuna TPE)
- `backtest_common.py` — Shared backtest infrastructure (JIT wrappers, API classes, CLI helpers)
- `backtest_utils.py` — Utilities (plotting, gap detection, config loaders, Sharpe ratio)

### Other
- `config.json` — Shared configuration (symbols, fees, output directory)
- `optuna_obi_config.json` — OBI optimization search space and settings
- `hftbacktest/` — Vendored hftbacktest source (Rust + Python bindings)
- `docs/` — API docs (`api_*.md`), strategy papers (PDFs), and technical notes
- `data/` — Parquet inputs and generated NPZ files
- `plots/` — Output plots and calibration CSVs

## Requirements

- Python 3.10+ (3.12 recommended)
- Rust toolchain (for compiling hftbacktest)
- Core deps: `numpy`, `pandas`, `pyarrow`, `websockets`
- Optional: `numba` (speed), `matplotlib` (plots), `optuna` (Bayesian optimization)

## Installation

### Linux (Ubuntu/Debian)

```bash
# Python and build tools
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev build-essential

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Build hftbacktest Rust extension
cd hftbacktest/py-hftbacktest
maturin develop --release
cd ../..

# Verify
python -c "from hftbacktest import BacktestAsset; print('hftbacktest OK')"
```

### Windows

1. Install Python 3.12 from [python.org](https://www.python.org/downloads/) (check "Add Python to PATH")
2. Install Rust from [rustup.rs](https://rustup.rs/)
3. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) ("Desktop development with C++")

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
cd hftbacktest\py-hftbacktest
maturin develop --release
cd ..\..
python -c "from hftbacktest import BacktestAsset; print('hftbacktest OK')"
```

### macOS

```bash
brew install python@3.12 rust
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd hftbacktest/py-hftbacktest
maturin develop --release
cd ../..
```

See `docs/install_hftbacktest.md` for detailed build troubleshooting.

## Quick Start

### 1. Collect Data

**Option A: Docker** (recommended for always-on collection)
```bash
cd data_collector
docker compose up -d
```

**Option B: Native Python**
```bash
# Runs until Ctrl+C, saves hourly Parquet files to data/
cd data_collector
python data_collector.py
```

Both methods write parquet files to the same `data/` directory (project root).

### 2. View and Validate Collected Data

```bash
cd data_collector
python read_parquet.py --orderbook --rows 5
python test_data_collector.py --verbose
```

### 3. Run Backtest (from project root)

```bash
cd ..
# Converts parquet → NPZ automatically, then runs OBI backtest
python backtest_standx_OBI.py --data-dir data --out data/btc_hft_obi.npz
```

## Parameter Optimization

### Grid Search

```bash
python backtest_standx_OBI_grid.py --vol-to-half-spread-min 2 --vol-to-half-spread-max 10
```

### Bayesian Optimization (Recommended)

Uses Optuna TPE sampler. Configure search space in `optuna_obi_config.json`.

```bash
# Run optimization
python backtest_standx_OBI_optuna.py --n-trials 100

# Start fresh
python backtest_standx_OBI_optuna.py --fresh --n-trials 100

# Resume interrupted run
python backtest_standx_OBI_optuna.py --n-trials 50
```

See `docs/better_searching_algo.md` for a guide to Bayesian optimization with Optuna.

## Configuration

`config.json` contains settings for both data collection and backtesting:

```json
{
  "threads": 6,
  "symbol": "BTC-USD",
  "max_rows": 10000000,
  "maker_fee": 0.0001,
  "taker_fee": 0.0004,
  "symbols": ["BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD"],
  "orderbook_levels": 20,
  "flush_interval_seconds": 5,
  "output_dir": "./data"
}
```

| Parameter | Used By | Description |
|-----------|---------|-------------|
| `threads` | Backtest | Thread count for parallel optimization |
| `symbol` | Backtest | Trading pair for backtesting |
| `max_rows` | Backtest | Max data rows to load |
| `maker_fee` / `taker_fee` | Backtest | Fee rates |
| `symbols` | Collector | Trading pairs to collect |
| `orderbook_levels` | Collector | Bid/ask depth levels |
| `flush_interval_seconds` | Collector | Parquet write interval |
| `output_dir` | Collector | Output directory |

## Data Schema

### Orderbook Parquet

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime64[ns, UTC] | Server timestamp |
| received_at | datetime64[ns, UTC] | Local receive time |
| symbol | string | Trading pair |
| bid_prices | list[float] | Bid prices (best first) |
| bid_quantities | list[float] | Bid quantities |
| ask_prices | list[float] | Ask prices (best first) |
| ask_quantities | list[float] | Ask quantities |
| best_bid / best_ask | float | Top of book |
| spread / mid_price | float | Derived fields |

### Trades Parquet

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime64[ns, UTC] | Server timestamp |
| symbol | string | Trading pair |
| price / quantity | float | Trade price and size |
| side | string | "BUY" or "SELL" |
| is_buyer_taker | bool | Aggressor side |

## Docker (Data Collection Only)

```bash
cd data_collector
docker compose up -d                          # Run collector in background
docker compose --profile test run --rm test-data   # Validate collected data
docker compose --profile tools run --rm read-data  # View collected data
docker compose down                           # Stop
```

## License

MIT
