# StandX Market Making Backtest

Utilities for converting order book + trade parquet files into hftbacktest NPZ format, then running market-making backtests with OBI (Order Book Imbalance) and GLFT strategies.

## Project Layout

- `backtest_standx.py`: converts parquet data to NPZ and runs a minimal market-making backtest.
- `backtest_standx_OBI.py`: runs an OBI-based market-making strategy with volatility-based spreads.
- `backtest_standx_OBI_grid.py`: grid search over OBI strategy parameters.
- `backtest_standx_OBI_optuna.py`: **Bayesian optimization** (Optuna TPE) for OBI parameters.
- `backtest_standx_GLFT.py`: runs a GLFT-based grid strategy with calibration output.
- `backtest_standx_GLFT_grid.py`: grid search over GLFT strategy parameters.
- `backtest_standx_GLFT_optuna.py`: **Bayesian optimization** (Optuna TPE) for GLFT parameters.
- `convert_standx.py`: converts parquet snapshots/trades into hftbacktest event array (`.npz`).
- `make_archive.py`: creates a zip archive with sensible default excludes.
- `data/`: parquet inputs and generated NPZ files.
- `plots/`: output plots and calibration CSVs.
- `DOC/`: background notes, strategy writeups, and architecture docs.
- `hftbacktest/`: vendored hftbacktest source (Rust + Python bindings).

## Requirements

- Python 3.10+ (3.12 recommended)
- Rust toolchain (for compiling hftbacktest)
- Core deps: `numpy`, `pandas`, `pyarrow`
- Optional: `numba` (speed), `matplotlib` (plots), `optuna` (Bayesian optimization)

## Installation

### Linux (Ubuntu/Debian)

1. **Install system dependencies:**

```bash
# Python and build tools
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev build-essential

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

2. **Create virtual environment:**

```bash
python3.12 -m venv venv
source venv/bin/activate
```

3. **Install Python dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Build hftbacktest Rust extension:**

```bash
cd hftbacktest/py-hftbacktest
maturin develop --release
cd ../..
```

5. **Verify installation:**

```bash
python -c "from hftbacktest import BacktestAsset; print('hftbacktest OK')"
```

### Linux (Fedora/RHEL)

1. **Install system dependencies:**

```bash
sudo dnf install python3.12 python3.12-devel gcc gcc-c++ make

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

2. Follow steps 2-5 from Ubuntu instructions above.

### Windows

1. **Install Python:**
   - Download Python 3.12 from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"

2. **Install Rust:**
   - Download and run [rustup-init.exe](https://rustup.rs/)
   - Follow the prompts (default installation)
   - Restart your terminal after installation

3. **Install Visual Studio Build Tools** (required for Rust on Windows):
   - Download [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Install "Desktop development with C++" workload

4. **Create virtual environment** (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Or Command Prompt:

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

5. **Install Python dependencies:**

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

6. **Build hftbacktest Rust extension:**

```powershell
cd hftbacktest\py-hftbacktest
maturin develop --release
cd ..\..
```

7. **Verify installation:**

```powershell
python -c "from hftbacktest import BacktestAsset; print('hftbacktest OK')"
```

### macOS

1. **Install Homebrew** (if not installed):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. **Install dependencies:**

```bash
brew install python@3.12 rust
```

3. **Create virtual environment:**

```bash
python3.12 -m venv venv
source venv/bin/activate
```

4. Follow steps 3-5 from Linux instructions above.

### Troubleshooting

**Rust not found after installation:**
- Linux/macOS: Run `source ~/.cargo/env` or restart terminal
- Windows: Restart terminal/IDE

**maturin build fails:**
- Ensure Rust is installed: `rustc --version`
- On Windows, ensure Visual Studio Build Tools are installed
- Try: `pip install --upgrade maturin`

**numba import errors:**
- Ensure compatible numpy version: `pip install "numpy>=1.24,<2.1"`
- On Apple Silicon: `pip install numba --no-cache-dir`

**Permission denied on Linux:**
- Don't use `sudo` with pip in a venv
- Check venv is activated: `which python` should show venv path

## Quickstart

### Convert parquet to NPZ

```bash
python convert_standx.py --data-dir data --out data/btc_hft.npz
```

### Run OBI backtest

```bash
python backtest_standx_OBI.py --data-dir data --out data/btc_hft_obi.npz
```

### Run GLFT backtest

```bash
python backtest_standx_GLFT.py --data-dir data --out data/btc_hft_glft.npz
```

## Parameter Optimization

### Grid Search

Brute-force search over parameter combinations:

```bash
# OBI grid search
python backtest_standx_OBI_grid.py --vol-to-half-spread-min 2 --vol-to-half-spread-max 10

# GLFT grid search
python backtest_standx_GLFT_grid.py --delta-min 1 --delta-max 10
```

### Bayesian Optimization (Recommended)

Uses Optuna TPE sampler for intelligent parameter search:

```bash
# OBI optimization (configure in optuna_obi_config.json)
python backtest_standx_OBI_optuna.py --n-trials 100

# GLFT optimization (configure in optuna_glft_config.json)
python backtest_standx_GLFT_optuna.py --n-trials 100

# Start fresh (delete previous results)
python backtest_standx_OBI_optuna.py --fresh --n-trials 100

# Resume interrupted optimization
python backtest_standx_OBI_optuna.py --n-trials 50
```

#### Optuna Configuration (`optuna_obi_config.json`)

```json
{
  "search_space": {
    "vol_to_half_spread": {"min": 2.0, "max": 100.0, "log": false},
    "skew": {"min": 0.1, "max": 100.0, "log": true},
    "c1_ticks": {"min": 10, "max": 1000, "log": false}
  },
  "optimization": {
    "n_trials": 200,
    "study_name": "obi_mm_opt",
    "storage": "sqlite:///optuna_obi.db",
    "min_trades": 100,
    "objective_metric": "equity"
  },
  "backtest": {
    "data_dir": "data",
    "out": "data/btc_hft_obi.npz",
    "max_rows": null,
    "grid_num": 1
  }
}
```

- `objective_metric`: `"equity"` or `"sharpe"` - what to maximize
- `min_trades`: penalize trials with fewer trades
- Results saved to SQLite for resumability

## Configuration

Edit `config.json` for global settings:

```json
{
  "threads": 2,
  "symbol": "BTC-USD",
  "max_rows": 10000000
}
```

## Outputs

- `data/*.npz`: compressed event arrays for hftbacktest
- `data/*.npz.meta.json`: cached metadata (tick size, lot size, row counts)
- `plots/*_balance_equity.png`: equity charts
- `plots/*_calibration.csv`: calibration snapshots
- `optuna_obi.db`: OBI Optuna optimization results (SQLite)
- `optuna_glft.db`: GLFT Optuna optimization results (SQLite)

## Notes

- The conversion step aligns price and trade windows and infers tick/lot sizes from the data.
- Supports two data formats: array-based orderbook (`orderbook_*.parquet`) and flat columns (`prices_*.parquet`).
- See `DOC/best_strategy.md` for model background and `DOC/live_trading_architecture.md` for live deployment.
