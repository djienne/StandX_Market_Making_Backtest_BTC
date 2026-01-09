# HftBacktest build + install (Windows, local repo)

This records the exact steps used to build and install `hftbacktest` from the local repo at
`C:\Users\david\Desktop\AS_params\hftbacktest`.

## Prereqs

- Rust toolchain via `rustup`
- Python 3.12 (project requires Python 3.11+; we used 3.12)
- PowerShell

## 1) Update Rust toolchain

```
rustup update stable
```

## 2) Install build tool (maturin) for Python 3.12

```
py -3.12 -m pip install --upgrade pip maturin
```

## 3) Build the wheel (from repo root)

```
cd C:\Users\david\Desktop\AS_params\hftbacktest\py-hftbacktest
$env:PYO3_PYTHON="C:\Users\david\AppData\Local\Programs\Python\Python312\python.exe"
py -3.12 -m maturin build --release
```

This produces a wheel under:

```
C:\Users\david\Desktop\AS_params\hftbacktest\target\wheels
```

## 4) Install the wheel

```
py -3.12 -m pip install C:\Users\david\Desktop\AS_params\hftbacktest\target\wheels\hftbacktest-2.4.4-cp312-cp312-win_amd64.whl
```

## 5) Verify install (run outside the repo folder)

```
cd C:\Users\david\Desktop\AS_params
py -3.12 -c "import hftbacktest; print(hftbacktest.__version__)"
```

## Optional: use a virtual environment (recommended)

```
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip maturin
# then follow steps 3-5 inside the venv
```

## Notes

- Building inside `py-hftbacktest` will fail with `maturin develop` unless a venv is active.
- The build may take a few minutes the first time (Rust crate compile).
- Pip may report dependency conflicts in a global Python install; use a venv to isolate.
