# Setup

You need **Python 3.12+** (3.14 is tested) and a **C compiler**. Nothing else is required beyond `numpy`, `scipy` and `pytest`.

## Windows
1. Install Python from python.org.
2. Install GCC (WinLibs MinGW-w64, user scope, no admin needed):
   ```powershell
   winget install BrechtSanders.WinLibs.POSIX.UCRT
   ```
   `scripts/build_c.py` finds it on `PATH`, or in the winget package folder if your shell hasn't picked up the new `PATH` yet. It also copies the OpenMP runtime DLLs next to the built library.
3. From the project folder:
   ```powershell
   python -m venv .venv
   .venv\Scripts\pip install numpy scipy pytest
   .venv\Scripts\python scripts\build_c.py
   .venv\Scripts\python -m pytest -q
   ```

## macOS (Intel or Apple Silicon)
```bash
xcode-select --install                 # Apple clang
brew install python@3.12 libomp        # libomp optional: without it Python threads parallelize instead
python3.12 -m venv .venv
.venv/bin/pip install numpy scipy pytest
.venv/bin/python scripts/build_c.py    # prints "openmp: yes" or "no - Python threads will parallelize"
.venv/bin/python -m pytest -q
```

## Linux
```bash
sudo apt install build-essential python3-venv    # or your distro's equivalent
python3 -m venv .venv
.venv/bin/pip install numpy scipy pytest
.venv/bin/python scripts/build_c.py
.venv/bin/python -m pytest -q
```

## Checking the install
```bash
.venv/bin/python scripts/bench.py -n 4000     # possessions per second
.venv/bin/python scripts/demo.py --replays-only
.venv/bin/python viz/server.py                # open http://127.0.0.1:8765/
```

## Build options
- `python scripts/build_c.py --no-openmp` builds a single-threaded C core; the Python thread pool then parallelizes.
- `--debug` builds with `-O0 -g`.
- Set the `CC` environment variable to pick a specific compiler.
