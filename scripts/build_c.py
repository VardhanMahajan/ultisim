"""Build csim/*.c into build/ultisim_core.(dll|so). Works on Windows (MinGW GCC), Linux (gcc/clang) and
macOS (Apple clang, with or without Homebrew libomp).

    python scripts/build_c.py               # optimized build (probes CPU + OpenMP flags)
    python scripts/build_c.py --debug       # -O0 -g
    python scripts/build_c.py --no-openmp   # force a single-threaded C core (Python then runs batches on threads)

Without OpenMP the engine is still fully parallel: ultisim/engine.py splits each batch over threads
(ctypes releases the GIL while C runs).
"""
from __future__ import annotations

import argparse
import glob
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSRC = ROOT / "csim"
BUILD = ROOT / "build"
IS_WIN = os.name == "nt"
IS_MAC = sys.platform == "darwin"
LIBNAME = "ultisim_core.dll" if IS_WIN else "ultisim_core.so"
# MinGW runtime DLLs the library may need at load time (copied next to it).
RUNTIME_DLLS = ["libgomp-1.dll", "libwinpthread-1.dll", "libgcc_s_seh-1.dll", "libdl.dll"]


def find_compiler() -> Path:
    env = os.environ.get("CC")
    if env and shutil.which(env):
        return Path(shutil.which(env))
    for n in (["clang", "cc", "gcc"] if IS_MAC else ["gcc", "cc", "clang"]):
        p = shutil.which(n)
        if p:
            return Path(p)
    if IS_WIN:
        hits = glob.glob(os.path.expandvars(
            r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs*\mingw64\bin\gcc.exe"))
        if hits:
            return Path(hits[0])
    sys.exit("No C compiler found. Windows: see docs/SETUP.md. macOS: xcode-select --install. Linux: install gcc.")


def try_compile(cc: Path, flags: list[str], code: str = "int main(void){return 0;}") -> bool:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "t.c"
        src.write_text(code)
        out = Path(d) / ("t.exe" if IS_WIN else "t")
        return subprocess.run([str(cc), *flags, str(src), "-o", str(out)], capture_output=True).returncode == 0


def cpu_flags(cc: Path) -> list[str]:
    for f in (["-march=native"], ["-mcpu=native"]):   # x86 / Apple Silicon & other ARM
        if try_compile(cc, f):
            return f
    return []


def openmp_flags(cc: Path) -> tuple[list[str], list[str]]:
    """(compile flags, link flags) for OpenMP, or ([], []) if unavailable."""
    code = "#include <omp.h>\nint main(void){return omp_get_max_threads() > 0 ? 0 : 1;}"
    if try_compile(cc, ["-fopenmp"], code):
        return ["-fopenmp"], ["-fopenmp"]
    if IS_MAC:  # Apple clang + Homebrew libomp (brew install libomp)
        prefixes = []
        if shutil.which("brew"):
            r = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True, text=True)
            if r.returncode == 0:
                prefixes.append(r.stdout.strip())
        prefixes += ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]
        for p in prefixes:
            cf = ["-Xpreprocessor", "-fopenmp", f"-I{p}/include"]
            lf = [f"-L{p}/lib", "-lomp", f"-Wl,-rpath,{p}/lib"]
            if Path(p).exists() and try_compile(cc, cf + lf, code):
                return cf, lf
    return [], []


def write_generated_header():
    """csim/us_structs.h is generated from ultisim/cabi.py (single source of truth for layouts)."""
    sys.path.insert(0, str(ROOT))
    from ultisim.cabi import generate_header
    hdr = CSRC / "us_structs.h"
    text = generate_header()
    if not hdr.exists() or hdr.read_text() != text:
        hdr.write_text(text)
        print(f"generated {hdr.relative_to(ROOT)}")


def build(debug: bool = False, use_openmp: bool = True) -> Path:
    write_generated_header()
    cc = find_compiler()
    BUILD.mkdir(exist_ok=True)
    out = BUILD / LIBNAME
    opt = ["-O0", "-g"] if debug else ["-O3", *cpu_flags(cc), "-DNDEBUG", "-fno-math-errno", "-ffast-math",
                                       "-fno-finite-math-only"]
    omp_c, omp_l = openmp_flags(cc) if use_openmp else ([], [])
    cmd = [str(cc), *opt, "-std=c11", "-Wall", "-Wextra", *omp_c, "-shared",
           "-o", str(out), *map(str, sorted(CSRC.glob("*.c"))), *omp_l, "-lm"]
    if not IS_WIN:
        cmd.insert(1, "-fPIC")
    print(f"compiler: {cc} ({platform.machine()}) | openmp: {'yes' if omp_c else 'no - Python threads will parallelize'}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    if IS_WIN and omp_c:
        for name in RUNTIME_DLLS:
            src = cc.parent / name
            if src.exists():
                shutil.copy2(src, BUILD / name)
    print(f"built {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-openmp", action="store_true")
    a = ap.parse_args()
    build(a.debug, use_openmp=not a.no_openmp)
