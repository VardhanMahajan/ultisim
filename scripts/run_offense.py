"""Phase 4 / Run A: coevolve offenses and defenses, find the best offensive structure.

    python scripts/run_offense.py --run smoke --gens 3 --off-pop 16 --def-pop 12      # quick smoke test
    python scripts/run_offense.py --run overnight --gens 300                           # full run
    python scripts/run_offense.py --run overnight --resume                             # continue after a stop

Watch it: python viz/server.py, then open http://127.0.0.1:8765/dashboard.html
Stop it any time with Ctrl+C: the checkpoint (runs/<run>/state.pkl) is saved every generation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ultisim.evolve import EvoConfig, Evolver  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--off-pop", type=int, default=48)
    ap.add_argument("--def-pop", type=int, default=32)
    ap.add_argument("--stage1", type=int, default=96, help="possessions every offense plays")
    ap.add_argument("--stage2", type=int, default=288, help="total possessions for the top offenses (racing)")
    ap.add_argument("--race-frac", type=float, default=0.34)
    ap.add_argument("--rosters", type=int, default=64)
    ap.add_argument("--replay-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    ec = EvoConfig(run=a.run, generations=a.gens, off_pop=a.off_pop, def_pop=a.def_pop, stage1=a.stage1,
                   stage2=a.stage2, race_frac=a.race_frac, n_rosters=a.rosters, replay_every=a.replay_every,
                   seed=a.seed, threads=a.threads)
    Evolver(ec, resume=a.resume).run()


if __name__ == "__main__":
    main()
