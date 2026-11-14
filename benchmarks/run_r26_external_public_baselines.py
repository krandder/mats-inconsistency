#!/usr/bin/env python3
"""
PARKED 2026-10-19 (Kelvin): not frozen, not run
Round 26 (`R26`) Master Wrapper: Dispatches to Split Runners `R26-A` (`OpenSpiel`) and `R26-B` (`Ataraxos` Stratego).

Following `outbox/r26-freeze-review-2026-10-19.md` and `docs/PROTOCOL.md`:
- `R26-A` (`OpenSpiel` Public Baselines, `5` Seeds `101..105`):
  - Pre-Registration: `benchmarks/PREREG-R26A-OPENSPIEL-PUBLIC-BASELINES.md`
  - Standalone Runner: `benchmarks/run_r26a_openspiel_public_baselines.py`
- `R26-B` (`Ataraxos` `arXiv:2511.07312` `10x10` Stratego, `3` Seeds `101..103`):
  - Pre-Registration: `benchmarks/PREREG-R26B-ATARAXOS-STRATEGO.md`
  - Standalone Runner: `benchmarks/run_r26b_ataraxos_stratego.py`
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Round 26 Split External Public Baselines (`R26-A` OpenSpiel and/or `R26-B` Ataraxos Stratego)"
    )
    parser.add_argument("--part", choices=("all", "r26a", "r26b"), default="all")
    parser.add_argument("--out-dir", type=str, default=str(_REPO_ROOT / "results" / "r26"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.part in ("all", "r26a"):
        subprocess.run(
            [
                sys.executable,
                str(_REPO_ROOT / "benchmarks" / "run_r26a_openspiel_public_baselines.py"),
                "--out-dir",
                str(out_dir / "r26a"),
            ],
            check=True,
        )

    if args.part in ("all", "r26b"):
        subprocess.run(
            [
                sys.executable,
                str(_REPO_ROOT / "benchmarks" / "run_r26b_ataraxos_stratego.py"),
                "--out-dir",
                str(out_dir / "r26b"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
