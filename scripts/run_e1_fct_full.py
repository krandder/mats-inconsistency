"""E1 full_context_transformer, fresh regime, unaligned, at a given provenance visibility (default full).
Saves the result record and per-episode forecasts (mean, prec) for paired comparisons.

    python3 scripts/run_e1_fct_full.py --visibility full --threads 2 --out results/explore/e1-floor-20261021
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggbench import make_env  # noqa: E402
from aggbench.transformer import evaluate_full_context_transformer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--visibility", default="full")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--out", default="results/explore/e1-floor-20261021")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
env = make_env("E1")
rec = evaluate_full_context_transformer(env, visibility=a.visibility, alignment="unaligned", regime="fresh",
                                        threads=a.threads, steps=a.steps, device="cpu")
f = rec.pop("forecasts")["trained"]
np.savez_compressed(out / f"fct_E1_unaligned_{a.visibility}_fresh_forecast.npz", mean=f["mean"], prec=f["prec"])
rec["wall_seconds"] = time.time() - t0
(out / f"fct_E1_unaligned_{a.visibility}_fresh.json").write_text(json.dumps(rec, indent=2, default=float) + "\n")
print(a.visibility, rec["methods"]["full_context_transformer"]["regret"], f"{rec['wall_seconds']:.0f}s", flush=True)
