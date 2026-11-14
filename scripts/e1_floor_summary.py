"""Summarise the E1 floor study: floor_none, transformer cells (regret to Bayes, paired excess over floor_none, train/dev
gap). Reads results/explore/e1-floor-20261021/ (floor_*_regret.npy, cell-*.json/.npy, fct_*_forecast.npz).

    python3 scripts/e1_floor_summary.py [DIR]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggbench import make_env  # noqa: E402
from aggbench.metrics import per_episode, summarize_values  # noqa: E402

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/explore/e1-floor-20261021")
floor = np.load(d / "floor_none_regret.npy")
dv = make_env("E1").sample("dev", len(floor))


def fmt(s):
    return f"{s['mean']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]"


rows = {}
for f in sorted(d.glob("fct_*_forecast.npz")):
    z = np.load(f)
    rows[f.stem.replace("_forecast", "") + " (64x3, 3k, CPU)"] = (per_episode({"mean": z["mean"], "prec": z["prec"]},
                                                                             dv)["regret"], None)
for f in sorted(d.glob("cell-*.json"), key=lambda p: (json.loads(p.read_text())["visibility"],
                                                      json.loads(p.read_text())["config"]["width"],
                                                      json.loads(p.read_text())["config"]["steps"])):
    rec = json.loads(f.read_text())
    rows[f.stem] = (np.load(f.with_name(f.stem + "-dev-regret.npy")), rec)
print(f"floor_none {fmt(summarize_values(floor))}")
for name, (reg, rec) in rows.items():
    gap = f" train/dev gap {rec['train_dev_gap']:+.3f} (seen {rec['seen_train_regret']['mean']:.3f})" if rec else ""
    print(f"{name}: regret {fmt(summarize_values(reg))}; minus floor_none {fmt(summarize_values(reg - floor))}{gap}")
