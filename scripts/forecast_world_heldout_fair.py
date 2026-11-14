"""Fair comparators for the held-out-source test (gate 2026-10-23, item b).

On the held-out test episodes (test split, chunks 1000.., forecasters may read the 10 sources never seen in
training), a learner that cannot read the new sources can at best ignore them. This script scores, per alpha:
  bayes_ignoring_heldout      exact Bayes given every TRAINING source read by anyone
  extremized_ignoring_heldout each forecaster's exact posterior from its training sources only, mean log-odds times
                              the extremizing factor fit on training episodes (as in the main cells)
  extremized_exact            the old comparator (exact posteriors that know the held-out sources)
KL is from (A), full-information Bayes that reads the held-out sources too.

    python scripts/forecast_world_heldout_fair.py results/explore/forecast-world-20261022/heldout_fair.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggbench import forecast_world as fw  # noqa: E402
from aggbench.metrics import summarize_values  # noqa: E402
from aggbench.results import harness_info  # noqa: E402


def main(out):
    torch.set_num_threads(2)
    res = {"harness": harness_info(), "n": 4096, "split": "test", "first_chunk": 1000, "cells": {}}
    for alpha in (0.0, 0.5, 1.0, 1.5, 2.0):
        env = fw.ForecastWorld(alpha=alpha, N=10, k=3)
        M = env.config.M
        tr = [env.sample("train", 1024, chunk=9_000_000 + i) for i in range(4)]
        a_ext = fw.fit_extremize(torch.cat([fw.probit_to_logit(e["probit"]) for e in tr]),
                                 torch.cat([e["y"] for e in tr]))
        rows = {k: [] for k in ("bayes_ignoring_heldout", "extremized_ignoring_heldout", "extremized_exact",
                                "mean_logodds_ignoring_heldout")}
        for i in range(4):
            ep = env.sample("test", 1024, chunk=1000 + i, heldout=True)
            # per-source information vectors, recomputed from the readings
            h = torch.einsum("jrd,njr->njd", env.B, ep["s"] / env.src_sd[None, :, None] ** 2)
            keep = torch.ones(env.config.M + env.config.M_heldout, dtype=torch.float64)
            keep[M:] = 0
            union_tr = ep["union"] * keep
            pu = env.local_posteriors(union_tr[:, None], h)["probit"][:, 0]
            loc = env.local_posteriors(ep["mask"] * keep, h)["probit"]        # forecasters ignoring new sources
            lo, lo_exact = fw.probit_to_logit(loc), fw.probit_to_logit(ep["probit"])
            for name, logit in (("bayes_ignoring_heldout", fw.probit_to_logit(pu)),
                                ("extremized_ignoring_heldout", a_ext * lo.mean(1)),
                                ("mean_logodds_ignoring_heldout", lo.mean(1)),
                                ("extremized_exact", a_ext * lo_exact.mean(1))):
                rows[name].append(fw.logit_metrics(logit, ep["bayes_probit"], ep["y"])["kl"])
        res["cells"][f"a{alpha:g}"] = {"alpha": alpha, "extremize_factor": a_ext,
                                       **{k: summarize_values(np.concatenate(v), n_boot=1000) for k, v in rows.items()}}
        print(alpha, {k: round(float(np.concatenate(v).mean()), 4) for k, v in rows.items()}, flush=True)
    Path(out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
