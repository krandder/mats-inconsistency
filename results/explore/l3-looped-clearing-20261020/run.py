"""EXPLORATORY lane L3 runner: one-layer attention vs stacked / looped attention vs Jacobi / exact clearing.

    cd /home/agent/workspaces/inconsistency-search
    PYTHONPATH=.:/usr/lib/python3/dist-packages OMP_NUM_THREADS=1 \
      /home/agent/.venvs/trader-cat/bin/python3 results/explore/l3-looped-clearing-20261020/run.py

Dev split only (train split for fitting). One result JSON per (environment config, visibility), written with
aggbench.results.write_result (validated). summary.json collects the rows the README tables are built from.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

from aggbench import make_env  # noqa: E402
from aggbench.baselines import all_baselines  # noqa: E402
from aggbench.controls import run_with_controls  # noqa: E402
from aggbench.envs import union_identified  # noqa: E402
from aggbench.metrics import paired_difference, summarize  # noqa: E402
from aggbench.results import make_result, split_record, write_result  # noqa: E402
from aggbench.verify import oracle_check  # noqa: E402

import arch  # noqa: E402

RUN = "l3-looped-clearing-20261020"
ARMS = ([("indep", 1), ("readout_softmax", 1)] + [("stacked", t) for t in (1, 2, 4, 8)]
        + [("looped", t) for t in (2, 4, 8)] + [("jacobi", t) for t in (1, 2, 4, 8)] + [("exact", 1)])
BUDGET_REF = ("stacked", 12, 8)  # parameter budget = stacked, width 12, T=8 (for the env's key size)


def jobs():
    out = []
    base = [("E1", {}), ("E2", {}), ("E3-sybil", {}), ("E3-collude", {})]
    for env, ov in base:
        for vis in ("full", "noisy"):
            out.append((env, ov, vis))
    for env in ("E2", "E3-sybil"):  # sweep of agents and overlap, full keys only (CPU budget)
        for ov in ({"n_agents": 3}, {"n_agents": 12}, {"pieces_per_agent": (1, 3)}, {"pieces_per_agent": (6, 10)}):
            out.append((env, ov, "full"))
    return out


def tag(env, ov, vis):
    parts = [env] + [f"{k}={'-'.join(map(str, v)) if isinstance(v, tuple) else v}" for k, v in ov.items()] + [vis]
    return "__".join(parts)


def run_job(job, steps, n_train, n_dev, n_boot, outdir):
    torch.set_num_threads(1)
    env_name, ov, vis = job
    t0 = time.time()
    env = make_env(env_name, **ov)
    tr, dv = env.sample("train", n_train), env.sample("dev", n_dev)
    kind = tr.kind
    key_dim = tr.keys[vis].shape[-1]
    budget = arch.n_params(arch.make_core(BUDGET_REF[0], key_dim, BUDGET_REF[1], BUDGET_REF[2]))
    forecasts, alpha = all_baselines(tr, dv, vis)
    learned, controls, meta = [], {}, {}
    for a, T in ARMS:
        name = f"{a}_T{T}" if a in ("stacked", "looped", "jacobi") else a
        d, npar = arch.matched_width(a, key_dim, T, budget)
        holder = {}

        def make(seed, a=a, T=T, d=d, holder=holder):
            agg = arch.Aggregator(seed, kind, a, key_dim, d, T, steps=steps)
            holder.setdefault("first", agg)
            return agg
        ta = time.time()
        arms = run_with_controls(make, tr.view(vis), tr.truth, dv.view(vis), seed=0)
        for k, f in arms.items():
            forecasts[f"{name}:{k}"] = f
        learned.append(f"{name}:trained")
        controls[f"{name}:trained"] = {"random_network": f"{name}:control:random_network",
                                       "untrained_init": f"{name}:control:untrained_init"}
        meta[name] = {"arch": a, "T": T, "width": d, "params": npar, "budget": budget,
                      "train_loss_trace": holder["first"].trace, "fit_seconds": round(time.time() - ta, 1)}
    methods = {k: summarize(f, dv, n_boot=n_boot) for k, f in forecasts.items()}
    paired = {}
    for k, f in forecasts.items():
        if k in ("bayes", "oracle_clearing", "extremized_log_pool"):
            continue
        paired[k] = {ref: paired_difference(f, forecasts[ref], dv, n_boot=n_boot)
                     for ref in ("oracle_clearing", "extremized_log_pool", "readout_softmax:trained",
                                 "stacked_T1:trained") if ref != k}
    extra = {
        "lane": "L3 looped clearing", "job_tag": tag(*job), "overrides": {k: list(v) if isinstance(v, tuple) else v
                                                                            for k, v in ov.items()},
        "training_target": "truth (log loss on train-split labels; no distillation)",
        "train_steps": steps, "batch": 128, "lr": 3e-3, "train_seed": 0, "arms": meta,
        "paired_regret_difference": paired, "extremize_alpha": alpha,
        "union_identified_dev_frac": (float(np.mean([union_identified(c) for c in dv.oracle["cov"]]))
                                      if kind == "bernoulli" else 1.0),
        "logit_guard": "predict clips logits to [-50, 50] and maps NaN to 0 for every learned arm and control",
        "seconds": None,
        "runner_sha256": {p: hashlib.sha256((HERE / p).read_bytes()).hexdigest() for p in ("run.py", "arch.py")},
        "runner_git_commit": subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True,
                                            text=True).stdout.strip(),
    }
    extra["seconds"] = round(time.time() - t0, 1)
    result = make_result(
        run_id=f"{RUN}/{tag(*job)}", track="EXPLORATORY", env=env, visibility=vis,
        splits={"train": split_record(tr, "train"), "eval": split_record(dv, "dev")},
        oracle_check=oracle_check(env), methods=methods, learned_arms=learned, controls=controls, extra=extra)
    path = write_result(outdir / "json", result, name=f"{tag(*job)}.json")
    print(f"done {tag(*job)} in {extra['seconds']}s -> {path.name}", flush=True)
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-dev", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="", help="substring filter on job tags")
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    outdir = Path(args.out)
    todo = [j for j in jobs() if args.only in tag(*j)]
    todo = [j for j in todo if not (outdir / "json" / f"{tag(*j)}.json").exists()]
    # expensive jobs first
    todo.sort(key=lambda j: -(j[0] == "E1") * 3 - j[1].get("n_agents", 5) / 4)
    print(f"{len(todo)} jobs", flush=True)
    work = [(j, args.steps, args.n_train, args.n_dev, args.n_boot, outdir) for j in todo]
    if args.workers == 1:
        for w in work:
            run_job(*w)
    else:
        with Pool(args.workers) as pool:
            pool.starmap(run_job, work, chunksize=1)


if __name__ == "__main__":
    main()
