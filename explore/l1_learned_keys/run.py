"""EXPLORATORY lane L1 runner: learned-key aggregators vs aggbench baselines, controls, all envs x visibilities.

    OMP_NUM_THREADS=4 PYTHONPATH=. /home/agent/.venvs/trader-cat/bin/python3 -m explore.l1_learned_keys.run \
        --out results/explore/l1-learned-keys-2026-10-20

Train split episodes 0..n_train-1 fit, dev split 0..n_dev-1 scored. No test split.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from aggbench import make_env
from aggbench.baselines import all_baselines
from aggbench.controls import run_with_controls
from aggbench.metrics import paired_difference, summarize
from aggbench.results import make_result, split_record, write_result
from aggbench.verify import oracle_check

from .arms import Aggregator

ARMS = {"bernoulli": ["lk_clear", "lk_clear_cal", "attn_pool"], "gaussian": ["lk_clear", "attn_pool"]}
HERE = Path(__file__).resolve().parent


def code_sha():
    h = hashlib.sha256()
    for p in sorted(HERE.glob("*.py")):
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=["E1", "E2", "E3-sybil", "E3-biased", "E3-collude"])
    ap.add_argument("--vis", nargs="+", default=["none", "noisy", "full"])
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-dev", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--track", default="EXPLORATORY")
    args = ap.parse_args()
    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sha = code_sha()
    for name in args.envs:
        env = make_env(name)
        t0 = time.time()
        tr, dv = env.sample("train", args.n_train), env.sample("dev", args.n_dev)
        ocheck = oracle_check(env)
        print(f"[{name}] sampled in {time.time() - t0:.1f}s; oracle check {ocheck['max_abs_diff']:.1e}", flush=True)
        for vis in args.vis:
            forecasts, alpha = all_baselines(tr, dv, vis)
            key_dim = tr.keys[vis].shape[-1]
            learned, controls, train_logs, times = [], {}, {}, {}
            for arm in args.arms or ARMS[tr.kind]:
                if arm not in ARMS[tr.kind]:
                    continue
                t1 = time.time()
                holder = {}

                def make(seed, arm=arm):
                    agg = Aggregator(arm, tr.kind, key_dim, seed, steps=args.steps)
                    holder.setdefault("first", agg)
                    return agg
                arms = run_with_controls(make, tr.view(vis), tr.truth, dv.view(vis), seed=args.seed)
                times[arm] = round(time.time() - t1, 1)
                train_logs[arm] = holder["first"].train_log
                for k, f in arms.items():
                    forecasts[f"{arm}:{k}"] = f
                learned.append(f"{arm}:trained")
                controls[f"{arm}:trained"] = {"random_network": f"{arm}:control:random_network",
                                              "untrained_init": f"{arm}:control:untrained_init"}
            methods = {k: summarize(f, dv, n_boot=args.n_boot) for k, f in forecasts.items()}
            paired = {}
            for k in forecasts:
                if k in ("bayes",):
                    continue
                paired[k] = {ref: paired_difference(forecasts[k], forecasts[ref], dv, n_boot=args.n_boot)
                             for ref in ("oracle_clearing", "extremized_log_pool") if ref != k}
            result = make_result(
                run_id=f"l1-learned-keys-{name}-{vis}", track=args.track, smoke=args.track == "SMOKE",
                env=env, visibility=vis,
                splits={"train": split_record(tr, "train"), "eval": split_record(dv, "dev")},
                oracle_check=ocheck, methods=methods, learned_arms=learned, controls=controls,
                extra={"lane": "L1", "label": "EXPLORATORY", "extremize_alpha": alpha,
                       "paired_regret_difference": paired, "train_loss_log": train_logs, "fit_seconds": times,
                       "code": "explore/l1_learned_keys", "code_sha256": sha, "steps": args.steps,
                       "training_label": "truth (no distillation)", "torch": torch.__version__})
            path = write_result(out, result, name=f"result-{name}-{vis}.json")
            msg = " ".join(f"{k}={methods[k]['regret']['mean']:.3f}" for k in methods)
            print(f"[{name}/{vis}] {msg} times={times} -> {path}", flush=True)


if __name__ == "__main__":
    main()
