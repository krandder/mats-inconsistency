"""EXPLORATORY lane L2 diagnostic (DECLARED DISTILLATION, README rule 2): the same pipeline as run_l2.py,
trained on soft targets sigmoid(bayes logit) of the train split instead of truth labels.

Purpose: separate "the messages and architecture cannot carry/compute Bayes" from "learning from noisy
0/1 labels in this budget is the bottleneck". Not an arm of the main question. E2/E3 only.
"""
import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from aggbench import make_env
from aggbench.baselines import all_baselines
from aggbench.controls import run_with_controls
from aggbench.messages import check_locality
from aggbench.metrics import paired_difference, sigmoid, summarize
from aggbench.results import make_result, split_record, write_result
from aggbench.verify import oracle_check

import run_l2
from run_l2 import VIS, VectorMessageAggregator, agent_inputs, cached_sample, local_features

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="E2", choices=["E2", "E3-sybil", "E3-biased", "E3-collude"])
    ap.add_argument("--dims", type=int, nargs="+", default=[16])
    ap.add_argument("--n-train", type=int, default=200000)
    ap.add_argument("--n-dev", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--cache", default="/tmp/l2-vecmsg-cache")
    ap.add_argument("--tag", default="", help="suffix for run_id and output file (parallel runs)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    log = lambda s: print(s, flush=True)
    env = make_env(args.env)
    tr = cached_sample(env, "train", args.n_train, args.cache)
    dv = cached_sample(env, "dev", args.n_dev, args.cache)
    tr_in, dv_in = agent_inputs(tr, args.env), agent_inputs(dv, args.env)
    in_dim = local_features(dv_in.obs[:1], dv_in.keys[:1], dv.kind).shape[-1]
    soft = sigmoid(tr.bayes["logit"])            # distillation targets (declared)
    forecasts, alpha = all_baselines(tr, dv, VIS)
    learned, controls, locality, paired, fit_info = [], {}, {}, {}, {}
    for m in args.dims:
        name = f"vecmsg_distill_m{m}"
        built = {}

        def make(seed, m=m):
            agg = VectorMessageAggregator(seed, dv.kind, in_dim, m, steps=args.steps, log=log)
            built.setdefault("first", agg)
            return agg

        arms = run_with_controls(make, tr_in, soft, dv_in, seed=0)
        for arm, f in arms.items():
            forecasts[f"{name}:{arm}"] = f
        learned.append(f"{name}:trained")
        controls[f"{name}:trained"] = {"random_network": f"{name}:control:random_network",
                                       "untrained_init": f"{name}:control:untrained_init"}
        locality[name] = {"passed": bool(check_locality(built["first"].encode_numpy, dv_in.obs[:500],
                                                        dv_in.keys[:500], trials=5)), "episodes": 500, "trials": 5}
        paired[name] = {f"trained_minus_{b}": paired_difference(arms["trained"], forecasts[b], dv)
                        for b in ("oracle_clearing", "extremized_log_pool")}
        fit_info[name] = built["first"].fit_info
        log(f"[{args.env} distill] m={m} regret {summarize(arms['trained'], dv, n_boot=200)['regret']['mean']:.4f} "
            f"minus oracle_clearing {paired[name]['trained_minus_oracle_clearing']['mean']:+.4f}")
    extra = {"lane": "L2", "diagnostic": "declared distillation to bayes (soft targets sigmoid(bayes logit) on train)",
             "runner": "results/explore/l2-vector-messages-20261020/run_l2_distill.py",
             "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "imports_sha256": {"run_l2.py": hashlib.sha256(Path(run_l2.__file__).read_bytes()).hexdigest()},
             "message_dims": args.dims, "steps": args.steps, "locality_check": locality,
             "paired_differences": paired, "fit_info": fit_info, "extremize_alpha": alpha}
    result = make_result(run_id=f"l2-vector-messages-20261020-{args.env}-distill{args.tag}", track="EXPLORATORY", env=env,
                         visibility=VIS, splits={"train": split_record(tr, "train"), "eval": split_record(dv, "dev")},
                         oracle_check=oracle_check(env), methods={k: summarize(f, dv) for k, f in forecasts.items()},
                         learned_arms=learned, controls=controls, extra=extra)
    log(f"wrote {write_result(HERE, result, name=f'result_{args.env}-distill{args.tag}.json')}")


if __name__ == "__main__":
    main()
