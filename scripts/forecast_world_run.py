"""ForecastWorld experiment runner: baselines, ceilings and the set-attention aggregator arms for a list of cells.

    python scripts/forecast_world_run.py --cells 1:10:3 --out results/explore/forecast-world-20261022/cells
    (cell = alpha:N:k; arms "nokey", "key0", "key0.5", "key1", "key3", "oracle")

Per cell: one test set (4096 test-split episodes, ceilings A and B), one held-out-sources test set (forecasters
may also read the M_heldout sources whose embeddings never appear in training), a 2048-episode dev set for
best-val selection, fresh training episodes every step (train-split chunks seed * 10^6 + step), 3 seeds.
Writes one aggbench.result/v1 JSON per cell (methods on the main test set; held-out test metrics, per-seed
numbers, controls and timings under extra).
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggbench import forecast_world as fw  # noqa: E402
from aggbench.controls import random_weights  # noqa: E402
from aggbench.metrics import summarize_values  # noqa: E402
from aggbench.results import make_result, write_result  # noqa: E402

EXT_CHUNK = 9_000_000          # train-split chunks used only to fit the extremizing factor


def parse_arm(a):
    """nokey, oracle, key<sigma>, or skey<sigma>: standardized key with its own input projection."""
    if a.startswith("skey"):
        return "key", float(a[4:])
    if a.startswith("key"):
        return "key", float(a[3:])
    return a, None


STD = {}   # arm name -> standardized split embedding


def sample_set(env, split, n, heldout=False, ceiling_b=True, start=0, per=1024):
    eps = [env.sample(split, min(per, n - i), chunk=start + i // per, heldout=heldout, ceiling_b=ceiling_b)
           for i in range(0, n, per)]
    keys = [k for k in eps[0] if k not in ("X", "mu", "s", "theta")]
    return {k: torch.cat([e[k] for e in eps]) for k in keys}


def summary(m):
    return {k: summarize_values(v, n_boot=1000) for k, v in m.items()}


class Wrapped:
    def __init__(self, module):
        self.module = module


def predict(model, ep, arm, sigma, batch=1024, std=False):
    model.eval()
    outs = []
    with torch.no_grad():
        n = ep["y"].shape[0]
        for a in range(0, n, batch):
            sub = {k: v[a:a + batch] for k, v in ep.items()}
            tok, lo = fw.token_features(sub, arm, sigma, standardize=getattr(model, "std", False))
            outs.append(model(tok, lo).double())
    return torch.cat(outs)


def bce(model, ep, arm, sigma):
    logit = predict(model, ep, arm, sigma)
    y = ep["y"]
    return float(torch.nn.functional.binary_cross_entropy_with_logits(logit, y))


def train(env, arm_name, seed, dev_set, args, device):
    arm, sigma = parse_arm(arm_name)
    std = arm_name.startswith("skey")
    c = env.config
    tok0, _ = fw.token_features({k: v[:2] for k, v in dev_set.items()}, arm, sigma)
    mk = dict(width=args.width, layers=args.layers, seed=seed, readout=args.readout, split=c.Q if std else None)
    model = fw.make_aggregator(tok0.shape[-1], c.Q, **mk).to(device)
    model.std = std
    init_state = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    best, best_state, best_step, log = math.inf, None, 0, []
    t0 = time.time()
    for step in range(args.steps):
        model.train()
        ep = env.sample("train", args.batch, chunk=seed * 1_000_000 + step, key_sigma=sigma or 0.0)
        tok, lo = fw.token_features(ep, arm, sigma, standardize=std)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(tok, lo), ep["y"].float())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            vl = bce(model, dev_set, arm, sigma)
            if vl < best:
                best, best_step, best_state = vl, step + 1, copy.deepcopy(model.state_dict())
            log.append({"step": step + 1, "val_logloss": vl, "train_loss": loss.item()})
    model.load_state_dict(best_state)
    init = fw.make_aggregator(tok0.shape[-1], c.Q, **mk).to(device)
    init.std = std
    init.load_state_dict(init_state)
    return model, init, {"best_step": best_step, "best_val": best, "train_seconds": time.time() - t0, "log": log}


def oracle_check(env, n=32):
    """Recompute ceiling A per episode with numpy (independent dense solve) and check the probit inversion."""
    ep = env.sample("dev", n, chunk=777)
    A, sq = env.A.cpu().numpy(), env.s_q.cpu().numpy()
    B, sd = env.B.cpu().numpy(), env.src_sd.cpu().numpy()
    s = ep["s"].cpu().numpy()
    worst = 0.0
    for e in range(n):
        idx = np.flatnonzero(ep["union"][e].cpu().numpy())
        Bu = B[idx].reshape(-1, env.config.D)
        r = np.repeat(sd[idx] ** 2, env.config.rows)
        P = np.eye(env.config.D) + Bu.T @ (Bu / r[:, None])
        mu = np.linalg.solve(P, Bu.T @ (s[e, idx].reshape(-1) / r))
        var = np.einsum("qd,qd->q", A, np.linalg.solve(P, A.T).T)
        u = A @ mu / np.sqrt(var + sq ** 2)
        worst = max(worst, float(np.abs(u - ep["bayes_probit"][e].cpu().numpy()).max()))
    z = (ep["probit"] * ep["denom"]).cpu().numpy()
    inv = float(np.abs(z - (ep["mu"] @ env.A.T).cpu().numpy()).max())
    return {"episodes_checked": n, "max_abs_diff": max(worst, inv), "verified": bool(max(worst, inv) < 1e-8),
            "what": "ceiling A probit vs independent numpy solve; probit inversion z = a_q . mu_i"}


def run_cell(alpha, N, k, args, device):
    env = fw.ForecastWorld(alpha=alpha, N=N, k=k, instance_seed=args.instance).to(device)
    t0 = time.time()
    test = sample_set(env, "test", args.n_test)
    held = sample_set(env, "test", args.n_test, heldout=True, start=1000)
    dev_set = sample_set(env, "dev", args.n_dev, ceiling_b=False)
    ext_tr = sample_set(env, "train", 4096, ceiling_b=False, start=EXT_CHUNK)
    t_sets = time.time() - t0
    a_ext = fw.fit_extremize(fw.probit_to_logit(ext_tr["probit"]), ext_tr["y"])

    def base_preds(ep):
        lo = fw.probit_to_logit(ep["probit"])
        return {"bayes": fw.probit_to_logit(ep["bayes_probit"]),
                "bayes_oracle_keys": fw.probit_to_logit(ep["oracle_probit"]),
                "prior": torch.zeros_like(ep["bayes_probit"]),
                "single_forecaster": lo[:, 0],
                "mean_prob": fw.pool_mean_prob(lo), "mean_logodds": fw.pool_mean_logodds(lo),
                "median": fw.pool_median(lo), "sum_logodds": fw.pool_sum_logodds(lo),
                "extremized_mean_logodds": a_ext * fw.pool_mean_logodds(lo)}

    def metrics(logit, ep):
        return fw.logit_metrics(logit, ep["bayes_probit"], ep["y"])

    methods, held_methods, per_ep = {}, {}, {}
    for name, lg in base_preds(test).items():
        m = metrics(lg, test)
        methods[name], per_ep[name] = summary(m), m["kl"]
    for name, lg in base_preds(held).items():
        held_methods[name] = summary(metrics(lg, held))
    learned, controls, training, seeds_tbl = [], {}, {}, {}
    for arm_name in args.arms:
        arm, sigma = parse_arm(arm_name)
        for seed in range(args.seed0, args.seed0 + args.seeds):
            mname = f"attn_{arm_name}/s{seed}"
            model, init, info = train(env, arm_name, seed, dev_set, args, device)
            m = metrics(predict(model, test, arm, sigma), test)
            methods[mname], per_ep[mname] = summary(m), m["kl"]
            held_methods[mname] = summary(metrics(predict(model, held, arm, sigma), held))
            rnd = random_weights(Wrapped(model.cpu()), seed + 1).module.to(device)
            model.to(device)
            methods[mname + ":control_random_network"] = summary(metrics(predict(rnd, test, arm, sigma), test))
            methods[mname + ":control_untrained_init"] = summary(metrics(predict(init, test, arm, sigma), test))
            learned.append(mname)
            controls[mname] = {"random_network": mname + ":control_random_network",
                               "untrained_init": mname + ":control_untrained_init"}
            training[mname] = info
            seeds_tbl.setdefault(arm_name, []).append(float(m["kl"].mean()))
            print(f"[{alpha}:{N}:{k}] {mname} kl={m['kl'].mean():.4f} held={held_methods[mname]['kl']['mean']:.4f} "
                  f"best_step={info['best_step']} {info['train_seconds']:.0f}s", flush=True)
    # paired differences (per episode, test set) of each learned seed vs ceiling B and vs extremizing
    paired = {}
    for mname in learned:
        paired[mname] = {ref: summarize_values(per_ep[mname] - per_ep[ref], n_boot=1000)
                         for ref in ("bayes_oracle_keys", "extremized_mean_logodds")}
    extra = {
        "cell": {"alpha": alpha, "N": N, "k": k},
        "top3_share": env.top_share(3), "top3_share_heldout": env.top_share(3, heldout=True),
        "distinct_sources_mean": float(test["union"].sum(1).mean()),
        "extremize_factor": a_ext,
        "heldout_test": {"methods": held_methods, "split": "test", "first_chunk": 1000, "n": args.n_test,
                         "heldout_source_share": float(held["mask"][..., env.config.M:].sum(-1).gt(0).double().mean())},
        "seed_kl": {a: {"per_seed": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                    for a, v in seeds_tbl.items()},
        "paired_vs_refs": paired, "training": training,
        "train_cfg": {"steps": args.steps, "batch": args.batch, "lr": args.lr, "width": args.width,
                      "layers": args.layers, "readout": args.readout, "eval_every": args.eval_every, "n_dev": args.n_dev,
                      "fresh_episodes": "train chunk seed*1e6+step, one chunk per step"},
        "seconds": {"sets": t_sets, "total": time.time() - t0}, "device": str(device),
    }
    splits = {"eval": {"split": "test", "first_seed": int(fw.episode_seeds("test", 1)[0]), "n": args.n_test},
              "val": {"split": "dev", "first_seed": int(fw.episode_seeds("dev", 1)[0]), "n": args.n_dev},
              "train": {"split": "train", "first_seed": 0, "n": args.steps * args.batch}}
    env_obj = type("E", (), {"name": env.name, "config": env.config})()
    res = make_result(f"forecast-world-a{alpha}-N{N}-k{k}", "SMOKE" if args.smoke else "EXPLORATORY", env_obj,
                      "full", splits, oracle_check(env), methods, learned, controls, smoke=args.smoke,
                      extra=extra, allow_dirty=args.allow_dirty)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["1:10:3"])
    ap.add_argument("--arms", nargs="*", default=["nokey", "key0", "key0.5", "key1", "key3", "oracle"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0, help="first training seed (to split seeds over processes)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--readout", default="diag", choices=["diag", "full"])
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--n_test", type=int, default=4096)
    ap.add_argument("--n_dev", type=int, default=2048)
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow_dirty", action="store_true")
    ap.add_argument("--tag", default="", help="part name: several processes may run arms of one cell")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    for cell in args.cells:
        a, n, k = cell.split(":")
        a = float(a)
        res = run_cell(a, int(n), int(k), args, args.device)
        tag = f"_{args.tag}" if args.tag else ""
        name = f"cell_a{a:g}_N{n}_k{k}{tag}{'_smoke' if args.smoke else ''}.json"
        write_result(args.out, res, name=name, allow_dirty=args.allow_dirty)
        print("wrote", Path(args.out) / name, flush=True)


if __name__ == "__main__":
    main()
