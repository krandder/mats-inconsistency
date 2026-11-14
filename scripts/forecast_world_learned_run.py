"""ForecastWorld with LEARNED forecasters (2026-10-22 redirect): forecasters read raw source observations and send
messages; the aggregator is trained jointly with them on outcomes only. See aggbench.forecast_world.make_learned_system.

    python scripts/forecast_world_learned_run.py --cells 1:10:3 --arms fc fckey wide bar fckey_rand \
        --out results/explore/forecast-world-20261022/learned

Arms: fc, fckey, wide, bar, and the same with suffix _rand (descriptions are unrelated random vectors: control).
Test sets are the same episodes as scripts/forecast_world_run.py (same seeds; raw observations come from a separate
stream), so the exact-local-posterior arms there are the labelled reference. Writes one aggbench.result/v1 JSON
per cell (part), with held-out-source test metrics and local forecast quality under extra.
"""
import argparse
import copy
import math
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

EXT_CHUNK = 9_000_000


def sample_set(env, split, n, heldout=False, ceiling_b=True, start=0, per=1024):
    eps = [env.sample(split, min(per, n - i), chunk=start + i // per, heldout=heldout, ceiling_b=ceiling_b, obs=True)
           for i in range(0, n, per)]
    keys = [k for k in eps[0] if k not in ("X", "mu", "s", "theta", "key", "key_clean", "key_eps")]
    return {k: torch.cat([e[k] for e in eps]) for k in keys}


def summary(m):
    return {k: summarize_values(v, n_boot=1000) for k, v in m.items()}


class Wrapped:
    def __init__(self, module):
        self.module = module


def predict(model, ep, batch=512):
    model.eval()
    outs, locs = [], []
    with torch.no_grad():
        n = ep["y"].shape[0]
        for a in range(0, n, batch):
            sub = {k: v[a:a + batch] for k, v in ep.items()}
            o, loc = model(sub)
            outs.append(o.double())
            if loc is not None:
                locs.append(loc.double())
    return torch.cat(outs), (torch.cat(locs) if locs else None)


def parse(arm_name):
    return (arm_name[:-5], "random") if arm_name.endswith("_rand") else (arm_name, "meaning")


def train(env, arm_name, seed, dev_set, args, device):
    arm, desc = parse(arm_name)
    mk = dict(desc=desc, width=args.width, layers=args.layers, msg_dim=args.msg_dim, seed=seed)
    model = fw.make_learned_system(env, arm, **mk).to(device)
    init_state = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    best, best_state, best_step, log = math.inf, None, 0, []
    t0 = time.time()
    for step in range(args.steps):
        model.train()
        ep = env.sample("train", args.batch, chunk=seed * 1_000_000 + step, obs=True)
        y = ep["y"].float()
        out, loc = model(ep)
        loss = bce(out, y)
        if loc is not None:
            loss = loss + bce(loc, y[:, None].expand_as(loc))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            o, _ = predict(model, dev_set)
            vl = float(bce(o, dev_set["y"]))
            if vl < best:
                best, best_step, best_state = vl, step + 1, copy.deepcopy(model.state_dict())
            log.append({"step": step + 1, "val_logloss": vl, "train_loss": loss.item()})
    model.load_state_dict(best_state)
    init = fw.make_learned_system(env, arm, **mk).to(device)
    init.load_state_dict(init_state)
    return model, init, {"best_step": best_step, "best_val": best, "train_seconds": time.time() - t0, "log": log}


def local_quality(loc, ep):
    """Learned forecasters' own forecasts vs their exact local posteriors: mean KL(exact local || learned), nats."""
    if loc is None:
        return None
    kl = fw.logit_metrics(loc.reshape(-1, loc.shape[-1]), ep["probit"].reshape(-1, loc.shape[-1]),
                          ep["y"][:, None].expand_as(loc).reshape(-1, loc.shape[-1]))["kl"]
    return float(kl.mean())


def run_cell(alpha, N, k, args, device):
    from forecast_world_run import oracle_check   # same checks as the exact-forecaster runner
    env = fw.ForecastWorld(alpha=alpha, N=N, k=k, instance_seed=args.instance, disjoint=args.disjoint).to(device)
    t0 = time.time()
    test = sample_set(env, "test", args.n_test)
    held = sample_set(env, "test", args.n_test, heldout=True, start=1000)
    dev_set = sample_set(env, "dev", args.n_dev, ceiling_b=False)
    ext_tr = sample_set(env, "train", 4096, ceiling_b=False, start=EXT_CHUNK)
    a_ext = fw.fit_extremize(fw.probit_to_logit(ext_tr["probit"]), ext_tr["y"])

    def metrics(logit, ep):
        return fw.logit_metrics(logit, ep["bayes_probit"], ep["y"])

    def base(ep):
        lo = fw.probit_to_logit(ep["probit"])
        return {"bayes": fw.probit_to_logit(ep["bayes_probit"]),
                "bayes_oracle_keys": fw.probit_to_logit(ep["oracle_probit"]),
                "prior": torch.zeros_like(ep["bayes_probit"]), "mean_logodds": fw.pool_mean_logodds(lo),
                "extremized_mean_logodds": a_ext * fw.pool_mean_logodds(lo)}
    methods = {m: summary(metrics(v, test)) for m, v in base(test).items()}
    held_methods = {m: summary(metrics(v, held)) for m, v in base(held).items()}
    learned, controls, training, local = [], {}, {}, {}
    for arm_name in args.arms:
        for seed in range(args.seed0, args.seed0 + args.seeds):
            mname = f"learned_{arm_name}/s{seed}"
            model, init, info = train(env, arm_name, seed, dev_set, args, device)
            o, loc = predict(model, test)
            m = metrics(o, test)
            methods[mname] = summary(m)
            ho, hloc = predict(model, held)
            held_methods[mname] = summary(metrics(ho, held))
            local[mname] = {"test": local_quality(loc, test), "heldout": local_quality(hloc, held)}
            rnd = random_weights(Wrapped(model.cpu()), seed + 1).module.to(device)
            model.to(device)
            methods[mname + ":control_random_network"] = summary(metrics(predict(rnd, test)[0], test))
            methods[mname + ":control_untrained_init"] = summary(metrics(predict(init, test)[0], test))
            learned.append(mname)
            controls[mname] = {"random_network": mname + ":control_random_network",
                               "untrained_init": mname + ":control_untrained_init"}
            training[mname] = info
            print(f"[{alpha}:{N}:{k}] {mname} kl={m['kl'].mean():.4f} held={held_methods[mname]['kl']['mean']:.4f} "
                  f"local={local[mname]['test']} best_step={info['best_step']} {info['train_seconds']:.0f}s", flush=True)
    extra = {
        "cell": {"alpha": alpha, "N": N, "k": k, "disjoint": bool(args.disjoint)}, "forecasters": "learned",
        "top3_share": env.top_share(3), "distinct_sources_mean": float(test["union"].sum(1).mean()),
        "extremize_factor": a_ext,
        "heldout_test": {"methods": held_methods, "split": "test", "first_chunk": 1000, "n": args.n_test,
                         "heldout_source_share": float(held["mask"][..., env.config.M:].sum(-1).gt(0).double().mean())},
        "local_forecast_kl_vs_exact": local, "training": training,
        "train_cfg": {"steps": args.steps, "batch": args.batch, "lr": args.lr, "width": args.width,
                      "layers": args.layers, "msg_dim": args.msg_dim, "eval_every": args.eval_every,
                      "n_dev": args.n_dev, "loss": "aggregate log loss + forecasters' own log loss (fc arms)",
                      "fresh_episodes": "train chunk seed*1e6+step, one chunk per step"},
        "seconds": time.time() - t0, "device": str(device),
    }
    splits = {"eval": {"split": "test", "first_seed": int(fw.episode_seeds("test", 1)[0]), "n": args.n_test},
              "val": {"split": "dev", "first_seed": int(fw.episode_seeds("dev", 1)[0]), "n": args.n_dev},
              "train": {"split": "train", "first_seed": 0, "n": args.steps * args.batch}}
    env_obj = type("E", (), {"name": env.name + "_learned", "config": env.config})()
    return make_result(f"forecast-world-learned{'-disjoint' if args.disjoint else ''}-a{alpha}-N{N}-k{k}", "SMOKE" if args.smoke else "EXPLORATORY",
                       env_obj, "full", splits, oracle_check(env), methods, learned, controls, smoke=args.smoke,
                       extra=extra, allow_dirty=args.allow_dirty)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["1:10:3"])
    ap.add_argument("--arms", nargs="*", default=["fc", "fckey", "wide", "bar", "fckey_rand"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--msg_dim", type=int, default=64)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--n_test", type=int, default=4096)
    ap.add_argument("--n_dev", type=int, default=2048)
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow_dirty", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--disjoint", action="store_true", help="no-overlap control: all N*k sources distinct")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    for cell in args.cells:
        a, n, k = cell.split(":")
        a = float(a)
        res = run_cell(a, int(n), int(k), args, args.device)
        tag = f"_{args.tag}" if args.tag else ""
        dj = "_disjoint" if args.disjoint else ""
        name = f"learned{dj}_a{a:g}_N{n}_k{k}{tag}{'_smoke' if args.smoke else ''}.json"
        write_result(args.out, res, name=name, allow_dirty=args.allow_dirty)
        print("wrote", Path(args.out) / name, flush=True)


if __name__ == "__main__":
    main()
