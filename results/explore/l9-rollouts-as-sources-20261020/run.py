"""EXPLORATORY lane L9: rollouts of an imperfect model as information sources (aggbench E4).

Phase 1 (forecasts; heavy, run on a spot L4 through run_on_l4.sh):
    python3 results/explore/l9-rollouts-as-sources-20261020/run.py forecasts --out /root/runs/<run-id>
writes one forecasts/<tag>.npz per condition: the dev-split logit of every method (baselines, ceilings,
attention arms and their controls) plus meta.json (commit, timings).

Phase 2 (results; light, local, clean checkout at the same harness commit):
    python3 results/explore/l9-rollouts-as-sources-20261020/run.py results --forecasts <dir>
regenerates the dev batches (deterministic seeds), scores every method with aggbench.metrics, and writes one
validated result JSON per condition (aggbench.results.write_result) plus summary.json.

Arms. Baselines: aggbench.baselines.all_baselines (fit on train, scored on dev). Ceilings: bayes (all K
pieces), oracle_clearing (= reports ceiling), ceiling:trace (envs.rollout_trace_posterior_logit). Learned:
a one-layer attention aggregator over tokens [report, key] whose output is prior + sum_i (w_i (r_i - l0) + c_i),
(w_i, c_i) read from each token after attention; trained on Y labels (no distillation), fixed hyperparameters,
3 seeds, each run through controls.run_with_controls.
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from aggbench import make_env  # noqa: E402
from aggbench.baselines import all_baselines  # noqa: E402
from aggbench.envs import rollout_trace_posterior_logit  # noqa: E402

RUN = "l9-rollouts-as-sources-20261020"
EXACT = dict(calibration=1.0, recency=1.0, read_noise=0.0, shared_bias=0.0, step_noise=0.0)
# Recipe v1 (pre-registered before launch): 50k train episodes, Adam 2e-3 cosine, 3000 steps, no early stopping.
# Recipe v2 (added AFTER v1 showed the keyed arms memorizing train, train regret 0.000 vs dev 0.2): 200k train
# episodes, AdamW (weight decay 1e-4), up to 6000 steps, early stopping on 10k held-out TRAIN-split episodes
# (seeds 200000..209999); dev is still only scored.
RECIPE = os.environ.get("L9_RECIPE", "v1")
N_TRAIN, N_DEV, STEPS, N_VAL = {"v1": (50_000, 5_000, 3000, 0), "v2": (200_000, 5_000, 6000, 10_000)}[RECIPE]
SEEDS = (0, 1, 2)
SUFFIX = "" if RECIPE == "v1" else f"__{RECIPE}"
if os.environ.get("L9_SMOKE"):  # interface check only, never a result
    N_TRAIN, N_DEV, STEPS, N_VAL = 2000, 300, 30, min(N_VAL, 300)


def conditions():
    """(tag, model, overrides, visibilities, seeds)."""
    out = [("imperfect__N6", "imperfect", {}, ("none", "noisy", "full"), SEEDS),
           ("exact__N6", "exact", dict(EXACT), ("none", "full"), SEEDS)]
    for n in (1, 2, 4, 12, 24):
        out.append((f"imperfect__N{n}", "imperfect", {"n_rollouts": n}, ("full",), (0,)))
        out.append((f"exact__N{n}", "exact", dict(EXACT, n_rollouts=n), ("full",), (0,)))
    return out


# ------------------------------------------------------------------------------------------- aggregator

def make_aggregator(seed, key_dim, steps=None, device="cpu", val=None):
    import torch

    class Net(torch.nn.Module):
        def __init__(self, in_dim, width=64, heads=4):
            super().__init__()
            self.embed = torch.nn.Sequential(torch.nn.Linear(in_dim, width), torch.nn.GELU(),
                                             torch.nn.Linear(width, width))
            self.ln1, self.ln2 = torch.nn.LayerNorm(width), torch.nn.LayerNorm(width)
            self.attn = torch.nn.MultiheadAttention(width, heads, batch_first=True)
            self.mlp = torch.nn.Sequential(torch.nn.Linear(width, 2 * width), torch.nn.GELU(),
                                           torch.nn.Linear(2 * width, width))
            self.head = torch.nn.Linear(width, 2)

        def forward(self, tokens, centered):        # tokens (n, N, in_dim), centered reports (n, N)
            z = self.embed(tokens)
            h = self.ln1(z)
            z = z + self.attn(h, h, h, need_weights=False)[0]
            z = z + self.mlp(self.ln2(z))
            w, c = self.head(z).unbind(-1)
            return (w * centered + c).sum(1)

    steps = steps or STEPS

    class Agg:
        def __init__(self):
            torch.manual_seed(seed)
            self.module = Net(1 + key_dim)

        def _inputs(self, view):
            l0 = view.public["prior_logit"]
            r = view.reports["logit"] - l0
            tok = np.concatenate([r[..., None] / 4, view.keys], axis=-1)
            return (torch.tensor(tok, dtype=torch.float32, device=device),
                    torch.tensor(r, dtype=torch.float32, device=device))

        def fit(self, view, truth):
            net = self.module.to(device)
            tok, r = self._inputs(view)
            y = torch.tensor(truth, dtype=torch.float32, device=device)
            if val is None:
                opt = torch.optim.Adam(net.parameters(), lr=2e-3)
            else:
                opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
                vtok, vr = self._inputs(val[0])
                vy = torch.tensor(val[1], dtype=torch.float32, device=device)
                best, best_state, self.best_step = float("inf"), copy.deepcopy(net.state_dict()), 0
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
            gen = torch.Generator(device="cpu").manual_seed(seed)
            bce = torch.nn.functional.binary_cross_entropy_with_logits
            for step in range(1, steps + 1):
                idx = torch.randint(0, len(y), (1024,), generator=gen).to(device)
                opt.zero_grad()
                bce(net(tok[idx], r[idx]), y[idx]).backward()
                opt.step()
                sched.step()
                if val is not None and step % 200 == 0:
                    with torch.no_grad():
                        loss = bce(net(vtok, vr), vy).item()
                    if loss < best:
                        best, best_state, self.best_step = loss, copy.deepcopy(net.state_dict()), step
            if val is not None:
                net.load_state_dict(best_state)
            self.module = net.cpu()

        def predict(self, view):
            net = copy.deepcopy(self.module).to(device)
            tok, r = self._inputs(view)
            with torch.no_grad():
                out = net(tok, r).double().cpu().numpy()
            return {"logit": view.public["prior_logit"] + out}

    return Agg()


# ------------------------------------------------------------------------------------------- phase 1

def trace_ceiling(env, batch):
    l0 = batch.public["prior_logit"]
    return np.array([rollout_trace_posterior_logit(batch.obs[e], batch.oracle["order"][e], batch.oracle["mu"][e],
                                                   env.config, l0) for e in range(len(batch))])


def run_condition(args):
    tag, model, ov, vises, seeds, outdir = args
    import torch
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from aggbench.controls import run_with_controls
    t0 = time.time()
    env = make_env("E4", **ov)
    tr, dv = env.sample("train", N_TRAIN), env.sample("dev", N_DEV)
    va = env.sample("train", N_VAL, start=N_TRAIN) if N_VAL else None
    t_gen = time.time() - t0
    out, alphas = {}, {}
    for vis in vises:
        f, alphas[vis] = all_baselines(tr, dv, vis)
        if vis == vises[0]:
            out.update({k: v["logit"] for k, v in f.items()})
            out["ceiling:trace"] = trace_ceiling(env, dv)
        key_dim = tr.keys[vis].shape[-1]
        for s in seeds:
            val = (va.view(vis), va.truth) if va is not None else None
            arms = run_with_controls(lambda sd: make_aggregator(sd, key_dim, device=device, val=val), tr.view(vis), tr.truth,
                                     dv.view(vis), seed=s)
            for arm, fc in arms.items():
                out[f"attn:{vis}:s{s}:{arm}"] = fc["logit"]
    np.savez(Path(outdir) / "forecasts" / f"{tag}{SUFFIX}.npz", **out)
    return tag, {"gen_s": t_gen, "total_s": time.time() - t0, "device": device,
                 "extremize_alpha": alphas, "n_train": N_TRAIN, "n_dev": N_DEV, "n_val": N_VAL, "recipe": RECIPE}


def phase_forecasts(outdir, workers):
    (Path(outdir) / "forecasts").mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    jobs = [c + (outdir,) for c in conditions()]
    from aggbench.results import harness_info
    meta = {"recipe": RECIPE, "commit_env": os.environ.get("RUN_COMMIT", ""), "commit_git": commit.strip(),
            "harness_sha256": harness_info()["sha256"], "conditions": {}}
    with get_context("spawn").Pool(workers) as pool:
        for tag, m in pool.imap_unordered(run_condition, jobs):
            meta["conditions"][tag] = m
            print("done", tag, m, flush=True)
    (Path(outdir) / f"meta{SUFFIX}.json").write_text(json.dumps(meta, indent=2) + "\n")


# ------------------------------------------------------------------------------------------- phase 2

def phase_results(fdir):
    from aggbench.metrics import paired_difference, summarize
    from aggbench.results import harness_info, make_result, split_record, write_result
    from aggbench.verify import oracle_check
    h = harness_info()
    meta = json.loads((Path(fdir) / f"meta{SUFFIX}.json").read_text())
    if h["git_dirty"] or h["sha256"] != meta["harness_sha256"]:
        raise SystemExit("harness must be clean and identical (sha256) to the one that computed the forecasts")
    summary = {"forecast_meta": meta, "rows": []}
    for tag, model, ov, vises, seeds in conditions():
        env = make_env("E4", **ov)
        dv = env.sample("dev", N_DEV)
        tr_rec = {"split": "train", "first_seed": 0, "n": N_TRAIN + N_VAL}
        fz = np.load(Path(fdir) / "forecasts" / f"{tag}{SUFFIX}.npz")
        fc = {k: {"logit": fz[k]} for k in fz.files}
        assert np.allclose(fc["bayes"]["logit"], dv.bayes["logit"])  # same episodes on both machines
        check = oracle_check(env)
        fc["ceiling:infinite_rollouts"] = {"logit": infinite_rollouts_floor(env, dv)}
        shared = ["bayes", "ceiling:infinite_rollouts", "oracle_clearing", "ceiling:trace", "linear_pool", "log_pool", "sum_logodds",
                  "extremized_log_pool"]
        for vis in vises:
            names = shared + [k for k in fc if k.startswith(f"attn:{vis}:")]
            methods = {k: summarize(fc[k], dv) for k in names}
            arms = [f"attn:{vis}:s{s}:trained" for s in seeds]
            controls = {a: {"random_network": a.replace("trained", "control:random_network"),
                            "untrained_init": a.replace("trained", "control:untrained_init")} for a in arms}
            paired = {}
            for a in arms:
                for ref in ("extremized_log_pool", "oracle_clearing"):
                    paired[f"{a} - {ref}"] = paired_difference(fc[a], fc[ref], dv)
            paired["oracle_clearing - extremized_log_pool"] = paired_difference(fc["oracle_clearing"],
                                                                                fc["extremized_log_pool"], dv)
            paired["ceiling:trace - oracle_clearing"] = paired_difference(fc["ceiling:trace"], fc["oracle_clearing"], dv)
            methods[f"attn:{vis}:seed_mean"] = seed_mean(fc, arms, dv)
            extra = {"model": model, "visibility_note": "learned arms see keys at this visibility; baselines never "
                     "use keys", "paired_regret_differences": paired,
                     "extremize_alpha": meta["conditions"][tag]["extremize_alpha"][vis],
                     "forecasts_computed_on": meta["conditions"][tag]["device"],
                     "forecast_commit": meta["commit_env"] or meta["commit_git"],
                     "rollouts_cover_context_fraction": float(dv.oracle["cov"].any(1).all(1).mean()),
                     "recipe": RECIPE, "training": {
                         "v1": "Y labels on 50k train episodes, Adam 2e-3 cosine, 3000 steps of 1024, width 64",
                         "v2": "Y labels on 200k train episodes, AdamW 2e-3 wd 1e-4 cosine, up to 6000 steps of 1024, "
                               "width 64, early stopping on train-split episodes 200000..209999 (post hoc, after v1 "
                               "overfit)"}[RECIPE]}
            res = make_result(run_id=f"{RUN}/{tag}__{vis}{SUFFIX}", track="EXPLORATORY", env=env, visibility=vis,
                              splits={"train": tr_rec, "eval": split_record(dv, "dev")}, oracle_check=check,
                              methods=methods, learned_arms=arms, controls=controls, extra=extra)
            write_result(HERE / "json", res, name=f"{tag}__{vis}{SUFFIX}.json")
            summary["rows"].append({"tag": tag, "model": model, "n_rollouts": env.config.n_rollouts, "visibility": vis,
                                    "regret": {k: v["regret"] for k, v in methods.items()}, "paired": paired,
                                    "covered": extra["rollouts_cover_context_fraction"]})
            print("wrote", tag, vis, flush=True)
    (HERE / f"summary{SUFFIX}.json").write_text(json.dumps(summary, indent=2) + "\n")


def infinite_rollouts_floor(env, dv):
    """Limit of both ceilings as the number of rollouts grows: infinitely many rollouts in varied orders reveal
    each piece's misread LLR z_k = lam_k + e_k and the bias b exactly, but never e_k itself. z_k | Y ~
    N(+-m_k, 2 m_k + s_e^2), so the floor is l0 + sum_k 2 m_k z_k / (2 m_k + s_e^2). (Equals bayes when s_e = 0.)"""
    mu, x, e = dv.oracle["mu"], dv.oracle["x"], dv.oracle["e"]
    m = 2 * mu ** 2
    z = 2 * mu * x + e
    return dv.public["prior_logit"] + (2 * m * z / (2 * m + env.config.read_noise ** 2)).sum(1)


def seed_mean(fc, arms, dv):
    """Seed-averaged per-episode metrics of the trained arms (CI over episodes, seeds averaged first)."""
    from aggbench.metrics import per_episode, summarize_values
    per = [per_episode(fc[a], dv) for a in arms]
    return {m: summarize_values(np.mean([p[m] for p in per], axis=0)) for m in per[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["forecasts", "results"])
    ap.add_argument("--out", default=f"/tmp/{RUN}")
    ap.add_argument("--forecasts", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.phase == "forecasts":
        phase_forecasts(a.out, a.workers)
    else:
        phase_results(a.forecasts)


if __name__ == "__main__":
    main()
