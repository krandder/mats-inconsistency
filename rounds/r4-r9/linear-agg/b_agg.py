"""Experiment B aggregators and analysis (written before any B result was read): applies PREREG-LA.md B-main, B-a, B-b.
Reads runs/b/b-seed{0,1,2}.npz (from b_rollouts.py). Trains LT-4, LT-1 and LT-4 without the conflict feature per seed on
cal outcomes (early stopping on validation outcomes), fits the round-3 scalar pools on cal + val outcomes, runs the round-3
replication pools, scores KL(P* || mu) on the 500 test instances per family. Writes runs/b-eval/.
   python3 b_agg.py [--seeds 0,1,2] [--threads 1]          (CPU, float64)
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("SMALLNET_DIR", os.path.join(HERE, "eca"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "eca"))
import ecarun as R  # noqa: E402  (round-3 pool code: fit_sw, sw_pool, fit_u, pool_feature, kl_bern, logloss)
from lt import LinTF  # noqa: E402

RUNS = os.environ.get("LA_B_RUNS", os.path.join(HERE, "runs", "b"))
OUT = os.environ.get("LA_B_OUT", os.path.join(HERE, "runs", "b-eval"))
FAMS = ("B1", "B2")
T975_2 = 4.302652729911275
PUBLISHED = {"B1": {"mix": 0.2527, "sw": 0.2276, "dd": 0.2768}, "B2": {"mix": 0.1982, "sw": 0.1981, "dd": 0.2231}}
DT = torch.float64


def tci(x):
    x = np.asarray(x, float)
    m = float(x.mean())
    h = T975_2 * float(x.std(ddof=1)) / math.sqrt(len(x))
    return m, m - h, m + h


def lg(p, eps=1e-6):
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def get(z, split, key):
    return np.concatenate([z[f"{split}|{f}|{key}"] for f in FAMS], 0)


# ------------------------------------------------------------------------------------------------ tokens
def build(z, split, stats=None, drop_conflict=False):
    """-> ctx [n, 16, D], qry [n, 1, D], base logit P0 [n], y [n], fam index [n]; D = 16 + 4 + 1 + 4 (- 1 if no conflict).
    Order: k (16), c (beta, conflict, traj log-prob, tau), 1, v (Y, bridge 1-3)."""
    k, c, v = get(z, split, "k"), get(z, split, "c"), get(z, split, "v")
    kq = get(z, split, "kq")
    cq = np.stack([get(z, split, "beta0"), get(z, split, "conf0")], 1)
    if stats is None:
        stats = {"k_mu": k.reshape(-1, 16).mean(0), "k_sd": k.reshape(-1, 16).std(0) + 1e-8,
                 "c_mu": c.reshape(-1, 4).mean(0), "c_sd": c.reshape(-1, 4).std(0) + 1e-8,
                 "v_sd": v.reshape(-1, 4).std(0) + 1e-8,
                 "kq_mu": kq.mean(0), "kq_sd": kq.std(0) + 1e-8, "cq_mu": cq.mean(0), "cq_sd": cq.std(0) + 1e-8}
    cl = lambda a: np.clip(a, -5, 5)
    ks = cl((k - stats["k_mu"]) / stats["k_sd"])
    cs = cl((c - stats["c_mu"]) / stats["c_sd"])
    vs = cl(v / stats["v_sd"])
    n, K = v.shape[:2]
    kqs = cl((kq - stats["kq_mu"]) / stats["kq_sd"])
    cqs = cl((cq - stats["cq_mu"]) / stats["cq_sd"])
    cq4 = np.concatenate([cqs, np.zeros((n, 2))], 1)
    if drop_conflict:
        cs = np.delete(cs, 1, axis=2)
        cq4 = np.delete(cq4, 1, axis=1)
    ctx = np.concatenate([ks, cs, np.ones((n, K, 1)), vs], 2)
    qry = np.concatenate([kqs, cq4, np.ones((n, 1)), np.zeros((n, 4))], 1)[:, None, :]
    base = lg(get(z, split, "P0"))
    y = get(z, split, "y")
    fam = np.concatenate([np.full(len(z[f"{split}|{f}|y"]), i) for i, f in enumerate(FAMS)])
    t = lambda a: torch.tensor(a, dtype=DT)
    return t(ctx), t(qry), t(base), t(y), fam, stats


def lt_out(m, ctx, qry, vscale):
    _, q, _ = m(ctx, qry)
    return -q[:, 0, -4] * vscale                                         # readout: the query's Y slot (paper: -y)


def train_lt(L, tr, va, vscale, seed, epochs=300, patience=30, lr0=1e-3, batch=256):
    ctx, qry, base, y = tr[:4]
    D = ctx.shape[2]
    restarts, lr, hist = 0, lr0, []
    while True:
        m = LinTF("full", L, D, heads=4, init=1e-3, ydim=4, seed=seed + 100 * restarts).to(DT)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        g = torch.Generator().manual_seed(5_000 + seed + 100 * restarts)
        best, best_state, bad, skipped, diverged = float("inf"), None, 0, 0, False
        for ep in range(epochs):
            perm = torch.randperm(ctx.shape[0], generator=g)
            m.train()
            for a in range(0, len(perm), batch):
                idx = perm[a:a + batch]
                out = base[idx] + lt_out(m, ctx[idx], qry[idx], vscale)
                loss = F.binary_cross_entropy_with_logits(out, y[idx])
                if not torch.isfinite(loss):                 # skipped (amendment A-3: no restarts)
                    skipped += 1
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            if diverged:
                break
            with torch.no_grad():
                vo = va[2] + lt_out(m, va[0], va[1], vscale)
                vl = float(F.binary_cross_entropy_with_logits(vo, va[3]))
            hist.append({"restart": restarts, "epoch": ep, "val_logloss": vl, "skipped": skipped})
            if math.isfinite(vl) and vl < best - 1e-7:     # amendment A-3: a non-finite / worse epoch is never selected
                best, best_state, bad = vl, copy.deepcopy(m.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if not diverged or restarts >= 2:
            break
        restarts += 1
        lr /= 3
    if best_state is not None:
        m.load_state_dict(best_state)
    m.eval()
    return m, {"best_val_logloss": best, "no_finite_epoch": best_state is None, "epochs_run": hist[-1]["epoch"] + 1, "restarts": restarts, "lr": lr,
               "params": m.n_params()}, hist


def influence_spearman(m, te, vscale, conflict):
    """Within each test instance: Spearman(conflict_r, |d out / d v_r[Y]|); mean over instances."""
    ctx = te[0].clone().requires_grad_(True)
    out = lt_out(m, ctx, te[1], vscale)
    (gr,) = torch.autograd.grad(out.sum(), ctx)
    I = gr[:, :, -4].abs().numpy()                                     # d out / d (standardised v_r[Y])
    sp = np.array([R.spearman(conflict[i], I[i]) for i in range(len(I))])
    return float(np.nanmean(sp)), I


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--only-seed", type=int, default=None, help="train/score one seed and write its part file")
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    if not a.combine:
        for s in ([a.only_seed] if a.only_seed is not None else seeds):
            per_seed(s)
    if a.combine or a.only_seed is None:
        combine(seeds)


def per_seed(seed):
    t0 = time.time()
    z = np.load(os.path.join(RUNS, f"b-seed{seed}.npz"))
    res = {"seed": seed, "kl": {}, "logloss": {}, "lt": {}, "pools": {}}
    arrays = {}
    pst = {f: z[f"test|{f}|pstar"] for f in FAMS}
    yt = {f: z[f"test|{f}|y"] for f in FAMS}

    def score(name, mu_by_fam):
        for f in FAMS:
            res["kl"][f"{name}|{f}"] = float(R.kl_bern(pst[f], mu_by_fam[f]).mean())
            res["logloss"][f"{name}|{f}"] = float(R.logloss(mu_by_fam[f], yt[f]).mean())
            arrays[f"{name}|{f}"] = mu_by_fam[f]

    score("P0", {f: z[f"test|{f}|P0"] for f in FAMS})
    # ---- scalar pools on the same rollouts (round-3 definitions), fitted on cal + val outcomes
    fitq = np.concatenate([get(z, "cal", "qY"), get(z, "val", "qY")])
    fits = np.concatenate([get(z, "cal", "s"), get(z, "val", "s")])
    fitk = np.concatenate([get(z, "cal", "k"), get(z, "val", "k")])
    fitP0 = np.concatenate([get(z, "cal", "P0"), get(z, "val", "P0")])
    fitp0 = np.concatenate([get(z, "cal", "p0"), get(z, "val", "p0")])
    fity = np.concatenate([get(z, "cal", "y"), get(z, "val", "y")])
    sw = R.fit_sw(fitq, fits, fitP0, fity)
    u_dd = R.fit_u(R.pool_feature(fitq, fits, fitk, fitp0, "dd"), R.logit_np(fitp0), fity)
    res["pools"] = {"sw": sw, "dd_u": u_dd}
    score("mix", {f: z[f"test|{f}|qY"].mean(1) for f in FAMS})
    score("sw", {f: R.sw_pool(z[f"test|{f}|qY"], z[f"test|{f}|s"], z[f"test|{f}|P0"], sw["lam"], sw["eta"]) for f in FAMS})
    score("dd", {f: R.sig_np(R.logit_np(z[f"test|{f}|p0"]) + u_dd * R.pool_feature(
        z[f"test|{f}|qY"], z[f"test|{f}|s"], z[f"test|{f}|k"], z[f"test|{f}|p0"], "dd")) for f in FAMS})
    tau = z["taus"]
    t1 = tau == 1.0
    score("mix_tau1", {f: z[f"test|{f}|qY"][:, t1].mean(1) for f in FAMS})
    # ---- round-3 replication (rand chains, tau 1; pools fitted on the first 500 validation instances of F, B1, B2)
    rf = ("F", "B1", "B2")
    qv = np.concatenate([z[f"rep|val|{f}|q"][:, :, 1] for f in rf])
    sv = np.concatenate([z[f"rep|val|{f}|s"][:, :, 1] for f in rf])
    kv = np.concatenate([z[f"rep|val|{f}|k"][:, :, 1] for f in rf])
    P0v = np.concatenate([z[f"rep|val|{f}|q"][:, 0, 0] for f in rf])
    p0v = np.concatenate([z[f"rep|val|{f}|p0"] for f in rf])
    yv = np.concatenate([z[f"rep|val|{f}|y"] for f in rf])
    rsw = R.fit_sw(qv, sv, P0v, yv)
    ru = R.fit_u(R.pool_feature(qv, sv, kv, p0v, "dd"), R.logit_np(p0v), yv)
    res["pools"]["rep"] = {"sw": rsw, "dd_u": ru}
    for f in FAMS:
        q, s, k = (z[f"rep|test|{f}|{x}"][:, :, 1] for x in ("q", "s", "k"))
        P0r, p0r, pr = z[f"rep|test|{f}|q"][:, 0, 0], z[f"rep|test|{f}|p0"], z[f"rep|test|{f}|pstar"]
        assert np.allclose(pr, pst[f])
        for name, mu in (("rep_mix", q.mean(1)), ("rep_sw", R.sw_pool(q, s, P0r, rsw["lam"], rsw["eta"])),
                         ("rep_dd", R.sig_np(R.logit_np(p0r) + ru * R.pool_feature(q, s, k, p0r, "dd"))),
                         ("rep_P0", P0r)):
            res["kl"][f"{name}|{f}"] = float(R.kl_bern(pr, mu).mean())
            arrays[f"{name}|{f}"] = mu
    # ---- linear-transformer aggregators
    for name, L, drop in (("LT4", 4, False), ("LT1", 1, False), ("LT4_noconf", 4, True)):
        tr = build(z, "cal", None, drop)
        stats = tr[5]
        va = build(z, "val", stats, drop)
        te = build(z, "test", stats, drop)
        vscale = float(stats["v_sd"][0])
        tc = time.time()
        m, info, hist = train_lt(L, tr, va, vscale, seed)
        info["sec"] = time.time() - tc
        res["lt"][name] = info
        with torch.no_grad():
            out = (te[2] + lt_out(m, te[0], te[1], vscale)).numpy()
        mu = 1 / (1 + np.exp(-np.clip(out, -40, 40)))
        fam = te[4]
        score(name, {f: mu[fam == i] for i, f in enumerate(FAMS)})
        with open(os.path.join(OUT, f"train-{name}-s{seed}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(hist[0].keys()))
            w.writeheader()
            w.writerows(hist)
        if name == "LT4":
            conflict = get(z, "test", "c")[:, :, 1]
            sp, I = influence_spearman(m, te, vscale, conflict)
            res["spearman_conflict_influence"] = sp
            arrays["LT4_influence"] = I
            arrays["test_conflict"] = conflict
            torch.save({"state": m.state_dict(), "stats": {k_: v_.tolist() for k_, v_ in stats.items()}},
                       os.path.join(OUT, f"lt4-s{seed}.pt"))
        print(seed, name, json.dumps(info), {f: round(res["kl"][f"{name}|{f}"], 4) for f in FAMS}, flush=True)
    res["sec"] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, f"seed{seed}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, f"arrays-s{seed}.npz"), **arrays,
                        **{f"pstar|{f}": pst[f] for f in FAMS}, **{f"y|{f}": yt[f] for f in FAMS})


def combine(seeds):
    rs = [json.load(open(os.path.join(OUT, f"seed{s}.json"))) for s in seeds]
    kl = lambda name, f: np.array([r["kl"][f"{name}|{f}"] for r in rs])
    arms = ["P0", "mix", "sw", "dd", "mix_tau1", "LT4", "LT1", "LT4_noconf", "rep_P0", "rep_mix", "rep_sw", "rep_dd"]
    table = {f"{a}|{f}": list(tci(kl(a, f))) for a in arms for f in FAMS}
    v = {"B-main": {}, "B-b": {}}
    for f in FAMS:
        means = {p: float(kl(p, f).mean()) for p in ("mix", "sw", "dd")}
        bestp = min(means, key=means.get)
        m, lo, hi = tci(kl(bestp, f) - kl("LT4", f))
        lt = float(kl("LT4", f).mean())
        pub = min(PUBLISHED[f].values())
        v["B-main"][f] = {"LT4": lt, "best_pool": bestp, "best_pool_kl": means[bestp], "diff": [m, lo, hi],
                          "published_best": pub, "pass_i": bool(lo > 0), "pass_ii": bool(lt < pub)}
        m2, lo2, hi2 = tci(kl("LT1", f) - kl("LT4", f))
        v["B-b"][f] = {"LT1": float(kl("LT1", f).mean()), "LT4": lt, "diff": [m2, lo2, hi2], "pass": bool(lo2 > 0)}
    v["B-main"]["pass"] = all(v["B-main"][f]["pass_i"] and v["B-main"][f]["pass_ii"] for f in FAMS)
    v["B-b"]["pass"] = all(v["B-b"][f]["pass"] for f in FAMS)
    sp = np.array([r["spearman_conflict_influence"] for r in rs])
    m, lo, hi = tci(sp)
    v["B-a"] = {"per_seed": sp.tolist(), "mean": m, "ci": [lo, hi], "pass": bool(m <= -0.10 and hi < 0),
                "noconf_minus_LT4": {f: list(tci(kl("LT4_noconf", f) - kl("LT4", f))) for f in FAMS}}
    out = {"seeds": seeds, "kl_3seed": table, "verdicts": v, "published_r3": PUBLISHED,
           "per_seed": {str(r["seed"]): {"kl": r["kl"], "lt": r["lt"], "pools": r["pools"]} for r in rs}}
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    lines = ["| arm | B1 KL (3-seed mean [95% t-CI]) | B2 KL |", "|---|---|---|"]
    for a_ in arms:
        lines.append(f"| {a_} | " + " | ".join(f"{table[f'{a_}|{f}'][0]:.4f} [{table[f'{a_}|{f}'][1]:.4f}, {table[f'{a_}|{f}'][2]:.4f}]"
                                                for f in FAMS) + " |")
    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "kl.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "family", "seed", "kl"])
        for r in rs:
            for key, val in sorted(r["kl"].items()):
                arm, f = key.split("|")
                w.writerow([arm, f, r["seed"], f"{val:.6f}"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["P0", "mix", "sw", "dd", "LT1", "LT4", "LT4_noconf", "rep_mix", "rep_sw", "rep_dd"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, f in zip(axs, FAMS):
        for i, a_ in enumerate(show):
            vals = kl(a_, f)
            ax.bar(i, vals.mean(), color="#2471a3" if a_.startswith("LT") else ("0.6" if a_.startswith("rep") else "0.35"))
            ax.plot([i] * len(vals), vals, "k.", ms=4)
        for p, val in PUBLISHED[f].items():
            ax.axhline(val, color="#c0392b", lw=0.8, ls=":")
        ax.set_xticks(range(len(show)))
        ax.set_xticklabels(show, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"{f}: test KL(P* || pool), 3 seeds (dots); red dotted = published R3 pools")
    axs[0].set_ylabel("mean KL (nats)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "b-kl.png"), dpi=130)
    plt.close(fig)
    print(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
