"""IG / Aumann-Shapley pay for additive-logit agents vs exact and sampled Shapley.

Agents i own additive logit contributions z_i (N examples x C classes).  Price
= softmax(b0 + sum_S z_i); baseline all z = 0.  v(S) = sum_n log p_S(y) - log p0(y).
IG pay: pay_i = sum_n <z_i, e_y - pbar>, pbar = int_0^1 softmax(b0 + a q) da,
q = sum_i z_i.  Midpoint rule with m steps, or "exact" (binary: closed form
(softplus(a+s)-softplus(a))/s; C=3: 200-node Gauss-Legendre).

Parts:
  static  designed populations (inf, substitutes, complements, noise, harmful,
          small noise), n = 10 / 100 / 1000, C = 2 / 3, 3 seeds.  Compare IG
          (m in 5,10,20,50,exact) and permutation Shapley (K in 1,4,16,64)
          against exact Shapley (n=10) or a K=1000 reference (disjoint orders).
  online  online softmax regression, agents = input columns; pay every step on
          all examples vs every N steps on s sampled examples (scaled).
Outputs results/ig_summary.json, charts/ig_*.json.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from core import mean_ci, spearman
from exp0 import shapley_exact_from

HERE = os.path.dirname(os.path.abspath(__file__))
NEX = 1000
MS = (5, 10, 20, 50)
KS = (1, 4, 16, 64)
KREF = 1000
SEEDS = (0, 1, 2)
CATS = ["inf", "inf", "sub", "sub", "comp", "comp", "noise", "noise", "harm", "harm"]
GL_X, GL_W = np.polynomial.legendre.leggauss(200)
GL_X, GL_W = (GL_X + 1) / 2, GL_W / 2


def softmax(Z):
    Z = Z - Z.max(-1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(-1, keepdims=True)


def lse(Z):
    M = Z.max(-1, keepdims=True)
    return (M + np.log(np.exp(Z - M).sum(-1, keepdims=True)))[..., 0]


def softplus(x):
    return np.logaddexp(0, x)


def direction(C, rng):
    if C == 2:
        return np.array([0.0, 1.0])
    d = rng.standard_normal(C)
    d -= d.mean()
    return 1.2 * d / np.linalg.norm(d)


def population(n, C, seed):
    """Designed agents: blocks of 10 (CATS) + small noise agents to reach n."""
    rng = np.random.default_rng(seed * 7919 + n * 13 + C)
    nb = 1 if n == 10 else n // 50
    g = 1 / np.sqrt(nb)
    Z, cats, truth = [], [], np.zeros((NEX, C))
    for _ in range(nb):
        f = rng.standard_normal((5, NEX))
        d = [direction(C, rng) for _ in range(8)]
        # informative
        for k, c in ((0, 1.2), (1, 0.6)):
            t = c * g * f[k][:, None] * d[k]
            truth += t
            Z.append(t)
        # substitutes: two noisy copies, each 0.7x the true term (sum 1.4x)
        t = 1.0 * g * f[2][:, None] * d[2]
        truth += t
        for _ in range(2):
            Z.append(0.7 * t + 0.1 * g * rng.standard_normal(NEX)[:, None] * d[2])
        # complements: z_a = w(u+e_a), z_b = -w(u+e_b), y depends on e_a - e_b
        u = 3 * rng.standard_normal(NEX)
        ea, eb = rng.standard_normal(NEX), rng.standard_normal(NEX)
        w = 1.0 * g / np.sqrt(2)
        truth += (w * (ea - eb))[:, None] * d[3]
        Z.append((w * (u + ea))[:, None] * d[3])
        Z.append((-w * (u + eb))[:, None] * d[3])
        # noise (independent of y)
        for k in (4, 5):
            Z.append(0.3 * g * rng.standard_normal(NEX)[:, None] * d[k])
        # harmful: anti-signal on features that do move y
        for k, c in ((3, 0.5), (4, 0.5)):
            t = c * g * f[k][:, None] * d[k + 3]
            truth += t
            Z.append(-t)
        cats += CATS
    while len(Z) < n:
        Z.append(0.05 * rng.standard_normal(NEX)[:, None] * direction(C, rng))
        cats.append("small")
    y = (rng.random(NEX)[:, None] > np.cumsum(softmax(truth), 1)).sum(1)
    y = np.minimum(y, C - 1)
    freq = np.bincount(y, minlength=C) + 1.0
    b0 = np.log(freq / freq.sum())
    return np.array(Z), np.array(cats), y, b0


def value_masks(Z, y, b0, A):
    """v for coalition masks A (M, n) in chunks."""
    n, N, C = Z.shape
    Zf = Z.reshape(n, -1)
    lp0 = (b0[y] - lse(b0[None])).sum()
    out = []
    for s in range(0, len(A), 64):
        L = (A[s:s + 64] @ Zf).reshape(-1, N, C) + b0
        out.append(L[:, np.arange(N), y].sum(1) - lse(L).sum(1) - lp0)
    return np.concatenate(out)


def perm_shapley(Z, y, b0, K, rng):
    n, N, C = Z.shape
    lp0 = (b0[y] - lse(b0[None])).sum()
    phi = np.zeros(n)
    ar = np.arange(N)
    Z32, b32 = Z.astype(np.float32), b0.astype(np.float32)
    for _ in range(K):
        o = rng.permutation(n)
        P = np.cumsum(Z32[o], 0) + b32
        v = P[:, ar, y].sum(1, dtype=np.float64) - lse(P).sum(1, dtype=np.float64) - lp0
        phi[o] += np.diff(np.concatenate([[0.0], v]))
    return phi / K


def pbar_mid(q, b0, m):
    a = (np.arange(m) + 0.5) / m
    return sum(softmax(b0 + ak * q) for ak in a) / m


def pbar_exact(q, b0):
    C = q.shape[1]
    if C == 2:
        a = b0[1] - b0[0]
        s = q[:, 1] - q[:, 0]
        small = np.abs(s) < 1e-8
        ss = np.where(small, 1.0, s)
        p1 = np.where(small, 1 / (1 + np.exp(-a)), (softplus(a + ss) - softplus(a)) / ss)
        return np.stack([1 - p1, p1], 1)
    return pbar_gl(q, b0)


def pbar_gl(q, b0):
    return sum(w * softmax(b0 + x * q) for x, w in zip(GL_X, GL_W))


def ig_pay(Z, y, pbar):
    R = -pbar
    R[np.arange(len(y)), y] += 1
    return np.einsum("inc,nc->i", Z, R)


def metrics(est, ref):
    return dict(spearman=spearman(est, ref),
                rel_l1=float(np.abs(est - ref).sum() / np.abs(ref).sum()),
                sign_agree=float(np.mean(np.sign(est) == np.sign(ref))))


def static_job(cfg):
    n, C, seed = cfg
    t0 = time.time()
    Z, cats, y, b0 = population(n, C, seed)
    q = Z.sum(0)
    vN = float(value_masks(Z, y, b0, np.ones((1, n)))[0])
    out = dict(n=n, C=C, seed=seed, vN=vN)
    # closed form check (binary) and budget balance
    pe = pbar_exact(q, b0)
    if C == 2:
        out["closed_form_vs_gl200_maxabs"] = float(np.abs(pe - pbar_gl(q, b0)).max())
        out["closed_form_vs_mid10000_maxabs"] = float(np.abs(pe - pbar_mid(q, b0, 10000)).max())
    ig = {"exact": ig_pay(Z, y, pe)}
    for m in MS:
        ig[m] = ig_pay(Z, y, pbar_mid(q, b0, m))
    out["budget_err"] = {str(k): float((v.sum() - vN) / abs(vN)) for k, v in ig.items()}
    # LMSR reading: sum_i <z_i, pbar> == C(b0+q) - C(b0) per example
    cost_i = np.einsum("inc,nc->in", Z, pe)
    out["lmsr_cost_err_maxabs"] = float(np.abs(cost_i.sum(0) - (lse(b0 + q) - lse(b0[None]))).max())
    # reference
    rng = np.random.default_rng(10_000 + seed)
    if n <= 12:
        A = np.array([[mm >> i & 1 for i in range(n)] for mm in range(1 << n)], float)
        vals = value_masks(Z, y, b0, A)
        ref = shapley_exact_from(vals, n)
        ref1000 = perm_shapley(Z, y, b0, KREF, rng)
        out["ref"] = "exact"
        out["kref_vs_exact"] = metrics(ref1000, ref)
    else:
        kref = KREF if n <= 100 else 400
        ref = perm_shapley(Z, y, b0, kref, rng)
        out["ref"] = f"perm K={kref}"
    out["ref_budget_err"] = float((ref.sum() - vN) / abs(vN))
    res = {}
    for k, v in ig.items():
        res[f"IG m={k}"] = dict(metrics(v, ref), passes=(200 if C > 2 else 1) if k == "exact" else k)
    for K in KS:
        est = perm_shapley(Z, y, b0, K, np.random.default_rng(20_000 + seed * 100 + K))
        res[f"perm K={K}"] = dict(metrics(est, ref), passes=K * (n + 1))
    res["IG exact vs IG m=20"] = metrics(ig[20], ig["exact"])
    out["cmp"] = res
    out["exact_passes"] = 2 ** n
    # categories (per agent, nats), IG exact vs Shapley ref
    cat = {}
    for c in ["inf", "sub", "comp", "noise", "harm", "small"]:
        sel = cats == c
        if sel.any():
            cat[c] = dict(ig=float(ig["exact"][sel].mean()), shap=float(ref[sel].mean()),
                          absdiff=float(np.abs(ig["exact"][sel] - ref[sel]).mean()),
                          ig_frac_vN=float(ig["exact"][sel].sum() / vN), shap_frac_vN=float(ref[sel].sum() / vN))
    # within-pair split: |pay_a - pay_b| for complements and substitutes
    for c in ("comp", "sub"):
        idx = np.flatnonzero(cats == c).reshape(-1, 2)
        cat[c]["pair_gap_ig"] = float(np.abs(ig["exact"][idx[:, 0]] - ig["exact"][idx[:, 1]]).mean())
        cat[c]["pair_gap_shap"] = float(np.abs(ref[idx[:, 0]] - ref[idx[:, 1]]).mean())
    out["cats"] = cat
    out["secs"] = time.time() - t0
    return out


# ------------------------------------------------------------------- online
def online_job(cfg):
    C, seed = cfg
    rng = np.random.default_rng(500 + seed * 31 + C)
    D, T, BS, LR = 10, 1000, 512, 0.05
    Wt = np.stack([direction(C, rng) for _ in range(D)])     # true directions

    def batch(r, nn):
        f = r.standard_normal((nn, 4))
        u = 3 * r.standard_normal(nn)
        ea, eb = r.standard_normal(nn), r.standard_normal(nn)
        x = np.stack([f[:, 0], f[:, 1],
                      f[:, 2] + 0.15 * r.standard_normal(nn), f[:, 2] + 0.15 * r.standard_normal(nn),
                      (u + ea) / 3, (u + eb) / 3,
                      r.standard_normal(nn), r.standard_normal(nn),
                      f[:, 3], r.standard_normal(nn)], 1)
        Zt = (1.2 * f[:, :1] * Wt[0] + 0.6 * f[:, 1:2] * Wt[1] + 1.0 * f[:, 2:3] * Wt[2]
              + (0.7 * (ea - eb))[:, None] * Wt[4] + 0.5 * f[:, 3:4] * Wt[8])
        yy = (r.random(nn)[:, None] > np.cumsum(softmax(Zt), 1)).sum(1)
        return x, np.minimum(yy, C - 1)

    names = ["inf", "inf", "sub", "sub", "comp", "comp", "noise", "noise", "inf_weak", "noise"]
    _, yb = batch(np.random.default_rng(1), 100000)
    freq = np.bincount(yb, minlength=C) + 1.0
    b0 = np.log(freq / freq.sum())
    W = np.zeros((D, C))
    Ns, Ss = (1, 10, 50), (32, 256, BS)
    ref = np.zeros(D)
    cheap = {(N, s): np.zeros(D) for N in Ns for s in Ss}
    srng = np.random.default_rng(900 + seed)
    for t in range(T):
        x, y = batch(rng, BS)
        Z = x.T[:, :, None] * W[:, None, :]                  # (D, BS, C)
        q = Z.sum(0)
        pay_all = ig_pay(Z, y, pbar_exact(q, b0)) if C == 2 else ig_pay(Z, y, pbar_gl32(q, b0))
        ref += pay_all
        for N in Ns:
            if t % N:
                continue
            for s in Ss:
                if s == BS:
                    cheap[(N, s)] += N * pay_all
                else:
                    idx = srng.choice(BS, s, replace=False)
                    Zs = Z[:, idx]
                    ps = pbar_exact(Zs.sum(0), b0) if C == 2 else pbar_gl32(Zs.sum(0), b0)
                    cheap[(N, s)] += N * (BS / s) * ig_pay(Zs, y[idx], ps)
        P = softmax(b0 + x @ W)
        R = -P
        R[np.arange(BS), y] += 1
        W += LR * x.T @ R / BS
    out = dict(C=C, seed=seed, names=names, ref=ref.tolist(), W_end=W.tolist(), res={})
    for (N, s), v in cheap.items():
        out["res"][f"N={N},s={s}"] = dict(metrics(v, ref), cost_frac=float(s / (N * BS)))
    return out


def pbar_gl32(q, b0, _c={}):
    if "x" not in _c:
        x, w = np.polynomial.legendre.leggauss(32)
        _c["x"], _c["w"] = (x + 1) / 2, w / 2
    return sum(w * softmax(b0 + x * q) for x, w in zip(_c["x"], _c["w"]))


# --------------------------------------------------------------------- main
def ci(vals):
    vals = [v for v in vals if v == v]
    m, h = mean_ci(vals)
    return [round(m, 4), round(h, 4)]


def summarize(st, on):
    S = {"setup": dict(NEX=NEX, MS=MS, KS=KS, KREF=KREF, seeds=list(SEEDS),
                       note="mean and 95% t half-width over seeds; rel_l1 = sum|est-ref|/sum|ref|")}
    groups = {}
    for r in st:
        groups.setdefault((r["n"], r["C"]), []).append(r)
    stat = {}
    for (n, C), rs in sorted(groups.items()):
        d = dict(ref=rs[0]["ref"], exact_passes=rs[0]["exact_passes"] if n <= 12 else None,
                 vN=ci([r["vN"] for r in rs]), secs=ci([r["secs"] for r in rs]))
        d["budget_err"] = {k: ci([r["budget_err"][k] for r in rs]) for k in rs[0]["budget_err"]}
        d["ref_budget_err"] = ci([r["ref_budget_err"] for r in rs])
        d["lmsr_cost_err_maxabs"] = max(r["lmsr_cost_err_maxabs"] for r in rs)
        if C == 2:
            d["closed_form_vs_gl200_maxabs"] = max(r["closed_form_vs_gl200_maxabs"] for r in rs)
            d["closed_form_vs_mid10000_maxabs"] = max(r["closed_form_vs_mid10000_maxabs"] for r in rs)
        if "kref_vs_exact" in rs[0]:
            d["kref_vs_exact"] = {k: ci([r["kref_vs_exact"][k] for r in rs]) for k in rs[0]["kref_vs_exact"]}
        d["cmp"] = {a: {k: (ci([r["cmp"][a][k] for r in rs]) if k != "passes" else rs[0]["cmp"][a][k])
                        for k in rs[0]["cmp"][a]} for a in rs[0]["cmp"]}
        d["cats"] = {c: {k: ci([r["cats"][c][k] for r in rs]) for k in rs[0]["cats"][c]} for c in rs[0]["cats"]}
        stat[f"n={n},C={C}"] = d
    S["static"] = stat
    og = {}
    for r in on:
        og.setdefault(r["C"], []).append(r)
    S["online"] = {f"C={C}": {k: {m: ci([r["res"][k][m] for r in rs]) for m in rs[0]["res"][k]}
                              for k in rs[0]["res"]} | {"ref_cum_pay_seed0": dict(zip(rs[0]["names"], [round(v, 2) for v in rs[0]["ref"]]))}
                   for C, rs in sorted(og.items())}
    return S


def charts(S):
    src = "trader-economy/results/ig_summary.json"
    cd = os.path.join(HERE, "charts")
    # 1 error vs reference by method, n = 10/100/1000, binary
    groups = ["n=10", "n=100", "n=1000"]
    arms = ["IG m=20", "perm K=1", "perm K=4", "perm K=16", "perm K=64"]
    series, cis = {}, {}
    for C in (2, 3):
        for a in arms:
            key = f"{a} (C={C})"
            series[key] = [S["static"][f"{g},C={C}"]["cmp"][a]["rel_l1"][0] for g in groups]
            cis[key] = [S["static"][f"{g},C={C}"]["cmp"][a]["rel_l1"][1] for g in groups]
    json.dump(dict(id="ig_error_vs_shapley", kind="grouped_bar",
                   title="Pay error vs Shapley: IG (20 passes) vs sampled orders",
                   unit="relative L1 error", source=src, groups=groups, series=series, ci=cis,
                   note="reference = exact Shapley at n=10, 1000 disjoint orders at n=100/1000; "
                        "passes: IG 20 at every n; perm K*(n+1)"),
              open(os.path.join(cd, "ig_error_vs_shapley.json"), "w"), indent=1)
    # 2 category gap at n=10
    bars = []
    for C in (2, 3):
        cat = S["static"][f"n=10,C={C}"]["cats"]
        for c in ("inf", "sub", "comp", "noise", "harm"):
            bars.append(dict(label=f"{c} C={C}: IG", value=cat[c]["ig"][0], ci=cat[c]["ig"][1], highlight=True))
            bars.append(dict(label=f"{c} C={C}: Shapley", value=cat[c]["shap"][0], ci=cat[c]["shap"][1], highlight=False))
    json.dump(dict(id="ig_categories", kind="bar", title="n=10 designed agents: pay per agent, IG vs exact Shapley",
                   unit="nats per agent (1000 examples)", source=src, bars=bars,
                   note="IG = exact path integral; categories: 2 agents each"),
              open(os.path.join(cd, "ig_categories.json"), "w"), indent=1)
    # 3 online subsampling
    bars = []
    for k, v in S["online"]["C=2"].items():
        if k.startswith("N="):
            bars.append(dict(label=f"{k} (cost {v['cost_frac'][0]:.4f})", value=v["rel_l1"][0], ci=v["rel_l1"][1],
                             highlight=False))
    json.dump(dict(id="ig_online_subsample", kind="bar",
                   title="Online IG pay every N steps on s examples: cumulative pay error",
                   unit="relative L1 error vs every step, all 512 examples", source=src, bars=bars,
                   note="binary, 1000 SGD steps, 10 feature agents; cost = fraction of example-passes"),
              open(os.path.join(cd, "ig_online_subsample.json"), "w"), indent=1)


def main():
    t0 = time.time()
    st_cfgs = [(n, C, s) for n in (1000, 100, 10) for C in (2, 3) for s in SEEDS]
    on_cfgs = [(C, s) for C in (2, 3) for s in SEEDS]
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        print(static_job((10, 3, 0))["cmp"])
        print(online_job((2, 0))["res"])
        return
    with Pool(6) as pool:
        st = pool.map(static_job, st_cfgs, chunksize=1)
        on = pool.map(online_job, on_cfgs, chunksize=1)
    S = summarize(st, on)
    S["runtime_s"] = round(time.time() - t0)
    json.dump(S, open(os.path.join(HERE, "results", "ig_summary.json"), "w"), indent=1)
    charts(S)
    print("done", S["runtime_s"], "s")


if __name__ == "__main__":
    main()
