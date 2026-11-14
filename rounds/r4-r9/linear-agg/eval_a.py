"""Experiment A evaluation (written before any A result was read): applies PREREG-LA.md checks A1-A3 mechanically.
Reads runs/a/<kind>-L<L>-s<seed>/model.pt; writes runs/a-eval/{summary.json, tables.md, mse.csv, arrays.npz, *.png}.
CPU, float64.   python3 eval_a.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synth as S  # noqa: E402
from lt import LinTF  # noqa: E402
from train_a import YSCALE, tokens  # noqa: E402

RUNS = os.environ.get("LA_A_RUNS", os.path.join(HERE, "runs", "a"))
OUT = os.environ.get("LA_A_OUT", os.path.join(HERE, "runs", "a-eval"))
SMALL = int(os.environ.get("LA_A_SMALL", "0"))
KINDS, LS, SEEDS = ("gdpp", "diag", "full"), (1, 2, 4, 6), (0, 1, 2)
SIGMAS = (0.1, 0.5, 1.0, 2.0, 3.0)
T975_2 = 4.302652729911275
DT = torch.float64
N_DIAG = 5000


def tci(x):
    x = np.asarray(x, float)
    m = float(x.mean())
    h = T975_2 * float(x.std(ddof=1)) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
    return m, m - h, m + h


# ------------------------------------------------------------------------------------------------ closed forms
def adarr_torch(X, Y):
    """Batched AdaRR (paper): OLS by pseudo-inverse, sigma^2 = RSS/(n - rank), ridge. X [n,K,D], Y [n,K] -> w [n,D]."""
    pinv = torch.linalg.pinv(X)
    wls = torch.einsum("ndk,nk->nd", pinv, Y)
    rss = ((Y - torch.einsum("nkd,nd->nk", X, wls)) ** 2).sum(1)
    rank = torch.linalg.matrix_rank(X).to(X.dtype)
    s2 = rss / (X.shape[1] - rank).clamp_min(1.0)
    A = torch.einsum("nkd,nke->nde", X, X) + s2[:, None, None] * torch.eye(X.shape[2], dtype=X.dtype)[None]
    return torch.linalg.solve(A, torch.einsum("nkd,nk->nd", X, Y)[..., None])[..., 0]


def claims(ep):
    """Per-rollout back-projected claims c_r = <k_q, k_r> v_r and precision proxy p_r = 1/m_r."""
    return np.einsum("nkd,d->nk", ep["keys"], S.KQ) * ep["v"], 1.0 / ep["m"]


def crowd(ep):
    G = np.einsum("nkd,nbd->nkb", ep["keys"], ep["keys"])
    return np.maximum(G, 0).sum(2)


def pool_feats(ep, kind, eta=0.0, lam=0.0):
    c, p = claims(ep)
    if kind == "mix":
        return c.mean(1)
    if kind == "sw":
        w = p ** eta
        return (w * c).sum(1) / (lam + w.sum(1))
    if kind == "dd":
        return (p * c / crowd(ep)).sum(1)
    raise ValueError(kind)


def fit_pools(ep):
    y = ep["yq"]
    fit = {}
    for kind in ("mix", "dd"):
        f = pool_feats(ep, kind)
        u = float((f * y).sum() / (f * f).sum())
        fit[kind] = {"u": u, "fit_mse": float(((u * f - y) ** 2).mean())}
    best = None
    for eta in (0.0, 0.5, 1.0, 2.0):
        for lam in (0.0,) + tuple(float(x) for x in np.logspace(-3, 3, 25)):
            f = pool_feats(ep, "sw", eta, lam)
            u = float((f * y).sum() / (f * f).sum())
            m = float(((u * f - y) ** 2).mean())
            if best is None or m < best["fit_mse"] - 1e-12:
                best = {"u": u, "eta": eta, "lam": lam, "fit_mse": m}
    fit["sw"] = best
    return fit


def pool_pred(ep, kind, fp):
    if kind == "sw":
        return fp["u"] * pool_feats(ep, "sw", fp["eta"], fp["lam"])
    return fp["u"] * pool_feats(ep, kind)


# ------------------------------------------------------------------------------------------------ models
def load(kind, L, seed):
    d = os.path.join(RUNS, f"{kind}-L{L}-s{seed}")
    ck = torch.load(os.path.join(d, "model.pt"), map_location="cpu", weights_only=False)
    m = LinTF(kind, L, S.D + 1, init=0.0).to(DT)
    m.load_state_dict(ck["state"])
    m.eval()
    info = json.load(open(os.path.join(d, "train.json")))
    return m, info


@torch.no_grad()
def model_pred(m, ep, chunk=10000):
    out, xq = [], []
    ctx, qry = tokens(ep, DT, "cpu")
    for a in range(0, ctx.shape[0], chunk):
        _, q, _ = m(ctx[a:a + chunk], qry[a:a + chunk])
        out.append(-YSCALE * q[:, 0, -1])
        xq.append(q[..., :S.D])
    return torch.cat(out).numpy(), torch.cat(xq).numpy()


def kappa_ratio(m, ep):
    """Median over episodes of kappa(Sigma^L)/kappa(Sigma^0) on range(Sigma^0), plus per-layer medians."""
    ctx, qry = tokens(ep, DT, "cpu")
    with torch.no_grad():
        _, _, st = m(ctx, qry, keep_states=True)
    S0 = torch.einsum("nkd,nke->nde", st[0], st[0])
    ev, U = torch.linalg.eigh(S0)
    keep = ev > 1e-9 * ev[:, -1:]
    r = int(keep.sum(1).min())
    assert bool((keep.sum(1) == r).all()), "rank of Sigma^0 varies"
    Ur = U[:, :, -r:]
    ks = []
    for x in st:
        Sl = torch.einsum("nkd,nke->nde", x, x)
        e = torch.linalg.eigvalsh(torch.einsum("ndr,nde,nes->nrs", Ur, Sl, Ur))
        ks.append((e[:, -1] / e[:, 0].clamp_min(1e-300)).numpy())
    ks = np.stack(ks, 1)
    return float(np.median(ks[:, -1] / ks[:, 0])), [float(np.median(k)) for k in ks.T], r


def influence(fn, ep):
    """|d yhat / d v_r| for every (episode, rollout) via autograd; fn(v_tensor) -> yhat."""
    v = torch.tensor(ep["v"], dtype=DT, requires_grad=True)
    y = fn(v)
    (g,) = torch.autograd.grad(y.sum(), v)
    return g.abs().numpy()


def lt_fn(m, ep):
    keys = torch.tensor(ep["keys"], dtype=DT)
    mm = torch.tensor(ep["m"], dtype=DT)
    cand = torch.tensor(ep["cand"], dtype=DT)

    def f(v):
        ctx, qry = tokens({"keys": keys, "m": mm, "v": v, "cand": cand}, DT, "cpu")
        _, q, _ = m(ctx, qry)
        return -YSCALE * q[:, 0, -1]
    return f


def adarr_fn(ep):
    X = torch.tensor(ep["keys"] / np.sqrt(ep["m"])[..., None], dtype=DT)
    sm = torch.tensor(np.sqrt(ep["m"]), dtype=DT)
    kq = torch.tensor(S.KQ, dtype=DT)
    return lambda v: adarr_torch(X, v / sm) @ kq


def ratio_bad_clean(I, bad):
    return float(I[bad].mean() / I[~bad].mean())


_BEST = {}


def select_stats(ep, S_post, picks):
    n = len(ep["yq"])
    val = S.pair_value(ep, S_post, picks)
    if id(ep) not in _BEST:
        _BEST[id(ep)] = np.max(np.stack([S.pair_value(ep, S_post, np.tile(np.array([a, b]), (n, 1)))
                                         for a in range(8) for b in range(a + 1, 8)], 1), 1)
    best = _BEST[id(ep)]
    eff = val / np.maximum(best, 1e-300)
    tv = (S.CAND_TYPE[picks] == 2).mean(1)
    both = (np.sort(picks, 1) == np.array([0, 1])).all(1)
    return {"eff": float(eff.mean()), "tv_share": float(tv.mean()), "both_bridges": float(both.mean())}, eff


def top2(score):
    return np.argsort(-score, 1, kind="stable")[:, :2]


def main():
    os.makedirs(OUT, exist_ok=True)
    nl, npool = (2000, 5000) if SMALL else (20000, 50000)
    tests = {f"s{s}": S.episodes(nl, 9_000_000 + i, sigma=s) for i, s in enumerate(SIGMAS)}
    tests["pooled"] = S.episodes(npool, 9_100_000)
    orc = {k: S.oracle(e) for k, e in tests.items()}
    res = {"sets": list(tests), "mse": {}, "fits": {}, "diag": {}, "select": {}}
    arrays = {}
    for k, e in tests.items():
        arrays[f"yq|{k}"] = e["yq"]
        arrays[f"oracle|{k}"] = orc[k]["yhat"]
        res["mse"][f"oracle|{k}"] = float(np.mean((orc[k]["yhat"] - e["yq"]) ** 2))
        res["mse"][f"prior0|{k}"] = float(np.mean(e["yq"] ** 2))
        X = torch.tensor(e["keys"] / np.sqrt(e["m"])[..., None], dtype=DT)
        ya = (adarr_torch(X, torch.tensor(e["v"] / np.sqrt(e["m"]), dtype=DT)) @ torch.tensor(S.KQ, dtype=DT)).numpy()
        arrays[f"adarr|{k}"] = ya
        res["mse"][f"adarr|{k}"] = float(np.mean((ya - e["yq"]) ** 2))
    for seed in SEEDS:
        fe = S.episodes(10000 if SMALL else 100000, 8_000_000 + seed)
        fp = fit_pools(fe)
        res["fits"][str(seed)] = fp
        for kind in ("mix", "sw", "dd"):
            for k, e in tests.items():
                yp = pool_pred(e, kind, fp[kind])
                arrays[f"{kind}|s{seed}|{k}"] = yp
                res["mse"][f"{kind}|s{seed}|{k}"] = float(np.mean((yp - e["yq"]) ** 2))
    pe = {k: v[:N_DIAG] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == npool else v
          for k, v in tests["pooled"].items()}
    S_post_d = orc["pooled"]["S"][:N_DIAG]
    # reference influence ratios
    w_or = np.einsum("d,nde,nke->nk", S.KQ, S_post_d, pe["keys"]) / pe["nvar"]
    res["diag"]["influence_ratio|oracle"] = ratio_bad_clean(np.abs(w_or), pe["bad"])
    res["diag"]["influence_ratio|adarr"] = ratio_bad_clean(influence(adarr_fn(pe), pe), pe["bad"])
    # reference selectors on the pooled set
    P = tests["pooled"]
    Spost = orc["pooled"]["S"]
    rng = np.random.default_rng(12345)
    refs = {"exact_voi": top2(orc["pooled"]["voi"]),
            "raw_cosine": top2(np.einsum("nzd,d->nz", P["cand"], S.KQ) ** 2 + 1e-12 * rng.random((npool, 8))),
            "uncertainty": top2(orc["pooled"]["pred_var_cand"]),
            "random": np.stack([rng.permutation(8)[:2] for _ in range(npool)])}
    for name, pk in refs.items():
        st, eff = select_stats(P, Spost, pk)
        res["select"][f"{name}"] = st
        arrays[f"sel_eff|{name}"] = eff
    for kind in KINDS:
        for L in LS:
            for seed in SEEDS:
                tag = f"{kind}-L{L}-s{seed}"
                if not os.path.exists(os.path.join(RUNS, tag, "model.pt")):
                    print("MISSING", tag, flush=True)
                    continue
                m, info = load(kind, L, seed)
                res.setdefault("train", {})[tag] = info
                for k, e in tests.items():
                    yp, xq = model_pred(m, e)
                    arrays[f"{tag}|{k}"] = yp
                    res["mse"][f"{tag}|{k}"] = float(np.mean((yp - e["yq"]) ** 2))
                    if k == "pooled":
                        sc = np.einsum("nd,nzd->nz", xq[:, 0], xq[:, 1:]) ** 2
                        st, eff = select_stats(e, Spost, top2(sc))
                        res["select"][tag] = st
                        arrays[f"sel_eff|{tag}"] = eff
                kr, kl, rank = kappa_ratio(m, pe)
                I = influence(lt_fn(m, pe), pe)
                res["diag"][tag] = {"kappa_ratio_median": kr, "kappa_median_per_layer": kl, "rank_Sigma0": rank,
                                    "influence_ratio_bad_clean": ratio_bad_clean(I, pe["bad"])}
                if kind != "full":
                    res["diag"][tag]["omegas_xx_xy_yx_yy"] = m.omegas().tolist()
                print(tag, "pooled MSE", round(res["mse"][f"{tag}|pooled"], 4), json.dumps(res["diag"][tag])[:200], flush=True)
    np.savez_compressed(os.path.join(OUT, "arrays.npz"), **arrays)
    try:
        res["verdicts"] = verdicts(res)
    except KeyError as e:
        res["verdicts"] = {"error": f"missing {e}"}
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    write_tables(res)
    plots(res)
    print(json.dumps(res["verdicts"], indent=1))


def mse3(res, tag_wo_seed, k):
    return np.array([res["mse"][f"{tag_wo_seed}-s{s}|{k}"] for s in SEEDS])


def pool3(res, kind, k):
    return np.array([res["mse"][f"{kind}|s{s}|{k}"] for s in SEEDS])


def verdicts(res):
    v = {}
    a1a = {}
    for arm in ("diag-L4", "diag-L6", "full-L4", "full-L6"):
        L = int(arm.split("L")[1])
        x = mse3(res, arm, "pooled")
        cmp_ = {}
        for b in ("mix", "sw", "dd", f"gdpp-L{L}"):
            bb = pool3(res, b, "pooled") if b in ("mix", "sw", "dd") else mse3(res, b, "pooled")
            m, lo, hi = tci(bb - x)
            cmp_[b] = {"baseline_mse": float(bb.mean()), "diff_mean": m, "ci": [lo, hi], "pass": bool(lo > 0)}
        a1a[arm] = {"arm_mse": float(x.mean()), "vs": cmp_, "pass": all(c["pass"] for c in cmp_.values())}
    a1b = {}
    for arm in ("diag-L6", "full-L6"):
        lv = {}
        for s in SIGMAS:
            x = float(mse3(res, arm, f"s{s}").mean())
            ad = res["mse"][f"adarr|s{s}"]
            lv[str(s)] = {"arm": x, "adarr": ad, "ratio": x / ad, "pass": bool(x <= 1.05 * ad)}
        a1b[arm] = {"levels": lv, "pass": all(l_["pass"] for l_ in lv.values())}
    v["A1"] = {"a": a1a, "b": a1b, "pass": all(a["pass"] for a in a1a.values()) and any(b["pass"] for b in a1b.values())}
    a2a, a2b = {}, {}
    for arm in ("diag-L4", "diag-L6"):
        ks = [res["diag"][f"{arm}-s{s}"]["kappa_ratio_median"] for s in SEEDS]
        a2a[arm] = {"per_seed": ks, "pass": all(k <= 0.1 for k in ks)}
        om = [np.array(res["diag"][f"{arm}-s{s}"]["omegas_xx_xy_yx_yy"]) for s in SEEDS]
        a2b[arm] = {"per_seed_any_wyy_neg": [bool((o[:, 3] < 0).any()) for o in om],
                    "per_seed_any_wxy_neg": [bool((o[:, 1] < 0).any()) for o in om]}
        a2b[arm]["pass"] = all(a2b[arm]["per_seed_any_wyy_neg"]) and all(a2b[arm]["per_seed_any_wxy_neg"])
    ir = [res["diag"][f"diag-L6-s{s}"]["influence_ratio_bad_clean"] for s in SEEDS]
    m, lo, hi = tci(ir)
    a2c = {"per_seed": ir, "mean": m, "ci": [lo, hi], "pass": bool(m <= 0.8 and hi < 1)}
    v["A2"] = {"a": a2a, "b": a2b, "c": a2c, "pass": all(x["pass"] for x in a2a.values()) and
               all(x["pass"] for x in a2b.values()) and a2c["pass"]}
    eff = np.array([res["select"][f"diag-L6-s{s}"]["eff"] for s in SEEDS])
    m, lo, hi = tci(eff - res["select"]["raw_cosine"]["eff"])
    tv = np.array([res["select"][f"diag-L6-s{s}"]["tv_share"] for s in SEEDS])
    v["A3"] = {"a": {"eff_per_seed": eff.tolist(), "eff_mean": float(eff.mean()), "raw_cosine_eff": res["select"]["raw_cosine"]["eff"],
                     "diff_vs_cosine": [m, lo, hi], "pass": bool(eff.mean() >= 0.90 and lo > 0)},
               "b": {"tv_share_per_seed": tv.tolist(), "mean": float(tv.mean()), "pass": bool(tv.mean() <= 0.05)}}
    v["A3"]["pass"] = v["A3"]["a"]["pass"] and v["A3"]["b"]["pass"]
    return v


def write_tables(res):
    L_ = []
    L_.append("| set | oracle | AdaRR | mix | sw_mix | dd | " + " | ".join(f"{k} L{L}" for k in KINDS for L in LS) + " |")
    L_.append("|" + "---|" * (6 + len(KINDS) * len(LS)))
    rows = []
    for k in [f"s{s}" for s in SIGMAS] + ["pooled"]:
        cells = [k, f"{res['mse'][f'oracle|{k}']:.4f}", f"{res['mse'][f'adarr|{k}']:.4f}"]
        cells += [f"{pool3(res, b, k).mean():.4f}" for b in ("mix", "sw", "dd")]
        for kind in KINDS:
            for L in LS:
                try:
                    cells.append(f"{mse3(res, f'{kind}-L{L}', k).mean():.4f}")
                except KeyError:
                    cells.append("n/a")
        L_.append("| " + " | ".join(cells) + " |")
        rows.append(cells)
    open(os.path.join(OUT, "tables.md"), "w").write("Test MSE (3-seed means; sigma levels s0.1 ... s3.0, pooled = sigma ~ U(0.1, 3))\n\n"
                                                     + "\n".join(L_) + "\n")
    import csv
    with open(os.path.join(OUT, "mse.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "set", "mse"])
        for key, val in sorted(res["mse"].items()):
            arm, rest = key.split("|", 1)
            if "|" in rest:
                seed, st = rest.split("|")
                w.writerow([arm, seed, st, f"{val:.6f}"])
            elif "-s" in arm and arm.split("-s")[-1].isdigit():
                w.writerow([arm.rsplit("-s", 1)[0], "s" + arm.rsplit("-s", 1)[1], rest, f"{val:.6f}"])
            else:
                w.writerow([arm, "", rest, f"{val:.6f}"])


def plots(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)
    xs = np.array(SIGMAS)
    for ax, L in zip(axs, LS):
        orc = np.array([res["mse"][f"oracle|s{s}"] for s in SIGMAS])
        ax.plot(xs, np.array([res["mse"][f"adarr|s{s}"] for s in SIGMAS]) - orc, "k--", label="AdaRR")
        for b, st in (("mix", ":"), ("sw", "-."), ("dd", (0, (1, 3)))):
            ax.plot(xs, np.array([pool3(res, b, f"s{s}").mean() for s in SIGMAS]) - orc, color="0.5", ls=st, label=b)
        for kind, col in (("gdpp", "#c0392b"), ("diag", "#2471a3"), ("full", "#1e8449")):
            try:
                ys = np.array([mse3(res, f"{kind}-L{L}", f"s{s}") for s in SIGMAS]) - orc[:, None]
            except KeyError:
                continue
            ax.errorbar(xs, ys.mean(1), yerr=ys.std(1, ddof=1), color=col, marker="o", ms=3, capsize=2, label=kind.replace("gdpp", "GD++"))
        ax.set_title(f"L = {L}")
        ax.set_xlabel("episode noise sigma")
        ax.axhline(0, color="0.8", lw=0.8)
    axs[0].set_ylabel("adjusted MSE (MSE - oracle MSE)")
    axs[-1].legend(fontsize=8)
    fig.suptitle("Experiment A: query MSE above the closed-form oracle (3 seeds, mean +- sd)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "a-adjusted-mse.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
