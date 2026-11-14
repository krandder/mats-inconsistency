#!/usr/bin/env python3
"""R8-E3 (E-MS-SAFE-ANCHOR) for one seed. Reuses the R5-E2 primary checkpoints (theta_real, slot trust, R5-E2 rollout
trust; nothing of theta_real is retrained). Everything fitted or selected uses the CALIBRATION split only, against
realised data (the true hidden piece types of calibration positions for belief aggregators; realised game results for the
trust head and the pooling hyperparameters). The exact oracle (P*_full, Q*) is read only by part_d1 and part_d2.
   python3 run_r8e3.py --seed 101 --ckpt /tmp/r5e2/ckpt/s101/primary --out OUT [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r8lib as Y  # noqa: E402  (puts r7, r6 and r5 on sys.path)
import r7lib as X  # noqa: E402
import e3lib as L  # noqa: E402
import run_e3 as R6  # noqa: E402
import eval6 as EV  # noqa: E402
import model6 as M  # noqa: E402
import train6 as T  # noqa: E402

DATA = T.DATA
KS = (1, 2, 4, 8, 16)
AGGS = ("raw_count", "union_t", "scaled_mix", "ataraxos_mmd", "ataraxos_mmd_fit", "kc_dedup_K3c", "attn_L3_K3c",
        "looped_attn_T3_K3c")
MMD_FIXED = 0.25
TAU_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, float("inf"))
ALPHA_GRID = (0.5, 1.0, 2.0)
CT_ALPHA = (0.0, 0.5, 1.0, 2.0)
M_GRID = (2, 4, 6, 8)
GAMMA_GRID = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
N_CLEAN, N_N1, N_N4 = 16, 3, 3                    # clean bag 16; mixed bag = clean 1-10 + 3 myopic + 3 tau-3.5 noisy


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------------------------- D1 pools
@torch.no_grad()
def d1_pools(Mo, R, ix, gen, dev, n=16):
    rt = Y.add_window(L.roots_e3(Mo.net, R, ix, dev))
    fa = T.sample_flank_actions(rt, n, gen, dev)
    pL = Y.d1_rollouts(Mo.net, Mo.strust, Mo.rtrust, rt, fa[:, 0], gen, dev)
    pR = Y.d1_rollouts(Mo.net, Mo.strust, Mo.rtrust, rt, fa[:, 1], gen, dev)
    return rt, pL, pR


def fit_d1(a, Mo, Rc, dev):
    """Belief aggregators anchored at ell_t, fitted on calibration two-flank roots against the TRUE hidden types
    (cross-entropy), bags = prefixes (K ~ U{1..16}) of 16-rollout sequences skewed with p_L ~ Beta(0.5, 0.5)
    (scaled_mix: p_L = 0.5)."""
    gen = torch.Generator(device=dev).manual_seed(a.seed + 810)
    roots = T.two_flank_roots(Rc, a.n_d1_roots, gen, dev)
    PL, PR, L0, HID, TRUE = [], [], [], [], []
    for c0 in range(0, roots.shape[0], 128):
        ix = roots[c0:c0 + 128]
        rt, pL, pR = d1_pools(Mo, Rc, ix, gen, dev)
        PL.append(pL)
        PR.append(pR)
        L0.append(rt["p0b"])
        HID.append(rt["hidden"])
        TRUE.append(Rc["slot_true"][ix].long())
    cat = lambda xs: {k: torch.cat([x[k] for x in xs]) for k in xs[0]}
    poolL, poolR = cat(PL), cat(PR)
    l0, hid, true = torch.cat(L0), torch.cat(HID), torch.cat(TRUE)
    N = l0.shape[0]
    res = {"n_roots": int(N), "cal_ce_anchor": float(Y.ce_true(l0, true, hid))}

    def bag(skew):
        pL = torch.full((N,), 0.5, device=dev) if skew == "unskewed" else torch.distributions.Beta(0.5, 0.5).sample((N,)).to(dev)
        s = Y.seq_bag(poolL, poolR, pL, gen)
        K = torch.randint(1, Y.BAG + 1, (N,), generator=gen, device=dev)
        return Y.prefix(s, K, l0)

    def fit(name, n_steps, fn, params, lr, skew="beta", extra=None):
        opt = torch.optim.Adam(params, lr=lr)
        hist = []
        for st in range(n_steps):
            b = bag(skew)
            loss = Y.ce_true(fn(b), true, hid)
            tot = loss + (extra(b) if extra is not None else 0.0)
            opt.zero_grad(set_to_none=True)
            tot.backward()
            opt.step()
            if st % 200 == 0 or st == n_steps - 1:
                hist.append([st, float(tot), float(loss)])
        res[name + "_hist"] = hist

    la = torch.tensor(math.log(0.1), device=dev, requires_grad=True)
    fit("scaled_mix", a.d1_steps, lambda b: Y.agg_scaled(b, la.exp()), [la], 1e-2, skew="unskewed")
    res["alpha"] = float(la.exp())
    le = torch.tensor(math.log(0.02), device=dev, requires_grad=True)
    fit("mmd_fit", a.d1_steps, lambda b: Y.agg_mmd(b, le.exp()), [le], 1e-2)
    res["eta_fit"] = float(le.exp())
    torch.manual_seed(a.seed + 820)
    key = X.KeyK3c().to(dev)
    lu = torch.tensor(math.log(0.1), device=dev, requires_grad=True)
    fit("kc_K3c", a.d1_steps, lambda b: Y.agg_kc(b, key(b), lu.exp()), list(key.parameters()) + [lu], 3e-3,
        extra=lambda b: L.key_align_loss(key(b), key.target(b), b["valid"]))
    res["u"] = float(lu.exp())
    for p in key.parameters():
        p.requires_grad_(False)
    nets = {}
    for nm, kw, so in (("attn_L3_K3c", dict(layers=3), 830), ("looped_attn_T3_K3c", dict(loops=3), 840)):
        torch.manual_seed(a.seed + so)
        m = Y.AnchorAttn(**kw).to(dev)
        fit(nm, a.d1_steps, lambda b, m=m: m(b, key(b)), list(m.parameters()), 1e-3)
        T.freeze(m)
        nets[nm] = m
        res[f"n_params_{nm}"] = int(sum(p.numel() for p in m.parameters()))
    with torch.no_grad():
        s = Y.seq_bag(poolL, poolR, torch.distributions.Beta(0.5, 0.5).sample((N,)).to(dev), gen)
        b = Y.prefix(s, 16, l0)
        fns = d1_fns(res, key, nets)
        for nm, fn in fns.items():
            res[f"cal_ce_K16_{nm}"] = float(Y.ce_true(fn(b), true, hid))
    return key, nets, res


def d1_fns(fit, key, nets):
    return {"raw_count": Y.agg_raw_count, "union_t": Y.agg_union_t, "scaled_mix": lambda b: Y.agg_scaled(b, fit["alpha"]),
            "ataraxos_mmd": lambda b: Y.agg_mmd(b, MMD_FIXED), "ataraxos_mmd_fit": lambda b: Y.agg_mmd(b, fit["eta_fit"]),
            "kc_dedup_K3c": lambda b: Y.agg_kc(b, key(b), fit["u"]),
            "attn_L3_K3c": lambda b: nets["attn_L3_K3c"](b, key(b)),
            "looped_attn_T3_K3c": lambda b: nets["looped_attn_T3_K3c"](b, key(b))}


@torch.no_grad()
def part_d1(a, Mo, key, nets, fit, R, dev, eo):
    """500 two-flank test positions; per policy (U_Left p_L 0.85, U_Right 0.15) an independent 16-rollout skewed sequence
    from its own pools (16 Left + 16 Right); bags = prefixes K in {1, 2, 4, 8, 16}; K = 0 is ell_t."""
    d = np.load(os.path.join(DATA, "oracle_d1_test.npz"))
    n = 32 if a.smoke else len(d["t"])
    roots = EV.rows_for(d, EV.row_index(R), dev, n)
    P = d["pstar_full"][:n]
    gen = torch.Generator(device=dev).manual_seed(a.seed + 860)
    fns = d1_fns(fit, key, nets)
    acc = {k: [] for k in ["game", "t", "n_hidden", "kl_anchor_t4", "kl_routeb_all", "nL_L", "nL_R"] +
           [f"{m}_{g}_K{K}" for g in AGGS for K in (0,) + KS for m in ("kl", "jsd")]}
    for c0 in range(0, n, 64):
        ix = roots[c0:c0 + 64]
        B = ix.shape[0]
        mus = {}
        hid = None
        for name, pLv in (("L", 0.85), ("R", 0.15)):
            rt, pL, pR = d1_pools(Mo, R, ix, gen, dev)
            hid = rt["hidden"].cpu().numpy()
            s = Y.seq_bag(pL, pR, torch.full((B,), pLv, device=dev), gen)
            acc[f"nL_{name}"].append(s["nL"].cpu().numpy())
            l0 = rt["p0b"]
            mus[name] = {(g, 0): l0.exp().cpu().numpy() for g in AGGS}
            for K in KS:
                b = Y.prefix(s, K, l0)
                for g, fn in fns.items():
                    mus[name][(g, K)] = fn(b).exp().cpu().numpy()
            if name == "L":
                acc["kl_anchor_t4"].append(None)
                rtL = rt
        Pc = P[c0:c0 + B]
        nh = np.maximum(hid.sum(1), 1)
        perslot = lambda x: np.where(hid, x, 0).sum(1) / nh
        for g in AGGS:
            for K in (0,) + KS:
                acc[f"jsd_{g}_K{K}"].append(perslot(EV.jsd_cat(mus["L"][(g, K)], mus["R"][(g, K)])))
                acc[f"kl_{g}_K{K}"].append(0.5 * (perslot(EV.kl_rows(Pc, mus["L"][(g, K)])) + perslot(EV.kl_rows(Pc, mus["R"][(g, K)]))))
        acc["kl_anchor_t4"][-1] = perslot(EV.kl_rows(Pc, rtL["l0b"].exp().cpu().numpy()))
        acc["kl_routeb_all"].append(perslot(EV.kl_rows(Pc, rtL["lmu"].exp().cpu().numpy())))
        acc["n_hidden"].append(hid.sum(1))
        acc["game"].append(d["game"][c0:c0 + B])
        acc["t"].append(d["t"][c0:c0 + B])
    acc = {k: np.concatenate(v) for k, v in acc.items()}
    EV.write_csv(os.path.join(eo, "d1_positions.csv"), acc)
    res = {"n": int(n), "kl_anchor_t4": float(acc["kl_anchor_t4"].mean()), "kl_routeb_all": float(acc["kl_routeb_all"].mean()),
           "mean_nL_L": float(acc["nL_L"].mean()), "mean_nL_R": float(acc["nL_R"].mean())}
    for g in AGGS:
        res[g] = {f"K{K}": {"kl": float(acc[f"kl_{g}_K{K}"].mean()), "jsd": float(acc[f"jsd_{g}_K{K}"].mean())} for K in (0,) + KS}
    return res


# --------------------------------------------------------------------------------------------- D2 calibration
def bags(Mo, R, ix, act, gen, dev, n_clean):
    rt = Y.add_window(T.make_roots(Mo.net, R, ix, dev))
    A = lambda k: act[:, None].expand(-1, k)
    c = Y.run_det(Mo.net, Mo.strust, Mo.rtrust, rt, A(n_clean), gen, dev, noisy=False, depth=4, temp=1.0)
    n1 = Y.run_det(Mo.net, Mo.strust, Mo.rtrust, rt, A(N_N1), gen, dev, noisy=True, depth=1, temp=1.0)
    n4 = Y.run_det(Mo.net, Mo.strust, Mo.rtrust, rt, A(N_N4), gen, dev, noisy=True, depth=4, temp=Y.TAU_NOISY)
    return c, n1, n4


def fit_trust_select(a, Mo, Rc, dev):
    """New theta_trust (r5/r7 regression: excess surprise of q_r against the REALISED result; inputs r5 trust features +
    cone_lp + ply_lp; no tier label) on calibration rollouts of the played move with the mixed-bag composition (10 clean +
    3 myopic noisy + 3 tau-3.5 noisy). Then on 1,000 DISJOINT calibration roots (16 clean + 3 + 3 noisy) every pooling
    hyperparameter is chosen by the log loss of the pooled effective win against the realised result."""
    gen = torch.Generator(device=dev).manual_seed(a.seed + 800)
    ok = (Rc["n_ply"] >= 1) & (Rc["t"] >= 4) & (Rc["t"] <= 36)
    cand = torch.nonzero(ok).squeeze(1)
    perm = cand[torch.randperm(cand.shape[0], generator=gen, device=dev)]
    tr_roots = perm[:a.n_trust_roots]
    sel_roots = perm[a.n_trust_roots:a.n_trust_roots + a.n_sel_roots]
    Xs, Ys, Ts = [], [], []
    for c0 in range(0, tr_roots.shape[0], 256):
        ix = tr_roots[c0:c0 + 256]
        y = Rc["result"][ix].long()
        c, n1, n4 = bags(Mo, Rc, ix, R6.played_index(Rc, ix), gen, dev, 10)
        for tier, o in ((0, c), (1, n1), (2, n4)):
            Xs.append(o["feats_x"].reshape(-1, o["feats_x"].shape[-1]).float())
            Ys.append(M.excess_surprise(o["q_leaf"].reshape(-1, 3), y.repeat_interleave(o["e"].shape[1])))
            Ts.append(torch.full((o["e"].numel(),), tier, device=dev))
    Xs, Ys, Ts = torch.cat(Xs), torch.cat(Ys), torch.cat(Ts)
    torch.manual_seed(a.seed + 805)
    tr, hist = X.fit_trust(Xs, Ys, gen, dev, steps=a.trust_steps)
    bt = X.beta_new(tr, Xs)
    res = {"n_train_roots": int(tr_roots.shape[0]), "n_train_rollouts": int(Ys.shape[0]), "trust_hist": hist,
           "target_mean_by_tier": [float(Ys[Ts == k].mean()) for k in range(3)],
           "train_beta_mean_by_tier": [float(bt[Ts == k].mean()) for k in range(3)],
           "train_spearman_beta_neg_target": EV.spearman(bt.cpu().numpy(), -Ys.cpu().numpy())}
    keep = ("e", "H", "ll")
    C, NX, Yr = {k: [] for k in keep + ("beta",)}, {k: [] for k in keep + ("beta",)}, []
    for c0 in range(0, sel_roots.shape[0], 256):
        ix = sel_roots[c0:c0 + 256]
        c, n1, n4 = bags(Mo, Rc, ix, R6.played_index(Rc, ix), gen, dev, N_CLEAN)
        c["beta"] = X.beta_new(tr, c["feats_x"])
        n1["beta"], n4["beta"] = X.beta_new(tr, n1["feats_x"]), X.beta_new(tr, n4["feats_x"])
        for k in C:
            C[k].append(c[k].float())
            NX[k].append(torch.cat([n1[k], n4[k]], 1).float())
        Yr.append(Rc["result"][ix].long())
    C = {k: torch.cat(v) for k, v in C.items()}
    NX = {k: torch.cat(v) for k, v in NX.items()}
    Yr = torch.cat(Yr)
    y = (Yr == 0).float() + 0.5 * (Yr == 1).float()
    bce = lambda p: float(torch.nn.functional.binary_cross_entropy(p.clamp(1e-5, 1 - 1e-5), y))
    mixb = {k: torch.cat([C[k][:, :10], NX[k]], 1) for k in C}
    sel = {}
    g_cons = {str(t): bce(X.wmean(C["e"], Y.cons_w(C["ll"], t))) for t in TAU_GRID}
    sel["tau_cons"] = min(TAU_GRID, key=lambda t: g_cons[str(t)])
    g_tp = {str(al): bce(X.wmean(C["e"], C["beta"] ** al)) for al in ALPHA_GRID}
    sel["alpha_trust"] = min(ALPHA_GRID, key=lambda al: g_tp[str(al)])
    g_gate = {str(m): bce(X.wmean(mixb["e"], Y.drop_bottom(mixb["beta"], m))) for m in M_GRID}
    sel["M_gate"] = min(M_GRID, key=lambda m: g_gate[str(m)])
    g_ct = {f"{t},{al}": bce(X.wmean(mixb["e"], Y.cons_w(mixb["ll"], t) * mixb["beta"] ** al)) for t in TAU_GRID for al in CT_ALPHA}
    best = min(((t, al) for t in TAU_GRID for al in CT_ALPHA), key=lambda z: g_ct[f"{z[0]},{z[1]}"])
    sel["tau_ct"], sel["alpha_ct"] = best
    g_st = {str(g): bce(X.wmean(C["e"], X.stiff_w(C["H"], g))) for g in GAMMA_GRID}
    sel["gamma"] = min(GAMMA_GRID, key=lambda g: g_st[str(g)])
    res.update(selected=sel, n_sel_roots=int(sel_roots.shape[0]), grid_cons=g_cons, grid_trust=g_tp, grid_gate=g_gate,
               grid_cons_trust=g_ct, grid_stiff=g_st,
               logloss_clean={"unif": bce(C["e"].mean(1)), "stiff2": bce(X.wmean(C["e"], X.stiff_w(C["H"], 2.0)))},
               logloss_mixed={"unif": bce(mixb["e"].mean(1)), "stiff2": bce(X.wmean(mixb["e"], X.stiff_w(mixb["H"], 2.0))),
                              "clean10": bce(C["e"][:, :10].mean(1))})
    return tr, res


# --------------------------------------------------------------------------------------------- D2
def delusional(qh, qstar_eff, pos, n):
    out = []
    for i in range(n):
        m = pos == i
        j = int(np.argmax(qh[m]))
        out.append(bool(qh[m][j] > 0.80 and qstar_eff[m][j] < 0.30))
    return out


@torch.no_grad()
def part_d2(a, Mo, tr, sres, R, dev, eo):
    """240 Q* test positions x every legal move. Per (s, a): 16 clean determinized rollouts (Z ~ belief at t-4) c1..c16
    and 6 noisy ones (3 myopic depth-1, 3 depth-4 at policy temperature 3.5; corrupted cone types). Clean bag = c1..c16
    (E3-D2 (1)); mixed bag = c1..c10 + the 6 noisy (E3-D2 (2)); the tier is never an input."""
    sel = sres["selected"]
    o = np.load(os.path.join(DATA, "oracle_q_test.npz"))
    n = 8 if a.smoke else len(o["t"])
    rows = EV.rows_for(o, EV.row_index(R), dev, n)
    lg = R["legal"][rows].long().cpu().numpy()
    assert np.array_equal(lg[..., 0] >= 0, o["canon"][:n, :, 0] >= 0) and \
        np.array_equal(lg[lg[..., 0] >= 0], o["canon"][:n][o["canon"][:n, :, 0] >= 0]), "legal order mismatch"
    qs = o["qstar"][:n]
    nl = o["n_legal"][:n]
    gen = torch.Generator(device=dev).manual_seed(a.seed + 870)
    pairs = [(i, j) for i in range(n) for j in range(int(nl[i]))]
    KEEP = ("e", "H", "q_leaf", "ll", "beta_r5", "cone_lp", "ply_lp", "terminal", "flag", "offsup", "n_ply")
    Z = {k: [] for k in KEEP + ("beta",)}
    for c0 in range(0, len(pairs), 512):
        pp = pairs[c0:c0 + 512]
        ix = rows[torch.tensor([p[0] for p in pp], device=dev)]
        act = torch.tensor([p[1] for p in pp], device=dev)
        c, n1, n4 = bags(Mo, R, ix, act, gen, dev, N_CLEAN)
        for oo in (c, n1, n4):
            oo["beta"] = X.beta_new(tr, oo["feats_x"])
        for k in Z:
            Z[k].append(torch.cat([c[k], n1[k], n4[k]], 1).float().cpu())
    Z = {k: torch.cat(v).double().numpy() for k, v in Z.items()}
    P_ = len(pairs)
    Pq = np.stack([qs[i, j] for i, j in pairs])
    qstar = Pq[:, 0] + 0.5 * Pq[:, 1]
    Z["kl"] = EV.kl_rows(Pq[:, None, :], Z["q_leaf"])
    tier = np.array([0] * N_CLEAN + [1] * N_N1 + [2] * N_N4)
    clean = slice(0, N_CLEAN)
    mixi = np.r_[0:10, N_CLEAN:N_CLEAN + N_N1 + N_N4]
    tt = torch.tensor
    qs_t = tt(qstar)
    res = {"n_positions": int(n), "n_pairs": P_, "selected": sel}
    # ---- (1) clean bags
    e, H, b, ll = (tt(Z[k][:, clean]) for k in ("e", "H", "beta", "ll"))
    W1 = {"unif": torch.ones_like(e), "stiff2": X.stiff_w(H, 2.0), "stiff_fit": X.stiff_w(H, sel["gamma"]),
          "trust_pool": b ** sel["alpha_trust"], "cons_pool": Y.cons_w(ll, sel["tau_cons"]),
          "cons_trust_pool": Y.cons_w(ll, sel["tau_ct"]) * b ** sel["alpha_ct"],
          "cons_tau1": Y.cons_w(ll, 1.0)}
    pools = {}
    for k, w in W1.items():
        mse, noise, b2 = Y.bias2(e, w, qs_t)
        res[f"clean_{k}"] = {"mse": mse, "noise": noise, "bias2": b2,
                             "neff": float((w.sum(-1) ** 2 / (w ** 2).sum(-1).clamp_min(1e-12)).mean())}
        pools[f"clean_{k}"] = (X.wmean(e, w)).numpy()
    # ---- (2) mixed bags
    em, Hm, bm, llm, klm = (tt(Z[k][:, mixi]) for k in ("e", "H", "beta", "ll", "kl"))
    y = -klm.numpy().ravel()
    res["spearman_trust"] = EV.spearman(bm.numpy().ravel(), y)
    res["spearman_stiff"] = EV.spearman(-Hm.numpy().ravel(), y)
    res["spearman_trust_r5head"] = EV.spearman(Z["beta_r5"][:, mixi].ravel(), y)
    res["spearman_cons"] = EV.spearman(Z["ll"][:, mixi].ravel(), y)
    wi = {k: [] for k in ("trust", "stiff", "cons")}
    for p in range(P_):
        yy = -klm[p].numpy()
        wi["trust"].append(EV.spearman(bm[p].numpy(), yy))
        wi["stiff"].append(EV.spearman(-Hm[p].numpy(), yy))
        wi["cons"].append(EV.spearman(llm[p].numpy(), yy))
    for k, v in wi.items():
        res[f"within_{k}"] = float(np.nanmean(v))
        res[f"within_{k}_n"] = int(np.isfinite(v).sum())
    Hr = Hm.numpy().ravel()
    order = np.argsort(Hr, kind="stable")
    strata = np.empty(y.size, int)
    strata[order] = (np.arange(y.size) * 10) // y.size
    st_t = [EV.spearman(bm.numpy().ravel()[strata == s], y[strata == s]) for s in range(10)]
    st_s = [EV.spearman(-Hr[strata == s], y[strata == s]) for s in range(10)]
    res.update(strata_trust=float(np.nanmean(st_t)), strata_stiff=float(np.nanmean(st_s)), strata_trust_each=st_t,
               strata_stiff_each=st_s)
    W2 = {"unif": torch.ones_like(em), "stiff2": X.stiff_w(Hm, 2.0), "trust_gate": Y.drop_bottom(bm, sel["M_gate"]),
          "cons_trust_pool": Y.cons_w(llm, sel["tau_ct"]) * bm ** sel["alpha_ct"],
          "cons_pool": Y.cons_w(llm, sel["tau_cons"]), "trust_pool": bm ** sel["alpha_trust"],
          "clean10": torch.cat([torch.ones(P_, 10, dtype=em.dtype), torch.zeros(P_, 6, dtype=em.dtype)], 1)}
    for k, w in W2.items():
        mse, noise, b2 = Y.bias2(em, w, qs_t)
        res[f"mixed_{k}"] = {"mse": mse, "noise": noise, "bias2": b2,
                             "neff": float((w.sum(-1) ** 2 / (w ** 2).sum(-1).clamp_min(1e-12)).mean())}
        pools[f"mixed_{k}"] = (X.wmean(em, w)).numpy()
    gate_drop = Y.drop_bottom(bm, sel["M_gate"]) == 0
    res["gate_dropped_noisy_share"] = float((gate_drop[:, 10:].sum() / gate_drop.sum().clamp_min(1)).item())
    pos = np.array([i for i, _ in pairs])
    actv = np.array([j for _, j in pairs])
    for k, qh in pools.items():
        res[f"{k}"]["delusional"] = float(np.mean(delusional(qh, qstar, pos, n)))
    for nm, sl in (("clean", tier == 0), ("noisy_myopic", tier == 1), ("noisy_tau35", tier == 2)):
        zz = {k: Z[k][:, sl] for k in ("e", "H", "kl", "beta", "beta_r5", "ll", "cone_lp", "ply_lp", "flag", "offsup", "n_ply")}
        res[f"tier_{nm}"] = {"kl": float(zz["kl"].mean()), "sq_err": float(((zz["e"] - qstar[:, None]) ** 2).mean()),
                             "terminal_like": float((zz["H"] < 0.15).mean()), "beta": float(zz["beta"].mean()),
                             "beta_r5": float(zz["beta_r5"].mean()), "ll": float(zz["ll"].mean()),
                             "cone_lp": float(zz["cone_lp"].mean()), "ply_lp": float(zz["ply_lp"].mean()),
                             "flag_end": float(zz["flag"].mean()), "H": float(zz["H"].mean()),
                             "offsup_per_move": float(zz["offsup"].sum() / max(1.0, float(np.clip(zz["n_ply"] - 1, 0, None).sum())))}
    EV.write_csv(os.path.join(eo, "d2_actions.csv"), {"pos": pos, "act": actv, "qstar_eff": qstar,
                                                     **{f"q_{k}": v for k, v in pools.items()}})
    K = N_CLEAN + N_N1 + N_N4
    slot = np.tile(np.arange(K), P_)
    EV.write_csv(os.path.join(eo, "d2_rollouts.csv"),
                 {"pair": np.repeat(np.arange(P_), K), "slot": slot, "tier": np.tile(tier, P_),
                  "in_mixed": np.isin(slot, mixi).astype(int),
                  **{k: Z[k].ravel() for k in ("e", "H", "beta", "beta_r5", "kl", "ll", "cone_lp", "ply_lp", "terminal", "flag")}})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-d1-roots", type=int, default=2000)
    ap.add_argument("--d1-steps", type=int, default=1200)
    ap.add_argument("--n-trust-roots", type=int, default=3000)
    ap.add_argument("--n-sel-roots", type=int, default=1000)
    ap.add_argument("--trust-steps", type=int, default=3000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_d1_roots, a.d1_steps, a.n_trust_roots, a.n_sel_roots, a.trust_steps = 32, 15, 48, 32, 20
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed)
    eo = os.path.join(a.out, "eval")
    os.makedirs(eo, exist_ok=True)
    Mo = EV.Models(a.ckpt, "primary", dev)
    h0 = T.param_hash(Mo.net)
    assert h0 == Mo.meta["param_hash_final"], "checkpoint differs from the R5-E2 run that produced it"
    R = T.load_split("test", dev)
    Rc = T.load_split("cal", dev)
    t0 = time.time()
    key, nets, fres = fit_d1(a, Mo, Rc, dev)
    fres["sec"] = time.time() - t0
    json.dump(fres, open(os.path.join(eo, "fit.json"), "w"), indent=1)
    log("fit", json.dumps({k: v for k, v in fres.items() if not k.endswith("hist")})[:1200])
    t1 = time.time()
    tr, sres = fit_trust_select(a, Mo, Rc, dev)
    sres["sec"] = time.time() - t1
    json.dump(sres, open(os.path.join(eo, "trust.json"), "w"), indent=1, default=float)
    log("trust", json.dumps({k: v for k, v in sres.items() if not k.startswith("grid")}, default=float)[:1500])
    for p, fn in (("d1", lambda: part_d1(a, Mo, key, nets, fres, R, dev, eo)), ("d2", lambda: part_d2(a, Mo, tr, sres, R, dev, eo))):
        t1 = time.time()
        r = fn()
        r["sec"] = time.time() - t1
        json.dump(r, open(os.path.join(eo, f"{p}.json"), "w"), indent=1, default=float)
        log(p, json.dumps({k: v for k, v in r.items() if not isinstance(v, list)}, default=float)[:1500])
    assert T.param_hash(Mo.net) == h0
    json.dump({"param_hash": h0, "unchanged": True, "sec": time.time() - t0}, open(os.path.join(eo, "done.json"), "w"))


if __name__ == "__main__":
    main()
