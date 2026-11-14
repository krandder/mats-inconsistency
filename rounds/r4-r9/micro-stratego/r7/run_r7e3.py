#!/usr/bin/env python3
"""R7-E3 (E-MS-TRUST-SEARCH) for one seed. Reuses the R5-E2 primary checkpoints (theta_real, slot trust, the R5-E2
rollout-trust head; nothing of theta_real is retrained) and r6's marginal-window code. Everything that is fitted or
selected (scaled_mix, ataraxos_mmd, K3/K3c key heads with kc_dedup, attn_L3, the new theta_trust head, the pooling
hyperparameters) uses the CALIBRATION split only, against model-only references or realised game results. The exact
oracle (P*_full, Q*) is read only by part_d0 (r6), part_d1 and part_d2.
   python3 run_r7e3.py --seed 101 --ckpt /tmp/r5e2/ckpt/s101/primary --out OUT [--smoke]
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
import r7lib as X  # noqa: E402  (puts r6/ and r5/ on sys.path)
import e3lib as L  # noqa: E402
import run_e3 as R6  # noqa: E402  (r6 runner: part_d0, played_index, key_gap)
import eval6 as EV  # noqa: E402
import model6 as M  # noqa: E402
import train6 as T  # noqa: E402

DATA = T.DATA
HEADS = ("K3", "K3c")
GAMMA_GRID = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
ALPHA_GRID = (0.5, 1.0, 2.0)
GATE_Q = (0.25, 0.35, 0.50)
GATE_TAU = (0.1, 0.25, 0.5)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------------------------- calibration: aggregators
def fit_aggregators(a, Mo, Rc, dev):
    """r6 run_e3.fit_aggregators (same pools, references, scaled_mix and ataraxos_mmd fits) with the K4 head replaced by
    two heads, K3 and K3c, each fitted jointly with its kc_dedup (+ L_key to its own salience target) and then given its
    own attn_L3. Also the calibration key gap of each head on 85/15 bags (model-only)."""
    gen = torch.Generator(device=dev).manual_seed(a.seed + 610)
    roots = T.two_flank_roots(Rc, a.n_agg_roots, gen, dev)
    net = Mo.net
    pools, refs = [], []
    for c0 in range(0, roots.shape[0], 128):
        ix = roots[c0:c0 + 128]
        rt = L.roots_e3(net, Rc, ix, dev)
        bact = L.base_cover(rt)
        base = L.e3_rollouts(net, Mo.strust, Mo.rtrust, rt, bact, gen, dev)
        fa = T.sample_flank_actions(rt, 32 + 64, gen, dev)
        pL = L.e3_rollouts(net, Mo.strust, Mo.rtrust, rt, fa[:, 0, :32], gen, dev)
        pR = L.e3_rollouts(net, Mo.strust, Mo.rtrust, rt, fa[:, 1, :32], gen, dev)
        rL = L.e3_rollouts(net, Mo.strust, Mo.rtrust, rt, fa[:, 0, 32:], gen, dev)
        rR = L.e3_rollouts(net, Mo.strust, Mo.rtrust, rt, fa[:, 1, 32:], gen, dev)
        refb = L.finish_bag(L.bag_cat([base, rL, rR]), rt)
        ref_lb, _, _ = M.coverage_reference(refb)
        ref_v = torch.cat([rL["e"], rR["e"]], 1).mean(1)
        keep = {k: rt[k] for k in ("l0b", "l0v", "hidden")}
        pools.append({"base": {k: base[k] for k in L.BAG_KEYS}, "L": {k: pL[k] for k in L.BAG_KEYS},
                      "R": {k: pR[k] for k in L.BAG_KEYS}, "rt": keep})
        refs.append((ref_lb, ref_v))
    cat = lambda f: torch.cat([f(p) for p in pools])
    base = {k: cat(lambda p: p["base"][k]) for k in L.BAG_KEYS}
    poolL = {k: cat(lambda p: p["L"][k]) for k in L.BAG_KEYS}
    poolR = {k: cat(lambda p: p["R"][k]) for k in L.BAG_KEYS}
    rt = {k: cat(lambda p: p["rt"][k]) for k in ("l0b", "l0v", "hidden")}
    ref_lb = torch.cat([r[0] for r in refs])
    ref_v = torch.cat([r[1] for r in refs])
    hidden = rt["hidden"]
    N = ref_lb.shape[0]
    res = {"n_roots": int(N)}

    def bag(skew):
        pL = torch.full((N,), 0.5, device=dev) if skew == "unskewed" else \
            torch.distributions.Beta(0.5, 0.5).sample((N,)).to(dev)
        b, _ = L.skewed_bag(base, poolL, poolR, rt, pL, gen)
        return b

    def fit(name, n_steps, fn, params, lr, skew="beta", extra=None):
        opt = torch.optim.Adam(params, lr=lr)
        hist = []
        for s in range(n_steps):
            b = bag(skew)
            lb, lv = fn(b)
            loss, l1, l2 = L.joint_loss(lb, lv, ref_lb, ref_v, hidden)
            if extra is not None:
                loss = loss + extra(b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if s % 250 == 0 or s == n_steps - 1:
                hist.append([s, float(loss), float(l1), float(l2)])
        res[name + "_hist"] = hist

    la_b = torch.tensor(math.log(0.3), device=dev, requires_grad=True)
    la_v = torch.tensor(math.log(0.05), device=dev, requires_grad=True)
    fit("scaled_mix", a.agg_steps, lambda b: L.agg_scaled_mix(b, la_b.exp(), la_v.exp()), [la_b, la_v], 1e-2, skew="unskewed")
    res["alpha_b"], res["alpha_v"] = float(la_b.exp()), float(la_v.exp())
    le_b = torch.tensor(math.log(0.05), device=dev, requires_grad=True)
    le_v = torch.tensor(math.log(0.05), device=dev, requires_grad=True)
    fit("mmd", a.agg_steps, lambda b: M.agg_mmd(b, le_b.exp(), le_v.exp()), [le_b, le_v], 1e-2)
    res["eta_b"], res["eta_v"] = float(le_b.exp()), float(le_v.exp())
    keys, atts = {}, {}
    for h, cls, so in (("K3", X.KeyK3, 620), ("K3c", X.KeyK3c, 624)):
        torch.manual_seed(a.seed + so)
        key = cls().to(dev)
        lu_b = torch.tensor(math.log(0.3), device=dev, requires_grad=True)
        lu_v = torch.tensor(math.log(0.3), device=dev, requires_grad=True)
        fit(f"kc_{h}", a.agg_steps, lambda b, key=key, lu_b=lu_b, lu_v=lu_v: M.agg_kc(b, key(b), lu_b.exp(), lu_v.exp()),
            list(key.parameters()) + [lu_b, lu_v], 3e-3,
            extra=lambda b, key=key: L.key_align_loss(key(b), key.target(b), b["valid"]))
        res[f"u_b_{h}"], res[f"u_v_{h}"] = float(lu_b.exp()), float(lu_v.exp())
        for p in key.parameters():
            p.requires_grad_(False)
        torch.manual_seed(a.seed + so + 10)
        att = M.AttnAgg().to(dev)
        fit(f"attn_{h}", a.agg_steps, lambda b, key=key, att=att: att(b, key(b)), list(att.parameters()), 1e-3)
        T.freeze(att)
        keys[h], atts[h] = key, att
    with torch.no_grad():
        bags = [L.skewed_bag(base, poolL, poolR, rt, torch.full((N,), pL, device=dev), gen)[0] for pL in (0.85, 0.15)]
        for h in HEADS:
            g = []
            for b in bags:
                w, ac = R6.key_gap(keys[h](b), b["fl"])
                g.append(w - ac)
            with np.errstate(invalid="ignore"):
                res[f"cal_key_gap_{h}"] = float(np.nanmean(np.nanmean(np.stack(g), 0)))
        b = bag("beta")
        fns = [("union", L.union_call), ("scaled_mix", lambda b: L.agg_scaled_mix(b, res["alpha_b"], res["alpha_v"])),
               ("mmd", lambda b: M.agg_mmd(b, res["eta_b"], res["eta_v"]))]
        for h in HEADS:
            fns += [(f"kc_{h}", lambda b, h=h: M.agg_kc(b, keys[h](b), res[f"u_b_{h}"], res[f"u_v_{h}"])),
                    (f"attn_{h}", lambda b, h=h: atts[h](b, keys[h](b)))]
        for nm, fn in fns:
            lb, lv = fn(b)
            _, l1, l2 = L.joint_loss(lb, lv, ref_lb, ref_v, hidden)
            res[f"cal_fit_{nm}"] = [float(l1), float(l2)]
    return keys, atts, res


# --------------------------------------------------------------------------------------------- calibration: theta_trust
def mixed_rollouts(Mo, R, ix, act, n_clean, n_x, gen, dev):
    rt = T.make_roots(Mo.net, R, ix, dev)
    oc = X.run_tier(Mo.net, Mo.strust, Mo.rtrust, rt, act[:, None].expand(-1, n_clean), gen, dev, False)
    ox = X.run_tier(Mo.net, Mo.strust, Mo.rtrust, rt, act[:, None].expand(-1, n_x), gen, dev, True)
    return oc, ox


def fit_trust_select(a, Mo, Rc, dev):
    """New theta_trust: r5's rollout-trust regression (target: excess surprise of the leaf claim against the REALISED
    result) on calibration rollouts of the move really played, 8 clean + 8 exploratory per root, features = r5's trust
    features + cone_lp + ply_lp (no tier label). Then, on 1,000 DISJOINT calibration roots (12 clean + 8 exploratory
    rollouts of the played move), the pooling hyperparameters are chosen by the log loss of the pooled effective win
    against the realised result: alpha (trust_pool), (q, tau_g) (trust_gate), and gamma (stiffness, descriptive)."""
    gen = torch.Generator(device=dev).manual_seed(a.seed + 700)
    ok = (Rc["n_ply"] >= 1) & (Rc["t"] >= 4) & (Rc["t"] <= 36)
    cand = torch.nonzero(ok).squeeze(1)
    perm = cand[torch.randperm(cand.shape[0], generator=gen, device=dev)]
    tr_roots = perm[:a.n_trust_roots]
    sel_roots = perm[a.n_trust_roots:a.n_trust_roots + a.n_sel_roots]
    Xs, Ys, Ts = [], [], []
    for c0 in range(0, tr_roots.shape[0], 256):
        ix = tr_roots[c0:c0 + 256]
        y = Rc["result"][ix].long()
        oc, ox = mixed_rollouts(Mo, Rc, ix, R6.played_index(Rc, ix), 8, 8, gen, dev)
        for tier, o in ((0, oc), (1, ox)):
            Xs.append(o["feats_x"].reshape(-1, o["feats_x"].shape[-1]).float())
            Ys.append(M.excess_surprise(o["q_leaf"].reshape(-1, 3), y.repeat_interleave(o["e"].shape[1])))
            Ts.append(torch.full((o["e"].numel(),), tier, device=dev))
    Xs, Ys, Ts = torch.cat(Xs), torch.cat(Ys), torch.cat(Ts)
    torch.manual_seed(a.seed + 710)
    tr, hist = X.fit_trust(Xs, Ys, gen, dev, steps=a.trust_steps)
    bt = X.beta_new(tr, Xs)
    res = {"n_train_roots": int(tr_roots.shape[0]), "n_train_rollouts": int(Ys.shape[0]), "trust_hist": hist,
           "target_mean_clean": float(Ys[Ts == 0].mean()), "target_mean_explore": float(Ys[Ts == 1].mean()),
           "train_beta_mean_clean": float(bt[Ts == 0].mean()), "train_beta_mean_explore": float(bt[Ts == 1].mean()),
           "train_spearman_beta_neg_target": EV.spearman(bt.cpu().numpy(), -Ys.cpu().numpy())}
    E, H, B, B5, Ec, Bc = [], [], [], [], [], []
    Yr = []
    for c0 in range(0, sel_roots.shape[0], 256):
        ix = sel_roots[c0:c0 + 256]
        oc, ox = mixed_rollouts(Mo, Rc, ix, R6.played_index(Rc, ix), 12, 8, gen, dev)
        bc, bx = X.beta_new(tr, oc["feats_x"]), X.beta_new(tr, ox["feats_x"])
        E.append(torch.cat([oc["e"][:, :8], ox["e"]], 1))
        H.append(torch.cat([oc["H"][:, :8], ox["H"]], 1))
        B.append(torch.cat([bc[:, :8], bx], 1))
        B5.append(torch.cat([oc["beta_r5"][:, :8], ox["beta_r5"]], 1))
        Ec.append(oc["e"])
        Bc.append(bc)
        Yr.append(Rc["result"][ix].long())
    E, H, B, B5, Ec, Bc, Yr = (torch.cat(v) for v in (E, H, B, B5, Ec, Bc, Yr))
    y = (Yr == 0).float() + 0.5 * (Yr == 1).float()
    bce = lambda p: float(torch.nn.functional.binary_cross_entropy(p.clamp(1e-5, 1 - 1e-5), y))
    g_st = {str(g): bce(X.wmean(E, X.stiff_w(H, g))) for g in GAMMA_GRID}
    g_al = {str(al): bce(X.wmean(E, B ** al)) for al in ALPHA_GRID}
    g_ga = {f"{q},{t}": bce(X.wmean(E, X.gate_w(B, q, t))) for q in GATE_Q for t in GATE_TAU}
    gamma = min(GAMMA_GRID, key=lambda g: g_st[str(g)])
    alpha = min(ALPHA_GRID, key=lambda al: g_al[str(al)])
    qt = min(((q, t) for q in GATE_Q for t in GATE_TAU), key=lambda z: g_ga[f"{z[0]},{z[1]}"])
    resample = X.prune_mean(torch.cat([E[:, :6], E[:, 8:14]], 1), torch.cat([B[:, :6], B[:, 8:14]], 1), 4, Ec[:, 8:12])
    res.update(n_sel_roots=int(sel_roots.shape[0]), gamma=gamma, alpha=alpha, gate_q=qt[0], gate_tau=qt[1],
               grid_stiff=g_st, grid_alpha=g_al, grid_gate=g_ga,
               logloss={"unif": bce(E.mean(1)), "stiff2": bce(X.wmean(E, X.stiff_w(H, 2.0))),
                        "trust_pool": g_al[str(alpha)], "trust_gate": g_ga[f"{qt[0]},{qt[1]}"],
                        "trust_resample": bce(resample), "clean8": bce(E[:, :8].mean(1)),
                        "trust_pool_r5head": bce(X.wmean(E, B5))},
               sel_beta_mean_clean=float(B[:, :8].mean()), sel_beta_mean_explore=float(B[:, 8:].mean()),
               sel_H_mean_clean=float(H[:, :8].mean()), sel_H_mean_explore=float(H[:, 8:].mean()))
    return tr, res


# --------------------------------------------------------------------------------------------- D1
@torch.no_grad()
def part_d1(a, Mo, keys, atts, fit, R, dev, eo):
    """r6 run_e3.part_d1 (same positions, base cover, pools and 85/15 bags) with both key heads."""
    d = np.load(os.path.join(DATA, "oracle_d1_test.npz"))
    n = 48 if a.smoke else len(d["t"])
    roots = EV.rows_for(d, EV.row_index(R), dev, n)
    P = d["pstar_full"][:n]
    gen = torch.Generator(device=dev).manual_seed(a.seed + 660)
    aggs = ("union", "mix", "scaled_mix", "ataraxos_mmd") + tuple(f"kc_dedup_{h}" for h in HEADS) + \
        tuple(f"attn_L3_{h}" for h in HEADS)
    acc = {k: [] for k in ["game", "t", "n_hidden", "kl_anchor", "kl_p0", "kl_routeb_all", "nL_L", "nL_R", "covered_L",
                          "covered_R"] + [f"key_{w}_{h}_{s}" for h in HEADS for w in ("within", "across") for s in "LR"] +
           [f"{m}_{g}" for g in aggs for m in ("jsd", "kl")]}
    for c0 in range(0, n, 64):
        ix = roots[c0:c0 + 64]
        B = ix.shape[0]
        rt = L.roots_e3(Mo.net, R, ix, dev)
        base = L.e3_rollouts(Mo.net, Mo.strust, Mo.rtrust, rt, L.base_cover(rt), gen, dev)
        mus = {}
        hid = rt["hidden"].cpu().numpy()
        Pc = P[c0:c0 + B]
        for name, pL in (("L", 0.85), ("R", 0.15)):
            fa = T.sample_flank_actions(rt, 24, gen, dev)
            poolL = L.e3_rollouts(Mo.net, Mo.strust, Mo.rtrust, rt, fa[:, 0], gen, dev)
            poolR = L.e3_rollouts(Mo.net, Mo.strust, Mo.rtrust, rt, fa[:, 1], gen, dev)
            b, nL = L.skewed_bag({k: base[k] for k in L.BAG_KEYS}, poolL, poolR, rt, torch.full((B,), pL, device=dev), gen)
            ks = {h: keys[h](b) for h in HEADS}
            outs = {"union": L.union_call(b), "mix": M.agg_mix(b),
                    "scaled_mix": L.agg_scaled_mix(b, fit["alpha_b"], fit["alpha_v"]),
                    "ataraxos_mmd": M.agg_mmd(b, fit["eta_b"], fit["eta_v"])}
            for h in HEADS:
                outs[f"kc_dedup_{h}"] = M.agg_kc(b, ks[h], fit[f"u_b_{h}"], fit[f"u_v_{h}"])
                outs[f"attn_L3_{h}"] = atts[h](b, ks[h])
            mus[name] = {g: lb.exp().cpu().numpy() for g, (lb, lv) in outs.items()}
            for h in HEADS:
                w, ac = R6.key_gap(ks[h], b["fl"])
                acc[f"key_within_{h}_{name}"].append(w)
                acc[f"key_across_{h}_{name}"].append(ac)
            acc[f"nL_{name}"].append(nL.cpu().numpy())
            acc[f"covered_{name}"].append((b["cov"].any(1).cpu().numpy() & hid).sum(1) / np.maximum(hid.sum(1), 1))
        nh = np.maximum(hid.sum(1), 1)
        perslot = lambda x: np.where(hid, x, 0).sum(1) / nh
        for g in aggs:
            acc[f"jsd_{g}"].append(perslot(EV.jsd_cat(mus["L"][g], mus["R"][g])))
            acc[f"kl_{g}"].append(0.5 * (perslot(EV.kl_rows(Pc, mus["L"][g])) + perslot(EV.kl_rows(Pc, mus["R"][g]))))
        acc["kl_anchor"].append(perslot(EV.kl_rows(Pc, rt["l0b"].exp().cpu().numpy())))
        acc["kl_p0"].append(perslot(EV.kl_rows(Pc, rt["p0b"].exp().cpu().numpy())))
        acc["kl_routeb_all"].append(perslot(EV.kl_rows(Pc, rt["lmu"].exp().cpu().numpy())))
        acc["n_hidden"].append(hid.sum(1))
        acc["game"].append(d["game"][c0:c0 + B])
        acc["t"].append(d["t"][c0:c0 + B])
    acc = {k: np.concatenate(v) for k, v in acc.items()}
    res = {"n": int(n)}
    with np.errstate(invalid="ignore"):
        for h in HEADS:
            acc[f"key_gap_{h}"] = np.nanmean(np.stack([acc[f"key_within_{h}_L"] - acc[f"key_across_{h}_L"],
                                                       acc[f"key_within_{h}_R"] - acc[f"key_across_{h}_R"]]), 0)
            res[f"key_gap_{h}"] = float(np.nanmean(acc[f"key_gap_{h}"]))
            res[f"key_cos_within_{h}"] = float(np.nanmean(np.concatenate([acc[f"key_within_{h}_L"], acc[f"key_within_{h}_R"]])))
            res[f"key_cos_across_{h}"] = float(np.nanmean(np.concatenate([acc[f"key_across_{h}_L"], acc[f"key_across_{h}_R"]])))
    EV.write_csv(os.path.join(eo, "d1_positions.csv"), acc)
    res.update({"kl_anchor": float(acc["kl_anchor"].mean()), "kl_p0": float(acc["kl_p0"].mean()),
                "kl_routeb_all": float(acc["kl_routeb_all"].mean()), "covered_L": float(acc["covered_L"].mean()),
                "covered_R": float(acc["covered_R"].mean()), "mean_nL_L": float(acc["nL_L"].mean()),
                "mean_nL_R": float(acc["nL_R"].mean())})
    for g in aggs:
        res[g] = {"jsd": float(acc[f"jsd_{g}"].mean()), "kl": float(acc[f"kl_{g}"].mean())}
    return res


# --------------------------------------------------------------------------------------------- D2
def delusional(qh, qstar_eff, pos, n):
    out = []
    for i in range(n):
        m = pos == i
        j = int(np.argmax(qh[m]))
        out.append(bool(qh[m][j] > 0.80 and qstar_eff[m][j] < 0.30))
    return out


def neff(w):
    return float((w.sum(-1) ** 2 / (w ** 2).sum(-1).clamp_min(1e-12)).mean())


@torch.no_grad()
def part_d2(a, Mo, tr, sel, R, dev, eo):
    """240 Q* test positions x every legal move. Per (s, a): 16 clean rollouts c1..c16 and 8 exploratory x1..x8.
    Mixed-reliability bag = c1..c8 + x1..x8 (scored for E3-D2). trust_resample = c1..c6 + x1..x6, prune the 4 lowest
    new-head beta, add c9..c12. Homogeneous clean bag c1..c16 (descriptive trust_trim12)."""
    o = np.load(os.path.join(DATA, "oracle_q_test.npz"))
    n = 12 if a.smoke else len(o["t"])
    rows = EV.rows_for(o, EV.row_index(R), dev, n)
    lg = R["legal"][rows].long().cpu().numpy()
    assert np.array_equal(lg[..., 0] >= 0, o["canon"][:n, :, 0] >= 0) and \
        np.array_equal(lg[lg[..., 0] >= 0], o["canon"][:n][o["canon"][:n, :, 0] >= 0]), "legal order mismatch"
    qs = o["qstar"][:n]
    nl = o["n_legal"][:n]
    gen = torch.Generator(device=dev).manual_seed(a.seed + 770)
    pairs = [(i, j) for i in range(n) for j in range(int(nl[i]))]
    KEEP = ("e", "H", "q_leaf", "beta_r5", "terminal", "flag", "uniq", "offsup", "cone_lp", "ply_lp", "n_ply")
    C = {k: [] for k in KEEP + ("beta",)}
    Xx = {k: [] for k in KEEP + ("beta",)}
    for c0 in range(0, len(pairs), 512):
        pp = pairs[c0:c0 + 512]
        ix = rows[torch.tensor([p[0] for p in pp], device=dev)]
        act = torch.tensor([p[1] for p in pp], device=dev)
        oc, ox = mixed_rollouts(Mo, R, ix, act, 16, 8, gen, dev)
        oc["beta"], ox["beta"] = X.beta_new(tr, oc["feats_x"]), X.beta_new(tr, ox["feats_x"])
        for dst, src in ((C, oc), (Xx, ox)):
            for k in dst:
                dst[k].append(src[k].float().cpu())
    C = {k: torch.cat(v).numpy() for k, v in C.items()}
    Xx = {k: torch.cat(v).numpy() for k, v in Xx.items()}
    P = len(pairs)
    Pq = np.stack([qs[i, j] for i, j in pairs])
    qstar_eff = Pq[:, 0] + 0.5 * Pq[:, 1]
    C["kl"] = EV.kl_rows(Pq[:, None, :], C["q_leaf"])
    Xx["kl"] = EV.kl_rows(Pq[:, None, :], Xx["q_leaf"])
    mix = {k: np.concatenate([C[k][:, :8], Xx[k]], 1) for k in ("e", "H", "beta", "beta_r5", "kl")}
    y = -mix["kl"].ravel()
    res = {"n_positions": int(n), "n_pairs": P, "n_rollouts_mixed": int(y.size), "n_rollouts_total": int(P * 24),
           "selected": {k: sel[k] for k in ("gamma", "alpha", "gate_q", "gate_tau")},
           "spearman_trust": EV.spearman(mix["beta"].ravel(), y), "spearman_stiff": EV.spearman(-mix["H"].ravel(), y),
           "spearman_trust_r5head": EV.spearman(mix["beta_r5"].ravel(), y)}
    within = {k: [] for k in ("trust", "stiff", "trust_r5head")}
    for p in range(P):
        yy = -mix["kl"][p]
        within["trust"].append(EV.spearman(mix["beta"][p], yy))
        within["stiff"].append(EV.spearman(-mix["H"][p], yy))
        within["trust_r5head"].append(EV.spearman(mix["beta_r5"][p], yy))
    for k, v in within.items():
        res[f"within_{k}"] = float(np.nanmean(v))
        res[f"within_{k}_n"] = int(np.isfinite(v).sum())
    Hr = mix["H"].ravel()
    order = np.argsort(Hr, kind="stable")
    strata = np.empty(y.size, int)
    strata[order] = (np.arange(y.size) * 10) // y.size
    st_t = [EV.spearman(mix["beta"].ravel()[strata == s], y[strata == s]) for s in range(10)]
    st_s = [EV.spearman(-Hr[strata == s], y[strata == s]) for s in range(10)]
    res.update(strata_trust=float(np.nanmean(st_t)), strata_stiff=float(np.nanmean(st_s)), strata_trust_each=st_t,
               strata_stiff_each=st_s, strata_sizes=np.bincount(strata, minlength=10).tolist())
    tt = lambda z: torch.tensor(z, dtype=torch.float64)
    e_m, H_m, b_m = tt(mix["e"]), tt(mix["H"]), tt(mix["beta"])
    pools = {"unif": e_m.mean(-1), "stiff2": X.wmean(e_m, X.stiff_w(H_m, 2.0)),
             "stiff_fit": X.wmean(e_m, X.stiff_w(H_m, sel["gamma"])),
             "trust_pool": X.wmean(e_m, b_m ** sel["alpha"]),
             "trust_gate": X.wmean(e_m, X.gate_w(b_m, sel["gate_q"], sel["gate_tau"])),
             "trust_resample": X.prune_mean(tt(np.concatenate([C["e"][:, :6], Xx["e"][:, :6]], 1)),
                                            tt(np.concatenate([C["beta"][:, :6], Xx["beta"][:, :6]], 1)), 4,
                                            tt(C["e"][:, 8:12])),
             "trust_pool_r5head": X.wmean(e_m, tt(mix["beta_r5"])),
             "clean8": tt(C["e"][:, :8]).mean(-1), "explore8": tt(Xx["e"]).mean(-1),
             "clean16": tt(C["e"]).mean(-1), "trust_trim12_clean16": X.prune_mean(tt(C["e"]), tt(C["beta"]), 4),
             "trust_trim12_clean16_r5head": X.prune_mean(tt(C["e"]), tt(C["beta_r5"]), 4),
             "trust_pool_clean16": X.wmean(tt(C["e"]), tt(C["beta"]) ** sel["alpha"])}
    pools = {k: v.numpy() for k, v in pools.items()}
    pos = np.array([i for i, _ in pairs])
    act = np.array([j for _, j in pairs])
    for k, qh in pools.items():
        res[f"mse_{k}"] = float(((qh - qstar_eff) ** 2).mean())
        dd = delusional(qh, qstar_eff, pos, n)
        res[f"delusional_{k}"] = float(np.mean(dd))
        res[f"delusional_count_{k}"] = int(np.sum(dd))
    for k in ("trust_gate", "trust_pool", "trust_resample"):
        res[f"mse_reduction_{k}_vs_stiff2"] = 1 - res[f"mse_{k}"] / res["mse_stiff2"]
        res[f"mse_unif_minus_{k}"] = res["mse_unif"] - res[f"mse_{k}"]
    res["neff_trust_pool"] = neff(b_m ** sel["alpha"])
    res["neff_trust_gate"] = neff(X.gate_w(b_m, sel["gate_q"], sel["gate_tau"]))
    res["neff_stiff2"] = neff(X.stiff_w(H_m, 2.0))
    pr = np.concatenate([C["beta"][:, :6], Xx["beta"][:, :6]], 1).argsort(1, kind="stable")[:, :4]
    res["resample_pruned_explore_share"] = float((pr >= 6).mean())
    for nm, Z in (("clean", C), ("explore", Xx)):
        nmoves = float(np.clip(Z["n_ply"] - 1, 0, None).sum())
        res[f"tier_{nm}"] = {"kl": float(Z["kl"].mean()), "abs_err": float(np.abs(Z["e"] - qstar_eff[:, None]).mean()),
                             "sq_err": float(((Z["e"] - qstar_eff[:, None]) ** 2).mean()),
                             "terminal_like_H_lt_0.15": float((Z["H"] < 0.15).mean()), "flag_end": float(Z["flag"].mean()),
                             "count_violation": float(Z["uniq"].mean()),
                             "offsup_per_move": float(Z["offsup"].sum() / max(1.0, nmoves)),
                             "cone_lp": float(Z["cone_lp"].mean()), "ply_lp": float(Z["ply_lp"].mean()),
                             "beta": float(Z["beta"].mean()), "beta_r5": float(Z["beta_r5"].mean()), "H": float(Z["H"].mean())}
    for nm, Z in (("clean", C), ("explore", Xx)):
        trap = Z["H"] < 0.15
        res[f"{nm}_terminal_like_kl"] = float(Z["kl"][trap].mean()) if trap.any() else float("nan")
        res[f"{nm}_other_kl"] = float(Z["kl"][~trap].mean()) if (~trap).any() else float("nan")
    EV.write_csv(os.path.join(eo, "d2_actions.csv"), {"pos": pos, "act": act, "qstar_eff": qstar_eff,
                                                     **{f"q_{k}": v for k, v in pools.items()}})
    K = 24
    allz = {k: np.concatenate([C[k], Xx[k]], 1) for k in ("e", "H", "beta", "beta_r5", "kl", "terminal", "flag", "uniq",
                                                          "offsup", "cone_lp", "ply_lp")}
    slot = np.tile(np.arange(K), P)
    EV.write_csv(os.path.join(eo, "d2_rollouts.csv"),
                 {"pair": np.repeat(np.arange(P), K), "slot": slot, "tier": (slot >= 16).astype(int),
                  "in_mixed": ((slot < 8) | (slot >= 16)).astype(int), **{k: v.ravel() for k, v in allz.items()}})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-agg-roots", type=int, default=1500)
    ap.add_argument("--agg-steps", type=int, default=1500)
    ap.add_argument("--n-trust-roots", type=int, default=4000)
    ap.add_argument("--n-sel-roots", type=int, default=1000)
    ap.add_argument("--trust-steps", type=int, default=3000)
    ap.add_argument("--n-gate", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_agg_roots, a.agg_steps, a.n_trust_roots, a.n_sel_roots, a.trust_steps, a.n_gate = 32, 20, 48, 32, 20, 48
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
    keys, atts, fitres = fit_aggregators(a, Mo, Rc, dev)
    fitres["sec"] = time.time() - t0
    a.alpha_b = fitres["alpha_b"]
    json.dump(fitres, open(os.path.join(eo, "fit.json"), "w"), indent=1)
    log("fit", json.dumps({k: v for k, v in fitres.items() if not k.endswith("hist")})[:900])
    t1 = time.time()
    tr, sel = fit_trust_select(a, Mo, Rc, dev)
    sel["sec"] = time.time() - t1
    json.dump(sel, open(os.path.join(eo, "trust.json"), "w"), indent=1, default=float)
    log("trust", json.dumps({k: v for k, v in sel.items() if not k.startswith("grid")}, default=float)[:1500])
    for p, fn in (("d0", lambda: R6.part_d0(a, Mo, R, Rc, dev, eo)), ("d1", lambda: part_d1(a, Mo, keys, atts, fitres, R, dev, eo)),
                  ("d2", lambda: part_d2(a, Mo, tr, sel, R, dev, eo))):
        t1 = time.time()
        r = fn()
        r["sec"] = time.time() - t1
        json.dump(r, open(os.path.join(eo, f"{p}.json"), "w"), indent=1, default=float)
        log(p, json.dumps({k: v for k, v in r.items() if not isinstance(v, list)}, default=float)[:1500])
    assert T.param_hash(Mo.net) == h0
    json.dump({"param_hash": h0, "unchanged": True, "sec": time.time() - t0}, open(os.path.join(eo, "done.json"), "w"))


if __name__ == "__main__":
    main()
