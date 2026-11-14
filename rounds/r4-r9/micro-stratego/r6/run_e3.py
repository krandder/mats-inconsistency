#!/usr/bin/env python3
"""R6-E3 (E-MS-MARGINAL) for one seed: reuses the R5-E2 primary checkpoints (no retraining of theta_real, theta_trust or
theta_amp), fits the pooling rules on the CALIBRATION split against model-only references, then scores E3-D0..D2 on
held-out data. The exact oracle (P*_full, Q*) is read only by the scoring functions (part_d0, part_d1, part_d2).
   python3 run_e3.py --seed 101 --ckpt /tmp/r5e2/ckpt/s101/primary --out OUT [--smoke]
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
import e3lib as L  # noqa: E402
sys.path.insert(0, L.R5)
import eval6 as EV  # noqa: E402
import model6 as M  # noqa: E402
import train6 as T  # noqa: E402

DATA = T.DATA
GAMMA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------------------------- calibration fits
def fit_aggregators(a, Mo, Rc, dev):
    gen = torch.Generator(device=dev).manual_seed(a.seed + 610)
    roots = T.two_flank_roots(Rc, a.n_agg_roots, gen, dev)
    net = Mo.net
    pools, refs = [], []
    for c0 in range(0, roots.shape[0], 128):
        ix = roots[c0:c0 + 128]
        rt = L.roots_e3(net, Rc, ix, dev)
        bact = L.base_cover(rt)
        base = L.e3_rollouts(net, Mo.strust, Mo.rtrust, rt, bact, gen, dev)
        fa = T.sample_flank_actions(rt, 32 + 64, gen, dev)                    # [b, 2, 96]
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
    torch.manual_seed(a.seed + 620)
    key = L.KeyK4(net.d).to(dev)
    lu_b = torch.tensor(math.log(0.3), device=dev, requires_grad=True)
    lu_v = torch.tensor(math.log(0.3), device=dev, requires_grad=True)
    fit("kc", a.agg_steps, lambda b: M.agg_kc(b, key(b), lu_b.exp(), lu_v.exp()), list(key.parameters()) + [lu_b, lu_v],
        3e-3, extra=lambda b: L.key_align_loss(key(b), b["sal"], b["valid"]))
    res["u_b"], res["u_v"] = float(lu_b.exp()), float(lu_v.exp())
    for p in key.parameters():
        p.requires_grad_(False)
    torch.manual_seed(a.seed + 630)
    att = M.AttnAgg().to(dev)
    fit("attn", a.agg_steps, lambda b: att(b, key(b)), list(att.parameters()), 1e-3)
    T.freeze(att)
    with torch.no_grad():
        b = bag("beta")
        for nm, fn in (("union", L.union_call), ("scaled_mix", lambda b: L.agg_scaled_mix(b, res["alpha_b"], res["alpha_v"])),
                       ("mmd", lambda b: M.agg_mmd(b, res["eta_b"], res["eta_v"])),
                       ("kc", lambda b: M.agg_kc(b, key(b), res["u_b"], res["u_v"])), ("attn", lambda b: att(b, key(b)))):
            lb, lv = fn(b)
            _, l1, l2 = L.joint_loss(lb, lv, ref_lb, ref_v, hidden)
            res[f"cal_fit_{nm}"] = [float(l1), float(l2)]
    return key, att, res


@torch.no_grad()
def played_index(R, ix):
    lg = R["legal"][ix].long()
    real = R["ply"][ix, 0, 1].long() * M.NSQ + R["ply"][ix, 0, 2].long()
    return ((lg[..., 0] * M.NSQ + lg[..., 1]) == real[:, None]).float().argmax(1)


@torch.no_grad()
def fit_gamma(a, Mo, Rc, dev):
    """stiff_pool's gamma on CAL, oracle-free: 16 rollouts of the action really played, pooled; log loss of the pooled
    effective win against the realised result; best gamma on GAMMA_GRID. unif and trust are reported on the same data."""
    gen = torch.Generator(device=dev).manual_seed(a.seed + 640)
    ok = (Rc["n_ply"] >= 1) & (Rc["t"] >= 4) & (Rc["t"] <= 36)
    cand = torch.nonzero(ok).squeeze(1)
    roots = cand[torch.randperm(cand.shape[0], generator=gen, device=dev)[:a.n_gamma_roots]]
    E, H, B = [], [], []
    for c0 in range(0, roots.shape[0], 256):
        ix = roots[c0:c0 + 256]
        rt = T.make_roots(Mo.net, Rc, ix, dev)
        act = played_index(Rc, ix)[:, None].expand(-1, 16)
        o = T.run_rollouts(Mo.net, Mo.strust, Mo.rtrust, rt, act, 4, gen, dev, with_belief=False)
        E.append(o["e"])
        H.append(L.entropy3(o["q_leaf"]))
        B.append(o["beta"])
    E, H, B = torch.cat(E), torch.cat(H), torch.cat(B)
    res_ = Rc["result"][roots].long()
    y = (res_ == 0).float() + 0.5 * (res_ == 1).float()
    bce = lambda p: float(torch.nn.functional.binary_cross_entropy(p.clamp(1e-5, 1 - 1e-5), y))
    grid = {str(g): bce(L.pool_values(E, H, B, "stiff", g)) for g in GAMMA_GRID}
    best = min(GAMMA_GRID, key=lambda g: grid[str(g)])
    return {"gamma": best, "grid_logloss": grid, "unif_logloss": bce(L.pool_values(E, H, B, "unif")),
            "trust_logloss": bce(L.pool_values(E, H, B, "trust")), "n_roots": int(roots.shape[0])}


# --------------------------------------------------------------------------------------------- D0
def part_d0(a, Mo, R, Rc, dev, eo):
    res = {}
    acc = EV.battle_acc(Mo.net, R, dev)
    o = np.load(os.path.join(DATA, "oracle_d0_test.npz"))
    rows = EV.rows_for(o, EV.row_index(R), dev)
    kl0, nll7, _ = EV.belief_metrics(Mo.net, R, rows, o["pstar_full"], dev)
    res.update(battle_acc_ply0=float(acc.mean()), n_battles=int(acc.size), nll_7known=float(nll7.mean()),
               p0_kl_test=float(kl0.mean()))
    EV.write_csv(os.path.join(eo, "d0_battles.csv"), {"correct": acc.astype(int)})
    EV.write_csv(os.path.join(eo, "d0_nll7.csv"), {"nll7": nll7})
    gen = torch.Generator(device=dev).manual_seed(a.seed + 650)
    for split, RR, n in (("cal", Rc, a.n_gate), ("test", R, a.n_gate)):
        oo = np.load(os.path.join(DATA, f"oracle_d0_{split}.npz"))
        rows = EV.rows_for(oo, EV.row_index(RR), dev, n)
        P = oo["pstar_full"][:n]
        out = {k: [] for k in ("kl_anchor", "kl_pool16", "kl_routeb_all", "kl_p0", "kl_scaled16", "covered_share", "n_hidden")}
        for c0 in range(0, rows.shape[0], 256):
            ix = rows[c0:c0 + 256]
            rt = L.roots_e3(Mo.net, RR, ix, dev)
            nl = rt["n_legal"]
            act = torch.minimum((torch.rand(ix.shape[0], 16, generator=gen, device=dev) * nl[:, None]).long(),
                                (nl - 1)[:, None])
            cov = L.cover_mask(L.cones_for(rt, act)) & rt["hidden"][:, None]
            dlb = rt["msg"][:, None] * cov[..., None].float()
            b = {"dlb": dlb, "cov": cov, "e": torch.zeros(cov.shape[:2], device=dev),
                 "valid": torch.ones(cov.shape[:2], dtype=torch.bool, device=dev), "l0b": rt["l0b"]}
            lb16, _, _ = M.coverage_reference(b)
            sm = torch.log_softmax(rt["l0b"] + a.alpha_b * dlb.sum(1), -1)
            hid = rt["hidden"].cpu().numpy()
            Pc = P[c0:c0 + ix.shape[0]]
            f = lambda q: np.where(hid, EV.kl_rows(Pc, q.exp().cpu().numpy()), 0).sum(1) / np.maximum(hid.sum(1), 1)
            out["kl_anchor"].append(f(rt["l0b"]))
            out["kl_pool16"].append(f(lb16))
            out["kl_routeb_all"].append(f(rt["lmu"]))
            out["kl_p0"].append(f(rt["p0b"]))
            out["kl_scaled16"].append(f(sm))
            out["covered_share"].append((cov.any(1).cpu().numpy() & hid).sum(1) / np.maximum(hid.sum(1), 1))
            out["n_hidden"].append(hid.sum(1))
        out = {k: np.concatenate(v) for k, v in out.items()}
        EV.write_csv(os.path.join(eo, f"d0_gate_{split}.csv"), {"game": oo["game"][:n], "t": oo["t"][:n], **out})
        r = {k: float(v.mean()) for k, v in out.items()}
        r["reduction_pool16_vs_anchor"] = 1 - r["kl_pool16"] / r["kl_anchor"]
        r["reduction_routeb_all_vs_anchor"] = 1 - r["kl_routeb_all"] / r["kl_anchor"]
        r["reduction_scaled16_vs_anchor"] = 1 - r["kl_scaled16"] / r["kl_anchor"]
        res[split] = r
    res["gate"] = {"battle": res["battle_acc_ply0"] >= 0.99, "nll7": res["nll_7known"] <= 0.01,
                   "anchor_kl_cal": res["cal"]["kl_anchor"] >= 0.12, "pool16_cut_cal": res["cal"]["reduction_pool16_vs_anchor"] >= 0.40}
    return res


# --------------------------------------------------------------------------------------------- D1
def key_gap(k, fl):
    sim = (k @ k.transpose(1, 2)).cpu().numpy()
    f = fl.cpu().numpy()
    same = f[:, :, None] == f[:, None, :]
    eye = np.eye(f.shape[1], dtype=bool)[None]
    with np.errstate(invalid="ignore"):
        w = np.nanmean(np.where(same & ~eye, sim, np.nan), (1, 2))
        ac = np.nanmean(np.where(~same, sim, np.nan), (1, 2))
    return w, ac


@torch.no_grad()
def part_d1(a, Mo, key, att, fit, R, dev, eo):
    d = np.load(os.path.join(DATA, "oracle_d1_test.npz"))
    n = 48 if a.smoke else len(d["t"])
    roots = EV.rows_for(d, EV.row_index(R), dev, n)
    P = d["pstar_full"][:n]
    gen = torch.Generator(device=dev).manual_seed(a.seed + 660)
    aggs = ("union", "mix", "scaled_mix", "ataraxos_mmd", "kc_dedup", "attn_L3")
    acc = {k: [] for k in ["game", "t", "n_hidden", "kl_anchor", "kl_p0", "kl_routeb_all", "nL_L", "nL_R", "key_within_L",
                          "key_across_L", "key_within_R", "key_across_R", "covered_L", "covered_R"] +
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
            k = key(b)
            outs = {"union": L.union_call(b), "mix": M.agg_mix(b),
                    "scaled_mix": L.agg_scaled_mix(b, fit["alpha_b"], fit["alpha_v"]),
                    "ataraxos_mmd": M.agg_mmd(b, fit["eta_b"], fit["eta_v"]),
                    "kc_dedup": M.agg_kc(b, k, fit["u_b"], fit["u_v"]), "attn_L3": att(b, k)}
            mus[name] = {g: lb.exp().cpu().numpy() for g, (lb, lv) in outs.items()}
            w, ac = key_gap(k, b["fl"])
            acc[f"key_within_{name}"].append(w)
            acc[f"key_across_{name}"].append(ac)
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
    acc["key_gap"] = np.nanmean(np.stack([acc["key_within_L"] - acc["key_across_L"],
                                          acc["key_within_R"] - acc["key_across_R"]]), 0)
    EV.write_csv(os.path.join(eo, "d1_positions.csv"), acc)
    res = {"n": int(n), "key_gap": float(np.nanmean(acc["key_gap"])),
           "key_cos_within": float(np.nanmean(np.concatenate([acc["key_within_L"], acc["key_within_R"]]))),
           "key_cos_across": float(np.nanmean(np.concatenate([acc["key_across_L"], acc["key_across_R"]]))),
           "kl_anchor": float(acc["kl_anchor"].mean()), "kl_p0": float(acc["kl_p0"].mean()),
           "kl_routeb_all": float(acc["kl_routeb_all"].mean()), "covered_L": float(acc["covered_L"].mean()),
           "covered_R": float(acc["covered_R"].mean()), "mean_nL_L": float(acc["nL_L"].mean()),
           "mean_nL_R": float(acc["nL_R"].mean())}
    for g in aggs:
        res[g] = {"jsd": float(acc[f"jsd_{g}"].mean()), "kl": float(acc[f"kl_{g}"].mean())}
    return res


# --------------------------------------------------------------------------------------------- D2
@torch.no_grad()
def part_d2(a, Mo, gamma, R, dev, eo):
    o = np.load(os.path.join(DATA, "oracle_q_test.npz"))
    n = 12 if a.smoke else len(o["t"])
    rows = EV.rows_for(o, EV.row_index(R), dev, n)
    lg = R["legal"][rows].long().cpu().numpy()
    assert np.array_equal(lg[..., 0] >= 0, o["canon"][:n, :, 0] >= 0) and \
        np.array_equal(lg[lg[..., 0] >= 0], o["canon"][:n][o["canon"][:n, :, 0] >= 0]), "legal order mismatch"
    qs = o["qstar"][:n]
    nl = o["n_legal"][:n]
    gen = torch.Generator(device=dev).manual_seed(a.seed + 670)
    pairs = [(i, j) for i in range(n) for j in range(int(nl[i]))]
    K = 16
    E, Q, H, Bt = [], [], [], []
    for c0 in range(0, len(pairs), 512):
        pp = pairs[c0:c0 + 512]
        ix = rows[torch.tensor([p[0] for p in pp], device=dev)]
        act = torch.tensor([p[1] for p in pp], device=dev)[:, None].expand(-1, K)
        rt = T.make_roots(Mo.net, R, ix, dev)
        r = T.run_rollouts(Mo.net, Mo.strust, Mo.rtrust, rt, act, 4, gen, dev, with_belief=False)
        E.append(r["e"].cpu())
        Q.append(r["q_leaf"].cpu())
        H.append(L.entropy3(r["q_leaf"]).cpu())
        Bt.append(r["beta"].cpu())
    E, Q, H, Bt = torch.cat(E), torch.cat(Q), torch.cat(H), torch.cat(Bt)
    Pq = np.stack([qs[i, j] for i, j in pairs])                                   # [pairs, 3]
    qstar_eff = Pq[:, 0] + 0.5 * Pq[:, 1]
    kl = EV.kl_rows(Pq[:, None, :], Q.numpy())                                      # [pairs, K]
    beta, Hn = Bt.numpy(), H.numpy()
    y = -kl.ravel()
    res = {"n_positions": int(n), "n_pairs": len(pairs), "n_rollouts": int(y.size), "gamma": gamma,
           "spearman_trust": EV.spearman(beta.ravel(), y), "spearman_stiff": EV.spearman(-Hn.ravel(), y)}
    order = np.argsort(Hn.ravel(), kind="stable")
    strata = np.empty(y.size, int)
    strata[order] = (np.arange(y.size) * 10) // y.size                             # exactly equal-count (ties by index)
    st_t = [EV.spearman(beta.ravel()[strata == s], y[strata == s]) for s in range(10)]
    st_s = [EV.spearman(-Hn.ravel()[strata == s], y[strata == s]) for s in range(10)]
    res.update(strata_trust=float(np.nanmean(st_t)), strata_stiff=float(np.nanmean(st_s)), strata_trust_each=st_t,
               strata_stiff_each=st_s, strata_sizes=np.bincount(strata, minlength=10).tolist())
    pools = {k: L.pool_values(E, H, Bt, k, gamma).numpy() for k in ("unif", "stiff", "trust")}
    pos = np.array([i for i, _ in pairs])
    act = np.array([j for _, j in pairs])
    for k, qh in pools.items():
        res[f"mse_{k}"] = float(((qh - qstar_eff) ** 2).mean())
        deluded = []
        for i in range(n):
            m = pos == i
            j = int(np.argmax(qh[m]))
            deluded.append(bool(qh[m][j] > 0.80 and qstar_eff[m][j] < 0.30))
        res[f"delusional_{k}"] = float(np.mean(deluded))
        res[f"delusional_count_{k}"] = int(np.sum(deluded))
    res["mse_reduction_trust_vs_stiff"] = 1 - res["mse_trust"] / res["mse_stiff"]
    res["terminal_like_share"] = float((Hn < 0.15).mean())
    EV.write_csv(os.path.join(eo, "d2_actions.csv"), {"pos": pos, "act": act, "qstar_eff": qstar_eff,
                                                     **{f"q_{k}": v for k, v in pools.items()}})
    EV.write_csv(os.path.join(eo, "d2_rollouts.csv"), {"pair": np.repeat(np.arange(len(pairs)), K), "beta": beta.ravel(),
                                                      "H": Hn.ravel(), "kl": kl.ravel(), "e": E.numpy().ravel(),
                                                      "strata": strata})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-agg-roots", type=int, default=1500)
    ap.add_argument("--agg-steps", type=int, default=1500)
    ap.add_argument("--n-gamma-roots", type=int, default=1000)
    ap.add_argument("--n-gate", type=int, default=1000)
    ap.add_argument("--parts", default="fit,d0,d1,d2")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_agg_roots, a.agg_steps, a.n_gamma_roots, a.n_gate = 32, 20, 32, 48
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
    key, att, fitres = fit_aggregators(a, Mo, Rc, dev)
    fitres["stiff"] = fit_gamma(a, Mo, Rc, dev)
    fitres["sec"] = time.time() - t0
    a.alpha_b = fitres["alpha_b"]
    json.dump(fitres, open(os.path.join(eo, "fit.json"), "w"), indent=1)
    log("fit", json.dumps({k: v for k, v in fitres.items() if not k.endswith("hist")})[:900])
    for p, fn in (("d0", lambda: part_d0(a, Mo, R, Rc, dev, eo)), ("d1", lambda: part_d1(a, Mo, key, att, fitres, R, dev, eo)),
                  ("d2", lambda: part_d2(a, Mo, fitres["stiff"]["gamma"], R, dev, eo))):
        t1 = time.time()
        r = fn()
        r["sec"] = time.time() - t1
        json.dump(r, open(os.path.join(eo, f"{p}.json"), "w"), indent=1, default=float)
        log(p, json.dumps({k: v for k, v in r.items() if not isinstance(v, list)}, default=float)[:1500])
    assert T.param_hash(Mo.net) == h0
    json.dump({"param_hash": h0, "unchanged": True, "sec": time.time() - t0}, open(os.path.join(eo, "done.json"), "w"))


if __name__ == "__main__":
    main()
