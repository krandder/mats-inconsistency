#!/usr/bin/env python3
"""E-MS evaluation for one seed on the TEST split (90 held-out setups). Oracle (P*_full, Q*) and simulator are used here
only for scoring. Writes <out>/eval/<part>.npz (per-item arrays) and <out>/eval/<part>.json (summary).
   python3 eval_ms.py --seed 101 --out runs/ms/s101 --parts d0,d1,d2,d3,d4,d4u,d5 [--smoke]
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
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_ms as M  # noqa: E402
import train_ms as T  # noqa: E402
import agents_ms as A  # noqa: E402

DATA = T.DATA


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def kl_rows(p, q, eps=1e-9):
    p = np.asarray(p, np.float64)
    q = np.clip(np.asarray(q, np.float64), eps, 1)
    return np.where(p > 0, p * (np.log(np.clip(p, eps, 1)) - np.log(q)), 0.0).sum(-1)


def jsd_bern(a, b, eps=1e-9):
    a, b = np.clip(a, eps, 1 - eps), np.clip(b, eps, 1 - eps)
    m = 0.5 * (a + b)
    kl = lambda p, q: p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).correlation)


class Models:
    def __init__(self, out, dev):
        self.dev = dev
        ld = lambda f: torch.load(os.path.join(out, f), map_location=dev)
        self.meta = json.load(open(os.path.join(out, "train_meta.json")))
        self.net = M.RealNet().to(dev)
        self.net.load_state_dict(ld("real_100.pt"))
        self.net30 = M.RealNet().to(dev)
        self.net30.load_state_dict(ld("real_30.pt"))
        self.fused = M.RealNet().to(dev)
        self.fused.load_state_dict(ld("real_fused.pt"))
        self.amp = M.AmpNet(self.net.d).to(dev)
        self.amp.load_state_dict(ld("amp.pt"))
        self.amp_f = M.AmpNet(self.net.d).to(dev)
        self.amp_f.load_state_dict(ld("amp_fused.pt"))
        self.strust = M.TrustNet(self.net.d + 2).to(dev)
        self.strust.load_state_dict(ld("slot_trust.pt"))
        self.strust30 = M.TrustNet(self.net.d + 2).to(dev)
        self.strust30.load_state_dict(ld("slot_trust_30.pt"))
        rt = ld("rollout_trust.pt")
        self.rtrust = M.TrustNet(rt["mu"].shape[0]).to(dev)
        self.rtrust.load_state_dict(rt["state"])
        self.rtrust.mu, self.rtrust.sd = rt["mu"], rt["sd"]
        self.att = M.AttnAgg().to(dev)
        self.att.load_state_dict(ld("attn_l3.pt"))
        for m in (self.net, self.net30, self.fused, self.amp, self.amp_f, self.strust, self.strust30, self.rtrust, self.att):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


def row_index(R):
    g = R["game"].cpu().numpy()
    t = R["t"].cpu().numpy()
    return {(int(a), int(b)): i for i, (a, b) in enumerate(zip(g, t))}


# --------------------------------------------------------------------------------------------- D0
@torch.no_grad()
def belief_metrics(net, R, rows, pstar, chunk=1024):
    """0-step KL(P* || q) per position (mean over hidden slots) and 5-known-slot NLL per (position, hidden slot)."""
    kl0, nll5 = [], []
    for c0 in range(0, rows.shape[0], chunk):
        ix = rows[c0:c0 + chunk]
        tp = M.root_tape(R, ix)
        h, _ = net(tp, root_only=True)
        q = torch.softmax(net.belief_logits(h).float(), -1).cpu().numpy()
        hid = (R["slot_pubrev"][ix] == 0).cpu().numpy()
        P = pstar[c0:c0 + chunk]
        k = kl_rows(P, q)
        kl0.append(np.where(hid, k, 0).sum(1) / np.maximum(hid.sum(1), 1))
        tru = (R["slot_true"][ix] - 1)
        for s in range(6):
            others = torch.ones(ix.shape[0], 6, dtype=torch.bool, device=ix.device)
            others[:, s] = False
            tp5 = M.root_tape(R, ix, reveal_hidden=others)
            tp5["slot_val"][:, s] = 0
            h5, _ = net(tp5, root_only=True)
            q5 = torch.softmax(net.belief_logits(h5).float(), -1)[:, s]
            nl = -torch.log(q5.gather(1, tru[:, s:s + 1]).squeeze(1).clamp_min(1e-9)).cpu().numpy()
            nll5.append(nl[hid[:, s]])
    return np.concatenate(kl0), np.concatenate(nll5)


@torch.no_grad()
def battle_acc(net, R, chunk=2048):
    """Outcome accuracy with Z fully materialised (true types): ply 0 battles (gate) and all-ply battles."""
    ok0, okall = [], []
    N = R["t"].shape[0]
    for c0 in range(0, N, chunk):
        ix = torch.arange(c0, min(N, c0 + chunk), device=R["t"].device)
        tp = M.root_tape(R, ix, reveal_hidden=torch.ones(ix.shape[0], 6, dtype=torch.bool, device=ix.device))
        ply = R["ply"][ix]
        npl = R["n_ply"][ix]
        valid = torch.arange(M.MAXP, device=ix.device)[None] < npl[:, None]
        for c, k in enumerate(("ply_actor", "ply_from", "ply_to", "ply_out", "ply_att", "ply_def")):
            tp[k] = ply[..., c] * valid
        h, _ = net(tp)
        for j in range(M.MAXP):
            lo, _, _ = net.trans_logits(h[:, M.a_idx(j)])
            bat = valid[:, j] & (ply[:, j, 3] >= 1)
            corr = (lo.argmax(1) == ply[:, j, 3])[bat].cpu().numpy()
            okall.append(corr)
            if j == 0:
                ok0.append(corr)
    return np.concatenate(ok0), np.concatenate(okall)


def part_d0(Mo, R, rix, out, net=None, tag="d0"):
    net = net or Mo.net
    o = np.load(os.path.join(DATA, "oracle_d1.npz"))
    rows = torch.tensor([rix[(int(g), int(t))] for g, t in zip(o["game"], o["t"])], device=Mo.dev)
    kl0, nll5 = belief_metrics(net, R, rows, o["pstar_full"])
    kl0c, _ = belief_metrics(net, R, rows, o["pstar_c"])
    a0, aall = battle_acc(net, R)
    hid = (R["slot_pubrev"][rows] == 0).cpu().numpy()
    pf, pc = o["pstar_full"], o["pstar_c"]
    gapc = (np.where(hid, kl_rows(pf, pc), 0).sum(1) / np.maximum(hid.sum(1), 1))
    res = {"battle_acc_ply0": float(a0.mean()), "n_battles_ply0": int(a0.size), "battle_acc_allply": float(aall.mean()),
           "nll_5known": float(nll5.mean()), "n_5known": int(nll5.size), "kl0_full": float(kl0.mean()),
           "kl0_constraint": float(kl0c.mean()), "kl_full_vs_constraint_oracle": float(gapc.mean()), "n_pos": int(rows.shape[0])}
    res["gate"] = {"battle": res["battle_acc_ply0"] >= 0.97, "five_known": res["nll_5known"] <= 0.03,
                   "belief_gap": res["kl0_full"] >= 0.12}
    np.savez(os.path.join(out, f"{tag}.npz"), kl0_full=kl0, kl0_constraint=kl0c, nll5=nll5, acc0=a0, accall=aall)
    return res


# --------------------------------------------------------------------------------------------- D1
def invalid_flags(zfull, flags):
    """zfull [B,6] values 1..6 (complete sample), flags [B,6] (0 none, 1 moved, 2 moved long)."""
    srt = np.sort(zfull, 1)
    notperm = ~(srt == np.arange(1, 7)[None]).all(1)
    bad_move = ((flags >= 1) & np.isin(zfull, [1, 2])).any(1)
    bad_long = ((flags == 2) & (zfull != 3)).any(1)
    return notperm | bad_move | bad_long


@torch.no_grad()
def part_d1(Mo, R, rix, out, seed, smoke=False):
    o = np.load(os.path.join(DATA, "oracle_d1.npz"))
    n = 200 if smoke else len(o["t"])
    rows = torch.tensor([rix[(int(g), int(t))] for g, t in zip(o["game"][:n], o["t"][:n])], device=Mo.dev)
    pf = o["pstar_full"][:n]
    K = 8
    gen = torch.Generator(device=Mo.dev).manual_seed(seed + 100)
    res, arr = {}, {}
    for order in ("row_major", "random", "stiff_voi"):
        kls, inv, mus = [], [], []
        for c0 in range(0, n, 512):
            ix = rows[c0:c0 + 512]
            B = ix.shape[0]
            tp = M.tape_repeat(M.root_tape(R, ix), K)
            tpz, info = M.sample_Z(Mo.net, tp, order, gen, Mo.strust)
            hid = info["hidden"].view(B, K, 6)[:, 0].cpu().numpy()
            cond = info["cond"].view(B, K, 6, 6).mean(1).cpu().numpy()
            k = kl_rows(pf[c0:c0 + B], cond)
            kls.append(np.where(hid, k, 0).sum(1) / np.maximum(hid.sum(1), 1))
            z = tpz["slot_val"].cpu().numpy()
            fl = tp["slot_flags"].cpu().numpy()
            inv.append(invalid_flags(z, fl).reshape(B, K).mean(1))
            mus.append(cond)
        arr[f"kl_{order}"] = np.concatenate(kls)
        arr[f"invalid_{order}"] = np.concatenate(inv)
        res[order] = {"kl": float(arr[f"kl_{order}"].mean()), "invalid_rate": float(arr[f"invalid_{order}"].mean())}
    kl0, _ = belief_metrics(Mo.net, R, rows, pf)
    arr["kl_0step"] = kl0
    res["zero_step_kl"] = float(kl0.mean())
    res["n"] = n
    rm, sv = res["row_major"], res["stiff_voi"]
    res["kl_reduction_vs_row_major"] = 1 - sv["kl"] / rm["kl"]
    res["invalid_reduction_vs_row_major"] = (1 - sv["invalid_rate"] / rm["invalid_rate"]) if rm["invalid_rate"] > 0 else float("nan")
    np.savez(os.path.join(out, "d1.npz"), **arr)
    return res


# --------------------------------------------------------------------------------------------- D2
@torch.no_grad()
def part_d2(Mo, R, rix, out, seed, smoke=False):
    d = np.load(os.path.join(DATA, "d2_pos.npz"))
    n = 64 if smoke else len(d["t"])
    roots = torch.tensor([rix[(int(g), int(t))] for g, t in zip(d["game"][:n], d["t"][:n])], device=Mo.dev)
    gen = torch.Generator(device=Mo.dev).manual_seed(seed + 200)
    pools = [T.flank_rollouts(Mo.net, Mo.strust, Mo.rtrust, Mo.amp, R, roots, 24, gen) for _ in range(2)]
    ref = T.flank_rollouts(Mo.net, Mo.strust, Mo.rtrust, Mo.amp, R, roots, 64, gen)
    vref = ref["q"].reshape(n, -1).mean(1).cpu().numpy()
    K = 24
    meta = Mo.meta["agg"]
    out_mu = {}
    arr = {"vref": vref}
    for name, pL, pool in (("L", 0.8, pools[0]), ("R", 0.2, pools[1])):
        nL = torch.distributions.Binomial(K, torch.full((n,), pL, device=Mo.dev)).sample().long()
        # first nL rollouts from the L pool, the rest from the R pool (all distinct)
        j = torch.arange(K, device=Mo.dev)[None].expand(n, -1)
        fl = (j >= nL[:, None]).long()
        jj = torch.where(fl == 0, j, j - nL[:, None])
        ar = torch.arange(n, device=Mo.dev)[:, None]
        g = lambda x: x[ar, fl, jj]
        b = {"q": g(pool["q"]), "dl": g(pool["dl"]), "beta": g(pool["beta"]), "ebar": pool["ebar"][ar, fl, jj],
             "l0": pool["l0"], "valid": torch.ones(n, K, dtype=torch.bool, device=Mo.dev)}
        k = Mo.amp.key(b["ebar"])
        mu = {"mix": M.agg_mix(b["q"], b["valid"]),
              "ataraxos_mmd": torch.sigmoid(M.agg_mmd(b["l0"], b["dl"], b["beta"], b["valid"], meta["eta"])),
              "kc_dedup": torch.sigmoid(M.agg_kc(b["l0"], b["dl"], b["beta"], k, b["valid"], meta["u_root"])),
              "attn_L3": torch.sigmoid(Mo.att(b["l0"], b["dl"], b["beta"], k, b["valid"]))}
        out_mu[name] = {kk: v.cpu().numpy() for kk, v in mu.items()}
        arr[f"nL_{name}"] = nL.cpu().numpy()
        # key diagnostics: mean cosine within flank vs across flank
        sim = (k @ k.transpose(1, 2)).cpu().numpy()
        f = fl.cpu().numpy()
        same = f[:, :, None] == f[:, None, :]
        eye = np.eye(K, dtype=bool)[None]
        arr[f"cos_within_{name}"] = np.nanmean(np.where(same & ~eye, sim, np.nan), (1, 2))
        arr[f"cos_across_{name}"] = np.nanmean(np.where(~same, sim, np.nan), (1, 2))
    res = {"n": n}
    for agg in ("mix", "ataraxos_mmd", "kc_dedup", "attn_L3"):
        a, b_ = out_mu["L"][agg], out_mu["R"][agg]
        js = jsd_bern(a, b_)
        mse = 0.5 * ((a - vref) ** 2 + (b_ - vref) ** 2)
        arr[f"jsd_{agg}"], arr[f"mse_{agg}"], arr[f"muL_{agg}"], arr[f"muR_{agg}"] = js, mse, a, b_
        res[agg] = {"jsd": float(js.mean()), "mse": float(mse.mean())}
    flank_gap = (pools[0]["q"][:, 0].mean(1) - pools[0]["q"][:, 1].mean(1)).abs().cpu().numpy()
    arr["flank_gap"] = flank_gap
    res["flank_gap_mean"] = float(flank_gap.mean())
    res["key_cos_within"] = float(np.nanmean(arr["cos_within_L"]))
    res["key_cos_across"] = float(np.nanmean(arr["cos_across_L"]))
    np.savez(os.path.join(out, "d2.npz"), **arr)
    return res


# --------------------------------------------------------------------------------------------- D3
@torch.no_grad()
def part_d3(Mo, R, rix, out, seed, smoke=False):
    o = np.load(os.path.join(DATA, "oracle_q.npz"))
    n = 40 if smoke else len(o["t"])
    rows = [rix[(int(g), int(t))] for g, t in zip(o["game"][:n], o["t"][:n])]
    gen = torch.Generator(device=Mo.dev).manual_seed(seed + 300)
    lg = R["legal"][torch.tensor(rows, device=Mo.dev)].cpu().numpy()
    qstar = o["qstar"][:n]
    nl = o["n_legal"][:n]
    # sanity: the recorded legal list order equals the oracle's legal order (canonical from/to)
    assert np.array_equal(lg[..., 0] >= 0, o["canon"][:n, :, 0] >= 0) and np.array_equal(lg[lg[..., 0] >= 0], o["canon"][:n][o["canon"][:n, :, 0] >= 0])
    items = [(i, j, d, ck, rep) for i in range(n) for j in range(int(nl[i])) for d in range(1, 7) for ck in (0, 1) for rep in range(2)]
    arr = {k: [] for k in ("pos", "act", "depth", "ckpt", "beta", "H", "kl", "qeff", "qstar_eff")}
    for ck, (net, st) in enumerate(((Mo.net30, Mo.strust30), (Mo.net, Mo.strust))):
        it = [x for x in items if x[3] == ck]
        pos = torch.tensor([rows[x[0]] for x in it], device=Mo.dev)
        fr = torch.tensor([lg[x[0], x[1], 0] for x in it], device=Mo.dev)
        to = torch.tensor([lg[x[0], x[1], 1] for x in it], device=Mo.dev)
        dd = torch.tensor([x[2] for x in it], device=Mo.dev)
        f, q, l0 = T.trust_rollouts(net, st, R, pos, dd, None, gen, order="random", root_action=(fr, to))
        beta = T.trust_beta(Mo.rtrust, f).cpu().numpy()
        q = q.cpu().numpy()
        P = np.stack([qstar[x[0], x[1]] for x in it])
        arr["pos"].append([x[0] for x in it])
        arr["act"].append([x[1] for x in it])
        arr["depth"].append([x[2] for x in it])
        arr["ckpt"].append([ck] * len(it))
        arr["beta"].append(beta)
        arr["H"].append(-(np.clip(q, 1e-9, 1) * np.log(np.clip(q, 1e-9, 1))).sum(1))
        arr["kl"].append(kl_rows(P, q))
        arr["qeff"].append(q[:, 0] + 0.5 * q[:, 1])
        arr["qstar_eff"].append(P[:, 0] + 0.5 * P[:, 1])
    arr = {k: np.concatenate([np.asarray(x) for x in v]) for k, v in arr.items()}
    y = -arr["kl"]
    res = {"n_rollouts": int(y.size), "spearman_trust": spearman(arr["beta"], y), "spearman_stiff": spearman(-arr["H"], y)}
    qs = np.quantile(arr["H"], np.linspace(0, 1, 11))
    strata = np.clip(np.searchsorted(qs, arr["H"], side="right") - 1, 0, 9)
    st_t, st_s = [], []
    for b in range(10):
        m = strata == b
        st_t.append(spearman(arr["beta"][m], y[m]))
        st_s.append(spearman(-arr["H"][m], y[m]))
    res["strata_trust"] = float(np.nanmean(st_t))
    res["strata_stiff"] = float(np.nanmean(st_s))
    res["strata_trust_each"], res["strata_stiff_each"] = st_t, st_s
    for ck in (0, 1):
        m = arr["ckpt"] == ck
        res[f"ckpt{ck}"] = {"spearman_trust": spearman(arr["beta"][m], y[m]), "spearman_stiff": spearman(-arr["H"][m], y[m]),
                            "mean_kl": float(arr["kl"][m].mean())}
    arr["strata"] = strata
    # delusional move selection: K = 16 rollouts round-robin, depth 4, three weightings
    dz = {k: [] for k in ("trust", "stiff", "uniform")}
    sel_q = {k: [] for k in dz}
    sel_true = {k: [] for k in dz}
    K = 16
    for ck, (net, st) in enumerate(((Mo.net30, Mo.strust30), (Mo.net, Mo.strust))):
        pos = torch.tensor(rows, device=Mo.dev).repeat_interleave(K)
        acts = np.concatenate([np.arange(K) % int(nl[i]) for i in range(n)])
        fr = torch.tensor([lg[i, a, 0] for i in range(n) for a in (np.arange(K) % int(nl[i]))], device=Mo.dev)
        to = torch.tensor([lg[i, a, 1] for i in range(n) for a in (np.arange(K) % int(nl[i]))], device=Mo.dev)
        dd = torch.full((pos.shape[0],), 4, device=Mo.dev)
        f, q, _ = T.trust_rollouts(net, st, R, pos, dd, None, gen, order="random", root_action=(fr, to))
        beta = T.trust_beta(Mo.rtrust, f).cpu().numpy().reshape(n, K)
        q = q.cpu().numpy()
        qe = (q[:, 0] + 0.5 * q[:, 1]).reshape(n, K)
        H = (-(np.clip(q, 1e-9, 1) * np.log(np.clip(q, 1e-9, 1))).sum(1)).reshape(n, K)
        ac = acts.reshape(n, K)
        W = {"trust": beta, "stiff": 1.0 / (H + 0.05), "uniform": np.ones_like(qe)}
        for wn, w in W.items():
            for i in range(n):
                best, bv = None, -1
                for a in range(int(nl[i])):
                    m = ac[i] == a
                    v = float((w[i, m] * qe[i, m]).sum() / w[i, m].sum())
                    if v > bv:
                        best, bv = a, v
                qt = qstar[i, best, 0] + 0.5 * qstar[i, best, 1]
                dz[wn].append(bv > 0.8 and qt < 0.3)
                sel_q[wn].append(bv)
                sel_true[wn].append(qt)
    for wn in dz:
        arr[f"delusional_{wn}"] = np.array(dz[wn])
        arr[f"selclaim_{wn}"] = np.array(sel_q[wn])
        arr[f"seltrue_{wn}"] = np.array(sel_true[wn])
        res[f"delusional_{wn}"] = float(np.mean(dz[wn]))
        res[f"selected_true_value_{wn}"] = float(np.mean(sel_true[wn]))
    np.savez(os.path.join(out, "d3.npz"), **arr)
    return res


# --------------------------------------------------------------------------------------------- D4
@torch.no_grad()
def part_d4(Mo, R, rix, out, seed, smoke=False):
    res = {"lir_param_hash_identical": Mo.meta["lir_param_hash_before"] == Mo.meta["lir_param_hash_after"],
           "lir_param_hash": Mo.meta["lir_param_hash_after"], "fused_param_hash": Mo.meta["fused_param_hash_after"]}
    fresh = M.RealNet().to(Mo.dev)
    fresh.load_state_dict(torch.load(os.path.join(os.path.dirname(out), "real_100.pt"), map_location=Mo.dev))
    fresh.eval()
    res["lir_saved_equals_used"] = T.param_hash(fresh) == Mo.meta["lir_param_hash_after"]
    res["lir_metrics"] = part_d0(Mo, R, rix, out, net=Mo.net, tag="d4_lir")
    res["fused_metrics"] = part_d0(Mo, R, rix, out, net=Mo.fused, tag="d4_fused")
    res["belief_kl_change_fused"] = res["fused_metrics"]["kl0_full"] / res["lir_metrics"]["kl0_full"] - 1
    res["battle_acc_change_fused"] = res["fused_metrics"]["battle_acc_ply0"] - res["lir_metrics"]["battle_acc_ply0"]
    o = np.load(os.path.join(DATA, "oracle_q.npz"))
    n = 40 if smoke else len(o["t"])
    rows = torch.tensor([rix[(int(g), int(t))] for g, t in zip(o["game"][:n], o["t"][:n])], device=Mo.dev)
    lg = R["legal"][rows]
    nl = o["n_legal"][:n]
    tp0 = M.root_tape(R, rows)
    qs = o["qstar"][:n]
    rec = {k: [] for k in ("real", "amp", "fused", "true")}
    for j in range(14):
        m = torch.tensor(nl > j, device=Mo.dev)
        if not m.any():
            continue
        sub = M.tape_index(tp0, m)
        lr_, _ = M.action_l0(Mo.net, None, sub, lg[m, j, 0], lg[m, j, 1])
        la_, _ = M.action_l0(Mo.net, Mo.amp, sub, lg[m, j, 0], lg[m, j, 1])
        lf_, _ = M.action_l0(Mo.fused, Mo.amp_f, sub, lg[m, j, 0], lg[m, j, 1])
        rec["real"].append(torch.sigmoid(lr_).cpu().numpy())
        rec["amp"].append(torch.sigmoid(la_).cpu().numpy())
        rec["fused"].append(torch.sigmoid(lf_).cpu().numpy())
        mm = m.cpu().numpy()
        rec["true"].append(qs[mm, j, 0] + 0.5 * qs[mm, j, 1])
    rec = {k: np.concatenate(v) for k, v in rec.items()}
    for k in ("real", "amp", "fused"):
        res[f"value_mse_{k}"] = float(((rec[k] - rec["true"]) ** 2).mean())
    res["value_mse_reduction_amp"] = 1 - res["value_mse_amp"] / res["value_mse_real"]
    np.savez(os.path.join(out, "d4.npz"), **{f"v_{k}": v for k, v in rec.items()})
    return res


def test_setups():
    return np.load(os.path.join(DATA, "split.npz"))["test"]


def ctx_for(Mo, seed):
    return A.Ctx(Mo.net, Mo.amp, Mo.strust, Mo.rtrust, Mo.meta["action_scales"], Mo.dev, seed)


def part_d4u(Mo, out, seed, smoke=False):
    n = 16 if smoke else 200
    res = {}
    arr = {}
    for u in ("flag", "miner"):
        ag = A.Unified(ctx_for(Mo, seed + 400), K=16, utility=u)
        r, cap, lost, ln = A.play_match(ag, A.Heuristic(seed + 401), n, test_setups(), seed + 402)
        res[u] = {"miner_capture_rate": float(cap.mean()), "own_miner_lost_rate": float(lost.mean()),
                  "effective_win": float(r.mean()), "games": n}
        arr[f"cap_{u}"], arr[f"lost_{u}"], arr[f"res_{u}"] = cap, lost, r
    res["capture_ratio"] = res["miner"]["miner_capture_rate"] / max(res["flag"]["miner_capture_rate"], 1e-9)
    np.savez(os.path.join(out, "d4u.npz"), **arr)
    return res


def part_d5(Mo, out, seed, smoke=False):
    n = 8 if smoke else 400
    res, arr = {}, {}
    for opp_name in ("P0", "Ataraxos-1Step", "Tabular-MCTS"):
        cu = ctx_for(Mo, seed + 500)
        co = ctx_for(Mo, seed + 501)
        opp = {"P0": A.P0, "Ataraxos-1Step": A.Ataraxos1, "Tabular-MCTS": A.MCTS}[opp_name](co, K=16)
        t0 = time.time()
        r, _, _, ln = A.play_match(A.Unified(cu, K=16), opp, n, test_setups(), seed + 502)
        res[opp_name] = {"effective_win": float(r.mean()), "wins": int((r == 1).sum()), "draws": int((r == 0.5).sum()),
                         "losses": int((r == 0).sum()), "mean_len": float(ln.mean()), "sec": time.time() - t0}
        arr[f"res_{opp_name}"] = r
        log("d5", opp_name, res[opp_name])
    np.savez(os.path.join(out, "d5.npz"), **arr)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="d0,d1,d2,d3,d4,d4u,d5")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    eo = os.path.join(a.out, "eval")
    os.makedirs(eo, exist_ok=True)
    Mo = Models(a.out, dev)
    R = T.load_split("test", dev)
    rix = row_index(R)
    for p in a.parts.split(","):
        t0 = time.time()
        if p == "d0":
            r = part_d0(Mo, R, rix, eo)
        elif p == "d1":
            r = part_d1(Mo, R, rix, eo, a.seed, a.smoke)
        elif p == "d2":
            r = part_d2(Mo, R, rix, eo, a.seed, a.smoke)
        elif p == "d3":
            r = part_d3(Mo, R, rix, eo, a.seed, a.smoke)
        elif p == "d4":
            r = part_d4(Mo, R, rix, eo, a.seed, a.smoke)
        elif p == "d4u":
            r = part_d4u(Mo, eo, a.seed, a.smoke)
        elif p == "d5":
            r = part_d5(Mo, eo, a.seed, a.smoke)
        r["sec"] = time.time() - t0
        json.dump(r, open(os.path.join(eo, f"{p}.json"), "w"), indent=1, default=float)
        log(p, json.dumps({k: v for k, v in r.items() if not isinstance(v, list)}, default=float)[:1500])


if __name__ == "__main__":
    main()
