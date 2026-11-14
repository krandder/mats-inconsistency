#!/usr/bin/env python3
"""R5-E2 evaluation for one seed and one architecture. The oracle (P*_full, P*_c, Q*, Q*_miner) is read here only
for scoring and gates; no fitted quantity depends on it. Writes <out>/eval/<part>.json (summary) and per-item CSVs.
   python3 eval6.py --seed 101 --arch primary --out OUT --parts d0,d0cal,d0b,d1,d2,d3a,d3b [--smoke]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import agents6 as A  # noqa: E402
import model6 as M  # noqa: E402
import train6 as T  # noqa: E402

DATA = T.DATA


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def kl_rows(p, q, eps=1e-9):
    p = np.asarray(p, np.float64)
    q = np.clip(np.asarray(q, np.float64), eps, 1)
    return np.where(p > 0, p * (np.log(np.clip(p, eps, 1)) - np.log(q)), 0.0).sum(-1)


def jsd_cat(p, q, eps=1e-12):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    m = 0.5 * (p + q)
    return 0.5 * (p * np.log(p / m)).sum(-1) + 0.5 * (q * np.log(q / m)).sum(-1)


def jsd_bern(a, b, eps=1e-9):
    a, b = np.clip(a, eps, 1 - eps), np.clip(b, eps, 1 - eps)
    return jsd_cat(np.stack([a, 1 - a], -1), np.stack([b, 1 - b], -1))


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).correlation)


def write_csv(path, cols: dict):
    keys = list(cols)
    n = len(cols[keys[0]])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            w.writerow([(f"{float(cols[k][i]):.6g}" if isinstance(cols[k][i], (float, np.floating)) else int(cols[k][i]))
                        for k in keys])


class Models:
    def __init__(self, out, arch, dev):
        self.dev = dev
        ld = lambda f: torch.load(os.path.join(out, f), map_location=dev)
        self.meta = json.load(open(os.path.join(out, "train_meta.json")))
        self.net = T.freeze(self._net(arch, ld("real_100.pt")))
        self.net30 = T.freeze(self._net(arch, ld("real_30.pt")))
        d = self.net.d
        self.amp = M.AmpNet(d).to(dev)
        self.amp.load_state_dict(ld("amp.pt"))
        T.freeze(self.amp)
        self.strust = M.TrustNet(d + 2).to(dev)
        self.strust.load_state_dict(ld("slot_trust.pt"))
        self.strust30 = M.TrustNet(d + 2).to(dev)
        self.strust30.load_state_dict(ld("slot_trust_30.pt"))
        rt = ld("rollout_trust.pt")
        self.rtrust = M.TrustNet(rt["mu"].shape[0]).to(dev)
        self.rtrust.load_state_dict(rt["state"])
        self.rtrust.mu, self.rtrust.sd = rt["mu"], rt["sd"]
        self.att = M.AttnAgg().to(dev)
        self.att.load_state_dict(ld("attn_l3.pt"))
        for m in (self.strust, self.strust30, self.rtrust, self.att):
            T.freeze(m)

    def _net(self, arch, sd):
        n = M.make_net(arch).to(self.dev)
        n.load_state_dict(sd)
        return n


def row_index(R):
    g = R["game"].cpu().numpy().astype(np.int64)
    t = R["t"].cpu().numpy().astype(np.int64)
    return {(int(a), int(b)): i for i, (a, b) in enumerate(zip(g, t))}


def rows_for(o, rix, dev, n=None):
    n = len(o["t"]) if n is None else n
    return torch.tensor([rix[(int(g), int(t))] for g, t in zip(o["game"][:n], o["t"][:n])], device=dev)


# --------------------------------------------------------------------------------------------- D0
@torch.no_grad()
def belief_metrics(net, R, rows, pstar, dev, chunk=1024):
    kl0, nll7, hidn = [], [], []
    for c0 in range(0, rows.shape[0], chunk):
        ix = rows[c0:c0 + chunk]
        tp = M.tape_from(R, ix)
        with T.autocast(dev):
            lq, _ = M.belief0(net, tp)
        q = lq.exp().cpu().numpy()
        hid = ((R["slot_pubrev"][ix] == 0) & (R["slot_loc"][ix] < M.NSQ)).cpu().numpy()
        k = kl_rows(pstar[c0:c0 + chunk], q)
        kl0.append(np.where(hid, k, 0).sum(1) / np.maximum(hid.sum(1), 1))
        hidn.append(hid.sum(1))
        tru = (R["slot_true"][ix].long() - 1).clamp_min(0)
        for s in range(M.NS):
            others = torch.ones(ix.shape[0], M.NS, dtype=torch.bool, device=dev)
            others[:, s] = False
            tp7 = M.tape_from(R, ix, reveal_hidden=others)
            tp7["slot_val"][:, s] = 0
            with T.autocast(dev):
                l7, _ = M.belief0(net, tp7)
            nl = -l7[:, s].gather(1, tru[:, s:s + 1]).squeeze(1).cpu().numpy()
            nll7.append(nl[hid[:, s]])
    return np.concatenate(kl0), np.concatenate(nll7), np.concatenate(hidn)


@torch.no_grad()
def battle_acc(net, R, dev, chunk=2048):
    ok0 = []
    for c0 in range(0, R["n"], chunk):
        ix = torch.arange(c0, min(R["n"], c0 + chunk), device=dev)
        alive = R["slot_loc"][ix] < M.NSQ
        tp = M.tape_from(R, ix, reveal_hidden=torch.ones(ix.shape[0], M.NS, dtype=torch.bool, device=dev))
        ply = R["ply"][ix].long()
        valid = torch.arange(M.MAXP, device=dev)[None] < R["n_ply"][ix].long()[:, None]
        for c, k in enumerate(M.PLY_KEYS):
            tp[k] = ply[..., c] * valid
        with T.autocast(dev):
            h, _ = net(tp)
            lo, _, _ = net.trans_logits(h[:, M.a_idx(0)])
        bat = valid[:, 0] & (ply[:, 0, 3] >= 1)
        ok0.append((lo.float().argmax(1) == ply[:, 0, 3])[bat].cpu().numpy())
    return np.concatenate(ok0)


def part_d0(Mo, R, rix, eo, split):
    o = np.load(os.path.join(DATA, f"oracle_d0_{split}.npz"))
    rows = rows_for(o, rix, Mo.dev)
    kl0, nll7, hidn = belief_metrics(Mo.net, R, rows, o["pstar_full"], Mo.dev)
    kl0c, _, _ = belief_metrics(Mo.net, R, rows, o["pstar_c"], Mo.dev)
    acc = battle_acc(Mo.net, R, Mo.dev)
    hid = ((R["slot_pubrev"][rows] == 0) & (R["slot_loc"][rows] < M.NSQ)).cpu().numpy()
    gap = np.where(hid, kl_rows(o["pstar_full"], o["pstar_c"]), 0).sum(1) / np.maximum(hid.sum(1), 1)
    res = {"split": split, "battle_acc_ply0": float(acc.mean()), "n_battles_ply0": int(acc.size),
           "nll_7known": float(nll7.mean()), "n_7known": int(nll7.size), "kl0_full_per_hidden_slot": float(kl0.mean()),
           "kl0_full_total_per_position": float((kl0 * hidn).mean()), "kl0_constraint": float(kl0c.mean()),
           "kl_full_vs_constraint_oracle": float(gap.mean()), "n_pos": int(rows.shape[0]), "mean_hidden": float(hidn.mean())}
    res["gate"] = {"battle": res["battle_acc_ply0"] >= 0.97, "seven_known": res["nll_7known"] <= 0.05,
                   "belief_gap": res["kl0_full_per_hidden_slot"] >= 0.05}
    write_csv(os.path.join(eo, f"d0_{split}_positions.csv"), {"game": o["game"], "t": o["t"], "kl0_full": kl0,
                                                              "kl0_constraint": kl0c, "n_hidden": hidn, "kl_pfull_pc": gap})
    write_csv(os.path.join(eo, f"d0_{split}_nll7.csv"), {"nll7": nll7})
    write_csv(os.path.join(eo, f"d0_{split}_battles.csv"), {"correct": acc.astype(int)})
    return res


# --------------------------------------------------------------------------------------------- D0 part 3b (calibration split)
@torch.no_grad()
def part_d0b(Mo, Rc, rix, eo, seed, smoke=False):
    dev = Mo.dev
    gen = torch.Generator(device=dev).manual_seed(seed + 150)
    o = np.load(os.path.join(DATA, "oracle_d0_cal.npz"))
    n = 64 if smoke else len(o["t"])
    rows = rows_for(o, rix, dev, n)
    K = 16
    kl0s, kl16s = [], []
    for c0 in range(0, n, 256):
        ix = rows[c0:c0 + 256]
        rt = T.make_roots(Mo.net, Rc, ix, dev)
        nl = rt["n_legal"]
        act = (torch.rand(ix.shape[0], K, generator=gen, device=dev) * nl[:, None]).long().clamp(max=M.NSQ)
        act = torch.minimum(act, (nl - 1)[:, None])
        rr = T.run_rollouts(Mo.net, Mo.strust, Mo.rtrust, rt, act, 4, gen, dev)
        bag = {"dlb": rr["dlb"], "cov": rr["cov"], "e": rr["e"], "valid": torch.ones_like(act, dtype=torch.bool),
               "l0b": rt["l0b"]}
        lb, _, _ = M.coverage_reference(bag)
        hid = rt["hidden"].cpu().numpy()
        P = o["pstar_full"][c0:c0 + ix.shape[0]]
        k0 = kl_rows(P, rt["l0b"].exp().cpu().numpy())
        k16 = kl_rows(P, lb.exp().cpu().numpy())
        kl0s.append(np.where(hid, k0, 0).sum(1) / np.maximum(hid.sum(1), 1))
        kl16s.append(np.where(hid, k16, 0).sum(1) / np.maximum(hid.sum(1), 1))
    kl0s, kl16s = np.concatenate(kl0s), np.concatenate(kl16s)
    # action values on the calibration Q* positions
    oq = np.load(os.path.join(DATA, "oracle_q_cal.npz"))
    nq = 16 if smoke else len(oq["t"])
    qrows = rows_for(oq, rix, dev, nq)
    rt = T.make_roots(Mo.net, Rc, qrows, dev)
    lg = rt["legal"].cpu().numpy()
    assert np.array_equal(lg[..., 0] >= 0, oq["canon"][:nq, :, 0] >= 0), "legal-list order mismatch with the oracle"
    nl = rt["n_legal"]
    act = torch.arange(K, device=dev)[None] % nl[:, None]
    rr = T.run_rollouts(Mo.net, Mo.strust, Mo.rtrust, rt, act, 4, gen, dev, with_belief=False)
    e = rr["e"].cpu().numpy()
    l0 = A.legal_l0(A.Ctx(Mo.net, None, None, None, None, dev, seed), rt, with_amp=False).cpu().numpy()
    qs = oq["qstar"][:nq]
    actn = act.cpu().numpy()
    se0, se16 = [], []
    for i in range(nq):
        for j in range(int(oq["n_legal"][i])):
            m = actn[i] == j
            if not m.any():
                continue
            qe = qs[i, j, 0] + 0.5 * qs[i, j, 1]
            se0.append((1 / (1 + np.exp(-l0[i, j])) - qe) ** 2)
            se16.append((e[i, m].mean() - qe) ** 2)
    se0, se16 = np.array(se0), np.array(se16)
    res = {"n_belief_pos": int(n), "kl0": float(kl0s.mean()), "kl16": float(kl16s.mean()),
           "belief_kl_reduction": float(1 - kl16s.mean() / kl0s.mean()), "n_action_values": int(se0.size),
           "mse0": float(se0.mean()), "mse16": float(se16.mean()), "action_mse_reduction": float(1 - se16.mean() / se0.mean())}
    write_csv(os.path.join(eo, "d0b_belief_positions.csv"), {"kl0": kl0s, "kl16": kl16s})
    write_csv(os.path.join(eo, "d0b_action_values.csv"), {"se0": se0, "se16": se16})
    return res


# --------------------------------------------------------------------------------------------- D1
def policy_bag(pool, rt, nL, K, dev):
    N_ = nL.shape[0]
    j = torch.arange(K, device=dev)[None].expand(N_, -1)
    fl = (j >= nL[:, None]).long()
    jj = torch.where(fl == 0, j, j - nL[:, None])
    return T.bag_from_pool(pool, rt, fl, jj)


def key_gap(k, fl):
    sim = (k @ k.transpose(1, 2)).cpu().numpy()
    f = fl.cpu().numpy()
    same = f[:, :, None] == f[:, None, :]
    eye = np.eye(f.shape[1], dtype=bool)[None]
    with np.errstate(invalid="ignore"):
        w = np.nanmean(np.where(same & ~eye, sim, np.nan), (1, 2))
        a = np.nanmean(np.where(~same, sim, np.nan), (1, 2))
    return w, a


@torch.no_grad()
def part_d1(Mo, R, rix, eo, seed, smoke=False):
    dev = Mo.dev
    d = np.load(os.path.join(DATA, "d1_pos.npz"))
    n = 48 if smoke else len(d["t"])
    roots = rows_for(d, rix, dev, n)
    gen = torch.Generator(device=dev).manual_seed(seed + 200)
    rt = T.make_roots(Mo.net, R, roots, dev)
    K = 24
    pools = [T.flank_pool(Mo.net, Mo.strust, Mo.rtrust, rt, K, gen, dev) for _ in range(2)]
    refp = T.flank_pool(Mo.net, Mo.strust, Mo.rtrust, rt, 64, gen, dev)
    ref_lb, ref_vl, ref_n = M.coverage_reference(T.full_bag(refp, rt))
    ref_p = ref_lb.exp().cpu().numpy()
    ref_v = torch.sigmoid(ref_vl).cpu().numpy()
    hid = rt["hidden"].cpu().numpy()
    meta = Mo.meta["agg"]
    mu, arr = {}, {}
    for name, pL, pool in (("L", 0.8, pools[0]), ("R", 0.2, pools[1])):
        nL = torch.distributions.Binomial(K, torch.full((n,), pL, device=dev)).sample().long()
        b = policy_bag(pool, rt, nL, K, dev)
        k = Mo.amp.key(b["ebar"].reshape(-1, b["ebar"].shape[-1])).view(n, K, -1)
        outs = {"mix": M.agg_mix(b), "ataraxos_mmd": M.agg_mmd(b, meta["eta_b"], meta["eta_v"]),
                "kc_dedup": M.agg_kc(b, k, meta["u_b"], meta["u_v"]), "attn_L3": Mo.att(b, k)}
        mu[name] = {a: (lb.exp().cpu().numpy(), torch.sigmoid(lv).cpu().numpy()) for a, (lb, lv) in outs.items()}
        w, ac = key_gap(k, b["fl"])
        arr[f"nL_{name}"] = nL.cpu().numpy()
        arr[f"cos_within_{name}"], arr[f"cos_across_{name}"] = w, ac
        arr[f"key_gap_{name}"] = w - ac
        cov = (b["cov"].float().sum(1) > 0).cpu().numpy()
        arr[f"covered_{name}"] = (cov & hid).sum(1) / np.maximum(hid.sum(1), 1)
    res = {"n": int(n), "ref_cov_per_hidden_slot": float((ref_n.cpu().numpy() * hid).sum() / hid.sum()),
           "ref_uncovered_share": float(((ref_n.cpu().numpy() == 0) & hid).sum() / hid.sum())}
    nh = np.maximum(hid.sum(1), 1)
    for agg in ("mix", "ataraxos_mmd", "kc_dedup", "attn_L3"):
        (pLb, pLv), (pRb, pRv) = mu["L"][agg], mu["R"][agg]
        jb = np.where(hid, jsd_cat(pLb, pRb), 0).sum(1) / nh
        jv = jsd_bern(pLv, pRv)
        klb = 0.5 * (np.where(hid, kl_rows(ref_p, pLb), 0).sum(1) + np.where(hid, kl_rows(ref_p, pRb), 0).sum(1)) / nh
        msev = 0.5 * ((pLv - ref_v) ** 2 + (pRv - ref_v) ** 2)
        arr[f"jsd_b_{agg}"], arr[f"jsd_v_{agg}"], arr[f"klref_b_{agg}"], arr[f"mse_v_{agg}"] = jb, jv, klb, msev
        res[agg] = {"jsd_belief": float(jb.mean()), "jsd_value": float(jv.mean()), "kl_ref_belief": float(klb.mean()),
                    "mse_ref_value": float(msev.mean())}
    kg = np.nanmean(np.stack([arr["key_gap_L"], arr["key_gap_R"]]), 0)
    res["key_cos_within"] = float(np.nanmean([np.nanmean(arr["cos_within_L"]), np.nanmean(arr["cos_within_R"])]))
    res["key_cos_across"] = float(np.nanmean([np.nanmean(arr["cos_across_L"]), np.nanmean(arr["cos_across_R"])]))
    res["key_gap"] = float(np.nanmean(kg))
    res["covered_share_L"] = float(arr["covered_L"].mean())
    res["covered_share_R"] = float(arr["covered_R"].mean())
    arr["key_gap"] = kg
    arr["game"], arr["t"] = d["game"][:n], d["t"][:n]
    write_csv(os.path.join(eo, "d1_positions.csv"), arr)
    return res


# --------------------------------------------------------------------------------------------- D2
@torch.no_grad()
def part_d2(Mo, R, rix, eo, seed, smoke=False):
    dev = Mo.dev
    o = np.load(os.path.join(DATA, "oracle_q_test.npz"))
    n = 12 if smoke else len(o["t"])
    rows = rows_for(o, rix, dev, n).cpu().numpy()
    gen = torch.Generator(device=dev).manual_seed(seed + 300)
    lg = R["legal"][torch.tensor(rows, device=dev)].long().cpu().numpy()
    assert np.array_equal(lg[..., 0] >= 0, o["canon"][:n, :, 0] >= 0) and \
        np.array_equal(lg[lg[..., 0] >= 0], o["canon"][:n][o["canon"][:n, :, 0] >= 0]), "legal order mismatch"
    qstar = o["qstar"][:n]
    nl = o["n_legal"][:n]
    items = [(i, j, d_, ck, rep) for i in range(n) for j in range(int(nl[i])) for d_ in range(1, 7) for ck in (0, 1)
             for rep in range(2)]
    arr = {k: [] for k in ("pos", "act", "depth", "ckpt", "beta", "H", "kl", "qeff", "qstar_eff", "terminal")}
    for ck, (net, st) in enumerate(((Mo.net30, Mo.strust30), (Mo.net, Mo.strust))):
        it = [x for x in items if x[3] == ck]
        pos = torch.tensor([rows[x[0]] for x in it], device=dev)
        fr = torch.tensor([lg[x[0], x[1], 0] for x in it], device=dev)
        to = torch.tensor([lg[x[0], x[1], 1] for x in it], device=dev)
        dd = torch.tensor([x[2] for x in it], device=dev)
        betas, qs, terms = [], [], []
        for c0 in range(0, pos.shape[0], 2048):
            sl = slice(c0, c0 + 2048)
            rt = T.make_roots(net, R, pos[sl], dev)
            b_ = torch.zeros(pos[sl].shape[0], device=dev)
            q_ = torch.zeros(pos[sl].shape[0], 3, device=dev)
            tm = torch.zeros(pos[sl].shape[0], dtype=torch.bool, device=dev)
            for dv in torch.unique(dd[sl]).tolist():
                sel = dd[sl] == dv
                sub = {k: (M.tape_index(v, sel) if isinstance(v, dict) else v[sel]) for k, v in rt.items()}
                with T.autocast(dev):
                    out = M.full_rollouts(net, st, sub, fr[sl][sel], to[sl][sel], int(dv), gen, order="random",
                                          with_belief=False)
                b_[sel] = T.trust_beta(Mo.rtrust, out["feats"].float())
                q_[sel] = out["q_leaf"].float()
                tm[sel] = out["terminal"]
            betas.append(b_)
            qs.append(q_)
            terms.append(tm)
        beta = torch.cat(betas).cpu().numpy()
        q = torch.cat(qs).cpu().numpy()
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
        arr["terminal"].append(torch.cat(terms).cpu().numpy().astype(int))
    arr = {k: np.concatenate([np.asarray(x) for x in v]) for k, v in arr.items()}
    y = -arr["kl"]
    res = {"n_rollouts": int(y.size), "spearman_trust": spearman(arr["beta"], y), "spearman_stiff": spearman(-arr["H"], y)}
    qq = np.quantile(arr["H"], np.linspace(0, 1, 11))
    strata = np.clip(np.searchsorted(qq, arr["H"], side="right") - 1, 0, 9)
    st_t, st_s = [], []
    for b in range(10):
        m = strata == b
        st_t.append(spearman(arr["beta"][m], y[m]))
        st_s.append(spearman(-arr["H"][m], y[m]))
    res["strata_trust"], res["strata_stiff"] = float(np.nanmean(st_t)), float(np.nanmean(st_s))
    res["strata_trust_each"], res["strata_stiff_each"] = st_t, st_s
    for ck in (0, 1):
        m = arr["ckpt"] == ck
        res[f"ckpt{ck}"] = {"spearman_trust": spearman(arr["beta"][m], y[m]), "spearman_stiff": spearman(-arr["H"][m], y[m]),
                            "mean_kl": float(arr["kl"][m].mean())}
    lo = arr["H"] <= np.quantile(arr["H"], 0.1)
    res["kl_lowest_entropy_tenth"], res["kl_other"] = float(arr["kl"][lo].mean()), float(arr["kl"][~lo].mean())
    res["terminal_share"] = float(arr["terminal"].mean())
    arr["strata"] = strata
    write_csv(os.path.join(eo, "d2_rollouts.csv"), arr)
    return res


# --------------------------------------------------------------------------------------------- D3a
@torch.no_grad()
def part_d3a(Mo, R, rix, eo, seed, smoke=False):
    dev = Mo.dev
    o = np.load(os.path.join(DATA, "oracle_q_test.npz"))
    n = 12 if smoke else len(o["t"])
    rows = rows_for(o, rix, dev, n)
    h_before = T.param_hash(Mo.net)
    rt = T.make_roots(Mo.net, R, rows, dev)
    l0b_before = rt["l0b"].clone()
    choices = {}
    for u in ("std", "miner"):
        ctx = A.Ctx(Mo.net, Mo.amp, Mo.strust, Mo.rtrust, Mo.att, dev, seed + 400)
        choices[u] = A.Unified(ctx, K=16, utility=u).choose(rt).cpu().numpy()
    rt2 = T.make_roots(Mo.net, R, rows, dev)
    drift = float((rt2["l0b"] - l0b_before).abs().max())
    h_after = T.param_hash(Mo.net)
    qm = o["miner"][:n]
    nl = o["n_legal"][:n]
    a_s, a_m = choices["std"], choices["miner"]
    q_s = np.array([qm[i, a_s[i]] for i in range(n)])
    q_m = np.array([qm[i, a_m[i]] for i in range(n)])
    q_max = np.array([np.nanmax(qm[i, :nl[i]]) for i in range(n)])
    relevant = (q_max - q_s) >= 0.05
    success = q_m > q_s
    res = {"n": int(n), "param_hash_identical": h_before == h_after, "belief_max_abs_change": drift,
           "n_relevant": int(relevant.sum()), "shift_rate_relevant": float(success[relevant].mean()) if relevant.any() else float("nan"),
           "changed_action_rate": float((a_s != a_m).mean()), "mean_miner_std": float(q_s.mean()),
           "mean_miner_miner": float(q_m.mean()), "shift_rate_all": float(success.mean()),
           "nonnegative_rate_all": float((q_m >= q_s).mean())}
    write_csv(os.path.join(eo, "d3a_positions.csv"), {"game": o["game"][:n], "t": o["t"][:n], "kind": o["kind"][:n],
                                                      "a_std": a_s, "a_miner": a_m, "q_miner_std": q_s, "q_miner_miner": q_m,
                                                      "q_miner_max": q_max, "relevant": relevant.astype(int),
                                                      "success": success.astype(int)})
    return res


# --------------------------------------------------------------------------------------------- D3b
def part_d3b(Mo, eo, seed, smoke=False):
    dev = Mo.dev
    sp = np.load(os.path.join(DATA, "split.npz"))["test"]
    rng = np.random.default_rng(seed + 500)
    npair = 4 if smoke else 200
    pairs = [(int(rng.choice(sp)), int(rng.choice(sp))) for _ in range(npair)]
    cu = A.Ctx(Mo.net, Mo.amp, Mo.strust, Mo.rtrust, Mo.att, dev, seed + 501)
    co = A.Ctx(Mo.net, Mo.amp, Mo.strust, Mo.rtrust, Mo.att, dev, seed + 502)
    t0 = time.time()
    r, side, ln = A.play_match(A.Unified(cu, K=16), A.P0(co), pairs, dev)
    res = {"games": int(r.size), "effective_win": float(r.mean()), "wins": int((r == 1).sum()), "draws": int((r == 0.5).sum()),
           "losses": int((r == 0).sum()), "mean_len": float(ln.mean()), "sec": time.time() - t0,
           "effective_win_as_red": float(r[side == 0].mean()), "effective_win_as_blue": float(r[side == 1].mean())}
    write_csv(os.path.join(eo, "d3b_games.csv"), {"pair": np.repeat(np.arange(npair), 2), "unified_colour": side,
                                                  "result": r, "length": ln})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arch", choices=list(M.ARCHS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="d0,d0cal,d0b,d1,d2,d3a,d3b")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed)
    eo = os.path.join(a.out, "eval")
    os.makedirs(eo, exist_ok=True)
    Mo = Models(a.out, a.arch, dev)
    R = T.load_split("test", dev)
    rix = row_index(R)
    Rc, rixc = None, None
    for p in a.parts.split(","):
        t0 = time.time()
        if p in ("d0cal", "d0b") and Rc is None:
            Rc = T.load_split("cal", dev)
            rixc = row_index(Rc)
        if p == "d0":
            r = part_d0(Mo, R, rix, eo, "test")
        elif p == "d0cal":
            r = part_d0(Mo, Rc, rixc, eo, "cal")
        elif p == "d0b":
            r = part_d0b(Mo, Rc, rixc, eo, a.seed, a.smoke)
        elif p == "d1":
            r = part_d1(Mo, R, rix, eo, a.seed, a.smoke)
        elif p == "d2":
            r = part_d2(Mo, R, rix, eo, a.seed, a.smoke)
        elif p == "d3a":
            r = part_d3a(Mo, R, rix, eo, a.seed, a.smoke)
        elif p == "d3b":
            r = part_d3b(Mo, eo, a.seed, a.smoke)
        else:
            raise ValueError(p)
        r["sec"] = time.time() - t0
        json.dump(r, open(os.path.join(eo, f"{p}.json"), "w"), indent=1, default=float)
        log(p, json.dumps({k: v for k, v in r.items() if not isinstance(v, list)}, default=float)[:1500])


if __name__ == "__main__":
    main()
