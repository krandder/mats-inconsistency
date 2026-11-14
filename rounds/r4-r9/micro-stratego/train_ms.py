#!/usr/bin/env python3
"""E-MS training for one seed (GPU). Stages (all pre-registered in PREREG-MS.md):
  1 theta_real on TRAIN real trajectories only: any-order masked CE on opponent slots + 1-step transition CE + policy CE
    (behaviour cloning, used by rollouts as the learned opponent/self model) + outcome CE (forecaster). Checkpoints at
    30% and 100% of the steps.
  2 slot-trust head (theta_trust, belief claims) on CAL: excess surprise of q(Z_s | context) vs the realised slot.
  3 theta_amp distillation on TRAIN roots from search consensus over theta_real (chi_real = 0, LIR arm) and the fused
    arm (chi_real = 1, optimistic target +0.2).
  4 rollout-trust head (theta_trust, value claims) on CAL: rollouts from both checkpoints, root action = realised action,
    target = excess surprise of the leaf claim vs the realised result.
  5 D2 aggregators on CAL two-flank roots: kc_dedup (W_K2 + scale), ataraxos_mmd (eta), attn_L3, fitted on bags with
    flank skew p_L ~ Beta(0.5, 0.5) against the unskewed K=128 mix reference (model-based, no simulator).
  6 per-action pooling scales for D5 on CAL (eta for Ataraxos-1Step, u for kc_dedup) against realised results.
   python3 train_ms.py --seed 101 --out runs/s101 [--smoke]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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

DATA = os.environ.get("MS_DATA", os.path.join(HERE, "data"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_split(name, dev):
    z = np.load(os.path.join(DATA, f"{name}.npz"))
    R = {k: torch.tensor(z[k].astype(np.int64), device=dev) for k in z.files}
    return R


def param_hash(mod):
    h = hashlib.sha256()
    for k, v in sorted(mod.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train_batch(R, idx, gen):
    """Real-trajectory batch: root tape with any-order random reveal of hidden slots + the next <= 6 real plies."""
    B = idx.shape[0]
    dev = idx.device
    hidden = R["slot_pubrev"][idx] == 0
    u = torch.rand(B, 1, generator=gen, device=dev)
    rev = (torch.rand(B, 6, generator=gen, device=dev) < u) & hidden
    tp = M.root_tape(R, idx, reveal_hidden=rev)
    ply = R["ply"][idx]
    npl = R["n_ply"][idx]
    valid = torch.arange(M.MAXP, device=dev)[None] < npl[:, None]
    for c, k in enumerate(("ply_actor", "ply_from", "ply_to", "ply_out", "ply_att", "ply_def")):
        tp[k] = ply[..., c] * valid
    return tp, hidden & ~rev, valid


def real_losses(net, R, idx, gen):
    tp, bmask, valid = train_batch(R, idx, gen)
    h, _ = net(tp)
    tru = R["slot_true"][idx] - 1
    bl = net.belief_logits(h)
    lb = (F.cross_entropy(bl.reshape(-1, 6), tru.reshape(-1), reduction="none") * bmask.reshape(-1).float()).sum() / bmask.sum().clamp_min(1)
    ply = R["ply"][idx]
    res = R["result"][idx]
    pre = torch.tensor([M.pre_idx(j) for j in range(M.MAXP)], device=idx.device)
    ai = torch.tensor([M.a_idx(j) for j in range(M.MAXP)], device=idx.device)
    oi = torch.tensor([M.o_idx(j) for j in range(M.MAXP)], device=idx.device)
    vf = valid.reshape(-1).float()
    nv = vf.sum().clamp_min(1)
    pol = net.h_policy(h[:, pre]).reshape(-1, 256)
    tgt = (ply[..., 1] * 16 + ply[..., 2]).reshape(-1)
    lp = (F.cross_entropy(pol, tgt, reduction="none") * vf).sum() / nv
    lo, la, ld = net.trans_logits(h[:, ai])
    lt = sum((F.cross_entropy(x.reshape(-1, x.shape[-1]), ply[..., c].reshape(-1), reduction="none") * vf).sum() / nv
             for x, c in ((lo, 3), (la, 4), (ld, 5)))
    vg = F.cross_entropy(net.h_value(h[:, M.GLOB]), res)
    rr = res[:, None].expand(-1, M.MAXP).reshape(-1)
    va = (F.cross_entropy(net.h_value(h[:, ai]).reshape(-1, 3), rr, reduction="none") * vf).sum() / nv
    vo = (F.cross_entropy(net.h_value(h[:, oi]).reshape(-1, 3), rr, reduction="none") * vf).sum() / nv
    lv = (vg + va + vo) / 3
    return lb, lp, lt, lv


def train_real(a, R, Rc, dev, out):
    torch.manual_seed(a.seed)
    net = M.RealNet().to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    N = R["t"].shape[0]
    S = a.steps
    warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, S) / S)))
    hist = []
    t0 = time.time()
    ck30 = int(0.3 * S)
    for st in range(S):
        net.train()
        idx = torch.randint(0, N, (a.batch,), device=dev, generator=gen)
        lb, lp, lt, lv = real_losses(net, R, idx, gen)
        loss = lb + lp + lt + lv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if st % 500 == 0 or st == S - 1:
            net.eval()
            with torch.no_grad():
                g2 = torch.Generator(device=dev).manual_seed(999)
                vi = torch.randint(0, Rc["t"].shape[0], (4096,), device=dev, generator=g2)
                vb, vp, vt, vv = real_losses(net, Rc, vi, g2)
            row = {"step": st, "train": [float(x) for x in (lb, lp, lt, lv)], "cal": [float(x) for x in (vb, vp, vt, vv)],
                   "sec": time.time() - t0}
            hist.append(row)
            log("real", json.dumps(row))
        if st + 1 == ck30:
            torch.save(net.state_dict(), os.path.join(out, "real_30.pt"))
    torch.save(net.state_dict(), os.path.join(out, "real_100.pt"))
    net.eval()
    return net, {"params": nparam, "hist": hist, "sec": time.time() - t0}


def fit_slot_trust(a, net, Rc, dev):
    gen = torch.Generator(device=dev).manual_seed(a.seed + 1)
    tr = M.TrustNet(net.d + 2).to(dev)
    opt = torch.optim.Adam(tr.parameters(), lr=1e-3)
    N = Rc["t"].shape[0]
    hist = []
    for st in range(a.trust_steps):
        idx = torch.randint(0, N, (512,), device=dev, generator=gen)
        hidden = Rc["slot_pubrev"][idx] == 0
        u = torch.rand(512, 1, generator=gen, device=dev)
        rev = (torch.rand(512, 6, generator=gen, device=dev) < u) & hidden
        tp = M.root_tape(Rc, idx, reveal_hidden=rev)
        with torch.no_grad():
            h, _ = net(tp, root_only=True)
            q = torch.softmax(net.belief_logits(h).float(), -1)
            tgt = M.excess_surprise(q, Rc["slot_true"][idx] - 1)
            f = M.slot_features(h, q, tp)
        m = (hidden & ~rev).float()
        pred = tr(f)
        l = (((pred - tgt) ** 2) * m).sum() / m.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        l.backward()
        opt.step()
        if st % 500 == 0:
            hist.append(float(l))
    tr.eval()
    for p in tr.parameters():
        p.requires_grad_(False)
    return tr, hist


def legal_lists(R, idx):
    lg = R["legal"][idx]                    # [B,14,2]
    n = (lg[..., 0] >= 0).sum(1)
    return lg, n


def search_roots(R, n, gen, dev, min_legal=2, tmax=14):
    ok = (R["t"] <= tmax) & ((R["legal"][..., 0] >= 0).sum(1) >= min_legal)
    cand = torch.nonzero(ok).squeeze(1)
    pick = torch.randperm(cand.shape[0], generator=gen, device=dev)[:n]
    return cand[pick]


@torch.no_grad()
def consensus_targets(a, net, strust, R, roots, K, dev, gen, depth=4, chunk=512):
    """Per (root, legal action): mean effective win of K rollouts in theta_real (stiff_voi Z)."""
    lg, nl = legal_lists(R, roots)
    rows_r, rows_a = [], []
    for i in range(roots.shape[0]):
        for j in range(int(nl[i])):
            rows_r.append(i)
            rows_a.append(j)
    rr = torch.tensor(rows_r, device=dev)
    ra = torch.tensor(rows_a, device=dev)
    tgt = torch.zeros(rr.shape[0], device=dev)
    for c0 in range(0, rr.shape[0], chunk):
        ri, ai = rr[c0:c0 + chunk], ra[c0:c0 + chunk]
        tp = M.tape_repeat(M.root_tape(R, roots[ri]), K)
        fr = lg[ri, ai, 0].repeat_interleave(K)
        to = lg[ri, ai, 1].repeat_interleave(K)
        tpz, _ = M.sample_Z(net, tp, "stiff_voi", gen, strust)
        r = M.rollout(net, tpz, fr, to, depth, gen)
        q = r["q_leaf"]
        tgt[c0:c0 + chunk] = (q[:, 0] + 0.5 * q[:, 1]).view(-1, K).mean(1)
    return rr, ra, tgt


def distill(a, net, R, roots, rr, ra, tgt, dev, fused=False):
    """theta_amp Q adapter from search consensus. LIR: theta_real frozen (chi_real = 0). Fused: theta_real trainable,
    optimistic target (+0.2)."""
    torch.manual_seed(a.seed + (7 if fused else 5))
    amp = M.AmpNet(net.d).to(dev)
    model = copy.deepcopy(net) if fused else net
    if fused:
        for p in model.parameters():
            p.requires_grad_(True)
        model.train()
    params = list(amp.q.parameters()) + (list(model.parameters()) if fused else [])
    opt = torch.optim.Adam(params, lr=3e-4)
    gen = torch.Generator(device=dev).manual_seed(a.seed + 11)
    lg, _ = legal_lists(R, roots)
    y = (tgt + 0.2).clamp(0.01, 0.99) if fused else tgt.clamp(0.01, 0.99)
    hist = []
    for st in range(a.distill_steps):
        b = torch.randint(0, rr.shape[0], (512,), device=dev, generator=gen)
        tp = M.root_tape(R, roots[rr[b]])
        tp["ply_from"][:, 0] = lg[rr[b], ra[b], 0]
        tp["ply_to"][:, 0] = lg[rr[b], ra[b], 1]
        if fused:
            h, _ = model(tp)
            l0 = M.eff_logit(model.h_value(h[:, M.a_idx(0)])) + amp.q(h[:, M.a_idx(0)]).squeeze(-1)
        else:
            with torch.no_grad():
                h, _ = model(tp)
                base = M.eff_logit(model.h_value(h[:, M.a_idx(0)]))
            l0 = base + amp.correction(h[:, M.a_idx(0)])
        loss = F.binary_cross_entropy_with_logits(l0, y[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if st % 500 == 0:
            hist.append(float(loss))
    model.eval()
    amp.eval()
    if fused:
        for p in model.parameters():
            p.requires_grad_(False)
    return amp, model, hist


@torch.no_grad()
def trust_rollouts(net, strust, R, roots, depth_t, amp, gen, order="random", root_action="real", chunk=1024):
    """Rollouts used to train / evaluate the rollout-trust head. depth_t [n] per root. Returns features, q_leaf, l0."""
    feats, qs, l0s, ds = [], [], [], []
    lg, nl = legal_lists(R, roots)
    for c0 in range(0, roots.shape[0], chunk):
        ix = roots[c0:c0 + chunk]
        tp = M.root_tape(R, ix)
        if root_action == "real":
            fr, to = R["ply"][ix, 0, 1], R["ply"][ix, 0, 2]
        else:
            fr, to = root_action[0][c0:c0 + chunk], root_action[1][c0:c0 + chunk]
        tpz, info = M.sample_Z(net, tp, order, gen, strust)
        lr_ = M.path_logp(net, tpz, info["hidden"], info["step"])
        nh = info["hidden"].sum(1).clamp_min(1).float()
        pdis = (info["logp"] - lr_).abs() / nh
        out_q = torch.zeros(ix.shape[0], 3, device=ix.device)
        out_f = None
        for d in torch.unique(depth_t[c0:c0 + chunk]).tolist():
            sel = depth_t[c0:c0 + chunk] == d
            sub = M.tape_index(tpz, sel)
            rd = M.rollout(net, sub, fr[sel], to[sel], int(d), gen)
            l0, _ = M.action_l0(net, None, M.tape_index(tp, sel), fr[sel], to[sel])
            f = M.trust_features(rd, l0, int(d), pdis[sel])
            if out_f is None:
                out_f = torch.zeros(ix.shape[0], f.shape[1], device=ix.device)
                out_l0 = torch.zeros(ix.shape[0], device=ix.device)
            out_f[sel] = f
            out_q[sel] = rd["q_leaf"]
            out_l0[sel] = l0
        feats.append(out_f)
        qs.append(out_q)
        l0s.append(out_l0)
    return torch.cat(feats), torch.cat(qs), torch.cat(l0s)


def fit_rollout_trust(a, nets, struts, Rc, dev):
    gen = torch.Generator(device=dev).manual_seed(a.seed + 21)
    roots = search_roots(Rc, a.n_trust_roots, gen, dev, min_legal=1, tmax=20)
    F_, Y = [], []
    for net, st in zip(nets, struts):
        for rep in range(2):
            d = torch.randint(1, 7, (roots.shape[0],), device=dev, generator=gen)
            f, q, _ = trust_rollouts(net, st, Rc, roots, d, None, gen, order="random")
            F_.append(f)
            Y.append(M.excess_surprise(q, Rc["result"][roots]))
    X = torch.cat(F_)
    Y = torch.cat(Y)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    tr = M.TrustNet(X.shape[1]).to(dev)
    tr.mu, tr.sd = mu, sd
    opt = torch.optim.Adam(tr.parameters(), lr=1e-3, weight_decay=1e-4)
    hist = []
    for s in range(a.trust_steps):
        b = torch.randint(0, X.shape[0], (512,), device=dev, generator=gen)
        l = ((tr((X[b] - mu) / sd) - Y[b]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        l.backward()
        opt.step()
        if s % 500 == 0:
            hist.append(float(l))
    tr.eval()
    return tr, {"n": int(X.shape[0]), "hist": hist, "target_mean": float(Y.mean())}


def trust_beta(tr, f):
    return M.beta_from(tr((f - tr.mu) / tr.sd))


@torch.no_grad()
def flank_rollouts(net, strust, rtrust, amp, R, roots, n_per_flank, gen, depth=4, chunk=256):
    """For each root: n_per_flank rollouts whose root action is uniform within each flank (canonical from-col 0..1 = L).
    Returns per root [2, n]: q_eff, dl (vs root 0-step state value), beta, ebar."""
    lg, nl = legal_lists(R, roots)
    out = {k: [] for k in ("q", "dl", "beta", "ebar", "l0")}
    for c0 in range(0, roots.shape[0], chunk):
        ix = roots[c0:c0 + chunk]
        B = ix.shape[0]
        L = lg[c0:c0 + chunk]
        isL = (L[..., 0] % 4 < 2) & (L[..., 0] >= 0)
        isR = (L[..., 0] % 4 >= 2) & (L[..., 0] >= 0)
        res = {k: [] for k in out}
        tp0 = M.root_tape(R, ix)
        h0, _ = net(tp0)
        l0root = M.eff_logit(net.h_value(h0[:, M.GLOB]))
        for fl, msk in ((0, isL), (1, isR)):
            w = msk.float().repeat_interleave(n_per_flank, 0)
            pick = torch.multinomial(w, 1, generator=gen).squeeze(1)
            rows = torch.arange(B, device=ix.device).repeat_interleave(n_per_flank)
            fr, to = L[rows, pick, 0], L[rows, pick, 1]
            tp = M.tape_repeat(tp0, n_per_flank)
            tpz, info = M.sample_Z(net, tp, "stiff_voi", gen, strust)
            lr_ = M.path_logp(net, tpz, info["hidden"], info["step"])
            pdis = (info["logp"] - lr_).abs() / info["hidden"].sum(1).clamp_min(1).float()
            r = M.rollout(net, tpz, fr, to, depth, gen)
            l0a, _ = M.action_l0(net, None, tp, fr, to)
            f = M.trust_features(r, l0a, depth, pdis)
            beta = trust_beta(rtrust, f)
            q = r["q_leaf"]
            qe = (q[:, 0] + 0.5 * q[:, 1]).clamp(1e-5, 1 - 1e-5)
            lr0 = l0root.repeat_interleave(n_per_flank)
            res["q"].append(qe.view(B, n_per_flank))
            res["dl"].append((torch.log(qe) - torch.log1p(-qe) - lr0).view(B, n_per_flank))
            res["beta"].append(beta.view(B, n_per_flank))
            res["ebar"].append(r["ebar"].view(B, n_per_flank, -1))
            res["l0"].append(l0root)
        for k in ("q", "dl", "beta", "ebar"):
            out[k].append(torch.stack(res[k], 1))
        out["l0"].append(l0root)
    return {k: torch.cat(v) for k, v in out.items()}


def skew_bag(P, K, pL, gen):
    """Draw a bag of K rollouts: each from flank L w.p. pL (per root), uniform index within the flank's pool."""
    N, _, n = P["q"].shape
    dev = P["q"].device
    fl = (torch.rand(N, K, generator=gen, device=dev) >= pL[:, None]).long()
    j = torch.randint(0, n, (N, K), generator=gen, device=dev)
    ar = torch.arange(N, device=dev)[:, None]
    g = lambda x: x[ar, fl, j]
    return {"q": g(P["q"]), "dl": g(P["dl"]), "beta": g(P["beta"]), "ebar": P["ebar"][ar, fl, j], "l0": P["l0"],
            "valid": torch.ones(N, K, dtype=torch.bool, device=dev), "fl": fl}


def fit_aggregators(a, net, strust, rtrust, amp, Rc, dev):
    gen = torch.Generator(device=dev).manual_seed(a.seed + 31)
    lg = Rc["legal"]
    both = ((lg[..., 0] % 4 < 2) & (lg[..., 0] >= 0)).any(1) & ((lg[..., 0] % 4 >= 2) & (lg[..., 0] >= 0)).any(1)
    ok = both & (Rc["t"] >= 2) & (Rc["t"] <= 14)
    cand = torch.nonzero(ok).squeeze(1)
    roots = cand[torch.randperm(cand.shape[0], generator=gen, device=dev)[:a.n_agg_roots]]
    pool = flank_rollouts(net, strust, rtrust, amp, Rc, roots, 32, gen)
    ref = flank_rollouts(net, strust, rtrust, amp, Rc, roots, 64, gen)      # K = 128 unskewed reference
    vref = ref["q"].reshape(roots.shape[0], -1).mean(1).clamp(1e-4, 1 - 1e-4)
    res = {"n_roots": int(roots.shape[0])}
    K = 24

    def bags(n_steps, fn, params, lr=1e-2):
        opt = torch.optim.Adam(params, lr=lr)
        hist = []
        for s in range(n_steps):
            pL = torch.distributions.Beta(0.5, 0.5).sample((roots.shape[0],)).to(dev)
            b = skew_bag(pool, K, pL, gen)
            l = F.binary_cross_entropy_with_logits(fn(b), vref)
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
            if s % 200 == 0:
                hist.append(float(l))
        return hist

    # ataraxos_mmd: eta
    log_eta = torch.tensor(math.log(0.05), device=dev, requires_grad=True)
    res["mmd_hist"] = bags(a.agg_steps, lambda b: M.agg_mmd(b["l0"], b["dl"], b["beta"], b["valid"], log_eta.exp()), [log_eta])
    res["eta"] = float(log_eta.exp())
    # kc_dedup: W_K2 + log u (E-CA k_v2 key head fitted on the calibration split, backbone frozen)
    log_u = torch.tensor(math.log(0.1), device=dev, requires_grad=True)

    def kc(b):
        k = amp.key(b["ebar"])
        return M.agg_kc(b["l0"], b["dl"], b["beta"], k, b["valid"], log_u.exp())
    res["kc_hist"] = bags(a.agg_steps, kc, list(amp.keyW2.parameters()) + [log_u], lr=3e-3)
    res["u_root"] = float(log_u.exp())
    # attn_L3
    torch.manual_seed(a.seed + 41)
    att = M.AttnAgg().to(dev)
    res["attn_hist"] = bags(a.agg_steps, lambda b: att(b["l0"], b["dl"], b["beta"], amp.key(b["ebar"]).detach(), b["valid"]),
                            list(att.parameters()), lr=1e-3)
    att.eval()
    for p in list(att.parameters()) + list(amp.keyW2.parameters()):
        p.requires_grad_(False)
    return att, res


@torch.no_grad()
def action_bags(net, strust, rtrust, amp, R, roots, K, gen, order, depth=4, chunk=256):
    """K rollouts per root, root actions round-robin over the legal list. Returns per root [K]: action index, q_eff,
    dl vs theta_amp l0(a), beta, key; and l0 per legal action [n,14]."""
    lg, nl = legal_lists(R, roots)
    outs = {k: [] for k in ("act", "q", "dl", "beta", "key", "l0")}
    for c0 in range(0, roots.shape[0], chunk):
        ix = roots[c0:c0 + chunk]
        B = ix.shape[0]
        L, n = lg[c0:c0 + chunk], nl[c0:c0 + chunk]
        tp0 = M.root_tape(R, ix)
        l0all = torch.full((B, 14), -1e4, device=ix.device)
        for j in range(14):
            m = n > j
            if m.any():
                lj, _ = M.action_l0(net, amp, M.tape_index(tp0, m), L[m, j, 0], L[m, j, 1])
                l0all[m, j] = lj
        act = torch.arange(K, device=ix.device)[None] % n[:, None]
        fr = L[torch.arange(B, device=ix.device)[:, None], act, 0].reshape(-1)
        to = L[torch.arange(B, device=ix.device)[:, None], act, 1].reshape(-1)
        tp = M.tape_repeat(tp0, K)
        tpz, info = M.sample_Z(net, tp, order, gen, strust)
        lr_ = M.path_logp(net, tpz, info["hidden"], info["step"])
        pdis = (info["logp"] - lr_).abs() / info["hidden"].sum(1).clamp_min(1).float()
        r = M.rollout(net, tpz, fr, to, depth, gen)
        l0r, _ = M.action_l0(net, None, tp, fr, to)
        beta = trust_beta(rtrust, M.trust_features(r, l0r, depth, pdis))
        q = r["q_leaf"]
        qe = (q[:, 0] + 0.5 * q[:, 1]).clamp(1e-5, 1 - 1e-5)
        l0a = l0all.gather(1, act).reshape(-1)
        outs["act"].append(act)
        outs["q"].append(qe.view(B, K))
        outs["dl"].append((torch.log(qe) - torch.log1p(-qe) - l0a).view(B, K))
        outs["beta"].append(beta.view(B, K))
        outs["key"].append(amp.key(r["ebar"]).view(B, K, -1))
        outs["l0"].append(l0all)
    return {k: torch.cat(v) for k, v in outs.items()}


def per_action_consensus(bag, kind, scale, gamma=2.0):
    """l*_a for every legal slot a: l0(a) + scale * sum over rollouts on a of (w_r * dl_r), with w = 1 (mmd, Ataraxos),
    or beta_r / c_r with crowding among the rollouts of the same action (kc)."""
    act, dl = bag["act"], bag["dl"]
    N, K = act.shape
    out = bag["l0"].clone()
    same = act[:, :, None] == act[:, None, :]
    if kind == "mmd":
        w = torch.ones_like(dl)
    elif kind == "kc":
        k = bag["key"]
        c = ((torch.relu(k @ k.transpose(1, 2)) ** gamma) * same.float()).sum(2).clamp_min(1e-6)
        w = bag["beta"] / c
    elif kind == "mix":
        return None
    contrib = torch.zeros(N, 14, device=dl.device)
    contrib.scatter_add_(1, act, w * dl)
    return out + scale * contrib


def fit_action_scales(a, net, strust, rtrust, amp, Rc, dev):
    gen = torch.Generator(device=dev).manual_seed(a.seed + 51)
    roots = search_roots(Rc, a.n_act_roots, gen, dev, min_legal=2, tmax=14)
    lg, nl = legal_lists(Rc, roots)
    real = Rc["ply"][roots, 0, 1] * 16 + Rc["ply"][roots, 0, 2]
    ja = ((lg[..., 0] * 16 + lg[..., 1]) == real[:, None]).float().argmax(1)
    y = (Rc["result"][roots] == 0).float() + 0.5 * (Rc["result"][roots] == 1).float()
    res = {}
    for kind, order in (("mmd", "row_major"), ("kc", "stiff_voi")):
        bag = action_bags(net, strust, rtrust, amp, Rc, roots, 16, gen, order)
        grid = np.exp(np.linspace(math.log(1e-3), math.log(10.0), 61))
        ll = []
        for s in grid:
            lstar = per_action_consensus(bag, kind, float(s))
            la = lstar.gather(1, ja[:, None]).squeeze(1)
            ll.append(float(F.binary_cross_entropy_with_logits(la, y)))
        ll0 = float(F.binary_cross_entropy_with_logits(bag["l0"].gather(1, ja[:, None]).squeeze(1), y))
        best = int(np.argmin(ll))
        res[kind] = {"scale": float(grid[best]), "logloss": ll[best], "logloss_l0": ll0, "grid_ll": ll}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--trust-steps", type=int, default=3000)
    ap.add_argument("--distill-roots", type=int, default=6000)
    ap.add_argument("--distill-steps", type=int, default=3000)
    ap.add_argument("--n-trust-roots", type=int, default=4000)
    ap.add_argument("--n-agg-roots", type=int, default=1500)
    ap.add_argument("--agg-steps", type=int, default=1500)
    ap.add_argument("--n-act-roots", type=int, default=3000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.trust_steps, a.distill_roots, a.distill_steps = 60, 30, 64, 30
        a.n_trust_roots, a.n_agg_roots, a.agg_steps, a.n_act_roots = 64, 48, 20, 64
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    meta = {"seed": a.seed, "args": vars(a), "device": str(dev)}
    t0 = time.time()
    R = load_split("train", dev)
    Rc = load_split("cal", dev)
    log("data", {k: int(v["t"].shape[0]) for k, v in (("train", R), ("cal", Rc))})
    net, meta["real"] = train_real(a, R, Rc, dev, a.out)
    for p in net.parameters():
        p.requires_grad_(False)
    h_before = param_hash(net)
    net30 = M.RealNet().to(dev)
    net30.load_state_dict(torch.load(os.path.join(a.out, "real_30.pt"), map_location=dev))
    net30.eval()
    for p in net30.parameters():
        p.requires_grad_(False)
    strust, meta["slot_trust_hist"] = fit_slot_trust(a, net, Rc, dev)
    strust30, _ = fit_slot_trust(a, net30, Rc, dev)
    torch.save(strust.state_dict(), os.path.join(a.out, "slot_trust.pt"))
    torch.save(strust30.state_dict(), os.path.join(a.out, "slot_trust_30.pt"))
    log("slot trust", meta["slot_trust_hist"][-3:], f"{time.time()-t0:.0f}s")
    # theta_amp distillation from search consensus on TRAIN roots
    gen = torch.Generator(device=dev).manual_seed(a.seed + 3)
    droots = search_roots(R, a.distill_roots, gen, dev)
    rr, ra, tgt = consensus_targets(a, net, strust, R, droots, 8, dev, gen)
    log("consensus targets", int(rr.shape[0]), float(tgt.mean()), f"{time.time()-t0:.0f}s")
    amp, _, meta["distill_lir_hist"] = distill(a, net, R, droots, rr, ra, tgt, dev, fused=False)
    amp_f, net_fused, meta["distill_fused_hist"] = distill(a, net, R, droots, rr, ra, tgt, dev, fused=True)
    meta["lir_param_hash_before"] = h_before
    meta["lir_param_hash_after"] = param_hash(net)
    meta["fused_param_hash_after"] = param_hash(net_fused)
    torch.save(net_fused.state_dict(), os.path.join(a.out, "real_fused.pt"))
    torch.save(amp_f.state_dict(), os.path.join(a.out, "amp_fused.pt"))
    log("distill", meta["lir_param_hash_before"] == meta["lir_param_hash_after"], f"{time.time()-t0:.0f}s")
    # rollout trust on CAL (both checkpoints)
    rtrust, meta["rollout_trust"] = fit_rollout_trust(a, [net30, net], [strust30, strust], Rc, dev)
    torch.save({"state": rtrust.state_dict(), "mu": rtrust.mu, "sd": rtrust.sd}, os.path.join(a.out, "rollout_trust.pt"))
    log("rollout trust", meta["rollout_trust"]["hist"][-3:], f"{time.time()-t0:.0f}s")
    # D2 aggregators on CAL
    att, meta["agg"] = fit_aggregators(a, net, strust, rtrust, amp, Rc, dev)
    torch.save(att.state_dict(), os.path.join(a.out, "attn_l3.pt"))
    log("aggregators", {k: v for k, v in meta["agg"].items() if not k.endswith("hist")}, f"{time.time()-t0:.0f}s")
    meta["action_scales"] = fit_action_scales(a, net, strust, rtrust, amp, Rc, dev)
    log("action scales", {k: (v["scale"], v["logloss"], v["logloss_l0"]) for k, v in meta["action_scales"].items()})
    torch.save(amp.state_dict(), os.path.join(a.out, "amp.pt"))
    meta["sec"] = time.time() - t0
    json.dump(meta, open(os.path.join(a.out, "train_meta.json"), "w"), indent=1, default=str)
    log("done", f"{meta['sec']:.0f}s")


if __name__ == "__main__":
    main()
