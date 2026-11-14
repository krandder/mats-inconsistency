#!/usr/bin/env python3
"""R5-E2 training for one seed and one architecture (GPU). Stages (pre-registered in PREREG-R5-E2.md):
  1 theta_real on TRAIN real trajectories only: any-order masked CE on the opponent's slots + policy CE (behaviour
    model of both sides) + transition CE + outcome CE. 16k steps; checkpoints at 30% and 100%.
  2 slot-trust heads (stiff_voi order) on CAL, one per checkpoint.
  3 theta_amp distillation (LIR, chi_real = 0) on TRAIN roots from cone-rollout consensus (8 per legal action).
  4 rollout-trust head theta_trust on CAL: cone rollouts from both checkpoints, root action = the real action,
    depth ~ U{1..6}, target = excess surprise of the leaf claim against the realised result.
  5 aggregators on CAL two-flank roots: ataraxos_mmd (eta_b, eta_v), kc_dedup (W_K2, u_b, u_v), attn_L3, fitted on
    bags of 24 with flank skew p_L ~ Beta(0.5, 0.5) against the unskewed K = 128 coverage reference on the joint
    8 x 6 belief logits + root value (model-only target: no oracle, no simulator).
   python3 train6.py --seed 101 --arch primary --out OUT [--smoke]
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
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
import model6 as M  # noqa: E402

DATA = os.environ.get("R5E2_DATA", "/tmp/r5e2/data")
AC = dict(device_type="cuda", dtype=torch.bfloat16)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def autocast_train(dev):
    """bf16 autocast for theta_real's training steps only."""
    return torch.autocast(**AC) if dev.type == "cuda" else contextlib.nullcontext()


def autocast(dev):
    """Inference (beliefs, likelihood ratios, rollouts) runs in fp32 (TF32 matmuls on GPU)."""
    return contextlib.nullcontext()


def load_split(name, dev):
    z = np.load(os.path.join(DATA, f"{name}.npz"))
    R = {k: torch.tensor(z[k].astype(np.int32 if k == "game" else np.int16), device=dev) for k in z.files}
    assert int(R["game"].max()) == int(z["game"].max()), "game index overflow"
    g = z["game"].astype(np.int64)
    t = z["t"].astype(np.int64)
    t0 = np.maximum(t - M.WIN, t % 2)
    i0 = np.arange(len(t)) - (t - t0)
    assert (g[i0] == g).all() and (t[i0] == t0).all(), "records are not stored game by game in time order"
    R["i0"] = torch.tensor(i0, device=dev)
    R["nw"] = torch.tensor(t - t0, device=dev)
    R["n"] = len(t)
    return R


def param_hash(mod):
    h = hashlib.sha256()
    for k, v in sorted(mod.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def freeze(m):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# --------------------------------------------------------------------------------------------- theta_real
def train_batch(R, idx, gen):
    B = idx.shape[0]
    dev = idx.device
    hidden = (R["slot_pubrev"][idx] == 0) & (R["slot_loc"][idx] < M.NSQ)
    u = torch.rand(B, 1, generator=gen, device=dev)
    rev = (torch.rand(B, M.NS, generator=gen, device=dev) < u) & hidden
    tp = M.tape_from(R, idx, reveal_hidden=rev)
    ply = R["ply"][idx].long()
    npl = R["n_ply"][idx].long()
    valid = torch.arange(M.MAXP, device=dev)[None] < npl[:, None]
    for c, k in enumerate(M.PLY_KEYS):
        tp[k] = ply[..., c] * valid
    return tp, hidden & ~rev, valid, ply


def real_losses(net, R, idx, gen):
    tp, bmask, valid, ply = train_batch(R, idx, gen)
    h, _ = net(tp)
    tru = R["slot_true"][idx].long() - 1
    bl = net.belief_logits(h).float()
    lb = (F.cross_entropy(bl.reshape(-1, 6), tru.clamp_min(0).reshape(-1), reduction="none") *
          bmask.reshape(-1).float()).sum() / bmask.sum().clamp_min(1)
    res = R["result"][idx].long()
    dev = idx.device
    pre = torch.tensor([M.pre_idx(j) for j in range(M.MAXP)], device=dev)
    ai = torch.tensor([M.a_idx(j) for j in range(M.MAXP)], device=dev)
    oi = torch.tensor([M.o_idx(j) for j in range(M.MAXP)], device=dev)
    vf = valid.reshape(-1).float()
    nv = vf.sum().clamp_min(1)
    pol = net.h_policy(h[:, pre]).float().reshape(-1, M.NPOL)
    tgt = (ply[..., 1] * M.NSQ + ply[..., 2]).reshape(-1)
    lp = (F.cross_entropy(pol, tgt, reduction="none") * vf).sum() / nv
    lo, la, ld = net.trans_logits(h[:, ai])
    lt = sum((F.cross_entropy(x.float().reshape(-1, x.shape[-1]), ply[..., c].reshape(-1), reduction="none") * vf).sum() / nv
             for x, c in ((lo, 3), (la, 4), (ld, 5)))
    vg = F.cross_entropy(net.h_value(h[:, M.GLOB]).float(), res)
    rr = res[:, None].expand(-1, M.MAXP).reshape(-1)
    va = (F.cross_entropy(net.h_value(h[:, ai]).float().reshape(-1, 3), rr, reduction="none") * vf).sum() / nv
    vo = (F.cross_entropy(net.h_value(h[:, oi]).float().reshape(-1, 3), rr, reduction="none") * vf).sum() / nv
    return lb, lp, lt, (vg + va + vo) / 3


def train_real(a, R, Rc, dev, out):
    torch.manual_seed(a.seed)
    net = M.make_net(a.arch).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    N = R["n"]
    S = a.steps
    warm = min(300, max(1, S // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, S) / S)))
    rows = []
    t0 = time.time()
    ck30 = int(0.3 * S)
    for st in range(S):
        net.train()
        idx = torch.randint(0, N, (a.batch,), device=dev, generator=gen)
        with autocast_train(dev):
            lb, lp, lt, lv = real_losses(net, R, idx, gen)
        loss = lb + lp + lt + lv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if st % 500 == 0 or st == S - 1:
            net.eval()
            with torch.no_grad(), autocast_train(dev):
                g2 = torch.Generator(device=dev).manual_seed(999)
                vi = torch.randint(0, Rc["n"], (4096,), device=dev, generator=g2)
                vb, vp, vt, vv = real_losses(net, Rc, vi, g2)
            row = {"step": st, "sec": round(time.time() - t0, 1)}
            for nm, x in zip(("belief", "policy", "trans", "value"), (lb, lp, lt, lv)):
                row["train_" + nm] = float(x.detach())
            for nm, x in zip(("belief", "policy", "trans", "value"), (vb, vp, vt, vv)):
                row["cal_" + nm] = float(x)
            rows.append(row)
            log("real", json.dumps(row))
        if st + 1 == ck30:
            torch.save(net.state_dict(), os.path.join(out, "real_30.pt"))
    torch.save(net.state_dict(), os.path.join(out, "real_100.pt"))
    with open(os.path.join(out, "train_curves.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return freeze(net), {"params": nparam, "sec": time.time() - t0, "last": rows[-1]}


def fit_slot_trust(a, net, Rc, dev):
    gen = torch.Generator(device=dev).manual_seed(a.seed + 1)
    tr = M.TrustNet(net.d + 2).to(dev)
    opt = torch.optim.Adam(tr.parameters(), lr=1e-3)
    hist = []
    for st in range(a.trust_steps):
        idx = torch.randint(0, Rc["n"], (512,), device=dev, generator=gen)
        hidden = (Rc["slot_pubrev"][idx] == 0) & (Rc["slot_loc"][idx] < M.NSQ)
        u = torch.rand(512, 1, generator=gen, device=dev)
        rev = (torch.rand(512, M.NS, generator=gen, device=dev) < u) & hidden
        tp = M.tape_from(Rc, idx, reveal_hidden=rev)
        with torch.no_grad(), autocast(dev):
            h, _ = net(tp, root_only=True)
            q = torch.softmax(net.belief_logits(h).float(), -1)
            tgt = M.excess_surprise(q, (Rc["slot_true"][idx].long() - 1).clamp_min(0))
            f = M.slot_features(h, q, tp)
        m = (hidden & ~rev).float()
        l = (((tr(f) - tgt) ** 2) * m).sum() / m.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        l.backward()
        opt.step()
        if st % 500 == 0:
            hist.append(float(l))
    return freeze(tr), hist


# --------------------------------------------------------------------------------------------- roots and rollouts
@torch.no_grad()
def make_roots(net, R, ix, dev):
    """Root batch for record indices ix: root tape at t, window root tape at t0, window plies, 0-step beliefs."""
    tp = M.tape_from(R, ix)
    i0 = R["i0"][ix]
    tpw = M.tape_from(R, i0)
    wply = R["ply"][i0][:, :M.WIN].long()
    nw = R["nw"][ix].long()
    with autocast(dev):
        l0b, _ = M.belief0(net, tp)
        l0v = M.root_value_logit(net, tp)
    hidden = (tp["slot_val"] == 0) & (tp["slot_loc"] < M.NSQ)
    lg = R["legal"][ix].long()
    nl = R["n_legal"][ix].long()
    return {"tp": tp, "tpw": tpw, "wply": wply, "nw": nw, "l0b": l0b.float(), "l0v": l0v.float(), "hidden": hidden,
            "legal": lg, "n_legal": nl}


def roots_repeat(rt, K):
    out = {}
    for k, v in rt.items():
        out[k] = M.tape_repeat(v, K) if isinstance(v, dict) else v.repeat_interleave(K, 0)
    return out


def flank_masks(lg):
    isL = (lg[..., 0] >= 0) & (lg[..., 0] % M.N <= 2)
    isR = (lg[..., 0] >= 0) & (lg[..., 0] % M.N >= 3)
    return isL, isR


@torch.no_grad()
def run_rollouts(net, strust, rtrust, rt, act, depth, gen, dev, order="stiff_voi", with_belief=True):
    """rt: root batch (B roots); act [B,K] indices into the legal list. Returns per-rollout tensors reshaped [B,K,...]
    plus beta (rollout trust, or 1 if rtrust is None)."""
    B, K = act.shape
    rr = roots_repeat(rt, K)
    ar = torch.arange(B, device=dev)[:, None]
    fr = rt["legal"][ar, act, 0].reshape(-1)
    to = rt["legal"][ar, act, 1].reshape(-1)
    with autocast(dev):
        o = M.full_rollouts(net, strust, rr, fr, to, depth, gen, order=order, with_belief=with_belief)
    beta = trust_beta(rtrust, o["feats"]) if rtrust is not None else torch.ones_like(o["e"])
    o["beta"] = beta
    o["act"] = act.reshape(-1)
    return {k: v.view(B, K, *v.shape[1:]) for k, v in o.items()}


def trust_beta(tr, f):
    return M.beta_from(tr((f - tr.mu) / tr.sd))


def sample_flank_actions(rt, n_per_flank, gen, dev):
    """[B, 2, n] legal-list indices: n uniform draws within the Left flank, n within the Right flank."""
    isL, isR = flank_masks(rt["legal"])
    out = []
    for msk in (isL, isR):
        w = msk.float()
        w = torch.where(w.sum(1, keepdim=True) > 0, w, torch.ones_like(w) * (rt["legal"][..., 0] >= 0).float())
        out.append(torch.multinomial(w, n_per_flank, replacement=True, generator=gen))
    return torch.stack(out, 1)


def two_flank_roots(R, n, gen, dev, tmin=4, tmax=36):
    isL, isR = flank_masks(R["legal"].long())
    ok = isL.any(1) & isR.any(1) & (R["t"].long() >= tmin) & (R["t"].long() <= tmax)
    cand = torch.nonzero(ok).squeeze(1)
    return cand[torch.randperm(cand.shape[0], generator=gen, device=dev)[:n]]


@torch.no_grad()
def flank_pool(net, strust, rtrust, rt, n_per_flank, gen, dev, chunk=128):
    """Per root [B, 2(flank), n] rollouts with every per-rollout tensor."""
    outs = []
    B = rt["l0v"].shape[0]
    for c0 in range(0, B, chunk):
        sub = {k: (M.tape_index(v, slice(c0, c0 + chunk)) if isinstance(v, dict) else v[c0:c0 + chunk]) for k, v in rt.items()}
        acts = sample_flank_actions(sub, n_per_flank, gen, dev)                     # [b,2,n]
        b = acts.shape[0]
        o = run_rollouts(net, strust, rtrust, sub, acts.reshape(b, -1), 4, gen, dev)
        outs.append({k: v.view(b, 2, n_per_flank, *v.shape[2:]) for k, v in o.items()})
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def bag_from_pool(pool, rt, fl, j):
    """Bag tensors: fl [N,K] flank of each draw, j [N,K] index within that flank's pool."""
    N_ = fl.shape[0]
    ar = torch.arange(N_, device=fl.device)[:, None]
    g = lambda x: x[ar, fl, j]
    return {"dlb": g(pool["dlb"]), "cov": g(pool["cov"]), "dlv": g(pool["dlv"]), "e": g(pool["e"]),
            "beta": g(pool["beta"]), "ebar": g(pool["ebar"]), "l0b": rt["l0b"], "l0v": rt["l0v"],
            "valid": torch.ones_like(fl, dtype=torch.bool), "fl": fl}


def skew_bag(pool, rt, K, pL, gen):
    N_, _, n = pool["e"].shape
    dev = pool["e"].device
    fl = (torch.rand(N_, K, generator=gen, device=dev) >= pL[:, None]).long()
    j = torch.randint(0, n, (N_, K), generator=gen, device=dev)
    return bag_from_pool(pool, rt, fl, j)


def full_bag(pool, rt):
    N_, _, n = pool["e"].shape
    dev = pool["e"].device
    fl = torch.arange(2, device=dev).repeat_interleave(n)[None].expand(N_, -1)
    j = torch.arange(n, device=dev).repeat(2)[None].expand(N_, -1)
    return bag_from_pool(pool, rt, fl, j)


def joint_loss(lb, lv, ref_lb, ref_v, hidden):
    """KL(ref || mu) summed over the 6 types, mean over hidden slots, + BCE of the value logit against the ref value."""
    kl = (ref_lb.exp() * (ref_lb - lb)).sum(-1)
    hb = hidden.float()
    lbel = ((kl * hb).sum(1) / hb.sum(1).clamp_min(1)).mean()
    lval = F.binary_cross_entropy_with_logits(lv, ref_v.clamp(1e-4, 1 - 1e-4))
    return lbel + lval, lbel, lval


def fit_aggregators(a, net, strust, rtrust, amp, Rc, dev):
    gen = torch.Generator(device=dev).manual_seed(a.seed + 31)
    roots = two_flank_roots(Rc, a.n_agg_roots, gen, dev)
    rt = make_roots(net, Rc, roots, dev)
    pool = flank_pool(net, strust, rtrust, rt, 32, gen, dev)
    refp = flank_pool(net, strust, rtrust, rt, 64, gen, dev)
    ref_lb, ref_vl, ref_n = M.coverage_reference(full_bag(refp, rt))
    ref_v = torch.sigmoid(ref_vl)
    hidden = rt["hidden"]
    res = {"n_roots": int(roots.shape[0]), "ref_cov_mean": float((ref_n * hidden.float()).sum() / hidden.float().sum()),
           "ref_slots_uncovered": float(((ref_n == 0) & hidden).float().sum() / hidden.float().sum())}
    K = 24

    def fit(n_steps, fn, params, lr):
        opt = torch.optim.Adam(params, lr=lr)
        hist = []
        for s in range(n_steps):
            pL = torch.distributions.Beta(0.5, 0.5).sample((roots.shape[0],)).to(dev)
            b = skew_bag(pool, rt, K, pL, gen)
            lb, lv = fn(b)
            l, l1, l2 = joint_loss(lb, lv, ref_lb, ref_v, hidden)
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
            if s % 200 == 0 or s == n_steps - 1:
                hist.append([s, float(l), float(l1), float(l2)])
        return hist

    le_b = torch.tensor(math.log(0.05), device=dev, requires_grad=True)
    le_v = torch.tensor(math.log(0.05), device=dev, requires_grad=True)
    res["mmd_hist"] = fit(a.agg_steps, lambda b: M.agg_mmd(b, le_b.exp(), le_v.exp()), [le_b, le_v], 1e-2)
    res["eta_b"], res["eta_v"] = float(le_b.exp()), float(le_v.exp())
    lu_b = torch.tensor(math.log(0.3), device=dev, requires_grad=True)
    lu_v = torch.tensor(math.log(0.3), device=dev, requires_grad=True)
    res["kc_hist"] = fit(a.agg_steps, lambda b: M.agg_kc(b, amp.key(b["ebar"]), lu_b.exp(), lu_v.exp()),
                         list(amp.keyW2.parameters()) + [lu_b, lu_v], 3e-3)
    res["u_b"], res["u_v"] = float(lu_b.exp()), float(lu_v.exp())
    torch.manual_seed(a.seed + 41)
    att = M.AttnAgg().to(dev)
    res["attn_hist"] = fit(a.agg_steps, lambda b: att(b, amp.key(b["ebar"]).detach()), list(att.parameters()), 1e-3)
    # calibration-split diagnostics on a fresh skew draw (not test numbers)
    with torch.no_grad():
        pL = torch.distributions.Beta(0.5, 0.5).sample((roots.shape[0],)).to(dev)
        b = skew_bag(pool, rt, K, pL, gen)
        for nm, fn in (("mix", lambda b: M.agg_mix(b)), ("mmd", lambda b: M.agg_mmd(b, res["eta_b"], res["eta_v"])),
                       ("kc", lambda b: M.agg_kc(b, amp.key(b["ebar"]), res["u_b"], res["u_v"])),
                       ("attn", lambda b: att(b, amp.key(b["ebar"])))):
            lb, lv = fn(b)
            _, l1, l2 = joint_loss(lb, lv, ref_lb, ref_v, hidden)
            res[f"cal_fit_{nm}"] = [float(l1), float(l2)]
    freeze(att)
    for p in amp.keyW2.parameters():
        p.requires_grad_(False)
    return att, res


# --------------------------------------------------------------------------------------------- theta_amp, theta_trust
@torch.no_grad()
def consensus_targets(net, strust, R, roots, K, gen, dev, chunk=2048):
    """Per (root, legal action) pair: mean leaf effective win of K cone rollouts of that action (depth 4)."""
    nl = R["n_legal"][roots].long()
    rr = torch.repeat_interleave(torch.arange(roots.shape[0], device=dev), nl)
    ra = torch.cat([torch.arange(int(n), device=dev) for n in nl.tolist()])
    tgt = torch.zeros(rr.shape[0], device=dev)
    for c0 in range(0, rr.shape[0], chunk):
        ri, ai = rr[c0:c0 + chunk], ra[c0:c0 + chunk]
        rt = make_roots(net, R, roots[ri], dev)
        o = run_rollouts(net, strust, None, rt, ai[:, None].expand(-1, K), 4, gen, dev, with_belief=False)
        tgt[c0:c0 + chunk] = o["e"].mean(1)
    return rr, ra, tgt


def distill(a, net, R, roots, rr, ra, tgt, dev):
    torch.manual_seed(a.seed + 5)
    amp = M.AmpNet(net.d).to(dev)
    opt = torch.optim.Adam(amp.q.parameters(), lr=3e-4)
    gen = torch.Generator(device=dev).manual_seed(a.seed + 11)
    lg = R["legal"][roots].long()
    y = tgt.clamp(0.01, 0.99)
    hist = []
    for st in range(a.distill_steps):
        b = torch.randint(0, rr.shape[0], (512,), device=dev, generator=gen)
        tp = M.tape_from(R, roots[rr[b]])
        tp["ply_from"][:, 0] = lg[rr[b], ra[b], 0]
        tp["ply_to"][:, 0] = lg[rr[b], ra[b], 1]
        with torch.no_grad(), autocast(dev):
            h, _ = net(tp)
            base = M.eff_logit(net.h_value(h[:, M.a_idx(0)]))
        l0 = base.float() + amp.correction(h[:, M.a_idx(0)])
        loss = F.binary_cross_entropy_with_logits(l0, y[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if st % 500 == 0:
            hist.append(float(loss))
    amp.eval()
    for p in amp.q.parameters():
        p.requires_grad_(False)
    return amp, hist


@torch.no_grad()
def real_action_rollouts(net, strust, R, roots, depth, gen, dev, chunk=1024):
    """One cone rollout per root whose root action is the action really played; depth [n]. Returns feats, q_leaf."""
    feats, qs = [], []
    for c0 in range(0, roots.shape[0], chunk):
        ix = roots[c0:c0 + chunk]
        rt = make_roots(net, R, ix, dev)
        fr, to = R["ply"][ix, 0, 1].long(), R["ply"][ix, 0, 2].long()
        dd = depth[c0:c0 + chunk]
        f_ = torch.zeros(ix.shape[0], net.d + M.N_TRUST_EXTRA, device=dev)
        q_ = torch.zeros(ix.shape[0], 3, device=dev)
        for d in torch.unique(dd).tolist():
            sel = dd == d
            sub = {k: (M.tape_index(v, sel) if isinstance(v, dict) else v[sel]) for k, v in rt.items()}
            with autocast(dev):
                o = M.full_rollouts(net, strust, sub, fr[sel], to[sel], int(d), gen, with_belief=False)
            f_[sel] = o["feats"].float()
            q_[sel] = o["q_leaf"].float()
        feats.append(f_)
        qs.append(q_)
    return torch.cat(feats), torch.cat(qs)


def fit_rollout_trust(a, nets, struts, Rc, dev):
    gen = torch.Generator(device=dev).manual_seed(a.seed + 21)
    ok = (Rc["n_ply"] >= 1)
    cand = torch.nonzero(ok).squeeze(1)
    roots = cand[torch.randperm(cand.shape[0], generator=gen, device=dev)[:a.n_trust_roots]]
    X, Y = [], []
    for net, st in zip(nets, struts):
        for rep in range(2):
            d = torch.randint(1, 7, (roots.shape[0],), device=dev, generator=gen)
            f, q = real_action_rollouts(net, st, Rc, roots, d, gen, dev)
            X.append(f)
            Y.append(M.excess_surprise(q, Rc["result"][roots].long()))
    X = torch.cat(X)
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
    freeze(tr)
    return tr, {"n": int(X.shape[0]), "hist": hist, "target_mean": float(Y.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arch", choices=list(M.ARCHS), required=True)
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
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.batch, a.trust_steps, a.distill_roots, a.distill_steps = 40, 64, 20, 24, 20
        a.n_trust_roots, a.n_agg_roots, a.agg_steps = 32, 16, 10
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    meta = {"seed": a.seed, "arch": a.arch, "arch_cfg": M.ARCHS[a.arch], "args": vars(a), "device": str(dev)}
    t0 = time.time()
    R = load_split("train", dev)
    Rc = load_split("cal", dev)
    log("data", {"train": R["n"], "cal": Rc["n"]})
    net, meta["real"] = train_real(a, R, Rc, dev, a.out)
    h_before = param_hash(net)
    net30 = M.make_net(a.arch).to(dev)
    net30.load_state_dict(torch.load(os.path.join(a.out, "real_30.pt"), map_location=dev))
    freeze(net30)
    strust, meta["slot_trust_hist"] = fit_slot_trust(a, net, Rc, dev)
    strust30, _ = fit_slot_trust(a, net30, Rc, dev)
    torch.save(strust.state_dict(), os.path.join(a.out, "slot_trust.pt"))
    torch.save(strust30.state_dict(), os.path.join(a.out, "slot_trust_30.pt"))
    log("slot trust", meta["slot_trust_hist"][-2:], f"{time.time()-t0:.0f}s")
    gen = torch.Generator(device=dev).manual_seed(a.seed + 3)
    ok = (R["n_legal"] >= 2) & (R["t"] <= 40)
    cand = torch.nonzero(ok).squeeze(1)
    droots = cand[torch.randperm(cand.shape[0], generator=gen, device=dev)[:a.distill_roots]]
    rr, ra, tgt = consensus_targets(net, strust, R, droots, 8, gen, dev)
    log("consensus targets", int(rr.shape[0]), float(tgt.mean()), f"{time.time()-t0:.0f}s")
    amp, meta["distill_hist"] = distill(a, net, R, droots, rr, ra, tgt, dev)
    meta["lir_param_hash_before"] = h_before
    meta["lir_param_hash_after"] = param_hash(net)
    log("distill", h_before == meta["lir_param_hash_after"], f"{time.time()-t0:.0f}s")
    rtrust, meta["rollout_trust"] = fit_rollout_trust(a, [net30, net], [strust30, strust], Rc, dev)
    torch.save({"state": rtrust.state_dict(), "mu": rtrust.mu, "sd": rtrust.sd}, os.path.join(a.out, "rollout_trust.pt"))
    log("rollout trust", meta["rollout_trust"]["hist"][-2:], f"{time.time()-t0:.0f}s")
    att, meta["agg"] = fit_aggregators(a, net, strust, rtrust, amp, Rc, dev)
    torch.save(att.state_dict(), os.path.join(a.out, "attn_l3.pt"))
    torch.save(amp.state_dict(), os.path.join(a.out, "amp.pt"))
    log("aggregators", {k: v for k, v in meta["agg"].items() if not k.endswith("hist")}, f"{time.time()-t0:.0f}s")
    meta["param_hash_final"] = param_hash(net)
    meta["sec"] = time.time() - t0
    json.dump(meta, open(os.path.join(a.out, "train_meta.json"), "w"), indent=1, default=str)
    log("done", f"{meta['sec']:.0f}s")


if __name__ == "__main__":
    main()
