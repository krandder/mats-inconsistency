r"""R6-E3 (E-MS-MARGINAL) library on top of r5/ (model6, train6, eval6; the r5 code is not modified).

Marginal 4-ply window messages (spec 3.2.1): for a root at time t (observer to move) the anchor is theta_real's 0-step
belief at t0 = max(t - 4, t mod 2) (a belief cache last updated 4 plies ago). For every hidden slot s and type k the
message is the log-likelihood of the opponent's REAL moves in the window t0..t-1 under theta_real's policy head with
Z_s = k and EVERY other hidden slot MASK (no neighbour sampling, so no uniqueness vetoes):
    log mu_s = log softmax(log P_anchor,s + LL_s),   msg_s = log mu_s - log P_anchor,s.
A cone rollout r carries msg_s for the slots s of its cone (the 3 hidden slots nearest the root move's target square,
as r5 model6.select_cone) and 0 elsewhere; its value claim, trust features and attention fingerprint come from the r5
value rollout (stiff_voi cone sample, depth 4). Because msg_s does not depend on the rollout, rollouts that cover the
same slot carry identical messages (exact duplicates).
Keys K4 (spec 3.2.2): k_r = normalise(W [LayerNorm(e_r - mean_bag e) || s_r]), s_r[s] = ||dl_r[s]||_2 (8-slot salience),
fitted with the kc_dedup calibration loss + L_key = mean_{r != b} (<k_r, k_b> - <s_r/|s_r|, s_b/|s_b|>)^2.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

R5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r5")
sys.path.insert(0, R5)
import model6 as M  # noqa: E402
import train6 as T  # noqa: E402

N_BASE = 4
N_SKEW = 20
K_BAG = N_BASE + N_SKEW
BAG_KEYS = ("dlb", "cov", "dlv", "e", "beta", "ebar", "sal", "fl")


# --------------------------------------------------------------------------------------------- marginal messages
@torch.no_grad()
def marginal_messages(net, rt, chunk=4096):
    """rt: root batch (train6.make_roots). Returns anchor [B,8,6] log-probs at t0, msg [B,8,6] (0 on non-hidden slots),
    lmu [B,8,6] = log softmax(anchor + msg)."""
    tpw, wply, nw = rt["tpw"], rt["wply"], rt["nw"]
    B = nw.shape[0]
    dev = nw.device
    base = M.tape_clone(tpw)
    live = torch.arange(M.WIN, device=dev)[None] < nw[:, None]
    for ci, k_ in enumerate(M.PLY_KEYS):
        z = torch.zeros(B, M.MAXP, dtype=torch.long, device=dev)
        z[:, :M.WIN] = torch.where(live, wply[..., ci], torch.zeros_like(wply[..., ci]))
        base[k_] = z
    opp = (live & (wply[..., 0] == 1)).float()
    mv = (wply[..., 1] * M.NSQ + wply[..., 2]).clamp(0, M.NPOL - 1)
    h0, _ = net(tpw, root_only=True)
    anchor = torch.log_softmax(net.belief_logits(h0).float(), -1)
    hid = rt["hidden"]
    ll = torch.zeros(B, M.NS, 6, device=dev)
    has_w = opp.sum(1) > 0
    tapes, rr_l, ss_l, kk_l = [], [], [], []
    for s in range(M.NS):
        r = torch.nonzero(hid[:, s] & has_w).squeeze(1)
        if r.numel() == 0:
            continue
        for k in range(1, 7):
            t_ = M.tape_index(base, r)
            v = t_["slot_val"].clone()
            v[:, s] = k
            t_["slot_val"] = v
            tapes.append(t_)
            rr_l.append(r)
            ss_l.append(torch.full_like(r, s))
            kk_l.append(torch.full_like(r, k))
    if tapes:
        big = M.tape_cat(tapes)
        rr, ss, kk = torch.cat(rr_l), torch.cat(ss_l), torch.cat(kk_l)
        g = torch.zeros(rr.shape[0], device=dev)
        for c0 in range(0, rr.shape[0], chunk):
            sl = slice(c0, c0 + chunk)
            h, _ = net(M.tape_index(big, sl))
            for j in range(M.WIN):
                lp = torch.log_softmax(net.h_policy(h[:, M.pre_idx(j)]).float(), -1)
                g[sl] += lp.gather(1, mv[rr[sl], j:j + 1]).squeeze(1) * opp[rr[sl], j]
        ll[rr, ss, kk - 1] = g
    lmu = torch.log_softmax(anchor + ll, -1)
    msg = torch.where(hid[..., None], lmu - anchor, torch.zeros_like(lmu)).clamp(-8, 8)
    return anchor, msg, torch.log_softmax(anchor + msg, -1)


def roots_e3(net, R, ix, dev):
    """Root batch + marginal messages; l0b := the t-4 anchor (beliefs are pooled on top of the anchor)."""
    rt = T.make_roots(net, R, ix, dev)
    anchor, msg, lmu = marginal_messages(net, rt)
    rt["p0b"] = rt["l0b"]                 # theta_real's current one-pass belief at t (reported, never pooled on)
    rt["l0b"] = anchor
    rt["msg"] = msg
    rt["lmu"] = lmu
    return rt


def rt_index(rt, sel):
    return {k: (M.tape_index(v, sel) if isinstance(v, dict) else v[sel]) for k, v in rt.items()}


# --------------------------------------------------------------------------------------------- cones, base cover
@torch.no_grad()
def cones_for(rt, act):
    """Cone [B,K,3] of every rollout whose root move is legal-list index act [B,K]."""
    B, K = act.shape
    tp = M.tape_repeat(rt["tp"], K)
    ar = torch.arange(B, device=act.device)[:, None]
    fr = rt["legal"][ar, act, 0].reshape(-1)
    to = rt["legal"][ar, act, 1].reshape(-1)
    cone, _ = M.select_cone(tp, fr, to)
    return cone.view(B, K, -1)


def cover_mask(cone):
    """cone [..., 3] (slot index or -1) -> [..., 8] bool."""
    out = torch.zeros(*cone.shape[:-1], M.NS, dtype=torch.bool, device=cone.device)
    out.scatter_(-1, cone.clamp_min(0), cone >= 0)
    # a padded entry (-1 -> index 0) must not mark slot 0 unless a real entry does
    real0 = (cone == 0).any(-1)
    out[..., 0] = real0
    return out


@torch.no_grad()
def base_cover(rt, n=N_BASE):
    """Per root: n legal moves picked greedily to maximise the number of hidden slots covered by their cones (ties: the
    lowest legal-list index; once nothing new can be covered, the lowest unpicked index). Returns [B, n] legal indices."""
    B = rt["legal"].shape[0]
    L = int(rt["n_legal"].max())
    dev = rt["legal"].device
    allact = torch.minimum(torch.arange(L, device=dev)[None].expand(B, -1), (rt["n_legal"] - 1)[:, None])
    cov = cover_mask(cones_for(rt, allact)) & rt["hidden"][:, None]          # [B,L,8]
    valid = torch.arange(L, device=dev)[None] < rt["n_legal"][:, None]
    covered = torch.zeros(B, M.NS, dtype=torch.bool, device=dev)
    picked = torch.zeros(B, L, dtype=torch.bool, device=dev)
    out = []
    ar = torch.arange(B, device=dev)
    for _ in range(n):
        gain = (cov & ~covered[:, None]).sum(-1).float()
        score = gain * 1000 - torch.arange(L, device=dev)[None].float()
        score = score.masked_fill(~valid | picked, -1e9)
        j = score.argmax(1)
        out.append(j)
        picked[ar, j] = True
        covered |= cov[ar, j]
    return torch.stack(out, 1)


# --------------------------------------------------------------------------------------------- rollouts and bags
@torch.no_grad()
def e3_rollouts(net, strust, rtrust, rt, act, gen, dev, depth=4):
    """r5 value rollouts (stiff_voi cone sample) + marginal belief residuals for their cones. Tensors [B, K, ...]."""
    o = T.run_rollouts(net, strust, rtrust, rt, act, depth, gen, dev, with_belief=False)
    cov = cover_mask(o["cone"]) & rt["hidden"][:, None]
    o["cov"] = cov
    o["dlb"] = rt["msg"][:, None] * cov[..., None].float()
    o["sal"] = o["dlb"].norm(dim=-1)
    ar = torch.arange(act.shape[0], device=dev)[:, None]
    o["fl"] = (rt["legal"][ar, act, 0] % M.N >= 3).long()
    return o


def bag_cat(parts):
    return {k: torch.cat([p[k] for p in parts], 1) for k in BAG_KEYS}


def bag_select(pool, idx):
    ar = torch.arange(idx.shape[0], device=idx.device)[:, None]
    return {k: pool[k][ar, idx] for k in BAG_KEYS}


def finish_bag(b, rt):
    b = dict(b)
    b["l0b"], b["l0v"] = rt["l0b"], rt["l0v"]
    b["valid"] = torch.ones(b["e"].shape, dtype=torch.bool, device=b["e"].device)
    return b


def skewed_bag(base, poolL, poolR, rt, pL, gen, n_skew=N_SKEW):
    """base cover rollouts + n_skew rollouts: nL ~ Binomial(n_skew, pL) from the Left pool and the rest from the Right
    pool, taken in pool order (pools are i.i.d. draws, so this is a random subset). Pools hold >= n_skew rollouts."""
    B = pL.shape[0]
    dev = pL.device
    nL = torch.distributions.Binomial(n_skew, pL).sample().long()
    j = torch.arange(n_skew, device=dev)[None].expand(B, -1)
    fromL = j < nL[:, None]
    sL = bag_select(poolL, j.clamp(max=poolL["e"].shape[1] - 1))
    sR = bag_select(poolR, (j - nL[:, None]).clamp(0, poolR["e"].shape[1] - 1))
    sk = {k: torch.where(fromL.view(B, n_skew, *([1] * (sL[k].dim() - 2))), sL[k], sR[k]) for k in BAG_KEYS}
    return finish_bag(bag_cat([base, sk]), rt), nL


# --------------------------------------------------------------------------------------------- K4 keys and aggregators
class KeyK4(nn.Module):
    def __init__(self, d, kdim=16):
        super().__init__()
        self.lin = nn.Linear(d + M.NS, kdim)

    def forward(self, b):
        e = b["ebar"].float()
        v = b["valid"].float()[..., None]
        mu = (e * v).sum(1, keepdim=True) / v.sum(1, keepdim=True).clamp_min(1)
        ec = F.layer_norm(e - mu, (e.shape[-1],))
        return F.normalize(self.lin(torch.cat([ec, b["sal"].float()], -1)), dim=-1)


def key_align_loss(k, sal, valid):
    s = F.normalize(sal.float(), dim=-1)
    g = k @ k.transpose(1, 2) - s @ s.transpose(1, 2)
    m = (valid[:, :, None] & valid[:, None, :]).float() * (1 - torch.eye(k.shape[1], device=k.device))[None]
    return ((g ** 2) * m).sum() / m.sum().clamp_min(1)


def agg_scaled_mix(b, a_b, a_v):
    v = b["valid"].float()
    lb = b["l0b"] + a_b * (b["dlb"] * v[..., None, None]).sum(1)
    lv = b["l0v"] + a_v * (b["dlv"] * v).sum(1)
    return torch.log_softmax(lb, -1), lv


def union_call(b):
    """Exact per-slot deduplication: each covered slot's (identical) message counted once; uncovered slots keep the
    anchor (r5 coverage_reference); value = plain mean of the leaf claims."""
    lb, lv, _ = M.coverage_reference(b)
    return lb, lv


def joint_loss(lb, lv, ref_lb, ref_v, hidden):
    kl = (ref_lb.exp() * (ref_lb - lb)).sum(-1)
    hb = hidden.float()
    lbel = ((kl * hb).sum(1) / hb.sum(1).clamp_min(1)).mean()
    lval = F.binary_cross_entropy_with_logits(lv, ref_v.clamp(1e-4, 1 - 1e-4))
    return lbel + lval, lbel, lval


# --------------------------------------------------------------------------------------------- action-value pooling
def entropy3(q):
    q = q.clamp_min(1e-9)
    return -(q * q.log()).sum(-1)


def pool_values(e, H, beta, kind, gamma=1.0):
    """e, H, beta [..., K] -> pooled effective win [...]: unif (mean), stiff (weights exp(-gamma H)), trust (weights beta)."""
    if kind == "unif":
        w = torch.ones_like(e)
    elif kind == "stiff":
        w = torch.exp(-gamma * H)
    elif kind == "trust":
        w = beta
    else:
        raise ValueError(kind)
    return (w * e).sum(-1) / w.sum(-1).clamp_min(1e-12)
