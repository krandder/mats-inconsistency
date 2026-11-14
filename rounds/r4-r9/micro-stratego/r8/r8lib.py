r"""R8-E3 (E-MS-SAFE-ANCHOR) library on top of r7/r7lib.py, r6/e3lib.py and r5/ (none of them is modified).

Anchor. ell_t = theta_real's one-pass 0-step belief at the current turn t (r6 roots_e3 "p0b"; KL to the exact posterior
about 0.047). Every belief aggregator here starts from ell_t: mu = log_softmax(ell_t + sum_r w_r[s] dl_r[s]), so K = 0
rollouts gives exactly ell_t.

Consistency rollouts (beliefs, E3-D1). Rollout r has the r5 cone C_r (3 hidden slots nearest its root move's target at
t), a determinization Z_r of C_r drawn from theta_real's belief at the window start t0 = t - 4 (stiff_voi any-order
unmasking on the t0 tape), its window log-likelihood LL_r = sum over the opponent's REAL moves in [t0, t) of log pi(move)
under theta_real's policy head with C_r := Z_r and the other hidden slots MASK, and the belief residual
dl_r[s] = lmu_s - ell_t,s on its cone slots (0 elsewhere), lmu_s = log softmax(log P_t0,s + LL_s) the r6 marginal window
posterior of slot s ("the rollout's claim"; identical for every rollout that covers s, as in R6/R7).

Determinized value rollouts (action values, E3-D2). Clean: cone types Z_r drawn from the belief at t0 as above, written
into the root tape at t, then r5's depth-4 value rollout (policy temperature 1). Noisy (epistemic corruption): the same
draw, then every cone slot's type replaced by a uniformly drawn DIFFERENT type ("random piece swaps"); either myopic
(depth 1) or a depth-4 random playout at policy temperature 3.5. Per rollout: e, q, H, LL_window(Z_r), r5 trust features,
cone_lp (log-prob of the final cone types under the UNTEMPERED belief at t, teacher-forced, per slot), ply_lp (mean
untempered policy log-prob of the sampled moves), descriptive flags.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

R7 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r7")
sys.path.insert(0, R7)
import r7lib as X  # noqa: E402  (puts r6 and r5 on sys.path)
import e3lib as L  # noqa: E402
import model6 as M  # noqa: E402
import train6 as T  # noqa: E402

TAU_NOISY = 3.5
LOG3 = math.log(3.0)
BAG = 16


# --------------------------------------------------------------------------------------------- window log-likelihood
@torch.no_grad()
def window_base(rt):
    """Window tape at t0 with the real window plies teacher-forced (as r6 marginal_messages), the opponent-ply mask and
    the real moves. Returns base tape [B], opp [B, WIN] float, mv [B, WIN] long."""
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
    return base, opp, mv


@torch.no_grad()
def ll_window(net, base, opp, mv, cone, zc, chunk=4096):
    """Per row: log-likelihood of the opponent's real window moves with the cone slots set to zc (others MASK)."""
    B = cone.shape[0]
    out = torch.zeros(B, device=cone.device)
    for c0 in range(0, B, chunk):
        sl = slice(c0, c0 + chunk)
        tp = M.tape_index(base, sl)
        v = tp["slot_val"].clone()
        cn, z = cone[sl], zc[sl]
        ok = cn >= 0
        ar = torch.arange(cn.shape[0], device=cn.device)[:, None].expand_as(cn)
        v[ar[ok], cn[ok]] = z[ok]
        tp["slot_val"] = v
        h, _ = net(tp)
        g = torch.zeros(cn.shape[0], device=cn.device)
        for j in range(M.WIN):
            lp = torch.log_softmax(net.h_policy(h[:, M.pre_idx(j)]).float(), -1)
            g += lp.gather(1, mv[sl, j:j + 1]).squeeze(1) * opp[sl, j]
        out[sl] = g
    return out


def cone_values(tp, cone):
    return torch.where(cone >= 0, tp["slot_val"].gather(1, cone.clamp_min(0)), torch.zeros_like(cone))


def corrupt(zc, cone, gen):
    """Every real cone entry's type -> a uniformly drawn DIFFERENT type in 1..6."""
    r = torch.randint(1, 6, zc.shape, generator=gen, device=zc.device)            # 1..5
    new = ((zc - 1 + r) % 6) + 1
    return torch.where(cone >= 0, new, zc)


def set_cone(tp, cone, zc):
    tp = M.tape_clone(tp)
    v = tp["slot_val"]
    ok = cone >= 0
    ar = torch.arange(cone.shape[0], device=cone.device)[:, None].expand_as(cone)
    v[ar[ok], cone[ok]] = zc[ok]
    tp["slot_val"] = v
    return tp


def step_from_cone(cone, step_of):
    """Order of the cone slots: the t0 sampling order (step_of from the t0 tape)."""
    return step_of


@torch.no_grad()
def det_rollouts(net, strust, roots, fr, to, gen, noisy=False, depth=4, temp=1.0, chunk=4096):
    """One determinized value rollout per row (roots repeated per row). See module docstring."""
    B = fr.shape[0]
    outs = []
    for c0 in range(0, B, chunk):
        sl = slice(c0, c0 + chunk)
        tp = M.tape_index(roots["tp"], sl)
        tpw = M.tape_index(roots["tpw"], sl)
        f_, t_ = fr[sl], to[sl]
        cone, _ = M.select_cone(tp, f_, t_)
        ncone = (cone >= 0).sum(1).clamp_min(1).float()
        tpw_z, step_of = X.sample_cone_tempered(net, tpw, cone, gen, strust, 1.0)
        zc = cone_values(tpw_z, cone)
        if noisy:
            zc = corrupt(zc, cone, gen)
        base = {k: v[sl] for k, v in roots["wbase"].items()}
        ll = ll_window(net, base, roots["wopp"][sl], roots["wmv"][sl], cone, zc)
        tpz = set_cone(tp, cone, zc)
        samp = step_of >= 0
        fwd = torch.where(samp, samp.sum(1, keepdim=True) - 1 - step_of, step_of)
        lfwd = M.reverse_logp(net, tpz, fwd)
        lrev = M.reverse_logp(net, tpz, step_of)
        pdis = (lfwd - lrev).abs() / ncone
        r = X.value_rollout_t(net, tpz, f_, t_, depth, gen, samp, temp=temp)
        q = r["q_leaf"]
        e = (q[:, 0] + 0.5 * q[:, 1]).clamp(1e-5, 1 - 1e-5)
        l0a, _ = M.action_l0(net, None, tp, f_, t_)
        feats = M.trust_features(r, l0a, pdis)
        cone_lp = lfwd / ncone
        ply_lp = torch.where(r["ply_n"] > 0, r["ply_lp_sum"] / r["ply_n"].clamp_min(1), torch.zeros_like(lfwd))
        outs.append({"e": e, "q_leaf": q, "feats": feats, "feats_x": torch.cat([feats, cone_lp[:, None], ply_lp[:, None]], 1),
                     "ll": ll, "cone_lp": cone_lp, "ply_lp": ply_lp, "terminal": r["terminal"], "flag": r["flag"],
                     "offsup": r["offsup"], "n_ply": r["n_ply"]})
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def add_window(rt):
    base, opp, mv = window_base(rt)
    rt["wbase"], rt["wopp"], rt["wmv"] = base, opp, mv
    return rt


@torch.no_grad()
def run_det(net, strust, rtrust_old, rt, act, gen, dev, noisy=False, depth=4, temp=1.0):
    """[B, K] determinized value rollouts of legal-list actions act (rt must carry the window fields, add_window)."""
    B, K = act.shape
    rr = T.roots_repeat(rt, K)
    ar = torch.arange(B, device=dev)[:, None]
    fr = rt["legal"][ar, act, 0].reshape(-1)
    to = rt["legal"][ar, act, 1].reshape(-1)
    o = det_rollouts(net, strust, rr, fr, to, gen, noisy, depth, temp)
    o["beta_r5"] = T.trust_beta(rtrust_old, o["feats"])
    o["H"] = L.entropy3(o["q_leaf"])
    del o["feats"]
    return {k: v.view(B, K, *v.shape[1:]) for k, v in o.items()}


# --------------------------------------------------------------------------------------------- consistency rollouts (D1)
D1_KEYS = ("dlb", "cov", "beta", "sal", "fl", "ll", "zoh")


@torch.no_grad()
def d1_rollouts(net, strust, rtrust, rt, act, gen, dev):
    """[B, K] consistency rollouts for legal-list actions act. rt from r6 roots_e3 + add_window."""
    o = L.e3_rollouts(net, strust, rtrust, rt, act, gen, dev)            # cone, cov, beta (R5-E2 head), fl, value claims
    B, K = act.shape
    cone = o["cone"].reshape(B * K, -1)
    rr = T.roots_repeat(rt, K)
    tpw_z, _ = X.sample_cone_tempered(net, rr["tpw"], cone, gen, strust, 1.0)
    zc = cone_values(tpw_z, cone)
    ll = ll_window(net, rr["wbase"], rr["wopp"], rr["wmv"], cone, zc)
    cov = o["cov"]
    dl = torch.where(cov[..., None], (rt["lmu"] - rt["p0b"])[:, None], torch.zeros_like(rt["lmu"])[:, None].expand(-1, K, -1, -1))
    zoh = torch.zeros(B * K, M.NS, 6, device=dev)
    ok = cone >= 0
    ar = torch.arange(B * K, device=dev)[:, None].expand_as(cone)
    zoh[ar[ok], cone[ok], (zc[ok] - 1).clamp(0, 5)] = 1.0
    zoh = zoh.view(B, K, M.NS, 6) * cov[..., None].float()
    return {"dlb": dl, "cov": cov, "beta": o["beta"], "sal": dl.norm(dim=-1), "fl": o["fl"], "ll": ll.view(B, K),
            "zoh": zoh}


def seq_bag(poolL, poolR, pL, gen, n=BAG):
    """A sequence of n rollouts per root: position j comes from the Left pool with probability pL (else Right), each pool
    used in order without replacement (pools hold >= n). Prefixes are skewed bags of any size."""
    B = pL.shape[0]
    dev = pL.device
    fromL = torch.rand(B, n, generator=gen, device=dev) < pL[:, None]
    iL = torch.cumsum(fromL.long(), 1) - 1
    iR = torch.cumsum((~fromL).long(), 1) - 1
    ar = torch.arange(B, device=dev)[:, None]
    out = {}
    for k in D1_KEYS:
        a, b = poolL[k][ar, iL.clamp_min(0)], poolR[k][ar, iR.clamp_min(0)]
        m = fromL.view(B, n, *([1] * (a.dim() - 2)))
        out[k] = torch.where(m, a, b)
    out["nL"] = fromL.sum(1)
    return out


def prefix(bag, K, l0):
    B = l0.shape[0]
    b = {k: v for k, v in bag.items() if k != "nL"}
    b["valid"] = torch.arange(BAG, device=l0.device)[None].expand(B, -1) < (K if torch.is_tensor(K) else torch.full((B,), K, device=l0.device))[:, None]
    b["l0b"] = l0
    return b


# --------------------------------------------------------------------------------------------- anchored aggregators
def agg_raw_count(b, floor=0.01):
    """Consistency-weighted raw determinization counts REPLACE the anchor on covered slots (weights exp(LL - max LL));
    1% uniform floor; uncovered slots keep ell_t."""
    v = b["valid"].float()
    ll = b["ll"].masked_fill(~b["valid"], -1e9)
    w = torch.exp(ll - ll.max(1, keepdim=True).values) * v
    cnt = (w[..., None, None] * b["zoh"]).sum(1)                                  # [B,8,6]
    tot = cnt.sum(-1, keepdim=True)
    p = (1 - floor) * cnt / tot.clamp_min(1e-30) + floor / 6
    return torch.where(tot > 0, p.clamp_min(1e-12).log(), b["l0b"])


def agg_union_t(b):
    """Each covered slot gets the rollouts' (identical) claim lmu; uncovered keep ell_t (descriptive)."""
    v = b["valid"].float()
    cov = b["cov"].float() * v[..., None]
    n = cov.sum(1)
    mean_dl = (cov[..., None] * b["dlb"]).sum(1) / n.clamp_min(1)[..., None]
    return torch.log_softmax(b["l0b"] + mean_dl, -1)


def agg_scaled(b, a):
    v = b["valid"].float()
    return torch.log_softmax(b["l0b"] + a * (b["dlb"] * v[..., None, None]).sum(1), -1)


def agg_mmd(b, eta):
    w = b["beta"] * b["valid"].float()
    return torch.log_softmax(b["l0b"] + eta * (w[..., None, None] * b["dlb"]).sum(1), -1)


def agg_kc(b, k, u, gamma=2.0):
    c = M.crowding(k, b["valid"], gamma)
    w = b["beta"] / c * b["valid"].float()
    return torch.log_softmax(b["l0b"] + u * (w[..., None, None] * b["dlb"]).sum(1), -1)


class AnchorAttn(nn.Module):
    """attn_L3 with an anchor token: tokens = [anchor (W_a ell_t), rollouts [k_r, W_v dl_r, log beta_r, log c_r, LL_r]];
    `layers` pre-LN residual self-attention blocks (untied: attn_L3), or ONE block applied `loops` times with a step
    embedding (weight-tied: looped_attn_T3, as R7-E1's LoopedAttnT3). Output: non-negative per-slot weights for each
    rollout; mu = log_softmax(ell_t + sum_r w_r[s] dl_r[s]). K = 0 returns ell_t exactly."""

    def __init__(self, kdim=16, d=64, layers=3, heads=4, loops=0, init_w=0.02):
        super().__init__()
        self.loops = loops
        self.wv = nn.Linear(M.NS * 6, d)
        self.wa = nn.Linear(M.NS * 6, d)
        self.inp = nn.Linear(kdim + 3, d)
        self.anchor = nn.Parameter(torch.zeros(d))
        n = 1 if loops else layers
        self.layers = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(n)])
        self.lns = nn.ModuleList([nn.LayerNorm(d) for _ in range(n)])
        self.ffs = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                                  for _ in range(n)])
        self.step_emb = nn.Parameter(torch.randn(max(loops, 1), d) * 0.02)
        self.out = nn.Linear(d, M.NS)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, math.log(math.expm1(init_w)))

    def weights(self, b, k, gamma=2.0):
        valid = b["valid"]
        Bn, K = valid.shape
        c = M.crowding(k, valid, gamma)
        ll = b["ll"].clamp(-30, 0) / 5.0
        x = self.inp(torch.cat([k.float(), b["beta"].clamp_min(1e-6).log()[..., None], c.log()[..., None], ll[..., None]], -1)) \
            + self.wv(b["dlb"].clamp(-8, 8).reshape(Bn, K, M.NS * 6))
        a0 = (self.wa(b["l0b"].reshape(Bn, M.NS * 6)) + self.anchor)[:, None]
        x = torch.cat([a0, x], 1)
        pad = torch.cat([torch.zeros(Bn, 1, dtype=torch.bool, device=valid.device), ~valid], 1)
        steps = [(0, t) for t in range(self.loops)] if self.loops else [(i, None) for i in range(len(self.layers))]
        for i, t in steps:
            if t is not None:
                x = x + self.step_emb[t]
            y = self.lns[i](x)
            x = x + self.layers[i](y, y, y, key_padding_mask=pad, need_weights=False)[0]
            x = x + self.ffs[i](x)
        return F.softplus(self.out(x[:, 1:])) * valid.float()[..., None]

    def forward(self, b, k):
        w = self.weights(b, k)
        return torch.log_softmax(b["l0b"] + (w[..., None] * b["dlb"].clamp(-8, 8)).sum(1), -1)


def ce_true(lb, true, hidden):
    """Mean over hidden slots of -log mu[s, true type] (true in 1..6)."""
    lp = lb.gather(-1, (true - 1).clamp(0, 5)[..., None]).squeeze(-1)
    h = hidden.float()
    return -((lp * h).sum(1) / h.sum(1).clamp_min(1)).mean()


# --------------------------------------------------------------------------------------------- value pooling
def cons_w(ll, tau):
    """exp(LL / tau) per bag (last dim), max-shifted; tau = inf -> uniform."""
    if math.isinf(tau):
        return torch.ones_like(ll)
    z = ll / tau
    return torch.exp(z - z.max(-1, keepdim=True).values)


def drop_bottom(beta, M_):
    """0/1 weights keeping all but the M_ lowest-beta rollouts (stable; ties drop the earlier index)."""
    idx = beta.argsort(dim=-1, stable=True)[..., :M_]
    w = torch.ones_like(beta)
    w.scatter_(-1, idx, 0.0)
    return w


def bias2(e, w, qstar):
    """(MSE, rollout-noise part, Bias^2) of the weighted mean, weights treated as fixed: noise = var_within * sum w~^2."""
    wn = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
    est = (wn * e).sum(-1)
    mse = ((est - qstar) ** 2).mean()
    noise = (e.var(-1, unbiased=True) * (wn ** 2).sum(-1)).mean()
    return float(mse), float(noise), float(mse - noise)
