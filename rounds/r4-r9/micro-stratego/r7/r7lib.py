r"""R7-E3 (E-MS-TRUST-SEARCH) library on top of r6/e3lib.py and r5/ (neither is modified).

Part A, pure salience keys (spec 2.1.1). Salience of rollout r: s_r,i = ||dl_r,i||_2 / (sqrt(sum_j ||dl_r,j||_2^2) + 1e-6)
over the 8 opponent slots, where dl_r is the rollout's 8x6 marginal window message (r6 e3lib: msg on its cone slots, 0
elsewhere). Bag-centred salience: st_r = (s_r - mean_bag s) / (||s_r - mean_bag s|| + 1e-6).
   K3:  k_r = normalize(W_K s_r)            (W_K 8 -> 16, no bias), alignment target s_r
   K3c: k_r = normalize(W_K [s_r || st_r])  (W_K 16 -> 16, no bias), alignment target st_r
Both are fitted like r6's K4: kc_dedup's calibration loss against the model-only reference + L_key.

Part B, mixed-reliability rollouts (spec 2.1.2). A CLEAN rollout is r6's D2 rollout unchanged (r5 stiff_voi cone sample
from theta_real's belief, depth 4, policy temperature 1). An EXPLORATORY rollout uses the same 3-slot cone (nearest the
root move's target), samples each cone type (same stiff_voi order) from softmax(belief logits / 2.5), then exchanges the
sampled types of 2 cone slots chosen uniformly ("1 random piece-type swap"), and plays its 3 further plies from
softmax(policy logits / 2.0). theta_real is not told the tier. Per rollout we also record, for the trust head:
   cone_lp = log-probability of the final cone types under theta_real's UNTEMPERED belief (teacher-forced in the sampling
             order) per cone slot (for clean rollouts this equals r5's sampling log-probability),
   ply_lp  = mean UNTEMPERED policy log-probability of the moves sampled at plies 1..3 (0 if the rollout ended at ply 0),
and, descriptively only, flag-capture endings, piece-count violations of the materialised opponent slots, and moves with
untempered probability below 1e-3.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

R6 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r6")
sys.path.insert(0, R6)
import e3lib as L  # noqa: E402  (puts r5/ on sys.path)
import model6 as M  # noqa: E402
import train6 as T  # noqa: E402

TAU_CONE = 2.5
TAU_PLY = 2.0
OFFSUP_P = 1e-3
LOG3 = math.log(3.0)
COUNTS = torch.tensor([1, 2, 2, 1, 1, 1])          # FLAG, BOMB, SCOUT, MINER, GENERAL, MARSHAL (r5 env6.COUNTS[1:])
N_EXTRA = 2                                        # cone_lp, ply_lp


# --------------------------------------------------------------------------------------------- Part A: K3 / K3c keys
def salience_unit(sal):
    s = sal.float()
    return s / (s.norm(dim=-1, keepdim=True) + 1e-6)


def salience_centred(s, valid):
    v = valid.float()[..., None]
    mu = (s * v).sum(1, keepdim=True) / v.sum(1, keepdim=True).clamp_min(1)
    c = s - mu
    return c / (c.norm(dim=-1, keepdim=True) + 1e-6)


class KeyK3(nn.Module):
    def __init__(self, kdim=16):
        super().__init__()
        self.lin = nn.Linear(M.NS, kdim, bias=False)

    def target(self, b):
        return salience_unit(b["sal"])

    def forward(self, b):
        return F.normalize(self.lin(salience_unit(b["sal"])), dim=-1)


class KeyK3c(nn.Module):
    def __init__(self, kdim=16):
        super().__init__()
        self.lin = nn.Linear(2 * M.NS, kdim, bias=False)

    def target(self, b):
        return salience_centred(salience_unit(b["sal"]), b["valid"])

    def forward(self, b):
        s = salience_unit(b["sal"])
        return F.normalize(self.lin(torch.cat([s, salience_centred(s, b["valid"])], -1)), dim=-1)


# --------------------------------------------------------------------------------------------- Part B: tiered rollouts
@torch.no_grad()
def sample_cone_tempered(net, tp, cone, gen, slot_trust, temp):
    """r5 model6.sample_cone with order stiff_voi (order from the untempered belief and slot trust, as r5), but each cone
    type drawn from softmax(belief logits / temp). Returns the filled tape and step_of [B,8] (-1 if not sampled)."""
    tp = M.tape_clone(tp)
    B, m = cone.shape
    dev = cone.device
    ar = torch.arange(B, device=dev)
    incone = torch.zeros(B, M.NS, dtype=torch.bool, device=dev)
    ok = cone >= 0
    incone[ar[:, None].expand(-1, m)[ok], cone[ok]] = True
    step_of = torch.full((B, M.NS), -1, dtype=torch.long, device=dev)
    for st in range(m):
        todo = incone & (tp["slot_val"] == 0)
        if not todo.any():
            break
        h, _ = net(tp, root_only=True)
        lg = net.belief_logits(h).float()
        q = torch.softmax(lg, -1)
        H = -(q.clamp_min(1e-9) * q.clamp_min(1e-9).log()).sum(-1)
        e = slot_trust(M.slot_features(h, q, tp)) if slot_trust is not None else torch.zeros_like(H)
        score = (H + e).masked_fill(~todo, float("inf"))
        s = score.argmin(1)
        act = todo.any(1)
        qs = torch.softmax(lg[ar, s] / temp, -1)
        v = torch.multinomial(qs.clamp_min(1e-9), 1, generator=gen).squeeze(1)
        step_of[ar[act], s[act]] = st
        nv = tp["slot_val"].clone()
        nv[ar[act], s[act]] = v[act] + 1
        tp["slot_val"] = nv
    return tp, step_of


@torch.no_grad()
def swap_one(tp, step_of, gen):
    """Exchange the sampled types of 2 distinct materialised cone slots chosen uniformly (no-op with < 2)."""
    samp = step_of >= 0
    B = samp.shape[0]
    dev = samp.device
    r = torch.rand(B, M.NS, generator=gen, device=dev).masked_fill(~samp, 2.0)
    o = r.argsort(1)
    i, j = o[:, 0], o[:, 1]
    okk = samp.sum(1) >= 2
    ar = torch.arange(B, device=dev)
    v = tp["slot_val"].clone()
    vi, vj = v[ar, i].clone(), v[ar, j].clone()
    v[ar[okk], i[okk]] = vj[okk]
    v[ar[okk], j[okk]] = vi[okk]
    tp = dict(tp)
    tp["slot_val"] = v
    return tp


@torch.no_grad()
def value_rollout_t(net, tp, root_from, root_to, depth, gen, src_slots, temp=1.0):
    """r5 model6.value_rollout, copied unchanged in sampling and outputs (identical random draws for temp=1), plus per
    rollout: ply_lp_sum / ply_n (untempered policy log-probabilities of the moves sampled at plies 1..depth-1), offsup
    (number of those moves with untempered probability < OFFSUP_P) and flag (the rollout ended on a flag capture)."""
    tp = M.tape_clone(tp)
    B = root_from.shape[0]
    dev = root_from.device
    ar = torch.arange(B, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    term_val = torch.zeros(B, 3, device=dev)
    n_ply = torch.zeros(B, dtype=torch.long, device=dev)
    ent = torch.zeros(B, depth, device=dev)
    miner_u = torch.zeros(B, device=dev)
    lp_sum = torch.zeros(B, device=dev)
    lp_n = torch.zeros(B, device=dev)
    offsup = torch.zeros(B, device=dev)
    flag_end = torch.zeros(B, dtype=torch.bool, device=dev)
    turn = tp["turn"]
    for j in range(depth):
        if j == 0:
            fr, to = root_from, root_to
        else:
            h, _ = net(tp)
            lg1 = net.h_policy(h[:, M.pre_idx(j)]).float()
            mv = torch.multinomial(torch.softmax(lg1 / temp, -1), 1, generator=gen).squeeze(1)
            lp = torch.log_softmax(lg1, -1).gather(1, mv[:, None]).squeeze(1)
            lv_ = (~done).float()
            lp_sum = lp_sum + lp * lv_
            lp_n = lp_n + lv_
            offsup = offsup + (lp.exp() < OFFSUP_P).float() * lv_
            fr, to = mv // M.NSQ, mv % M.NSQ
        live = ~done
        actor = j % 2
        for k_, v_ in (("ply_actor", torch.full_like(fr, actor)), ("ply_from", fr), ("ply_to", to)):
            t_ = tp[k_].clone()
            t_[live, j] = v_[live]
            tp[k_] = t_
        h, _ = net(tp)
        lo, la, ld = net.trans_logits(h[:, M.a_idx(j)])
        po, pa, pd = (torch.softmax(x.float(), -1) for x in (lo, la, ld))
        oc = torch.multinomial(po, 1, generator=gen).squeeze(1)
        at = torch.multinomial(pa, 1, generator=gen).squeeze(1)
        de = torch.multinomial(pd, 1, generator=gen).squeeze(1)
        ent[:, j] = torch.where(live, -(po.clamp_min(1e-9) * po.clamp_min(1e-9).log()).sum(-1), torch.zeros_like(ent[:, j]))
        for k_, v_ in (("ply_out", oc), ("ply_att", at), ("ply_def", de)):
            t_ = tp[k_].clone()
            t_[live, j] = v_[live]
            tp[k_] = t_
        n_ply = torch.where(live, torch.full_like(n_ply, j + 1), n_ply)
        cap = (oc == 1) | (oc == 3) | (oc == 4)
        lose = (oc == 2) | (oc == 3)
        if actor == 0:
            ev = (cap & (de == M.MINER)).float() - (lose & (at == M.MINER)).float()
        else:
            ev = -(cap & (de == M.MINER)).float() + (lose & (at == M.MINER)).float()
        miner_u = miner_u + torch.where(live, ev, torch.zeros_like(ev))
        flag = live & (oc == 4)
        if flag.any():
            tv = torch.zeros(B, 3, device=dev)
            tv[:, 0 if actor == 0 else 2] = 1.0
            term_val = torch.where(flag[:, None], tv, term_val)
        flag_end = flag_end | flag
        clock = live & ~flag & (turn + j + 1 >= M.MAX_TURNS)
        if clock.any():
            tv = torch.zeros(B, 3, device=dev)
            tv[:, 1] = 1.0
            term_val = torch.where(clock[:, None], tv, term_val)
        done = done | flag | clock
    h, attn = net(tp, need_attn=True)
    ri = M.o_idx(0) + 2 * (n_ply - 1)
    q_leaf = torch.softmax(net.h_value(h[ar, ri]).float(), -1)
    q_leaf = torch.where(done[:, None], term_val * 0.98 + 0.02 / 3, q_leaf)
    src = torch.zeros(B, M.NTOK, dtype=torch.bool, device=dev)
    src[:, M.SLOT0:M.SLOT0 + M.NS] = src_slots
    for j in range(depth):
        okj = n_ply > j
        src[ar[okj], tp["ply_from"][okj, j]] = True
        src[ar[okj], tp["ply_to"][okj, j]] = True
    a = attn[ar, ri].float() * src.float()
    a = a / a.sum(-1, keepdim=True).clamp_min(1e-8)
    ebar = a @ net.pos.detach().float()
    return {"q_leaf": q_leaf, "terminal": done, "n_ply": n_ply, "ent": ent, "miner_u": miner_u,
            "h_root": h[:, :M.NROOT].float().mean(1), "ebar": ebar, "ply_lp_sum": lp_sum, "ply_n": lp_n,
            "offsup": offsup, "flag": flag_end}


def count_violation(slot_val):
    """[B,8] opponent slot values (0 = MASK) -> [B] bool: some type occurs more often than the piece set allows."""
    cnt = F.one_hot(slot_val.clamp(0, 6), 7)[..., 1:].sum(1)
    return (cnt > COUNTS.to(slot_val.device)[None]).any(1)


@torch.no_grad()
def tier_rollouts(net, strust, roots, fr, to, depth, gen, explore, chunk=4096):
    """One rollout per row (roots repeated per row as r5 model6.full_rollouts). explore=False reproduces r5's clean
    rollout draw for draw. Returns e, q_leaf, feats (r5 trust features), feats_x (feats || cone_lp, ply_lp) and the
    descriptive flags."""
    B = fr.shape[0]
    outs = []
    for c0 in range(0, B, chunk):
        sl = slice(c0, c0 + chunk)
        tp = M.tape_index(roots["tp"], sl)
        f_, t_ = fr[sl], to[sl]
        cone, in_range = M.select_cone(tp, f_, t_)
        ncone = (cone >= 0).sum(1).clamp_min(1).float()
        if explore:
            tpz, step_of = sample_cone_tempered(net, tp, cone, gen, strust, TAU_CONE)
            tpz = swap_one(tpz, step_of, gen)
            samp = step_of >= 0
            fwd = torch.where(samp, samp.sum(1, keepdim=True) - 1 - step_of, step_of)
            lfwd = M.reverse_logp(net, tpz, fwd)
        else:
            tpz, _, lfwd, step_of = M.sample_cone(net, tp, cone, gen, "stiff_voi", strust)
        lrev = M.reverse_logp(net, tpz, step_of)
        pdis = (lfwd - lrev).abs() / ncone
        r = value_rollout_t(net, tpz, f_, t_, depth, gen, step_of >= 0, temp=TAU_PLY if explore else 1.0)
        q = r["q_leaf"]
        e = (q[:, 0] + 0.5 * q[:, 1]).clamp(1e-5, 1 - 1e-5)
        l0a, _ = M.action_l0(net, None, tp, f_, t_)
        feats = M.trust_features(r, l0a, pdis)
        cone_lp = lfwd / ncone
        ply_lp = torch.where(r["ply_n"] > 0, r["ply_lp_sum"] / r["ply_n"].clamp_min(1), torch.zeros_like(lfwd))
        outs.append({"e": e, "q_leaf": q, "feats": feats, "feats_x": torch.cat([feats, cone_lp[:, None], ply_lp[:, None]], 1),
                     "cone_lp": cone_lp, "ply_lp": ply_lp, "terminal": r["terminal"], "flag": r["flag"],
                     "uniq": count_violation(tpz["slot_val"]), "offsup": r["offsup"], "n_ply": r["n_ply"]})
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


@torch.no_grad()
def run_tier(net, strust, rtrust_old, rt, act, gen, dev, explore, depth=4):
    """As r5 train6.run_rollouts (roots repeated K times), for one tier. Tensors [B, K, ...]; beta_r5 = the R5-E2
    rollout-trust head on r5's features (descriptive)."""
    B, K = act.shape
    rr = T.roots_repeat(rt, K)
    ar = torch.arange(B, device=dev)[:, None]
    fr = rt["legal"][ar, act, 0].reshape(-1)
    to = rt["legal"][ar, act, 1].reshape(-1)
    o = tier_rollouts(net, strust, rr, fr, to, depth, gen, explore)
    o["beta_r5"] = T.trust_beta(rtrust_old, o["feats"])
    o["H"] = L.entropy3(o["q_leaf"])
    del o["feats"]
    return {k: v.view(B, K, *v.shape[1:]) for k, v in o.items()}


# --------------------------------------------------------------------------------------------- theta_trust (new head)
def fit_trust(X, Y, gen, dev, steps=3000, batch=512):
    """r5 train6.fit_rollout_trust's regression (TrustNet, standardised inputs, MSE to excess surprise, Adam 1e-3,
    weight decay 1e-4) on the given features/targets."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    tr = M.TrustNet(X.shape[1]).to(dev)
    tr.mu, tr.sd = mu, sd
    opt = torch.optim.Adam(tr.parameters(), lr=1e-3, weight_decay=1e-4)
    hist = []
    for s in range(steps):
        b = torch.randint(0, X.shape[0], (batch,), device=dev, generator=gen)
        loss = ((tr((X[b] - mu) / sd) - Y[b]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if s % 500 == 0 or s == steps - 1:
            hist.append([s, float(loss)])
    T.freeze(tr)
    return tr, hist


@torch.no_grad()
def beta_new(tr, fx):
    return M.beta_from(tr((fx.float() - tr.mu) / tr.sd))


# --------------------------------------------------------------------------------------------- pooling rules
def wmean(e, w):
    return (w * e).sum(-1) / w.sum(-1).clamp_min(1e-12)


def stiff_w(H, gamma):
    """(1 - H/log 3)^gamma; H is the entropy of the 3-outcome leaf claim (max log 3)."""
    return (1 - H / LOG3).clamp_min(0) ** gamma


def gate_w(beta, q, tau):
    """sigmoid((beta - Q_q(beta)) / (tau * std(beta) + 1e-6)) within each (s, a) bag (last dim)."""
    Qq = torch.quantile(beta, q, dim=-1, keepdim=True)
    sd = beta.std(-1, keepdim=True)
    return torch.sigmoid((beta - Qq) / (tau * sd + 1e-6))


def prune_mean(e, beta, n_prune, e_add=None):
    """Drop the n_prune lowest-beta rollouts of each bag (stable: ties drop the earlier index), uniform mean of the rest
    together with e_add (replacement rollouts)."""
    idx = beta.argsort(dim=-1, stable=True)[..., n_prune:]
    keep = e.gather(-1, idx)
    if e_add is not None:
        keep = torch.cat([keep, e_add], -1)
    return keep.mean(-1)
