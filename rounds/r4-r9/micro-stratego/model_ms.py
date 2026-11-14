"""E-MS model: theta_real (any-order world-model transformer over one rollout tape), theta_amp (stop-grad Q adapter +
k_v2 key projection), theta_trust (stop-grad excess-surprise MLPs), aggregators (mix, ataraxos_mmd, kc_dedup, attn_L3)
and the vectorised rollout engine that searches ONLY over theta_real (no simulator, no oracle).

Tape (35 tokens, observer = player to move at the root, canonical frame):
  0..15  squares: content (empty / own type / opponent unknown / opponent revealed type), own public flags, slot pointer
  16..21 opponent slots Z_0..Z_5: current square, public moved / moved-long flags, move count (0..3+), public-revealed
         bit, value (MASK or type 1..6; materialised hypothetically during search)
  22     global: turn
  23+2j  A_j: actor (observer / opponent), from square, to square           (j = 0..5)
  24+2j  O_j: outcome (move / attacker wins / defender wins / tie / flag taken), revealed attacker type, defender type
Attention: root tokens (0..22) attend only to root tokens; ply tokens attend to the root and causally to earlier plies.
Heads (theta_real): belief at slot tokens (6), policy for ply j at token 22 (j=0) or O_{j-1} (256 = from x to),
transition at A_j (5 + 7 + 7), value (win / draw / loss for the observer) at 22 and every A_j / O_j.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MAXP = 6
NROOT = 23
NTOK = NROOT + 2 * MAXP
GLOB = 22
SLOT0 = 16
MINER = 4
ROW_MAJOR = [5, 4, 3, 2, 1, 0]
TAPE_KEYS = ("sq_content", "sq_ownflag", "sq_slot", "slot_loc", "slot_flags", "slot_mcount", "slot_pubrev", "slot_val",
             "turn", "ply_actor", "ply_from", "ply_to", "ply_out", "ply_att", "ply_def")


def a_idx(j):
    return NROOT + 2 * j


def o_idx(j):
    return NROOT + 2 * j + 1


def pre_idx(j):
    return GLOB if j == 0 else o_idx(j - 1)


def attn_mask(dev):
    m = torch.zeros(NTOK, NTOK, dtype=torch.bool, device=dev)
    m[:NROOT, :NROOT] = True
    for i in range(NROOT, NTOK):
        m[i, :NROOT] = True
        m[i, NROOT:i + 1] = True
    return m


class Block(nn.Module):
    def __init__(self, d, h, ff, drop):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask, need_attn=False):
        B, n, d = x.shape
        hd = d // self.h
        q, k, v = self.qkv(self.ln1(x)).view(B, n, 3, self.h, hd).permute(2, 0, 3, 1, 4)
        attn = None
        if need_attn:
            s = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
            s = s.masked_fill(~mask, float("-inf"))
            a = torch.softmax(s, -1)
            attn = a.mean(1)
            o = a @ v
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        x = x + self.drop(self.proj(o.transpose(1, 2).reshape(B, n, d)))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x, attn


class RealNet(nn.Module):
    """theta_real: trunk + belief / policy / transition / value heads."""

    def __init__(self, d=128, layers=4, heads=4, ff=512, drop=0.1):
        super().__init__()
        self.d = d
        self.pos = nn.Parameter(torch.randn(NTOK, d) * 0.02)
        E = lambda n: nn.Embedding(n, d)
        self.e_sqc, self.e_own, self.e_sslot = E(14), E(5), E(7)
        self.e_loc, self.e_sfl, self.e_mc, self.e_rev, self.e_val = E(17), E(3), E(4), E(2), E(7)
        self.e_turn = E(25)
        self.e_actor, self.e_from, self.e_to = E(2), E(16), E(16)
        self.e_out, self.e_att, self.e_def = E(5), E(7), E(7)
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(d, heads, ff, drop) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.h_belief = nn.Linear(d, 6)
        self.h_policy = nn.Linear(d, 256)
        self.h_trans = nn.Linear(d, 5 + 7 + 7)
        self.h_value = nn.Linear(d, 3)
        self._mask = None

    def embed(self, tp: Dict[str, torch.Tensor]):
        sq = self.e_sqc(tp["sq_content"]) + self.e_own(tp["sq_ownflag"]) + self.e_sslot(tp["sq_slot"])
        sl = (self.e_loc(tp["slot_loc"]) + self.e_sfl(tp["slot_flags"]) + self.e_mc(tp["slot_mcount"]) +
              self.e_rev(tp["slot_pubrev"]) + self.e_val(tp["slot_val"]))
        gl = self.e_turn(tp["turn"].clamp(0, 24))[:, None]
        A = self.e_actor(tp["ply_actor"]) + self.e_from(tp["ply_from"]) + self.e_to(tp["ply_to"])
        O = self.e_out(tp["ply_out"]) + self.e_att(tp["ply_att"]) + self.e_def(tp["ply_def"])
        ply = torch.stack([A, O], 2).reshape(A.shape[0], 2 * MAXP, self.d)
        return torch.cat([sq, sl, gl, ply], 1) + self.pos[None]

    def forward(self, tp, need_attn=False, root_only=False):
        x = self.embed(tp)
        if self._mask is None or self._mask.device != x.device:
            self._mask = attn_mask(x.device)
        mask = self._mask
        if root_only:
            x = x[:, :NROOT]
            mask = mask[:NROOT, :NROOT]
        attn = None
        for li, blk in enumerate(self.blocks):
            x, a = blk(x, mask, need_attn and li == len(self.blocks) - 1)
            if a is not None:
                attn = a
        return self.lnf(x), attn

    def belief_logits(self, h):
        return self.h_belief(h[:, SLOT0:SLOT0 + 6])

    def trans_logits(self, hA):
        z = self.h_trans(hA)
        return z[..., :5], z[..., 5:12], z[..., 12:]


def eff_logit(vlogits):
    """3-way value logits (win, draw, loss) -> logit of effective win p = P(win) + 0.5 P(draw)."""
    p = torch.softmax(vlogits.float(), -1)
    e = (p[..., 0] + 0.5 * p[..., 1]).clamp(1e-5, 1 - 1e-5)
    return torch.log(e) - torch.log1p(-e)


class AmpNet(nn.Module):
    """theta_amp: 2-layer MLP on stop_grad(h_real) at A_0 -> additive correction to theta_real's A_0 value logit
    (0-step action value l0(a)), plus the k_v2 key projection W_K2 (d -> 16)."""

    def __init__(self, d=128, kdim=16):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.q[-1].weight)
        nn.init.zeros_(self.q[-1].bias)
        self.keyW2 = nn.Linear(d, kdim)

    def correction(self, hA0):
        return self.q(hA0.detach()).squeeze(-1)

    def key(self, ebar):
        return F.normalize(self.keyW2(F.layer_norm(ebar, (ebar.shape[-1],))), dim=-1)


class TrustNet(nn.Module):
    """theta_trust: predicts excess surprise e >= 0 from stop-grad features; beta = 1 / (e_hat + 0.05)."""

    def __init__(self, nin, hid=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(nin, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))

    def forward(self, x):
        return F.softplus(self.f(x)).squeeze(-1)


def beta_from(e):
    return 1.0 / (e + 0.05)


def excess_surprise(q, y):
    """q [..., C] probabilities, y [...] long -> max(0, -log q_y - H(q)) + max(0, 2(|q|^2 - q_y))."""
    q = q.clamp_min(1e-6)
    qy = q.gather(-1, y[..., None]).squeeze(-1)
    H = -(q * q.log()).sum(-1)
    return (-qy.log() - H).clamp_min(0) + (2 * ((q * q).sum(-1) - qy)).clamp_min(0)


# --------------------------------------------------------------------------------------------- aggregators
def crowding(k, gamma=2.0, valid=None):
    """k [N, K, kd] unit keys -> c_r = sum_b max(0, <k_r, k_b>)^gamma (over valid b)."""
    g = torch.relu(k @ k.transpose(1, 2)) ** gamma
    if valid is not None:
        g = g * valid[:, None, :].float()
    return g.sum(2).clamp_min(1e-6)


def agg_mix(q, valid):
    v = valid.float()
    return (q * v).sum(1) / v.sum(1).clamp_min(1)


def agg_mmd(l0, dl, beta, valid, eta):
    return l0 + eta * (beta * dl * valid.float()).sum(1)


def agg_kc(l0, dl, beta, k, valid, u, gamma=2.0):
    c = crowding(k, gamma, valid)
    return l0 + u * (beta * dl / c * valid.float()).sum(1)


class AttnAgg(nn.Module):
    """attn_L3: 3 pre-LN residual self-attention layers over the bag tokens (k_r, dl_r, log beta_r, log c_r) and a
    softplus weight per rollout; l* = l0 + sum_r w_r dl_r."""

    def __init__(self, kdim=16, d=32, layers=3, heads=2):
        super().__init__()
        self.inp = nn.Linear(kdim + 3, d)
        self.layers = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers)])
        self.lns = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ffs = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                                  for _ in range(layers)])
        self.out = nn.Linear(d, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, -2.0)

    def forward(self, l0, dl, beta, k, valid, gamma=2.0):
        c = crowding(k, gamma, valid)
        x = self.inp(torch.cat([k, dl[..., None].clamp(-8, 8), beta.clamp_min(1e-6).log()[..., None],
                                c.log()[..., None]], -1))
        pad = ~valid
        for at, ln, ff in zip(self.layers, self.lns, self.ffs):
            y = ln(x)
            x = x + at(y, y, y, key_padding_mask=pad, need_weights=False)[0]
            x = x + ff(x)
        w = F.softplus(self.out(x)).squeeze(-1) * valid.float()
        return l0 + (w * dl).sum(1)


# --------------------------------------------------------------------------------------------- tapes and search
def root_tape(rec: Dict[str, torch.Tensor], idx: torch.Tensor, reveal_hidden: Optional[torch.Tensor] = None):
    """Build root tapes for record indices idx; hidden slots MASK unless reveal_hidden (bool [B,6]) says otherwise."""
    pub = rec["slot_pubrev"][idx].long()
    tru = rec["slot_true"][idx].long()
    vis = pub.bool() if reveal_hidden is None else (pub.bool() | reveal_hidden)
    B = idx.shape[0]
    z = torch.zeros(B, MAXP, dtype=torch.long, device=idx.device)
    return {"sq_content": rec["sq_content"][idx].long(), "sq_ownflag": rec["sq_ownflag"][idx].long(),
            "sq_slot": rec["sq_slot"][idx].long(), "slot_loc": rec["slot_loc"][idx].long(),
            "slot_flags": rec["slot_flags"][idx].long(), "slot_mcount": rec["slot_mcount"][idx].long(),
            "slot_pubrev": pub, "slot_val": torch.where(vis, tru, torch.zeros_like(tru)), "turn": rec["turn"][idx].long(),
            "ply_actor": z.clone(), "ply_from": z.clone(), "ply_to": z.clone(), "ply_out": z.clone(), "ply_att": z.clone(),
            "ply_def": z.clone()}


def tape_repeat(tp, K):
    return {k: v.repeat_interleave(K, 0) for k, v in tp.items()}


def tape_index(tp, idx):
    return {k: v[idx] for k, v in tp.items()}


def tape_clone(tp):
    return {k: v.clone() for k, v in tp.items()}


@torch.no_grad()
def sample_Z(net: RealNet, tp, order: str, gen: torch.Generator, slot_trust: Optional[TrustNet] = None,
             record_cond=False):
    """Any-order unmasking of the opponent's hidden slots from theta_real's belief head (no rule-based pruning).
    order: row_major | random | stiff_voi (lowest predicted surprise H(q_s) + e_hat_s first) | given (tp['_order']).
    Returns the filled tape and dict(cond [B,6,6] conditional used when each slot was unmasked, logp [B],
    order [B,6] (unmask step of each slot, -1 if not hidden), n_hidden [B])."""
    tp = tape_clone(tp)
    B = tp["slot_val"].shape[0]
    dev = tp["slot_val"].device
    hidden = tp["slot_val"] == 0
    cond = torch.zeros(B, 6, 6, device=dev)
    step_of = torch.full((B, 6), -1, dtype=torch.long, device=dev)
    logp = torch.zeros(B, device=dev)
    if order == "random":
        rnd = torch.rand(B, 6, generator=gen, device=dev)
    rm_rank = torch.tensor([ROW_MAJOR.index(s) for s in range(6)], device=dev, dtype=torch.float)
    ar = torch.arange(B, device=dev)
    for step in range(6):
        todo = tp["slot_val"] == 0
        if not todo.any():
            break
        h, _ = net(tp, root_only=True)
        q = torch.softmax(net.belief_logits(h).float(), -1)               # [B,6,6]
        if order == "row_major":
            score = rm_rank[None].expand(B, 6).clone()
        elif order == "random":
            score = rnd.clone()
        elif order == "stiff_voi":
            H = -(q.clamp_min(1e-9) * q.clamp_min(1e-9).log()).sum(-1)
            e = slot_trust(slot_features(h, q, tp)) if slot_trust is not None else torch.zeros_like(H)
            score = H + e
        elif order == "given":
            score = tp["_order"].float()
        else:
            raise ValueError(order)
        score = score.masked_fill(~todo, float("inf"))
        s = score.argmin(1)
        act = todo.any(1)
        qs = q[ar, s]
        v = torch.multinomial(qs.clamp_min(1e-9), 1, generator=gen).squeeze(1)
        cond[ar[act], s[act]] = qs[act]
        step_of[ar[act], s[act]] = step
        logp[act] += qs[act].gather(1, v[act, None]).squeeze(1).clamp_min(1e-9).log()
        newv = tp["slot_val"].clone()
        newv[ar[act], s[act]] = v[act] + 1
        tp["slot_val"] = newv
    return tp, {"cond": cond, "logp": logp, "step": step_of, "hidden": hidden}


@torch.no_grad()
def path_logp(net: RealNet, tp_filled, hidden, step_of, reverse=True):
    """log q of the materialised Z under the reverse unmasking order (teacher-forced), for the 2-path disagreement."""
    B = hidden.shape[0]
    dev = hidden.device
    ns = hidden.sum(1)
    rstep = torch.where(hidden, (ns[:, None] - 1 - step_of), torch.full_like(step_of, 99))
    logp = torch.zeros(B, device=dev)
    ar = torch.arange(B, device=dev)
    base = tape_clone(tp_filled)
    full = tp_filled["slot_val"]
    for st in range(6):
        cur = base["slot_val"].clone()
        # slots with rstep < st are revealed; others hidden -> MASK
        cur = torch.where(hidden & (rstep >= st), torch.zeros_like(cur), full)
        tp = dict(base)
        tp["slot_val"] = cur
        h, _ = net(tp, root_only=True)
        q = torch.softmax(net.belief_logits(h).float(), -1)
        m = hidden & (rstep == st)
        if not m.any():
            break
        idx = m.float().argmax(1)
        act = m.any(1)
        qv = q[ar, idx].gather(1, (full[ar, idx] - 1).clamp_min(0)[:, None]).squeeze(1)
        logp[act] += qv[act].clamp_min(1e-9).log()
    return logp


def slot_features(h, q, tp):
    """Per-slot stop-grad features for the slot-trust head: h_slot (d), H(q_s), number of non-MASK slots."""
    H = -(q.clamp_min(1e-9) * q.clamp_min(1e-9).log()).sum(-1)
    nrev = (tp["slot_val"] > 0).float().sum(1, keepdim=True).expand(-1, 6) / 6.0
    return torch.cat([h[:, SLOT0:SLOT0 + 6].detach(), H[..., None], nrev[..., None]], -1)


@torch.no_grad()
def rollout(net: RealNet, tp, root_from, root_to, depth: int, gen: torch.Generator, temp=1.0, real_plies=None):
    """Roll `depth` plies inside theta_real from tapes whose Z is already materialised. Root action given (canonical
    from/to); later moves sampled from theta_real's policy head (both sides), outcomes from its transition head.
    real_plies: optional dict of [B, MAXP] tensors forcing actions (teacher-forced moves; outcomes still predicted).
    Returns dict: q_leaf [B,3], terminal [B] (bool), n_ply [B], ent [B, depth] transition entropies, miner_u [B],
    h_root [B,d] (mean root hidden state), attn_src fingerprint ebar [B,d], read index."""
    tp = tape_clone(tp)
    B = root_from.shape[0]
    dev = root_from.device
    ar = torch.arange(B, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    term_val = torch.zeros(B, 3, device=dev)
    n_ply = torch.zeros(B, dtype=torch.long, device=dev)
    ent = torch.zeros(B, depth, device=dev)
    miner_u = torch.zeros(B, device=dev)
    turn = tp["turn"]
    for j in range(depth):
        if j == 0:
            fr, to = root_from, root_to
        elif real_plies is not None:
            fr, to = real_plies["from"][:, j], real_plies["to"][:, j]
        else:
            h, _ = net(tp)
            lg = net.h_policy(h[:, pre_idx(j)]).float() / temp
            mv = torch.multinomial(torch.softmax(lg, -1), 1, generator=gen).squeeze(1)
            fr, to = mv // 16, mv % 16
        live = ~done
        actor = j % 2
        for k_, v_ in (("ply_actor", torch.full_like(fr, actor)), ("ply_from", fr), ("ply_to", to)):
            t_ = tp[k_].clone()
            t_[live, j] = v_[live]
            tp[k_] = t_
        h, _ = net(tp)
        lo, la, ld = net.trans_logits(h[:, a_idx(j)])
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
        # miner utility events (observer view)
        cap = (oc == 1) | (oc == 3)
        lose = (oc == 2) | (oc == 3)
        if actor == 0:
            ev = (cap & (de == MINER)).float() - (lose & (at == MINER)).float()
        else:
            ev = -(cap & (de == MINER)).float() + (lose & (at == MINER)).float()
        miner_u = miner_u + torch.where(live, ev, torch.zeros_like(ev))
        flag = live & (oc == 4)
        if flag.any():
            tv = torch.zeros(B, 3, device=dev)
            tv[:, 0 if actor == 0 else 2] = 1.0
            term_val = torch.where(flag[:, None], tv, term_val)
        clock = live & ~flag & (turn + j + 1 >= 24)
        if clock.any():
            tv = torch.zeros(B, 3, device=dev)
            tv[:, 1] = 1.0
            term_val = torch.where(clock[:, None], tv, term_val)
        done = done | flag | clock
    h, attn = net(tp, need_attn=True)
    ri = o_idx(0) + 2 * (n_ply - 1)
    q_leaf = torch.softmax(net.h_value(h[ar, ri]).float(), -1)
    q_leaf = torch.where(done[:, None], term_val * 0.98 + 0.02 / 3, q_leaf)
    # k_v2 source set: hypothetically materialised slots (hidden at the root) + squares touched by the rollout's moves
    src = torch.zeros(B, NTOK, dtype=torch.bool, device=dev)
    src[:, SLOT0:SLOT0 + 6] = (tp["slot_pubrev"] == 0) & (tp["slot_val"] > 0)
    for j in range(depth):
        ok = n_ply > j
        src[ar[ok], tp["ply_from"][ok, j]] = True
        src[ar[ok], tp["ply_to"][ok, j]] = True
    a = attn[ar, ri] * src.float()
    a = a / a.sum(-1, keepdim=True).clamp_min(1e-8)
    ebar = a @ net.pos.detach()
    return {"q_leaf": q_leaf, "terminal": done, "n_ply": n_ply, "ent": ent, "miner_u": miner_u,
            "h_root": h[:, :NROOT].mean(1), "ebar": ebar, "tape": tp}


@torch.no_grad()
def action_l0(net: RealNet, amp: Optional[AmpNet], tp_root, fr, to):
    """0-step action value logit l0(a) = theta_real A_0 value logit (+ theta_amp correction), Z masked."""
    tp = tape_clone(tp_root)
    tp["ply_actor"][:, 0] = 0
    tp["ply_from"][:, 0] = fr
    tp["ply_to"][:, 0] = to
    h, _ = net(tp)
    l = eff_logit(net.h_value(h[:, a_idx(0)]))
    if amp is not None:
        l = l + amp.correction(h[:, a_idx(0)]).float()
    return l, h[:, a_idx(0)]


def trust_features(r, l0, depth, pathdis, dmax=6):
    """Rollout-level stop-grad features for theta_trust."""
    q = r["q_leaf"]
    e = (q[:, 0] + 0.5 * q[:, 1]).clamp(1e-5, 1 - 1e-5)
    l = torch.log(e) - torch.log1p(-e)
    H = -(q.clamp_min(1e-9) * q.clamp_min(1e-9).log()).sum(-1)
    n = r["n_ply"].clamp_min(1).float()
    ent = r["ent"]
    f = torch.stack([r["n_ply"].float() / dmax, ent.sum(1) / n, ent.max(1).values, pathdis, (l - l0).abs().clamp(max=10),
                     H, e, r["terminal"].float()], 1)
    return torch.cat([r["h_root"].detach(), f], 1)


N_TRUST_EXTRA = 8
