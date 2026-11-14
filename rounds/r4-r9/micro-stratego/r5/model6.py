r"""R5-E2 model for the 6x6 board: theta_real (any-order world-model transformer over one tape, as ../model_ms.py with the
6x6 sizes), theta_amp (stop-grad action-value adapter + key projection W_K2), theta_trust (stop-grad excess-surprise MLPs),
the four aggregators (mix, ataraxos_mmd, kc_dedup, attn_L3) over 8x6 belief + root-value residuals, and the cone-rollout
engine that searches ONLY inside theta_real (no simulator, no oracle).

Tape (57 tokens, observer = player to move at the root, canonical frame):
  0..35  squares: content (empty / own type / opponent unknown / opponent revealed type / lake), own public flags,
         opponent slot pointer
  36..43 opponent slots Z_0..Z_7: square, public moved / moved-long flags, move count (0..3+), public-revealed bit,
         value (MASK or type 1..6; materialised hypothetically during search)
  44     global: turn
  45+2j  A_j: actor (observer / opponent), from square, to square            (j = 0..5)
  46+2j  O_j: outcome (move / attacker wins / defender wins / tie / flag taken), revealed attacker type, defender type
Root tokens attend to the root only; ply tokens attend to the root and causally to earlier plies.
Heads: belief at slot tokens (6), policy for ply j at token 44 (j = 0) or O_{j-1} (36 x 36 = 1296 from-to logits),
transition at A_j (5 + 7 + 7), value (win / draw / loss for the observer) at 44 and every A_j / O_j.

R5 fixes relative to R4-E1 (spec section 2.2, item 3):
 (1) a rollout materialises only its tactical cone: the m_sub = 3 hidden opponent slots nearest (Chebyshev) to the
     root move's target square; every other hidden slot stays MASK, and the key's source set is those cone slot
     tokens plus the squares the rollout's moves touch (R4-E1 marked every hidden slot);
 (2) each rollout returns an 8x6 belief residual (non-zero on its cone) and a root-value residual, and the aggregators
     (and W_K2) are fitted on the joint 48 belief logits + value (R4-E1: the scalar root value only).
The belief residual of rollout r for cone slot s is a 4-ply consistency check of the cone hypothesis against the REAL
last 4 plies (the public history): log mu_r[s, k] = log softmax( log q_real(Z_s = k | root at t-4, Z_{C\s} = sampled)
+ sum over the opponent's real moves in the window of [log pi_real(move | Z_s = k, Z_{C\s}) - log pi_real(move | Z_s
masked, Z_{C\s})] ), and dl_r[s] = log mu_r[s] - log q_real(Z_s | root at t). q_real and pi_real are theta_real's
belief and policy heads (the policy head is the learned behaviour model); no oracle, no simulator.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

N = 6
NSQ = 36
NS = 8
SLOT0 = 36
GLOB = 44
NROOT = 45
MAXP = 6
NTOK = NROOT + 2 * MAXP
NPOL = NSQ * NSQ
MAX_TURNS = 60
MINER = 4
M_SUB = 3
WIN = 4
TAPE_KEYS = ("sq_content", "sq_ownflag", "sq_slot", "slot_loc", "slot_flags", "slot_mcount", "slot_pubrev", "slot_val",
             "turn", "ply_actor", "ply_from", "ply_to", "ply_out", "ply_att", "ply_def")
PLY_KEYS = ("ply_actor", "ply_from", "ply_to", "ply_out", "ply_att", "ply_def")
ARCHS = {"primary": dict(d=128, layers=4, heads=4, ff=512), "fallback": dict(d=96, layers=3, heads=4, ff=384)}


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
            s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(hd)
            s = s.masked_fill(~mask, float("-inf"))
            a = torch.softmax(s, -1)
            attn = a.mean(1)
            o = (a @ v.float()).to(v.dtype)
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
        self.e_sqc, self.e_own, self.e_sslot = E(15), E(5), E(NS + 1)
        self.e_loc, self.e_sfl, self.e_mc, self.e_rev, self.e_val = E(NSQ + 1), E(3), E(4), E(2), E(7)
        self.e_turn = E(MAX_TURNS + 1)
        self.e_actor, self.e_from, self.e_to = E(2), E(NSQ), E(NSQ)
        self.e_out, self.e_att, self.e_def = E(5), E(7), E(7)
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(d, heads, ff, drop) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.h_belief = nn.Linear(d, 6)
        self.h_policy = nn.Linear(d, NPOL)
        self.h_trans = nn.Linear(d, 5 + 7 + 7)
        self.h_value = nn.Linear(d, 3)
        self._mask = None

    def embed(self, tp):
        sq = self.e_sqc(tp["sq_content"]) + self.e_own(tp["sq_ownflag"]) + self.e_sslot(tp["sq_slot"])
        sl = (self.e_loc(tp["slot_loc"]) + self.e_sfl(tp["slot_flags"]) + self.e_mc(tp["slot_mcount"]) +
              self.e_rev(tp["slot_pubrev"]) + self.e_val(tp["slot_val"]))
        gl = self.e_turn(tp["turn"].clamp(0, MAX_TURNS))[:, None]
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
        return self.h_belief(h[:, SLOT0:SLOT0 + NS])

    def trans_logits(self, hA):
        z = self.h_trans(hA)
        return z[..., :5], z[..., 5:12], z[..., 12:]


def make_net(arch: str) -> RealNet:
    return RealNet(**ARCHS[arch])


def eff_logit(vlogits):
    p = torch.softmax(vlogits.float(), -1)
    e = (p[..., 0] + 0.5 * p[..., 1]).clamp(1e-5, 1 - 1e-5)
    return torch.log(e) - torch.log1p(-e)


def logit(e):
    e = e.clamp(1e-5, 1 - 1e-5)
    return torch.log(e) - torch.log1p(-e)


class AmpNet(nn.Module):
    """theta_amp: 2-layer MLP on stop_grad(h_real at A_0) -> additive correction to theta_real's 0-step action-value
    logit l0(a); plus the key projection W_K2 (d -> 16)."""

    def __init__(self, d=128, kdim=16):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.q[-1].weight)
        nn.init.zeros_(self.q[-1].bias)
        self.keyW2 = nn.Linear(d, kdim)

    def correction(self, hA0):
        return self.q(hA0.detach().float()).squeeze(-1)

    def key(self, ebar):
        ebar = ebar.float()
        return F.normalize(self.keyW2(F.layer_norm(ebar, (ebar.shape[-1],))), dim=-1)


class TrustNet(nn.Module):
    def __init__(self, nin, hid=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(nin, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))

    def forward(self, x):
        return F.softplus(self.f(x.float())).squeeze(-1)


def beta_from(e):
    return 1.0 / (e + 0.05)


def excess_surprise(q, y):
    q = q.clamp_min(1e-6)
    qy = q.gather(-1, y[..., None]).squeeze(-1)
    H = -(q * q.log()).sum(-1)
    return (-qy.log() - H).clamp_min(0) + (2 * ((q * q).sum(-1) - qy)).clamp_min(0)


# --------------------------------------------------------------------------------------------- aggregators
def crowding(k, valid, gamma=2.0):
    g = torch.relu(k @ k.transpose(1, 2)) ** gamma
    g = g * valid[:, None, :].float()
    return g.sum(2).clamp_min(1e-6)


def agg_mix(b):
    """Naive rollout average: mean over the bag of each rollout's 8-slot belief claim softmax(l0 + dl_r) (a rollout's
    claim on a slot outside its cone is the 0-step belief) and of the leaf effective-win claims."""
    v = b["valid"].float()
    pb = torch.softmax(b["l0b"][:, None] + b["dlb"], -1)                       # [N,K,8,6]
    mb = (pb * v[..., None, None]).sum(1) / v.sum(1).clamp_min(1)[:, None, None]
    mv = (b["e"] * v).sum(1) / v.sum(1).clamp_min(1)
    return mb.clamp_min(1e-8).log(), logit(mv)


def agg_mmd(b, eta_b, eta_v):
    w = (b["beta"] * b["valid"].float())
    lb = b["l0b"] + eta_b * (w[..., None, None] * b["dlb"]).sum(1)
    lv = b["l0v"] + eta_v * (w * b["dlv"]).sum(1)
    return torch.log_softmax(lb, -1), lv


def agg_kc(b, k, u_b, u_v, gamma=2.0):
    c = crowding(k, b["valid"], gamma)
    w = b["beta"] / c * b["valid"].float()
    lb = b["l0b"] + u_b * (w[..., None, None] * b["dlb"]).sum(1)
    lv = b["l0v"] + u_v * (w * b["dlv"]).sum(1)
    return torch.log_softmax(lb, -1), lv


class AttnAgg(nn.Module):
    """attn_L3: 3 pre-LN residual self-attention layers (d 64, 4 heads) over rollout tokens
    z_r = [k_r, W_v dl^belief_r, dl^value_r, log beta_r, log c_r]; per-rollout non-negative weights for each of the 8
    slots and for the value: l*_b[s] = l0_b[s] + sum_r w_r[s] dl_r[s], l*_v = l0_v + sum_r w_r[8] dl^v_r."""

    def __init__(self, kdim=16, d=64, layers=3, heads=4):
        super().__init__()
        self.wv = nn.Linear(NS * 6, d)
        self.inp = nn.Linear(kdim + 3, d)
        self.layers = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers)])
        self.lns = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ffs = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                                  for _ in range(layers)])
        self.out = nn.Linear(d, NS + 1)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, -2.0)

    def weights(self, b, k, gamma=2.0):
        c = crowding(k, b["valid"], gamma)
        N_, K = b["dlv"].shape
        x = self.inp(torch.cat([k.float(), b["dlv"].clamp(-8, 8)[..., None], b["beta"].clamp_min(1e-6).log()[..., None],
                                c.log()[..., None]], -1)) + self.wv(b["dlb"].reshape(N_, K, NS * 6))
        pad = ~b["valid"]
        for at, ln, ff in zip(self.layers, self.lns, self.ffs):
            y = ln(x)
            x = x + at(y, y, y, key_padding_mask=pad, need_weights=False)[0]
            x = x + ff(x)
        return F.softplus(self.out(x)) * b["valid"].float()[..., None]          # [N,K,9]

    def forward(self, b, k):
        w = self.weights(b, k)
        lb = b["l0b"] + (w[..., :NS, None] * b["dlb"]).sum(1)
        lv = b["l0v"] + (w[..., NS] * b["dlv"]).sum(1)
        return torch.log_softmax(lb, -1), lv


def coverage_reference(b):
    """Unskewed reference consensus (model-only): per slot, the 0-step logits plus the mean residual over the rollouts
    whose cone contains the slot (0-step belief if none does); value = plain mean of the leaf claims."""
    v = b["valid"].float()
    cov = b["cov"].float() * v[..., None]                                      # [N,K,8]
    n = cov.sum(1)                                                             # [N,8]
    mean_dl = (cov[..., None] * b["dlb"]).sum(1) / n.clamp_min(1)[..., None]
    lb = torch.log_softmax(b["l0b"] + mean_dl, -1)
    mv = (b["e"] * v).sum(1) / v.sum(1).clamp_min(1)
    return lb, logit(mv), n


# --------------------------------------------------------------------------------------------- tapes
def tape_from(R, idx, reveal_hidden: Optional[torch.Tensor] = None):
    """Root tapes for record indices idx (records as int tensors); hidden opponent slots MASK unless reveal_hidden."""
    L = lambda k: R[k][idx].long()
    pub = L("slot_pubrev")
    tru = L("slot_true")
    vis = pub.bool() if reveal_hidden is None else (pub.bool() | reveal_hidden)
    B = idx.shape[0]
    z = torch.zeros(B, MAXP, dtype=torch.long, device=idx.device)
    tp = {k: L(k) for k in ("sq_content", "sq_ownflag", "sq_slot", "slot_loc", "slot_flags", "slot_mcount", "slot_pubrev")}
    tp["slot_val"] = torch.where(vis, tru, torch.zeros_like(tru))
    tp["turn"] = L("turn")
    for k in PLY_KEYS:
        tp[k] = z.clone()
    return tp


def tape_repeat(tp, K):
    return {k: v.repeat_interleave(K, 0) for k, v in tp.items()}


def tape_index(tp, idx):
    return {k: v[idx] for k, v in tp.items()}


def tape_clone(tp):
    return {k: v.clone() for k, v in tp.items()}


def tape_cat(tps):
    return {k: torch.cat([t[k] for t in tps], 0) for k in tps[0]}


def slot_features(h, q, tp):
    H = -(q.clamp_min(1e-9) * q.clamp_min(1e-9).log()).sum(-1)
    nrev = (tp["slot_val"] > 0).float().sum(1, keepdim=True).expand(-1, NS) / NS
    return torch.cat([h[:, SLOT0:SLOT0 + NS].detach().float(), H[..., None], nrev[..., None]], -1)


@torch.no_grad()
def belief0(net, tp):
    """0-step belief log-probabilities [B,8,6] and root hidden states (nothing materialised beyond the tape)."""
    h, _ = net(tp, root_only=True)
    return torch.log_softmax(net.belief_logits(h).float(), -1), h


# --------------------------------------------------------------------------------------------- cone rollouts
def select_cone(tp, fr, to, m=M_SUB):
    """The m hidden opponent slots nearest (Chebyshev) to the root move's target square; ties by distance to the
    from-square, then slot index. Returns cone [B,m] (slot index, -1 if fewer hidden slots) and in_range [B,m]."""
    loc = tp["slot_loc"]
    hidden = (tp["slot_val"] == 0) & (loc < NSQ)
    r, c = loc // N, loc % N
    tr, tc = (to // N)[:, None], (to % N)[:, None]
    fr_r, fr_c = (fr // N)[:, None], (fr % N)[:, None]
    dto = torch.maximum((r - tr).abs(), (c - tc).abs())
    dfr = torch.maximum((r - fr_r).abs(), (c - fr_c).abs())
    key = (dto * 100 + dfr * 8 + torch.arange(NS, device=loc.device)[None]).float()
    key = key.masked_fill(~hidden, float("inf"))
    vals, idx = key.topk(m, dim=1, largest=False)
    ok = torch.isfinite(vals)
    cone = torch.where(ok, idx, torch.full_like(idx, -1))
    in_range = ok & (torch.div(vals, 100, rounding_mode="floor") <= 2)
    return cone, in_range


@torch.no_grad()
def sample_cone(net, tp, cone, gen, order="stiff_voi", slot_trust=None):
    """Any-order unmasking of the cone slots only, from theta_real's belief head. Returns the filled tape and
    (sampled types zc [B,m] (0 if none), logp [B], step_of [B,8] (-1 if not sampled))."""
    tp = tape_clone(tp)
    B, m = cone.shape
    dev = cone.device
    ar = torch.arange(B, device=dev)
    incone = torch.zeros(B, NS, dtype=torch.bool, device=dev)
    ok = cone >= 0
    incone[ar[:, None].expand(-1, m)[ok], cone[ok]] = True
    step_of = torch.full((B, NS), -1, dtype=torch.long, device=dev)
    logp = torch.zeros(B, device=dev)
    rnd = torch.rand(B, NS, generator=gen, device=dev) if order == "random" else None
    for st in range(m):
        todo = incone & (tp["slot_val"] == 0)
        if not todo.any():
            break
        h, _ = net(tp, root_only=True)
        q = torch.softmax(net.belief_logits(h).float(), -1)
        if order == "random":
            score = rnd.clone()
        elif order == "stiff_voi":
            H = -(q.clamp_min(1e-9) * q.clamp_min(1e-9).log()).sum(-1)
            e = slot_trust(slot_features(h, q, tp)) if slot_trust is not None else torch.zeros_like(H)
            score = H + e
        else:
            raise ValueError(order)
        score = score.masked_fill(~todo, float("inf"))
        s = score.argmin(1)
        act = todo.any(1)
        qs = q[ar, s]
        v = torch.multinomial(qs.clamp_min(1e-9), 1, generator=gen).squeeze(1)
        step_of[ar[act], s[act]] = st
        logp[act] += qs[act].gather(1, v[act, None]).squeeze(1).clamp_min(1e-9).log()
        nv = tp["slot_val"].clone()
        nv[ar[act], s[act]] = v[act] + 1
        tp["slot_val"] = nv
    zc = torch.where(ok, tp["slot_val"].gather(1, cone.clamp_min(0)), torch.zeros_like(cone))
    return tp, zc, logp, step_of


@torch.no_grad()
def reverse_logp(net, tp_filled, step_of):
    """log q of the sampled cone values under the reverse unmasking order (teacher-forced), for the 2-path disagreement."""
    B = step_of.shape[0]
    dev = step_of.device
    samp = step_of >= 0
    ns = samp.sum(1)
    rstep = torch.where(samp, ns[:, None] - 1 - step_of, torch.full_like(step_of, 99))
    full = tp_filled["slot_val"]
    logp = torch.zeros(B, device=dev)
    ar = torch.arange(B, device=dev)
    for st in range(int(ns.max().item()) if B else 0):
        tp = dict(tp_filled)
        tp["slot_val"] = torch.where(samp & (rstep >= st), torch.zeros_like(full), full)
        h, _ = net(tp, root_only=True)
        q = torch.softmax(net.belief_logits(h).float(), -1)
        msk = samp & (rstep == st)
        act = msk.any(1)
        idx = msk.float().argmax(1)
        qv = q[ar, idx].gather(1, (full[ar, idx] - 1).clamp_min(0)[:, None]).squeeze(1)
        logp[act] += qv[act].clamp_min(1e-9).log()
    return logp


@torch.no_grad()
def window_messages(net, tpw, wply, nw, cone, zc, l0b, chunk=8192):
    """4-ply consistency belief residuals. tpw: root tapes at t0 = max(t - 4, t mod 2) (same observer); wply [B,4,6]
    real plies t0 .. t-1 (observer frame), nw [B] = t - t0; cone/zc from the rollout; l0b [B,8,6] 0-step log-beliefs at
    t. Returns dlb [B,8,6] (zero outside the cone) and cov [B,8]."""
    B, m = cone.shape
    dev = cone.device
    dlb = torch.zeros(B, NS, 6, device=dev)
    cov = torch.zeros(B, NS, dtype=torch.bool, device=dev)
    ar = torch.arange(B, device=dev)
    ok = cone >= 0
    for c in range(m):
        sel = ok[:, c]
        cov[ar[sel], cone[sel, c]] = True
    # base tapes at t0 with the cone materialised (window plies teacher-forced up to nw)
    base = tape_clone(tpw)
    for k_ in PLY_KEYS:
        base[k_] = torch.zeros(B, MAXP, dtype=torch.long, device=dev)
    jj = torch.arange(WIN, device=dev)[None]
    live = jj < nw[:, None]
    for ci, k_ in enumerate(PLY_KEYS):
        base[k_][:, :WIN] = torch.where(live, wply[..., ci], torch.zeros_like(wply[..., ci]))
    sv = base["slot_val"].clone()
    for c in range(m):
        sel = ok[:, c]
        sv[ar[sel], cone[sel, c]] = zc[sel, c]
    base["slot_val"] = sv
    opp = live & (wply[..., 0] == 1)                                               # opponent's real moves in the window
    mv_idx = (wply[..., 1] * NSQ + wply[..., 2]).clamp(0, NPOL - 1)
    for c in range(m):
        sel = ok[:, c]
        if not sel.any():
            continue
        rows = ar[sel]
        s_c = cone[sel, c]
        nb = rows.shape[0]
        # belief at t0 for slot s_c given the other cone slots (s_c masked)
        tb = tape_index(base, rows)
        v0 = tb["slot_val"].clone()
        v0[torch.arange(nb, device=dev), s_c] = 0
        tb["slot_val"] = v0
        h0, _ = net(tb, root_only=True)
        lq0 = torch.log_softmax(net.belief_logits(h0).float(), -1)[torch.arange(nb, device=dev), s_c]   # [nb,6]
        # window log-likelihood of the opponent's real moves for Z_s = k, k = 1..6 (the Z_s-masked reference term is
        # constant over k and cancels in the normalisation over types)
        ll = torch.zeros(nb, 6, device=dev)
        has_w = opp[rows].any(1)
        if has_w.any():
            rr = torch.nonzero(has_w).squeeze(1)
            tps = []
            for k in range(1, 7):
                t_ = tape_index(tb, rr)
                vv = t_["slot_val"].clone()
                vv[torch.arange(rr.shape[0], device=dev), s_c[rr]] = k
                t_["slot_val"] = vv
                tps.append(t_)
            big = tape_cat(tps)
            nr = rr.shape[0]
            mi_all = mv_idx[rows[rr]].repeat(6, 1)                                   # [6*nr, WIN]
            op_all = opp[rows[rr]].repeat(6, 1).float()
            g = torch.zeros(6 * nr, device=dev)
            for c0 in range(0, 6 * nr, chunk):
                sl = slice(c0, c0 + chunk)
                h, _ = net(tape_index(big, sl))
                for j in range(WIN):
                    lp = torch.log_softmax(net.h_policy(h[:, pre_idx(j)]).float(), -1)
                    g[sl] += lp.gather(1, mi_all[sl, j:j + 1]).squeeze(1) * op_all[sl, j]
            ll[rr] = g.view(6, nr).T
        lmu = torch.log_softmax(lq0 + ll, -1)
        dlb[rows, s_c] = (lmu - l0b[rows, s_c]).clamp(-8, 8)
    return dlb, cov


@torch.no_grad()
def value_rollout(net, tp, root_from, root_to, depth, gen, src_slots, temp=1.0):
    """Roll `depth` plies inside theta_real from tapes whose cone is materialised; root action given; later moves from
    the policy head (both sides), outcomes from the transition head. src_slots [B,8] bool: the key source slot tokens
    (the cone). Returns q_leaf [B,3], terminal, n_ply, ent [B,depth], miner_u, h_root [B,d], ebar [B,d]."""
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
        else:
            h, _ = net(tp)
            lg = net.h_policy(h[:, pre_idx(j)]).float() / temp
            mv = torch.multinomial(torch.softmax(lg, -1), 1, generator=gen).squeeze(1)
            fr, to = mv // NSQ, mv % NSQ
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
        cap = (oc == 1) | (oc == 3) | (oc == 4)
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
        clock = live & ~flag & (turn + j + 1 >= MAX_TURNS)
        if clock.any():
            tv = torch.zeros(B, 3, device=dev)
            tv[:, 1] = 1.0
            term_val = torch.where(clock[:, None], tv, term_val)
        done = done | flag | clock
    h, attn = net(tp, need_attn=True)
    ri = o_idx(0) + 2 * (n_ply - 1)
    q_leaf = torch.softmax(net.h_value(h[ar, ri]).float(), -1)
    q_leaf = torch.where(done[:, None], term_val * 0.98 + 0.02 / 3, q_leaf)
    src = torch.zeros(B, NTOK, dtype=torch.bool, device=dev)
    src[:, SLOT0:SLOT0 + NS] = src_slots
    for j in range(depth):
        okj = n_ply > j
        src[ar[okj], tp["ply_from"][okj, j]] = True
        src[ar[okj], tp["ply_to"][okj, j]] = True
    a = attn[ar, ri].float() * src.float()
    a = a / a.sum(-1, keepdim=True).clamp_min(1e-8)
    ebar = a @ net.pos.detach().float()
    return {"q_leaf": q_leaf, "terminal": done, "n_ply": n_ply, "ent": ent, "miner_u": miner_u,
            "h_root": h[:, :NROOT].float().mean(1), "ebar": ebar}


@torch.no_grad()
def action_l0(net, amp, tp_root, fr, to):
    """0-step action-value logit l0(a) = theta_real's A_0 value logit (+ theta_amp correction), hidden slots MASK."""
    tp = tape_clone(tp_root)
    tp["ply_actor"][:, 0] = 0
    tp["ply_from"][:, 0] = fr
    tp["ply_to"][:, 0] = to
    h, _ = net(tp)
    l = eff_logit(net.h_value(h[:, a_idx(0)]))
    if amp is not None:
        l = l + amp.correction(h[:, a_idx(0)]).float()
    return l, h[:, a_idx(0)]


@torch.no_grad()
def root_value_logit(net, tp_root):
    h, _ = net(tp_root)
    return eff_logit(net.h_value(h[:, GLOB]))


def trust_features(r, l0a, pathdis, dmax=6):
    q = r["q_leaf"]
    e = (q[:, 0] + 0.5 * q[:, 1]).clamp(1e-5, 1 - 1e-5)
    l = torch.log(e) - torch.log1p(-e)
    H = -(q.clamp_min(1e-9) * q.clamp_min(1e-9).log()).sum(-1)
    n = r["n_ply"].clamp_min(1).float()
    ent = r["ent"]
    f = torch.stack([r["n_ply"].float() / dmax, ent.sum(1) / n, ent.max(1).values, pathdis, (l - l0a).abs().clamp(max=10),
                     H, e, r["terminal"].float()], 1)
    return torch.cat([r["h_root"].detach().float(), f], 1)


@torch.no_grad()
def full_rollouts(net, strust, roots, fr, to, depth, gen, order="stiff_voi", with_belief=True, chunk=4096):
    """Cone rollouts for B rows (one rollout per row). roots: dict with tp (root tapes at t), tpw (window root tapes),
    wply [B,4,6], nw [B], l0b [B,8,6] 0-step log-belief at t, l0v [B] root state-value logit. Returns per-rollout tensors:
    e (leaf effective win), q_leaf, dlb, cov, dlv, ebar, miner_u, pathdis, feats (trust features), cone, in_range."""
    B = fr.shape[0]
    outs = []
    for c0 in range(0, B, chunk):
        sl = slice(c0, c0 + chunk)
        tp = tape_index(roots["tp"], sl)
        f_, t_ = fr[sl], to[sl]
        cone, in_range = select_cone(tp, f_, t_)
        tpz, zc, logp, step_of = sample_cone(net, tp, cone, gen, order, strust)
        lrev = reverse_logp(net, tpz, step_of)
        ncone = (cone >= 0).sum(1).clamp_min(1).float()
        pdis = (logp - lrev).abs() / ncone
        src = step_of >= 0
        r = value_rollout(net, tpz, f_, t_, depth, gen, src)
        q = r["q_leaf"]
        e = (q[:, 0] + 0.5 * q[:, 1]).clamp(1e-5, 1 - 1e-5)
        l0a, _ = action_l0(net, None, tp, f_, t_)
        o = {"e": e, "q_leaf": q, "dlv": logit(e) - roots["l0v"][sl], "ebar": r["ebar"], "miner_u": r["miner_u"],
             "pathdis": pdis, "feats": trust_features(r, l0a, pdis), "cone": cone, "in_range": in_range,
             "terminal": r["terminal"]}
        if with_belief:
            dlb, cov = window_messages(net, tape_index(roots["tpw"], sl), roots["wply"][sl], roots["nw"][sl], cone, zc,
                                       roots["l0b"][sl])
            o["dlb"], o["cov"] = dlb, cov
        outs.append(o)
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


N_TRUST_EXTRA = 8
