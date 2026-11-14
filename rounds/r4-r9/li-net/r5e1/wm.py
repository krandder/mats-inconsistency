"""R5-E1 networks. The four LIR blocks (spec 1.2):
  theta_real  WorldModel: pre-LN transformer (smallnet netdef.Block, unchanged) over 36 cell tokens + 1 note token; outputs 36
              occupancy logits. Trained only on real board realizations (any-order masked BCE).
  theta_trust TrustNet: 2-hidden-layer MLP on stop_grad rollout features -> e_hat >= 0, beta = 1 / (e_hat + 0.05).
  theta_amp   KeyHead: k_r = normalize(W_K LN(hbar_r)), hbar_r = attention-pooled stop_grad hidden states of theta_real on the
              cells unmasked by rollout r; v_r = Delta l_r (theta_real's rollout logits minus root logits).
  theta_agg   AttnL3: 3 pre-LN residual self-attention layers (d 64, 4 heads) over the rollout tokens
              z_r = [k_r, W_v Delta l_r, log beta_r, log c_r] -> per-rollout per-cell weights w_r in R_+^36,
              l* = l0 + sum_r w_r * Delta l_r. Baselines: mix, ataraxos_mmd, kc_dedup.
  ActionHead  (Part B) pi_act(c | mu, U): per-cell MLP on [h_c, l_c, u * l_c, u], softmax over the 36 cells.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

SMALLNET = os.environ.get("SMALLNET_DIR", "/home/agent/workspaces/eurisko-study/li-net/smallnet")
if SMALLNET not in sys.path:
    sys.path.insert(0, SMALLNET)
from netdef import Block  # noqa: E402  (smallnet's pre-LN block, unchanged)

NC = 36
W = 6
NREL = (2 * W - 1) ** 2          # 121 relative offsets between cells; +3 for note->cell, cell->note, note->note


def rel_index() -> torch.Tensor:
    r = torch.arange(NC) // W
    c = torch.arange(NC) % W
    dy = r[:, None] - r[None, :] + (W - 1)
    dx = c[:, None] - c[None, :] + (W - 1)
    idx = torch.empty(NC + 1, NC + 1, dtype=torch.long)
    idx[:NC, :NC] = dy * (2 * W - 1) + dx
    idx[NC, :NC] = NREL          # note token (query) -> cells
    idx[:NC, NC] = NREL + 1      # cells -> note token
    idx[NC, NC] = NREL + 2
    return idx


class WorldModel(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, ff=384, drop=0.0):
        super().__init__()
        self.d, self.layers, self.heads, self.ff = d, layers, heads, ff
        self.pos = nn.Parameter(torch.randn(NC, d) * 0.02)
        self.ev = nn.Embedding(3, d)          # evidence: none / probe reported miss / probe reported hit
        self.rev = nn.Embedding(3, d)         # reveal state: masked / revealed water / revealed ship
        self.nflag = nn.Embedding(2, d)       # cell named by the commander's note
        self.ntok = nn.Parameter(torch.randn(d) * 0.02)
        for e in (self.ev, self.rev, self.nflag):
            nn.init.normal_(e.weight, std=0.02)
        self.relb = nn.Parameter(torch.zeros(layers, heads, NREL + 3))
        self.register_buffer("relidx", rel_index(), persistent=False)
        self.blocks = nn.ModuleList([Block(d, heads, ff, drop) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.occ = nn.Linear(d, 1)

    def forward(self, ev, note, rev, val, need_attn=False):
        """ev [B,36] long, note [B] long, rev [B,36] bool, val [B,36] long/bool (read only where rev).
        Returns logits [B,36], hidden h [B,37,d] (token 36 = note), head-mean last-layer attention [B,37,37] or None."""
        B = ev.shape[0]
        rs = torch.where(rev, 1 + val.long(), torch.zeros_like(ev))
        nf = F.one_hot(note, NC).long()
        x = self.pos[None] + self.ev(ev) + self.rev(rs) + self.nflag(nf)
        xn = (self.ntok[None] + self.pos[note])[:, None]
        x = torch.cat([x, xn], 1)
        attn = None
        for li, blk in enumerate(self.blocks):
            bias = self.relb[li][:, self.relidx]                      # [heads, 37, 37]
            if need_attn and li == len(self.blocks) - 1:
                n, d = x.shape[1], x.shape[2]
                hd = d // blk.h
                q, k, _ = blk.qkv(blk.ln1(x)).view(B, n, 3, blk.h, hd).permute(2, 0, 3, 1, 4)
                attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(hd) + bias, -1).mean(1)
            x = blk(x, bias)
        h = self.lnf(x)
        return self.occ(h[:, :NC]).squeeze(-1), h, attn

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class TrustNet(nn.Module):
    """theta_trust: stop_grad features [hbar (d), mean rollout entropy, temperature flag, forward-reverse disagreement]."""

    def __init__(self, nin, hid=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(nin, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))

    def forward(self, x):
        return F.softplus(self.f(x)).squeeze(-1)


def beta_from(e):
    return 1.0 / (e + 0.05)


class KeyHead(nn.Module):
    """theta_amp key: k_r = normalize(W_K LN(hbar_r)) in R^16 (hbar_r is already stop_grad)."""

    def __init__(self, d=128, kdim=16):
        super().__init__()
        self.W = nn.Linear(d, kdim)

    def forward(self, hbar):
        return F.normalize(self.W(F.layer_norm(hbar, (hbar.shape[-1],))), dim=-1)


def crowding(k, valid, gamma=2.0):
    """c_r = sum_b max(0, <k_r, k_b>)^gamma over valid b (includes b = r, so c_r >= 1)."""
    g = torch.relu(k @ k.transpose(1, 2)) ** gamma
    g = g * valid[:, None, :].float()
    return g.sum(2).clamp_min(1e-6)


def agg_mix(lr, valid):
    """naive rollout average in probability space -> logit."""
    q = torch.sigmoid(lr) * valid[..., None].float()
    m = q.sum(1) / valid.float().sum(1, keepdim=True).clamp_min(1)
    m = m.clamp(1e-6, 1 - 1e-6)
    return torch.log(m) - torch.log1p(-m)


def agg_mmd(l0, dl, beta, valid, eta):
    """ataraxos_mmd: l0 + eta * sum_r beta_r Delta l_r."""
    return l0 + eta * ((beta * valid.float())[..., None] * dl).sum(1)


def agg_kc(l0, dl, beta, k, valid, u, gamma=2.0):
    """kc_dedup: l0 + u * sum_r (beta_r / c_r) Delta l_r."""
    c = crowding(k, valid, gamma)
    return l0 + u * ((beta / c * valid.float())[..., None] * dl).sum(1)


class AttnL3(nn.Module):
    def __init__(self, kdim=16, vdim=16, d=64, layers=3, heads=4, init_w=1.0 / 24):
        super().__init__()
        self.Wv = nn.Linear(NC, vdim)
        self.inp = nn.Linear(kdim + vdim + 2, d)
        self.att = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers)])
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ff = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                                 for _ in range(layers)])
        self.out = nn.Linear(d, NC)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, math.log(math.expm1(init_w)))

    def weights(self, dl, beta, k, valid, gamma=2.0):
        """per-rollout per-cell weights w [B, K, 36] (0 on padding)."""
        c = crowding(k, valid, gamma)
        dlc = dl.clamp(-8, 8)
        x = self.inp(torch.cat([k, self.Wv(dlc), beta.clamp_min(1e-6).log()[..., None], c.log()[..., None]], -1))
        pad = ~valid
        for at, ln, ff in zip(self.att, self.ln, self.ff):
            y = ln(x)
            x = x + at(y, y, y, key_padding_mask=pad, need_weights=False)[0]
            x = x + ff(x)
        return F.softplus(self.out(x)) * valid[..., None].float()

    def forward(self, l0, dl, beta, k, valid):
        w = self.weights(dl, beta, k, valid)
        return l0 + (w * dl.clamp(-8, 8)).sum(1), w


class ActionHead(nn.Module):
    """pi_act(c | mu, U): logits_c = MLP([h_c, l_c, u l_c, u]), u = +1 (U+ seek ship) / -1 (U- avoid ship)."""

    def __init__(self, d=128, hid=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d + 3, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))

    def forward(self, h, l, u):
        """h [B,36,d], l [B,36], u [B] (+-1) -> log pi [B,36]."""
        uu = u[:, None].expand_as(l)
        x = torch.cat([h, l[..., None], (uu * l)[..., None], uu[..., None]], -1)
        return torch.log_softmax(self.f(x).squeeze(-1), -1)


# ------------------------------------------------------------------------------------------------ metrics helpers
def bern_kl(p, q, eps=1e-6):
    """KL(Bern(p) || Bern(q)) elementwise (nats)."""
    p = p.clamp(0, 1)
    q = q.clamp(eps, 1 - eps)
    a = torch.where(p > 0, p * (torch.log(p.clamp_min(1e-300)) - torch.log(q)), torch.zeros_like(p))
    b = torch.where(p < 1, (1 - p) * (torch.log((1 - p).clamp_min(1e-300)) - torch.log1p(-q)), torch.zeros_like(p))
    return a + b


def bern_ent(p):
    p = p.clamp(0, 1)
    return -(torch.special.xlogy(p, p) + torch.special.xlogy(1 - p, 1 - p))


def bern_jsd(p, q):
    return bern_ent(0.5 * (p + q)) - 0.5 * (bern_ent(p) + bern_ent(q))


def excess_surprise(q, y):
    """Bernoulli version of R4-E1's excess surprise: max(0, -log q(y) - H(q)) + max(0, 2(|q|^2 - q(y)))."""
    q = q.clamp(1e-6, 1 - 1e-6)
    qy = torch.where(y, q, 1 - q)
    Hq = bern_ent(q)
    sq = q * q + (1 - q) * (1 - q)
    return (-qy.log() - Hq).clamp_min(0) + (2 * (sq - qy)).clamp_min(0)
