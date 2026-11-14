"""P10 models.

B2: one pre-LN transformer block with shared weights applied T times, input recall (the input
embedding is re-added at every step: u_t = LN_s(h_t) + e(x)), readout after every step.
Tokens: digit or MASK embedding + learned position embedding; no Sudoku structure is given.
Optional slot tokens (M2): phase 1 (passes 1..T) slots carry a query embedding and yield the prior
p(e_k|X) from their state after pass T; phase 2 (passes T+1..2T) slots carry the event value embedding
(teacher-forced e in training, argmax prior at test). B2-reg: slots carry learned constants.

ValueHead (M1): MLP on [h_c, pooled h, p_c] -> v(s, c); pooled -> solve-probability logit.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, u):
        a = self.ln1(u)
        u = u + self.attn(a, a, a, need_weights=False)[0]
        return u + self.mlp(self.ln2(u))


class B2(nn.Module):
    def __init__(self, ncell, ndig, d=64, heads=4, nslot=0):
        super().__init__()
        self.ncell, self.ndig, self.d, self.nslot = ncell, ndig, d, nslot
        self.tok = nn.Embedding(ndig + 1, d)
        self.pos = nn.Parameter(torch.randn(ncell, d) * 0.02)
        self.ln_s = nn.LayerNorm(d)
        self.block = Block(d, heads)
        self.ln_o = nn.LayerNorm(d)
        self.out = nn.Linear(d, ndig)
        if nslot:
            self.slot_pos = nn.Parameter(torch.randn(nslot, d) * 0.02)
            self.slot_query = nn.Parameter(torch.randn(nslot, d) * 0.02)
            self.slot_val = nn.Parameter(torch.randn(nslot, 2, d) * 0.02)
            self.slot_const = nn.Parameter(torch.randn(nslot, d) * 0.02)
            self.prior_w = nn.Parameter(torch.zeros(nslot, d))
            self.prior_b = nn.Parameter(torch.zeros(nslot))
            self.ln_p = nn.LayerNorm(d)

    def embed(self, x):
        return self.tok(x) + self.pos

    def step(self, h, e):
        return self.block(self.ln_s(h) + e)

    def readout(self, h):
        return self.out(self.ln_o(h[:, : self.ncell]))

    def run(self, x, T, h=None, e=None, all_logits=False):
        """Run T passes on input tokens x (B,C). Returns final hidden and logits (last or list)."""
        if e is None:
            e = self.embed(x)
        if h is None:
            h = torch.zeros_like(e)
        outs = []
        for _ in range(T):
            h = self.step(h, e)
            if all_logits:
                outs.append(self.readout(h))
        return h, (outs if all_logits else self.readout(h))

    # ---- slots (M2)
    def slot_inputs(self, B, mode, ev=None):
        """mode 'query' | 'const' | 'value' (ev: (B,K) in [0,1], straight-through values)."""
        if mode == 'query':
            s = self.slot_query[None].expand(B, -1, -1)
        elif mode == 'const':
            s = self.slot_const[None].expand(B, -1, -1)
        else:
            v0, v1 = self.slot_val[:, 0], self.slot_val[:, 1]
            s = v0[None] + ev[..., None] * (v1 - v0)[None]
        return s + self.slot_pos[None]

    def prior_logits(self, h):
        hs = self.ln_p(h[:, self.ncell:])
        return (hs * self.prior_w[None]).sum(-1) + self.prior_b[None]

    def run_slots(self, x, T, ev_fn=None, reg=False):
        """Two-phase run. Phase 1: T passes with query slots (or const for B2-reg) -> prior logits.
        Phase 2: T passes with value slots given by ev_fn(prior_logits) (or const for B2-reg).
        Returns (cell logits after 2T passes, prior logits after T passes, hidden after T)."""
        B = x.shape[0]
        ec = self.embed(x)
        e1 = torch.cat([ec, self.slot_inputs(B, 'const' if reg else 'query')], 1)
        h, _ = self.run(None, T, e=e1)
        pl = self.prior_logits(h)
        if reg:
            e2 = e1
        else:
            ev = ev_fn(pl)
            e2 = torch.cat([ec, self.slot_inputs(B, 'value', ev)], 1)
        h2, logits = self.run(None, T, h=h, e=e2)
        return logits, pl, h


class ValueHead(nn.Module):
    def __init__(self, d, ndig, hid=128):
        super().__init__()
        self.ln = nn.LayerNorm(2 * d)
        self.mlp = nn.Sequential(nn.Linear(2 * d + ndig, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
        self.solve = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, 1))

    def forward(self, h, p):
        """h (B,C,d) hidden state (detached), p (B,C,ndig) probabilities. Returns v (B,C), solve logit (B,)."""
        pooled = h.mean(1, keepdim=True).expand_as(h)
        z = self.ln(torch.cat([h, pooled], -1))
        v = self.mlp(torch.cat([z, p], -1)).squeeze(-1)
        return v, self.solve(h.mean(1)).squeeze(-1)


def progressive_loss(model, x, y, lm, T, x_pre=None, alpha=0.5):
    """Bansal et al. 2022 progressive loss mixed with the full-depth loss (alpha).
    x (B,C) tokens, y (B,C) targets 0..ndig-1, lm (B,C) float mask of supervised (masked) cells.
    x_pre: optional more-masked input used during the no-grad prefix (continuation augmentation)."""
    def ce(logits):
        l = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction='none').view_as(lm)
        return (l * lm).sum() / lm.sum().clamp_min(1)
    _, lf = model.run(x, T)
    full = ce(lf)
    n = int(torch.randint(0, T, (1,)))
    k = int(torch.randint(1, T - n + 1, (1,)))
    h = None
    if n > 0:
        with torch.no_grad():
            h, _ = model.run(x_pre if x_pre is not None else x, n)
        h = h.detach()
    _, lp = model.run(x, k, h=h)
    prog = ce(lp)
    return (1 - alpha) * full + alpha * prog, full.detach()
