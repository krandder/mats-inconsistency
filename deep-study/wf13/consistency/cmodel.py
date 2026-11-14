"""cmodel.py: the any-order amplified predictor = wf10's learned aggregator (free form, compact tokens), extended to
forecast any check i outside any context S. It reads K sampled trajectories with M's numbers along them and one pass of
M. No exact inference, no hand-built score.

Token per trajectory (9 numbers): three route totals (M's check numbers on the context, M's empty-context numbers of the
assumption values, the searcher's log probability), two summaries (how many context checks the trajectory contradicts,
its worst check number), and M's two forecasts of the target (u: all assumptions in context; v: assumptions and the
context checks in context) as logit and probability.
Global token (37 numbers): one pass of M for the target, which check is the target, the compute level K, context size.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

PW = os.environ.get('PROBWORLD', '/home/agent/workspaces/mats/work/deep-study/wf7/probworld')
sys.path.insert(0, PW)
from model import Block  # noqa: E402  (wf7's pre-LN attention block, as in wf10)

N_TOK, N_GLOB = 9, 37
LEVELS = [int(v) for v in os.environ.get('LEVELS', '1,4,16,64').split(',')]      # at most 4 levels


def lgt(p):
    p = p.clamp(1e-6, 1 - 1e-6)
    return p.log() - (1 - p).log()


def load_set(path):
    """One set's tensors; if `path` is missing, its shards path[:-3]_K.pt (written by separate processes) are joined."""
    if os.path.exists(path):
        return torch.load(path)
    import glob
    parts = [torch.load(f) for f in sorted(glob.glob(path[:-3] + '_*.pt'), key=lambda f: int(f[:-3].split('_')[-1]))]
    return {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}


class CS:
    """A context set on the device, with the per-trajectory context features precomputed."""

    def __init__(self, path, dev):
        d = load_set(path)
        self.meta = {k: v for k, v in d.items() if k not in ('lu', 'lv', 'la', 'lq', 'p_one', 'x', 'mask')}
        self.x, self.mask = d['x'].long().to(dev), d['mask'].to(dev)
        self.lu, self.lv = d['lu'].to(dev), d['lv'].to(dev)
        self.la, self.lq, self.p_one = d['la'].to(dev), d['lq'].to(dev), d['p_one'].float().to(dev)
        self.N, self.K = self.lu.shape[0], self.lu.shape[1]
        se, nv, mn = [], [], []
        for i in range(0, self.N, 4096):
            lu = self.lu[i:i + 4096].float()
            sgn = (2 * self.x[i:i + 4096] - 1)[:, None, :].float()
            ell = F.logsigmoid(lu * sgn) * self.mask[i:i + 4096][:, None, :].float()
            se.append(ell.sum(-1))
            nv.append((ell < -0.6931).float().sum(-1))
            mn.append(ell.min(-1).values)
        self.se, self.nv, self.mn = torch.cat(se), torch.cat(nv), torch.cat(mn)
        self.size = self.mask.float().sum(1) / 30.0


def features(D, idx, tgt, sel, level):
    """Queries (context idx (B,), target tgt (B,), trajectory selection sel (B,K)) -> tok, g, raw, uv."""
    ii = idx[:, None]
    lu = D.lu[ii, sel, tgt[:, None]].float()
    lv = D.lv[ii, sel, tgt[:, None]].float()
    se, la, lq = D.se[ii, sel], D.la[ii, sel], D.lq[ii, sel]
    u, v = torch.sigmoid(lu), torch.sigmoid(lv)
    tok = torch.stack([se / 10, la / 10, lq / 10, D.nv[ii, sel] * 0.3, D.mn[ii, sel] * 0.6, lu / 3, lv / 3, u, v], -1)
    po = D.p_one[idx, tgt]
    g = torch.zeros(idx.shape[0], N_GLOB, device=idx.device)
    g[:, 0], g[:, 1] = lgt(po) / 3, po
    g[torch.arange(len(tgt), device=idx.device), 2 + tgt] = 1.0
    g[:, 32 + level] = 1.0
    g[:, 36] = D.size[idx]
    return tok, g, torch.stack([se, la, lq], -1), torch.stack([u, v], -1)


class Agg(nn.Module):
    """wf10's free form: a small transformer over [global token, K trajectory tokens]; readout = MLP over the global
    state and `npool` learned softmax-weighted averages of the token states and of M's two forecasts; the pooling logits
    also get a linear term in the three route totals (initialised at zero)."""

    def __init__(self, d=64, L=2, nh=4, npool=4, dv=16):
        super().__init__()
        self.tok = nn.Sequential(nn.Linear(N_TOK, d), nn.GELU(), nn.Linear(d, d))
        self.glob = nn.Sequential(nn.Linear(N_GLOB, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(L)])
        self.ln = nn.LayerNorm(d)
        self.pool_s = nn.Linear(d, npool)
        self.pool_raw = nn.Linear(3, npool, bias=False)
        nn.init.zeros_(self.pool_raw.weight)
        self.pool_v = nn.Linear(d, dv)
        self.head = nn.Sequential(nn.Linear(d + npool * (dv + 2) + N_GLOB, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, tok, g, raw, uv):
        x = torch.cat([self.glob(g)[:, None], self.tok(tok)], 1)
        for b in self.blocks:
            x = b(x)
        x = self.ln(x)
        al = torch.softmax(self.pool_s(x[:, 1:]) + self.pool_raw(raw), 1)
        val = torch.cat([self.pool_v(x[:, 1:]), uv], -1)
        pooled = torch.einsum('bkp,bkv->bpv', al, val).flatten(1)
        return self.head(torch.cat([x[:, 0], pooled, g], 1)).squeeze(-1)


def forecast(model, D, idx, tgt, sel):
    """Logit of the amplified P(x_tgt = 1 | context idx) from the trajectories sel (B,K), K in LEVELS."""
    return model(*features(D, idx, tgt, sel, LEVELS.index(sel.shape[1])))


# ---------------------------------------------------------------- fixed estimators with the same interface
def fixed_forecast(name, D, idx, tgt, sel):
    """one: one pass of M.  meanu: plain average of M(x | h).  exu: the score's exact weights R(h, x_S) / psi(h) on
    M(x | h) (self-normalised).  exuv: the same weights on M's full-context number."""
    if name == 'one':
        return lgt(D.p_one[idx, tgt])
    ii = idx[:, None]
    u = torch.sigmoid(D.lu[ii, sel, tgt[:, None]].float())
    if name == 'meanu':
        return lgt(u.mean(1))
    w = torch.softmax((D.se[ii, sel] + D.la[ii, sel] - D.lq[ii, sel]).double(), 1).float()
    if name == 'exu':
        return lgt((w * u).sum(1))
    if name == 'exuv':
        return lgt((w * torch.sigmoid(D.lv[ii, sel, tgt[:, None]].float())).sum(1))
    raise KeyError(name)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
