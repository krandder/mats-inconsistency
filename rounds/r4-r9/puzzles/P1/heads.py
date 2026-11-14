"""Anytime-value heads on detached backbone features, settlement targets, and losses.

Features at step t (all detached): LayerNorm(h_t) (no affine), p_t, |p_t - p_{t-2}|,
|p_t - p_{t-1}|, Bernoulli entropy H(p_t), log(#positions).  No step counter.
Trunk: 3-conv (dilation 1) to 32 ch, then dilated 3-convs (2, 4, 8) with residuals, plus
global mean / max / log1p(sum) pooled context broadcast back to every position.
Output: one value per position per horizon (summed over positions for a total), or a
global logit (mean of per-position logits) for probability heads.
Every probability is clamped to [1e-4, 1 - 1e-4].
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-4
HORIZONS = [1, 2, 4, 8, 16, 32, 64, "cap"]
KL_HORIZONS = [2, 4, 8, 16, 32, 64, "cap"]      # H=1 excluded (2-cycles earn nothing)


def clampp(p):
    return p.clamp(EPS, 1 - EPS)


def logq(p, y):
    """log probability of label y (0/1) under Bernoulli p (clamped)."""
    p = clampp(p)
    return y * torch.log(p) + (1 - y) * torch.log1p(-p)


def kl_bern(p1, p0):
    """KL(Bern(p1) || Bern(p0)), elementwise, clamped."""
    p1, p0 = clampp(p1), clampp(p0)
    return p1 * (torch.log(p1) - torch.log(p0)) + (1 - p1) * (torch.log1p(-p1) - torch.log1p(-p0))


def ent_bern(p):
    p = clampp(p)
    return -(p * torch.log(p) + (1 - p) * torch.log1p(-p))


def features(h, p, p1, p2):
    """h: (B, C, *S) latent; p, p1, p2: (B, *S) probs at t, t-1, t-2. Returns (B, C+5, *S)."""
    hn = F.layer_norm(h.transpose(1, -1).float(), (h.shape[1],)).transpose(1, -1)
    npos = p[0].numel()
    extra = torch.stack([p, (p - p2).abs(), (p - p1).abs(), ent_bern(p),
                         torch.full_like(p, math.log(npos))], 1)
    return torch.cat([hn, extra.to(hn.dtype)], 1)


class HeadNet(nn.Module):
    """nout per-position outputs (+ optional global outputs). dim = 1 or 2.
    ablate=True drops the sum-pooled context and zeroes the log(#positions) input."""

    def __init__(self, cin, nout, dim, ch=32, kind="value", ablate=False, cdim=0):
        super().__init__()
        Conv = nn.Conv1d if dim == 1 else nn.Conv2d
        self.dim, self.kind, self.ablate, self.nout = dim, kind, ablate, nout
        self.c0 = Conv(cin, ch, 3, padding=1)
        self.cs = nn.ModuleList([Conv(ch, ch, 3, padding=d, dilation=d) for d in (2, 4, 8)])
        self.ctx = nn.Linear(3 * ch, ch)
        self.o1 = Conv(ch, ch, 1)
        self.o2 = Conv(ch, nout, 1)
        self.cdim = cdim
        if cdim:   # Q-ACT: continuation advantage MLP on pooled context + log c
            self.qm = nn.Sequential(nn.Linear(3 * ch + 1, 64), nn.ReLU(), nn.Linear(64, 1))

    def trunk(self, f):
        if self.ablate:
            f = f.clone()
            f[:, -1] = 0
        z = F.relu(self.c0(f))
        for c in self.cs:
            z = z + F.relu(c(z))
        zf = z.flatten(2)
        mean = zf.mean(2)
        mx = zf.amax(2)
        sm = torch.log1p(F.relu(zf).sum(2))
        if self.ablate:
            sm = torch.zeros_like(sm)
        pooled = torch.cat([mean, mx, sm], 1)
        g = self.ctx(pooled)
        z = F.relu(z + g.view(g.shape[0], g.shape[1], *([1] * self.dim)))
        return z, pooled

    def forward(self, f):
        """Returns per-position outputs (B, nout, *S) (softplus for kind='kl') and pooled."""
        z, pooled = self.trunk(f)
        o = self.o2(F.relu(self.o1(z)))
        if self.kind == "kl":
            o = F.softplus(o)
        return o, pooled

    def totals(self, f):
        o, pooled = self.forward(f)
        if self.kind in ("prob",):
            return o.flatten(2).mean(2), pooled         # global logit(s)
        return o.flatten(2).sum(2), pooled              # totals per horizon

    def qact_adv(self, pooled, npos, logc):
        """Continuation advantage A(t, c) total. pooled (B, 3ch); logc (K,) -> (B, K)."""
        B, K = pooled.shape[0], logc.shape[0]
        inp = torch.cat([pooled.unsqueeze(1).expand(B, K, pooled.shape[1]),
                         logc.view(1, K, 1).expand(B, K, 1)], 2)
        return self.qm(inp).squeeze(2) * npos


def head_macs_per_pos(cin, nout, dim, ch=32):
    k = 3 if dim == 1 else 9
    return cin * ch * k + 3 * ch * ch * k + ch * ch + ch * nout


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- arms
# name: (kind, outputs, settlement)
ARMS_T2M = {
    "R4": ("value", 1, "G1"),
    "R7": ("value", 8, "G"),
    "R7abl": ("value", 8, "G"),
    "R8": ("kl", 7, "K"),
    "R11": ("prob", 1, "exact"),
    "R12": ("prob", 1, "cons"),
    "R13": ("qact", 1, "qact"),
    "R5_30": ("prob", 1, "ponder30"),
    "R5_100": ("prob", 1, "ponder100"),
}
ARMS_MAZE = {
    "R4": ("value", 1, "G1"),
    "R7": ("value", 8, "G"),
    "R7abl": ("value", 8, "G"),
    "R8": ("kl", 7, "K"),
    "R9s": ("value", 8, "P"),       # pseudo-label signed, base fit on training distribution
    "R11": ("prob", 1, "exact"),
    "R12": ("prob", 1, "cons"),
    "R13": ("qact", 1, "qact"),
}
TT_ARMS = {"R9": "R8", "R9s": "R9s", "R12TT": "R12", "R7TT": "R7"}   # TT arm: base arm
LABEL_FREE = {"K", "P", "cons"}


def make_head(kind, nout, cin, dim, ablate=False):
    if kind == "qact":
        return HeadNet(cin, 2, dim, kind="qact", cdim=1)   # ch0 u_halt, ch1 per-pos advantage
    return HeadNet(cin, nout, dim, kind=kind, ablate=ablate)


# ---------------------------------------------------------------- targets
def horizon_index(t, H, T):
    return T if H == "cap" else min(t + H, T)


def perpos_targets(Z, t, T, y=None, yhat=None, kinds=("G", "K", "P")):
    """Z: (T+1, B, *S) stored LOGITS (fp16 ok; Z[0] = 0 means p_0 = 0.5).
    Returns dict kind -> (B, nH, *S) float targets at t."""
    P = lambda i: torch.sigmoid(Z[i].float())  # noqa: E731
    pt = P(t)
    out = {}
    if "G" in kinds:
        base = logq(pt, y)
        out["G"] = torch.stack([logq(P(horizon_index(t, H, T)), y) - base for H in HORIZONS], 1)
    if "K" in kinds:
        out["K"] = torch.stack([kl_bern(P(horizon_index(t, H, T)), pt) for H in KL_HORIZONS], 1)
    if "P" in kinds:
        base = logq(pt, yhat)
        out["P"] = torch.stack([logq(P(horizon_index(t, H, T)), yhat) - base for H in HORIZONS], 1)
    return out


def decision_times(T_max, n, gen, even=False, lo=1):
    """n log-uniform, stratified decision times in [lo, T_max] (distinct when possible)."""
    grid = torch.arange(lo, T_max + 1)
    if even:
        grid = grid[grid % 2 == 0]
    if len(grid) <= n:
        return grid.tolist()
    lg = torch.log(grid.float())
    edges = torch.linspace(lg[0].item(), lg[-1].item() + 1e-6, n + 1)
    u = edges[:-1] + (edges[1:] - edges[:-1]) * torch.rand(n, generator=gen)
    idx = torch.searchsorted(lg, u).clamp(max=len(grid) - 1)
    ts = sorted(set(grid[idx].tolist()))
    # fill duplicates with random unused points
    rest = [g for g in grid.tolist() if g not in ts]
    while len(ts) < n and rest:
        ts.append(rest.pop(int(torch.randint(len(rest), (1,), generator=gen))))
    return sorted(ts)
