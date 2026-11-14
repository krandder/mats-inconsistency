"""P12 arbitrageurs, semantic loss, true arbitrage, shared decode and metrics (torch).

Tensor layout: (B, n, n, 4) per (cell, slot d in E,S,W,N); an edge appears twice, as (c,d) and
(nb(c,d), opp d). nb(x)[c,d] = x[nb(c,d), opp d] (0 off-grid).

Directed-flow BP (the arbitrageur). Each open edge is one 3-state variable {off, u->v, v->u};
seen from cell c the two on-states are "out of c" and "in to c". Cell factor: #out - #in = +1 at S,
-1 at G, 0 elsewhere. This is flow conservation on directed arcs (Gamarnik-Shah-Wei 2012); the
3-state variable forbids an edge carrying flow both ways (a 2-cycle), which the 8-binary-arc form
(70 / 56 configurations) allows and which learned positive log-odds would exploit. Configurations
per cell: 19 ordinary, 16 at S, 16 at G. Messages factor -> edge are 8 numbers per cell
(out E,S,W,N, in E,S,W,N), log-odds relative to off, so the state is the same size as 8 arcs.
Unary: one log-odds theta_e per open edge, shared by both directions (true costs: theta = -cost).
Variants: max-sum (tau = 0) or sum-product at temperature tau; damping on messages.
"""
import itertools, math
import numpy as np
import torch
import torch.nn.functional as F

BIG = 1e4
OPP = [2, 3, 0, 1]


def nb(x):
    out = torch.zeros_like(x)
    out[:, :, :-1, 0] = x[:, :, 1:, 2]
    out[:, :-1, :, 1] = x[:, 1:, :, 3]
    out[:, :, 1:, 2] = x[:, :, :-1, 0]
    out[:, 1:, :, 3] = x[:, :-1, :, 1]
    return out


def es_to_slots(uE, uS):
    """(B,n,n) east and south edge values -> symmetric (B,n,n,4)."""
    B, n, _ = uE.shape
    z = uE.new_zeros(B, n, n, 4)
    z[..., 0] = uE; z[..., 1] = uS
    z[:, :, 1:, 2] = uE[:, :, :-1]; z[:, 1:, :, 3] = uS[:, :-1, :]
    z[:, :, -1, 0] = 0; z[:, -1, :, 1] = 0
    return z


def slots_to_es(x):
    return x[..., 0], x[..., 1]


# ------------------------------------------------------------------ configurations
def _cfgs(net):
    V = []
    for st in itertools.product([0, 1, 2], repeat=4):  # 0 off, 1 out, 2 in
        o = [1 if s == 1 else 0 for s in st]; i = [1 if s == 2 else 0 for s in st]
        if sum(o) - sum(i) == net:
            V.append(o + i)
    return torch.tensor(V, dtype=torch.float32)


CFG = {0: _cfgs(0), 1: _cfgs(1), -1: _cfgs(-1)}
assert [len(CFG[k]) for k in (0, 1, -1)] == [19, 16, 16]
CFG_ALL = torch.cat([CFG[0], CFG[1], CFG[-1]], 0)                      # (51, 8)
CFG_NET = (CFG_ALL[:, :4].sum(1) - CFG_ALL[:, 4:].sum(1)).long()       # (51,)


class FlowBP:
    """Persistent-message directed-flow BP. State: (mo, mi) each (B,n,n,4).
    One table of the 51 configurations with net flow in {-1,0,+1}; each cell uses those whose net
    flow equals its own (19 ordinary, 16 at S, 16 at G)."""

    def __init__(self, open_, S, G, tau=0.0, damp=0.5):
        self.open = open_.bool()
        B, n = open_.shape[0], open_.shape[1]
        dev = open_.device
        net = torch.zeros(B, n, n, dtype=torch.long, device=dev)
        ar = torch.arange(B, device=dev)
        net[ar, S[:, 0].long(), S[:, 1].long()] = 1
        net[ar, G[:, 0].long(), G[:, 1].long()] = -1
        self.net = net
        self.tau, self.damp = tau, damp
        self.V = CFG_ALL.to(dev)
        self.cellmask = (CFG_NET.to(dev)[None, None, None, :] == net[..., None])     # (B,n,n,51)
        self.on_mask = self.V.t().bool()                                            # (8,51)
        self.off_mask = ((self.V[:, :4] == 0) & (self.V[:, 4:] == 0)).t()          # (4,51)
        self.mask_on = self.cellmask.unsqueeze(-2) & self.on_mask                   # (B,n,n,8,51)
        self.mask_off = self.cellmask.unsqueeze(-2) & self.off_mask                 # (B,n,n,4,51)

    def init_state(self):
        z = torch.zeros(self.open.shape, device=self.open.device)
        return z, z.clone()

    def _red(self, x, mask):
        x = x.masked_fill(~mask, -12 * BIG)
        if self.tau == 0:
            return x.max(-1).values
        return self.tau * torch.logsumexp(x / self.tau, -1)

    def step(self, state, theta):
        """One synchronous BP step. theta (B,n,n,4) symmetric edge log-odds (per on-state)."""
        mo, mi = state
        th = theta.masked_fill(~self.open, -BIG)
        A = torch.cat([th + nb(mi), th + nb(mo)], -1).clamp(-BIG, BIG)   # (B,n,n,8): out slots, in slots
        E = A @ self.V.t()                                                 # (B,n,n,51)
        on = self._red(E.unsqueeze(-2) - A.unsqueeze(-1), self.mask_on)    # (B,n,n,8)
        off = self._red(E.unsqueeze(-2), self.mask_off)                    # (B,n,n,4)
        new = (on - torch.cat([off, off], -1)).clamp(-BIG, BIG)
        nmo = new[..., :4].masked_fill(~self.open, 0.0); nmi = new[..., 4:].masked_fill(~self.open, 0.0)
        if self.damp:
            nmo = (1 - self.damp) * nmo + self.damp * mo
            nmi = (1 - self.damp) * nmi + self.damp * mi
        return nmo, nmi

    def delta(self, state):
        """Message contribution to the edge log-odds (B,n,n,4): edge belief = theta + delta.
        beta_uv = theta + mo[c] + nb(mi)[c];  beta_vu = theta + mi[c] + nb(mo)[c];
        max-sum: max(x1, x2); sum-product: tau*(log(e^(x1/tau) + e^(x2/tau)) - log 2), x1 = mo + nb(mi),
        x2 = mi + nb(mo) (so with zero messages delta = 0 and the price is theta). Under the model
        P ~ exp(sum_e theta_e [e on] / tau) the exact on/off log-odds is (theta + delta)/tau + log 2."""
        mo, mi = state
        x1 = mo + nb(mi); x2 = mi + nb(mo)
        if self.tau == 0:
            d = torch.maximum(x1, x2)
        else:
            d = self.tau * (torch.logaddexp(x1 / self.tau, x2 / self.tau) - math.log(2.0))
        return d.masked_fill(~self.open, 0.0)


def unary(z, mode="logodds", c=0.0):
    """Edge unary for the flow arbitrageur from proposer logits z.
    logodds: theta = z - c (MAP of the independent-edge model restricted to flows; positive weights on a
             cycle make max-sum diverge on the computation tree, since the flow can loop for ever).
    logprob: theta = log sigma(z) - c <= -c < 0 (every cycle has negative weight, the Gamarnik-Shah-Wei
             setting: max-sum converges to the min-cost path sum_e (c - log p_e) when it is unique)."""
    if mode == "logodds":
        return z - c
    return F.logsigmoid(z) - c


# ------------------------------------------------------------------ undirected degree BP (A0-deg, weak arm)
def deg_bp_step(m, unary, open_, endm, damp=0.5, maxprod=True):
    """pilot.py bp_step: edges are binary variables, cell factor degree 1 at S/G, {0,2} elsewhere."""
    nin = unary + nb(m)
    nin = torch.where(open_, nin, torch.full_like(nin, -30.0))
    red = (lambda x, dim: x.max(dim).values) if maxprod else torch.logsumexp
    new = []
    for d in range(4):
        o = [k for k in range(4) if k != d]; a, b, c = (nin[..., k] for k in o)
        l1 = red(torch.stack([a, b, c], -1), -1)
        l2 = red(torch.stack([torch.zeros_like(a), a + b, a + c, b + c], -1), -1)
        new.append(torch.where(endm, -l1, l1 - l2))
    new = torch.stack(new, -1).clamp(-30, 30)
    new = torch.where(open_, new, torch.zeros_like(new))
    return (1 - damp) * new + damp * m


def deg_beliefs(m, unary, open_):
    b = unary + m + nb(m)
    return torch.where(open_, b, torch.full_like(b, -30.0))


# ------------------------------------------------------------------ semantic loss and true arbitrage
def semantic_loss(p, open_, endm):
    """Per cell -log P_indep(valid local degree configuration): degree 1 at S/G, {0,2} elsewhere,
    edge probabilities p (B,n,n,4), closed slots forced off. Returns (B,n,n)."""
    p = torch.where(open_, p.clamp(1e-6, 1 - 1e-6), torch.zeros_like(p))   # closed slots exactly off
    q = 1 - p; P0 = q.prod(-1); r = p / q
    e1 = r.sum(-1)
    e2 = sum(r[..., i] * r[..., j] for i in range(4) for j in range(i + 1, 4))
    valid = torch.where(endm, P0 * e1, P0 * (1 + e2))
    return -torch.log(valid.clamp_min(1e-30))


def _deg_cfgs():
    V = np.array(list(itertools.product([0, 1], repeat=4)), np.float32)
    return torch.tensor(V)


DEGV = _deg_cfgs()


def true_arbitrage(p, open_, endm, iters=300):
    """Per cell min over mu in conv(valid local configurations) of sum_e binaryKL(mu_e || p_e),
    by Frank-Wolfe over the vertex set (closed slots must be 0). Returns (B,n,n)."""
    B, n = p.shape[:2]
    P = torch.where(open_, p, torch.zeros_like(p)).clamp(1e-6, 1 - 1e-6).reshape(-1, 4)
    o = open_.reshape(-1, 4); e = endm.reshape(-1)
    V = DEGV.to(p.device)                                     # (16,4)
    deg = V.sum(1)
    ok_deg = torch.where(e[:, None], (deg == 1)[None, :], ((deg == 0) | (deg == 2))[None, :])
    ok_open = ((V[None, :, :] <= o[:, None, :].float())).all(-1)
    ok = ok_deg & ok_open                                     # (m,16)
    has = ok.any(1)
    # start: uniform over allowed vertices
    w = ok.float() / ok.float().sum(1, keepdim=True).clamp_min(1)
    lp, l1p = torch.log(P), torch.log1p(-P)

    def kl(mu):
        mu = mu.clamp(1e-9, 1 - 1e-9)
        return (mu * (torch.log(mu) - lp) + (1 - mu) * (torch.log1p(-mu) - l1p)) * o.float()

    for it in range(iters):
        mu = w @ V
        g = (torch.log(mu.clamp(1e-9, 1 - 1e-9)) - torch.log1p(-mu.clamp(1e-9, 1 - 1e-9)) - lp + l1p) * o.float()
        s = (g @ V.t()).masked_fill(~ok, 1e9)                  # linear minimiser over allowed vertices
        j = s.argmin(1)
        tgt = torch.zeros_like(w); tgt[torch.arange(len(j)), j] = 1.0
        gam = 2.0 / (it + 3.0)
        w = (1 - gam) * w + gam * tgt
    val = kl(w @ V).sum(1)
    val = torch.where(has, val, torch.full_like(val, float("inf")))
    return val.reshape(B, n, n)


# ------------------------------------------------------------------ shared decode and metrics
def endpoints(S, G, B, n, device):
    endm = torch.zeros(B, n, n, dtype=torch.bool, device=device)
    ar = torch.arange(B, device=device)
    endm[ar, S[:, 0].long(), S[:, 1].long()] = True
    endm[ar, G[:, 0].long(), G[:, 1].long()] = True
    return endm


def decode(logits, open_):
    """THE decode, shared by every arm: an edge is on iff its logit > 0 and it is open.
    logits (B,n,n,4) symmetric. Returns on (B,n,n,4) bool, symmetric."""
    on = (logits > 0) & open_
    return on & nb(on.float()).bool()   # symmetric by construction; the AND guards asymmetric inputs


def syndrome_ok(on, endm):
    """Every cell degree-valid (1 at S/G, 0 or 2 elsewhere). (B,) bool, and per-cell ok (B,n,n)."""
    deg = on.sum(-1)
    okc = torch.where(endm, deg == 1, (deg == 0) | (deg == 2))
    return okc.all(-1).all(-1), okc


def single_component(on):
    """(B,) bool: the on-cells (cells with >= 1 on edge) form one connected component of the on-edge graph.
    Union-find with min-root hooking and full pointer jumping (O(log) rounds)."""
    B, n = on.shape[:2]; N = n * n; dev = on.device
    idx = torch.arange(B * N, device=dev).view(B, n, n)
    uE = idx[:, :, :-1]; vE = torch.where(on[:, :, :-1, 0], idx[:, :, 1:], uE)
    uS = idx[:, :-1, :]; vS = torch.where(on[:, :-1, :, 1], idx[:, 1:, :], uS)
    u = torch.cat([uE.reshape(-1), uS.reshape(-1)]); v = torch.cat([vE.reshape(-1), vS.reshape(-1)])
    keep = u != v
    u = u[keep]; v = v[keep]
    par = torch.arange(B * N, device=dev)
    for _ in range(64):
        pu = par[u]; pv = par[v]
        new = par.scatter_reduce(0, torch.maximum(pu, pv), torch.minimum(pu, pv), reduce="amin")
        while True:
            nn_ = new[new]
            if torch.equal(nn_, new):
                break
            new = nn_
        if torch.equal(new, par):
            break
        par = new
    root = par.view(B, n, n)
    oncell = on.any(-1)
    big = B * N
    rmin = torch.where(oncell, root, torch.full_like(root, big)).amin((1, 2))
    rmax = torch.where(oncell, root, torch.full_like(root, -1)).amax((1, 2))
    return rmin == rmax


def valid_path(on, endm, need=None):
    """Exactly one simple S-G path: degree-valid (1 at S/G, 0 or 2 elsewhere) and the on-cells form one
    connected component (so no stray cycle). #edges == #on-cells - 1 is a cheap necessary prefilter; the
    connectivity check runs only on mazes passing it (and in `need`, if given; others return False)."""
    synd, _ = syndrome_ok(on, endm)
    deg = on.sum(-1)
    ne = on.sum((1, 2, 3)) // 2
    nc = (deg > 0).sum((1, 2))
    cand = synd & (ne == nc - 1)
    if need is not None:
        cand = cand & need
    out = torch.zeros_like(cand)
    if bool(cand.any()):
        out[cand] = single_component(on[cand])
    return out


def exact(on, path):
    return (on == path).all(-1).all(-1).all(-1)


def edge_f1(on, path):
    tp = (on & path).sum((1, 2, 3)).float(); fp = (on & ~path).sum((1, 2, 3)).float(); fn = (~on & path).sum((1, 2, 3)).float()
    return 2 * tp / (2 * tp + fp + fn).clamp_min(1)


def path_cost(on, ecost):
    """Sum of edge costs of the on-set (each edge once). ecost (B,n,n,4) finite on open slots."""
    return (torch.where(on, ecost, torch.zeros_like(ecost))).sum((1, 2, 3)) / 2


def edge_cost_tensor(cost, open_):
    """cell costs (B,n,n) -> slot edge costs (B,n,n,4) (mean of the two cells), 0 where closed."""
    c = cost
    E = torch.zeros_like(c); S_ = torch.zeros_like(c)
    E[:, :, :-1] = (c[:, :, :-1] + c[:, :, 1:]) / 2
    S_[:, :-1, :] = (c[:, :-1, :] + c[:, 1:, :]) / 2
    ec = es_to_slots(E, S_)
    return torch.where(open_, ec, torch.zeros_like(ec))
