"""spol.py -- 6x6 Sudoku: rollout policies (search side) and the distillation code. No exact inference, no solver.

A policy maps boards [k, 36] (-1 = empty) and the books' candidates [k, 36, 6] to value probabilities q [k, 36, 6] and
the cell to fill next j [k]. Hand-made: 'mrv' = fewest candidates first (the classic heuristic), 'random'.
NetPolicy = the trained rollout policy: a small network that is told which cells share a row, column or box (the same
relational prior as M) and reads each cell's value or its candidates. Its output is a distribution over all six values
per cell and is NOT masked by the candidates, so an untrained network proposes at random; order rule "most confident
cell first" on that distribution. The books drop proposals that collide (score 0, known at once).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K, N = 6, 36


def groups():
    r, c = np.arange(N) // K, np.arange(N) % K
    b = (r // 2) * 2 + c // 3
    out = []
    for g in (r, c, b):
        m = (g[:, None] == g[None, :]).astype(np.float32)
        out.append(m / m.sum(1, keepdims=True))
    return torch.from_numpy(np.stack(out))                  # [3, 36, 36]


class SNet(nn.Module):
    def __init__(self, d=48, L=3):
        super().__init__()
        self.inp = nn.Linear(2 * K + 1, d)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(4 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) for _ in range(L)])
        self.out = nn.Linear(d, K)
        self.register_buffer('G', groups())

    def forward(self, f):                                    # f [B, 36, 13]
        h = self.inp(f)
        for blk in self.blocks:
            h = h + blk(torch.cat([h, self.G[0] @ h, self.G[1] @ h, self.G[2] @ h], -1))
        return self.out(h)


def feats(b, c):
    emp = b < 0
    oh = (b[:, :, None] == np.arange(K)[None, None, :])
    return np.concatenate([oh, emp[:, :, None], c & emp[:, :, None]], -1).astype(np.float32)


class Hand:
    calls_per_state = 0

    def __init__(self, order='mrv'):
        self.order, self.name = order, 'hand-' + order

    def __call__(self, b, c, rng):
        emp = b < 0
        q = c / np.maximum(c.sum(-1, keepdims=True), 1)
        if self.order == 'mrv':
            key = np.where(emp, c.sum(-1) + 1e-3 * rng.random(b.shape), 99.0)
            j = key.argmin(1)
        else:
            j = np.where(emp, rng.random(b.shape), -1.0).argmax(1)
        return q, j


class NetPolicy:
    """mode: 'ok' | 'perm' (wrong goal: its value preferences are shifted by one digit) |
    'avoid' (never proposes digit 0 in the first two rows; with no floor it has no support there)."""
    calls_per_state = 1

    def __init__(self, net, mode='ok', order='conf', name='net'):
        self.net, self.mode, self.order, self.name = net, mode, order, name

    @torch.no_grad()
    def __call__(self, b, c, rng):
        emp = b < 0
        lg = self.net(torch.from_numpy(feats(b, c))).double().numpy()
        if self.mode == 'perm':
            lg = np.roll(lg, 1, axis=-1)
        p = np.exp(lg - lg.max(-1, keepdims=True))
        p = p / p.sum(-1, keepdims=True)              # the network's own distribution over all six values (unmasked)
        if self.mode == 'avoid':
            p[:, :12, 0] = 0.0
        q = p                                         # the books drop non-candidates and renormalise (sk.grow, sk.run_is)
        if self.order == 'conf':
            key = np.where(emp, p.max(-1) + 1e-6 * rng.random(b.shape), -1.0)
            j = key.argmax(1)
        elif self.order == 'confmask':
            # POST HOC order rule (not registered): confidence after the books have dropped the values that collide
            pm = p * c
            pm = pm / np.maximum(pm.sum(-1, keepdims=True), 1e-300)
            j = np.where(emp, pm.max(-1) + 1e-6 * rng.random(b.shape), -1.0).argmax(1)
        elif self.order == 'mrv':
            j = np.where(emp, c.sum(-1) + 1e-3 * rng.random(b.shape), 99.0).argmin(1)
        else:
            j = np.where(emp, rng.random(b.shape), -1.0).argmax(1)
        return q, j


def collect(T, S, rng):
    """Distillation targets from one tree: for every opened state under which the search found at least one finished
    demonstration, the share of the found ones with value v in cell j, for every empty cell j (the amplified next-step
    value distribution at that state). Up to S states per puzzle."""
    n = T.N
    st = T.status[:n]
    cnt = np.zeros(n)
    val = np.zeros((n, N, K), dtype=np.float32)
    done = np.flatnonzero(st == 2)
    if len(done) == 0:
        return None
    for d in done:
        oh = (T.board[d][:, None] == np.arange(K)[None, :]).astype(np.float32)
        i = T.parent[d]
        while i >= 0:
            cnt[i] += 1
            val[i] += oh
            i = T.parent[i]
    cand = np.flatnonzero((st == 1) & (cnt > 0))
    pick = cand if len(cand) <= S else rng.choice(cand, size=S, replace=False)
    b, c = T.board[pick], T.cand[pick]
    return feats(b, c), val[pick] / cnt[pick, None, None], (b < 0).astype(np.float32), c


def train(net, buf, steps, bs, lr, seed, log=None):
    f, tg, em, cd = [torch.from_numpy(np.concatenate([x[i] for x in buf])) for i in range(4)]
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    net.train()
    run = None
    for s in range(steps):
        for pg in opt.param_groups:
            pg['lr'] = lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * s / steps)))
        idx = torch.randint(0, len(f), (bs,), generator=g)
        lg = net(f[idx])                              # unmasked: the network has to learn what the candidates mean
        l = (-(tg[idx] * F.log_softmax(lg, -1)).sum(-1) * em[idx]).sum() / em[idx].sum()
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run = l.item() if run is None else 0.98 * run + 0.02 * l.item()
    net.eval()
    if log:
        log('    trained %d steps on %d states: running loss %.4f' % (steps, len(f), run))
    return run
