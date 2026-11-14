"""ppol.py -- parity world: rollout policies (search side). No exact inference.

A policy maps a batch of partial states (with the books' local numbers) to
  q [k, n]  its probability that each open assumption is 1 (the value part of the next step), and
  j [k]     the assumption to assign next (the order part).
Hand-made policies read the books only. NetPolicy is the trained rollout policy: a small message-passing network on the
score's own graph (which check has which parents is part of the hand-built score, wf7). It reads, per assumption:
assigned 0 / 1 / open, the score's prior number, the books' local log-odds; per check: the observed value, how many
parents are open, the observed value against the parity of the assigned parents. Its order rule is "most confident
first" on its own value head. Deliberately bad policies wrap it.
"""
import numpy as np
import torch
import torch.nn as nn

from pw import UN

D3 = np.array([[(code // 3 ** t) % 3 for t in range(3)] for code in range(27)])
RES = ((D3 == 1).sum(1) & 1)                                   # parity of the assigned parents, per pattern


def pick_conf(q, op, rng):
    c = np.abs(q - 0.5) + 1e-6 * rng.random(q.shape)
    return np.where(op, c, -1.0).argmax(1)


def pick_random(op, rng):
    return np.where(op, rng.random(op.shape), -1.0).argmax(1)


def pick_complete(inst, e, rng):
    """Hand-made order: the open assumption that completes the most checks; ties: most checks left with one open
    parent; then the books' confidence."""
    B, n = e['op'].shape
    P = inst.checks
    comp, near = np.zeros((B, n)), np.zeros((B, n))
    for t in range(3):
        idx = np.broadcast_to(P[None, :, t], e['nun'].shape)
        np.add.at(comp, (np.arange(B)[:, None], idx), e['nun'] == 1)
        np.add.at(near, (np.arange(B)[:, None], idx), e['nun'] == 2)
    key = comp * 100 + near * 10 + np.abs(e['p1'] - 0.5) + 1e-6 * rng.random((B, n))
    return np.where(e['op'], key, -1.0).argmax(1)


def pick_near(inst, e, rng):
    """Hand-made order: the open assumption in the most checks that have exactly two open parents (assigning it turns
    those checks from 'bounded' into 'summed out exactly'); ties: books' confidence."""
    B, n = e['op'].shape
    P = inst.checks
    near = np.zeros((B, n))
    for t in range(3):
        idx = np.broadcast_to(P[None, :, t], e['nun'].shape)
        np.add.at(near, (np.arange(B)[:, None], idx), e['nun'] == 2)
    key = near * 10 + np.abs(e['p1'] - 0.5) + 1e-6 * rng.random((B, n))
    return np.where(e['op'], key, -1.0).argmax(1)


def pick_greedy1(inst, e, rng):
    """Hand-made order by one-step lookahead in the books: the open assumption whose two children have the smallest
    total upper bound. Pure table arithmetic of the specification (2 x open states evaluated per opening)."""
    A = e['A']
    B, n = A.shape
    best, bj = np.full(B, np.inf), np.zeros(B, dtype=np.int64)
    for j in range(n):
        rows = np.flatnonzero(e['op'][:, j])
        if len(rows) == 0:
            continue
        A0, A1 = A[rows].copy(), A[rows].copy()
        A0[:, j], A1[:, j] = 0, 1
        ch = inst.ev(np.concatenate([A0, A1]))
        tot = np.logaddexp(ch['logU'][:len(rows)], ch['logU'][len(rows):]) + 1e-9 * rng.random(len(rows))
        upd = tot < best[rows]
        best[rows[upd]], bj[rows[upd]] = tot[upd], j
    return bj


ORDERS = {'complete': pick_complete, 'near': pick_near, 'greedy1': pick_greedy1}


class Hand:
    """order: 'conf' (books' confidence) | 'complete' | 'random';  value: 'books' (local numbers) | 'uniform'."""
    calls_per_state = 0

    def __init__(self, order='complete', value='books'):
        self.order, self.value, self.name = order, value, 'hand-%s-%s' % (order, value)

    def __call__(self, inst, e, rng):
        q = e['p1'] if self.value == 'books' else np.where(e['op'], 0.5, e['p1'])
        if self.order == 'conf':
            j = pick_conf(e['p1'], e['op'], rng)
        elif self.order in ORDERS:
            j = ORDERS[self.order](inst, e, rng)
        else:
            j = pick_random(e['op'], rng)
        return q, j


class GNN(nn.Module):
    def __init__(self, d=32, T=3):
        super().__init__()
        self.T = T
        self.v0 = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        self.c0 = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.cu = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) for _ in range(T)])
        self.vu = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) for _ in range(T)])
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, fv, fc, inc):                         # fv [B,n,5], fc [B,m,6], inc [m,n] or [B,m,n]
        hv, hc = self.v0(fv), self.c0(fc)
        incT = inc.transpose(-1, -2)
        for t in range(self.T):
            hc = hc + self.cu[t](torch.cat([hc, inc @ hv], -1))
            hv = hv + self.vu[t](torch.cat([hv, incT @ hc], -1))
        return self.out(hv).squeeze(-1)


def features(inst, A, tl, codes, nun, flip=False):
    """Network inputs from the books (float32). flip=True shows the network the OPPOSITE observations (wrong goal)."""
    x = inst.x if not flip else 1 - inst.x
    op = A == UN
    fv = np.stack([A == 0, A == 1, op, np.broadcast_to(inst.prior_logit[None] / 3.0, A.shape),
                   np.clip((tl if not flip else 2 * inst.prior_logit[None] - tl) / 3.0, -3, 3) * op], -1).astype(np.float32)
    res = (x[None, :] ^ RES[codes])
    fc = np.stack([np.broadcast_to(x[None], nun.shape), nun == 0, nun == 1, nun == 2, nun == 3, res], -1).astype(np.float32)
    return fv, fc


def incidence(inst):
    inc = np.zeros((inst.m, inst.n), dtype=np.float32)
    for t in range(3):
        inc[np.arange(inst.m), inst.checks[:, t]] = 1.0
    return torch.from_numpy(inc)


class NetPolicy:
    """mode: 'ok' | 'flip' (trained toward a wrong goal: it is shown the opposite observations) |
    'avoid' (never proposes h_j = 1 for the assumptions in `avoid`; with no floor it has no support there)."""
    calls_per_state = 1

    def __init__(self, net, mode='ok', avoid=(), order='conf', name='net'):
        self.net, self.mode, self.avoid, self.order, self.name = net, mode, list(avoid), order, name
        self._inc = {}

    @torch.no_grad()
    def q(self, inst, e):
        key = id(inst)
        if key not in self._inc:
            self._inc = {key: incidence(inst)}
        fv, fc = features(inst, e['A'], e['tl'], e['codes'], e['nun'], flip=self.mode == 'flip')
        lg = self.net(torch.from_numpy(fv), torch.from_numpy(fc), self._inc[key])
        q = torch.sigmoid(lg).double().numpy()
        q = np.where(e['op'], q, e['p1'])
        if self.mode == 'avoid':
            q[:, self.avoid] = np.where(e['op'][:, self.avoid], 0.0, q[:, self.avoid])
        return q

    def __call__(self, inst, e, rng):
        q = self.q(inst, e)
        if self.order == 'conf':
            j = pick_conf(q, e['op'], rng)
        elif self.order in ORDERS:
            j = ORDERS[self.order](inst, e, rng)
        else:
            j = pick_random(e['op'], rng)
        return q, j


class Table:
    """Pilot / evaluation-only stand-in: q given by a function (used by pilots with the exact conditional)."""
    calls_per_state = 1

    def __init__(self, fn, order='conf', name='table'):
        self.fn, self.order, self.name = fn, order, name

    def __call__(self, inst, e, rng):
        q = self.fn(inst, e)
        j = pick_conf(q, e['op'], rng) if self.order == 'conf' else ORDERS[self.order](inst, e, rng)
        return q, j
