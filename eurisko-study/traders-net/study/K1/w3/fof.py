"""K1 worker3: funds of funds on worker1's Kelly economy (K1/econ.py, imported, not edited).

A fraction phi of the leaves (slots) is grouped into funds of m members at t=0; the rest are grouped into
*virtual* buy-and-hold groups of the same size and topology (pure bookkeeping, no wealth moves), so both sit in
the same market. Fund rules (applied in econ.post_settle, after settlement + compute sink, before die/split):
  bh  : no transfers (fixed fractions bought at t=0; T1 lemma 6.1: identical to the flat economy)
  crp : rebalance fund value to equal fractions over live members every step (flat CRP, lemma 6.2)
  eg  : fractions follow online exponentiated gradient on the members' wealth relatives (Helmbold et al. 1998,
        unverified recollection), eta=0.05; the nonlinear rule of lemma 6.3
Topology: within = members trade the same question (niche); cross = members drawn at random.
Membership follows the population: a split child joins its parent's fund (EG fraction halved between them), the
dead leave it. Group value V_g = sum of its members' wealth (children included), V_g(0) = m.
Transfers are zero-sum within a fund, so the ledger stays exact (cons_resid_ledger).
"""
import os
import sys

import numpy as np

K1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, K1)
import econ  # noqa: E402


def make_groups(e, rng, m, topo):
    ids = np.flatnonzero(e.alive)
    if topo == 'within' and e.niches:
        order = ids[np.lexsort((rng.random(len(ids)), e.niche[ids]))]
        groups = []
        for q in np.unique(e.niche[order]):
            mem = order[e.niche[order] == q]
            groups += [mem[i:i + m] for i in range(0, len(mem), m)]
    else:
        order = rng.permutation(ids)
        groups = [order[i:i + m] for i in range(0, len(order), m)]
    return groups


class Funds:
    def __init__(self, e, rule, topo, phi, m, seed, eta=0.05, k_top=4):
        rng = np.random.default_rng([seed, 44])            # same grouping for main and null (same population)
        groups = make_groups(e, rng, m, 'within' if rule.startswith('fof') else topo)
        self.sub = None
        if rule.startswith('fof'):                          # funds of funds: top funds hold k_top subfunds
            perm = rng.permutation(len(groups))
            sub_of = np.full(e.N, -1)
            for j, mem in enumerate(groups):
                sub_of[mem] = j
            top = [np.concatenate([groups[j] for j in perm[i:i + k_top]]) for i in range(0, len(perm), k_top)]
            self.sub, self.S = sub_of, len(groups)
            groups = top
        G = len(groups)
        active = np.zeros(G, bool)
        active[rng.permutation(G)[:int(round(phi * G))]] = True
        self.e, self.rule, self.eta, self.G, self.active = e, rule, eta, G, active
        self.gid = np.full(e.N, -1)
        for g, mem in enumerate(groups):
            self.gid[mem] = g
        self.V0 = np.bincount(self.gid[self.gid >= 0], e.w[self.gid >= 0], G)
        self.a = np.where(self.gid >= 0, 1.0, 0.0)            # EG weights (unnormalised within group)
        self.pre = e.w.copy()
        self.moved = 0.0
        orig = e.step

        def step(*args):
            self.pre = e.w.copy()
            orig(*args)
            self._membership()
        e.step = step                                       # instance attribute shadows the method
        e.post_settle = self._settle

    def _members(self, act_only=True):
        e = self.e
        sel = e.alive & (self.gid >= 0)
        if act_only:
            sel &= self.active[np.maximum(self.gid, 0)]
        return np.flatnonzero(sel)

    def _settle(self, e):
        if self.rule == 'bh':
            return
        I = self._members()
        if len(I) == 0:
            return
        g = self.gid[I]
        V = np.bincount(g, e.w[I], self.G)
        if self.sub is not None:                            # top-level CRP over buy-and-hold subfunds
            sb = self.sub[I]
            Vs = np.bincount(sb, e.w[I], self.S)
            live = Vs > 0
            top_of_sub = np.full(self.S, -1); top_of_sub[sb] = g
            ns = np.bincount(top_of_sub[live], None, self.G)
            target = np.where(live, V[np.maximum(top_of_sub, 0)] / np.maximum(ns[np.maximum(top_of_sub, 0)], 1), 0.0)
            fac = np.where(live, target / np.maximum(Vs, 1e-300), 1.0)
            new = e.w[I] * fac[sb]
            self.moved += float(np.abs(new - e.w[I]).sum()) / 2
            e.w[I] = new
            return
        if self.rule == 'eg':
            x = np.where(self.pre[I] > 0, e.w[I] / np.maximum(self.pre[I], 1e-300), 1.0)
            a = self.a[I]
            an = a / np.bincount(g, a, self.G)[g]
            fx = np.bincount(g, an * x, self.G)[g]            # fund wealth relative
            a = an * np.exp(self.eta * x / np.maximum(fx, 1e-300))
            a = a / np.bincount(g, a, self.G)[g]
            self.a[I] = a
            frac = a
        else:
            frac = 1.0 / np.bincount(g, None, self.G)[g]
        new = frac * V[g]
        self.moved += float(np.abs(new - e.w[I]).sum()) / 2
        e.w[I] = new

    def _membership(self):
        e = self.e
        d = e.last_deaths
        par, ch = e.last_splits
        if len(d):                                          # dead leave their group (slot may be reused below)
            self.gid[d] = -1
            self.a[d] = 0.0
            if self.sub is not None:
                self.sub[d] = -1
        if len(par):
            self.gid[ch] = self.gid[par]
            if self.sub is not None:
                self.sub[ch] = self.sub[par]
            half = self.a[par] / 2
            self.a[par] = half
            self.a[ch] = half

    def metrics(self, tag=''):
        e = self.e
        sel = e.alive & (self.gid >= 0)
        I = np.flatnonzero(sel)
        V = np.bincount(self.gid[I], e.w[I], self.G)
        lv = np.log(np.maximum(V, 1e-12) / self.V0)
        act, bh = self.active, ~self.active
        row = {}
        for name, s in (('fund', act), ('bhgrp', bh)):
            if s.any():
                row[f'{tag}logV_{name}_mean'] = float(lv[s].mean())
                row[f'{tag}logV_{name}_median'] = float(np.median(lv[s]))
                row[f'{tag}alive_{name}_frac'] = float((V[s] >= e.die_at).mean())
                row[f'{tag}wealth_{name}'] = float(V[s].sum())
                row[f'{tag}n_{name}'] = int(np.isin(self.gid[I], np.flatnonzero(s)).sum())
        if act.any() and bh.any():
            row[f'{tag}premium_mean'] = row[f'{tag}logV_fund_mean'] - row[f'{tag}logV_bhgrp_mean']
            row[f'{tag}premium_median'] = row[f'{tag}logV_fund_median'] - row[f'{tag}logV_bhgrp_median']
            row[f'{tag}share_fund'] = float(V[act].sum() / max(V.sum(), 1e-300))
        row[f'{tag}moved_cum'] = self.moved
        return row


def niche_diversity(e):
    """Mean over occupied questions of 1/sum(share^2): leaf level and species level."""
    al = np.flatnonzero(e.alive & (e.w > 0))
    if len(al) == 0 or not e.niches:
        return {'eff_leaves_q': float('nan'), 'eff_species_q': float('nan')}
    q, w = e.niche[al], e.w[al]
    Wq = np.bincount(q, w, e.Q)
    s2 = np.bincount(q, w * w, e.Q)
    occ = Wq > 0
    sp = e.typ[al] * 5 + e.k[al]
    Wqs = np.bincount(q * 10 + sp, w, e.Q * 10).reshape(e.Q, 10)
    ss = ((Wqs / np.maximum(Wq, 1e-300)[:, None]) ** 2).sum(1)
    return {'eff_leaves_q': float((Wq[occ] ** 2 / s2[occ]).mean()), 'eff_species_q': float((1 / ss[occ]).mean()),
            'q_occupied': int(occ.sum())}


def run_arm(rule, topo, N, seed, T, phi=0.5, m=8, out_root=None, run_id=None, **kw):
    holder = {}

    def setup(main, nul):
        holder['m'] = Funds(main, rule, topo, phi, m, seed)
        if nul is not None:
            holder['n'] = Funds(nul, rule, topo, phi, m, seed)

    def extra(main, nul):
        row = holder['m'].metrics()
        row.update(niche_diversity(main))
        if nul is not None:
            row.update(holder['n'].metrics(tag='null_'))
            row.update({f'{k}_null': v for k, v in niche_diversity(nul).items()})
        return row

    return econ.run(N=N, Q=64, niches=True, seed=seed, T=T, out_root=out_root, run_id=run_id, setup=setup,
                    extra_metrics=extra, verbose=False, **kw)


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('--rule', required=True, choices=['bh', 'crp', 'eg', 'fof'])
    ap.add_argument('--topo', default='within', choices=['within', 'cross'])
    ap.add_argument('--N', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--T', type=int, default=10000)
    ap.add_argument('--phi', type=float, default=0.5)
    ap.add_argument('--out_root', default=os.path.join(K1, 'runs'))
    a = ap.parse_args()
    rid = f'w3-{a.rule}-{a.topo}-N{a.N}-s{a.seed}'
    s = run_arm(a.rule, a.topo, a.N, a.seed, a.T, phi=a.phi, out_root=a.out_root, run_id=rid)
    f = s['final']
    print(rid, json.dumps({k: f.get(k) for k in ('alive', 'brier_all', 'premium_mean', 'eff_species_q',
                                                  'top_clade_share', 'cons_resid_ledger', 'wall_s')}))
