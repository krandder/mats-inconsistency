"""Exact inference for a binary log-linear model by junction tree (torch, differentiable).

log p(x) = sum_i h_i x_i + sum_(i,j) J_ij x_i x_j + sum_(i,j,k) K_ijk x_i x_j x_k - log Z,  x in {0,1}^n.

The tree decomposition comes from networkx's min-fill heuristic on the graph whose cliques
contain every pair and triple factor, so every factor and every queried pair/triple sits in
one bag.  Messages are log-space tensors over separators; bag beliefs are calibrated by one
upward and one downward pass per connected component.  All operations are torch ops, so any
query probability can be differentiated with respect to h, J and K.
"""

from __future__ import annotations

import itertools

import networkx as nx
import torch
from networkx.algorithms.approximation import treewidth_min_fill_in

torch.set_default_dtype(torch.float64)


class JunctionTree:
    def __init__(self, n: int, pairs: list[tuple[int, int]], triples: list[tuple[int, int, int]]):
        self.n = n
        self.pairs = [tuple(sorted(p)) for p in pairs]
        self.triples = [tuple(sorted(t)) for t in triples]
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(self.pairs)
        for t in self.triples:
            g.add_edges_from(itertools.combinations(t, 2))
        self.graph = g
        self.bags: list[tuple[int, ...]] = []
        self.children: dict[int, list[int]] = {}
        self.parent: dict[int, int | None] = {}
        self.roots: list[int] = []
        self.treewidth = 0
        for comp in nx.connected_components(g):
            sub = g.subgraph(comp)
            if len(comp) == 1:
                tw, dec = 0, nx.Graph(); dec.add_node(frozenset(comp))
            else:
                tw, dec = treewidth_min_fill_in(sub)
            self.treewidth = max(self.treewidth, tw)
            ids = {}
            for b in dec.nodes:
                ids[b] = len(self.bags)
                self.bags.append(tuple(sorted(b)))
            root = next(iter(dec.nodes))
            self.roots.append(ids[root])
            self.parent[ids[root]] = None
            for a, b in nx.bfs_edges(dec, root):
                self.parent[ids[b]] = ids[a]
                self.children.setdefault(ids[a], []).append(ids[b])
        self.order = []  # bags in BFS order (parents before children)
        for r in self.roots:
            q = [r]
            while q:
                b = q.pop(0); self.order.append(b); q += self.children.get(b, [])
        # home bag for every variable, pair and triple
        self.home: dict[tuple, int] = {}
        for k, b in enumerate(self.bags):
            s = b
            for r in (1, 2, 3):
                for c in itertools.combinations(s, r):
                    self.home.setdefault(c, k)
        for f in self.pairs + self.triples + [(i,) for i in range(n)]:
            assert f in self.home, f
        # per bag: the factors living there and a 0/1 matrix (bag assignments x factors) so that
        # the bag's log-potential is one matrix-vector product over theta = cat(h, J, K)
        factors = [(i,) for i in range(n)] + self.pairs + self.triples
        per_bag: dict[int, list[int]] = {}
        for fi, f in enumerate(factors):
            per_bag.setdefault(self.home[f], []).append(fi)
        self.bag_factors, self.bag_ind = {}, {}
        for k, fis in per_bag.items():
            bag = self.bags[k]
            grid = torch.tensor(list(itertools.product([0, 1], repeat=len(bag))), dtype=torch.float64)  # (2^b, b)
            cols = []
            for fi in fis:
                c = torch.ones(len(grid))
                for v in factors[fi]:
                    c = c * grid[:, bag.index(v)]
                cols.append(c)
            self.bag_factors[k] = torch.tensor(fis)
            self.bag_ind[k] = torch.stack(cols, 1)

    @staticmethod
    def _marg(bel, keep):
        """logsumexp over every dim not in keep (sorted dim indices), as one 2-D reduction (much faster
        than torch's multi-dim logsumexp over many size-2 dims)."""
        rest = [d for d in range(bel.dim()) if d not in keep]
        if not rest:
            return bel
        x = bel.permute(*keep, *rest).reshape(2 ** len(keep), -1)
        return torch.logsumexp(x, 1).reshape([2] * len(keep))

    def _expand(self, vals, vars_, bag):
        """Reshape a tensor over sorted vars_ (subset of bag) to broadcast over bag's dims."""
        shape = [2 if v in vars_ else 1 for v in bag]
        return vals.reshape(shape)

    def _cube(self, bag, vars_, coef):
        """Log-potential tensor over bag for coef * prod x_v, v in vars_."""
        ind = torch.zeros([2 if v in vars_ else 1 for v in bag])
        idx = tuple(1 if v in vars_ else 0 for v in bag)
        ind[idx] = 1.0
        return ind * coef

    def calibrate(self, h, J, K):
        """h: (n,), J: (len(pairs),), K: (len(triples),).  Returns normalised log bag beliefs and log Z."""
        theta = torch.cat([h, J, K])
        pot = [(self.bag_ind[k] @ theta[self.bag_factors[k]]).reshape([2] * len(b)) if k in self.bag_ind
               else torch.zeros([2] * len(b)) for k, b in enumerate(self.bags)]
        up: dict[int, torch.Tensor] = {}  # message child -> parent, over the separator
        for b in reversed(self.order):
            bel = pot[b]
            for c in self.children.get(b, []):
                bel = bel + up[c]
            p = self.parent[b]
            if p is not None:
                sep = [v for v in self.bags[b] if v in self.bags[p]]
                m = self._marg(bel, [d for d, v in enumerate(self.bags[b]) if v in sep])
                up[b] = self._expand(m, sep, self.bags[p])
        logZ = torch.zeros(())
        for r in self.roots:
            bel = pot[r]
            for c in self.children.get(r, []):
                bel = bel + up[c]
            logZ = logZ + torch.logsumexp(bel.reshape(-1), 0)
        down: dict[int, torch.Tensor] = {}
        beliefs = [None] * len(self.bags)
        for b in self.order:
            bel = pot[b] + (down[b] if b in down else 0)
            full = bel
            for c in self.children.get(b, []):
                full = full + up[c]
            beliefs[b] = full - torch.logsumexp(full.reshape(-1), 0)
            for c in self.children.get(b, []):
                excl = full - up[c]
                sep = [v for v in self.bags[c] if v in self.bags[b]]
                m = self._marg(excl, [d for d, v in enumerate(self.bags[b]) if v in sep])
                down[c] = self._expand(m, sep, self.bags[c])
        self.beliefs = beliefs
        return beliefs, logZ

    def logp(self, lits):
        """log P(conjunction of literals [(i, x)]) from calibrated beliefs; the variables must share a bag."""
        vs = tuple(sorted(i for i, _ in lits))
        k = self.home[vs]
        bag = self.bags[k]
        val = dict(lits)
        bel = self.beliefs[k]
        idx = tuple(val[v] if v in val else slice(None) for v in bag)
        sub = bel[idx]
        return torch.logsumexp(sub.reshape(-1), 0)


def brute_force(n, pairs, triples, h, J, K, lits):
    xs = torch.tensor(list(itertools.product([0, 1], repeat=n)), dtype=torch.float64)
    e = xs @ h
    for k, (i, j) in enumerate(pairs):
        e = e + J[k] * xs[:, i] * xs[:, j]
    for k, (i, j, l) in enumerate(triples):
        e = e + K[k] * xs[:, i] * xs[:, j] * xs[:, l]
    lp = e - torch.logsumexp(e, 0)
    m = torch.ones(len(xs), dtype=torch.bool)
    for i, x in lits:
        m &= xs[:, i] == x
    return torch.logsumexp(lp[m], 0)


if __name__ == "__main__":  # self-test against brute force
    import random
    rng = random.Random(0)
    n = 12
    pairs = sorted({tuple(sorted(rng.sample(range(n), 2))) for _ in range(20)})
    adj = {i: set() for i in range(n)}
    for i, j in pairs:
        adj[i].add(j); adj[j].add(i)
    tri = sorted({tuple(sorted((i, j, k))) for i, j in pairs for k in adj[i] & adj[j]})
    torch.manual_seed(0)
    h, J, K = torch.randn(n), torch.randn(len(pairs)), torch.randn(len(tri))
    jt = JunctionTree(n, pairs, tri)
    jt.calibrate(h, J, K)
    worst = 0.0
    for q in [[(i, 1)] for i in range(n)] + [[(i, 1), (j, 0)] for i, j in pairs] + [[(a, 1), (b, 1), (c, 1)] for a, b, c in tri]:
        worst = max(worst, abs(float(jt.logp(q).exp() - brute_force(n, pairs, tri, h, J, K, q).exp())))
    print(f"n={n} pairs={len(pairs)} triangles={len(tri)} treewidth={jt.treewidth} bags={len(jt.bags)} max abs err={worst:.2e}")
