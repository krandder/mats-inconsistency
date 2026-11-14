"""N3 (scale-free-theory, 2026-10-29).

(a) Misaligned hierarchies. If the input law is a latent tree along T1 and the computation is a tree along T2 (same
    leaves = inputs), exact contraction cost is exp(O(treewidth(T1 u T2))) (Markov-Shi). We measure treewidth of
    the union graph: lower bound by MMD+ (min-d contraction degeneracy), upper bound by min-fill elimination,
    for (i) T2 = copy of T1, (ii) T2 = T1 after k random subtree prune-and-regraft moves, (iii) T2 independent.
(b) GF(2) pairwise functions f(x) = x^T U x mod 2 (U = upper-triangular adjacency): brute-force count of Nerode
    classes of x_S across a balanced cut vs 2^(rank_GF2 of the cut block (+1)), for K_n, G(n,1/2), sparse random.
"""
import json, sys, time, random, itertools
import numpy as np
import networkx as nx
from networkx.algorithms.approximation import treewidth_min_fill_in


def random_tree(leaves, rng):
    """Random rooted binary tree by random pairwise merging (Kingman-like). Returns (children dict, root)."""
    clusters = list(leaves)
    ch = {}
    nid = 0
    while len(clusters) > 1:
        i, j = rng.sample(range(len(clusters)), 2)
        a, b = clusters[i], clusters[j]
        node = ('n', nid); nid += 1
        ch[node] = (a, b)
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [node]
    return ch, clusters[0]


def tree_edges(ch, tag):
    rn = lambda v: (tag, v) if v[0] == 'n' else v                 # internal nodes are per tree; leaves are shared
    E = []
    for p, (a, b) in ch.items():
        E.append((rn(p), rn(a))); E.append((rn(p), rn(b)))
    return E


def spr(ch, root, rng):
    """One random subtree prune-and-regraft on a rooted binary tree (dict of children). Returns new (ch, root)."""
    ch = {k: tuple(v) for k, v in ch.items()}
    parent = {}
    for p, (a, b) in ch.items():
        parent[a] = p; parent[b] = p
    nodes = list(parent.keys())                       # every non-root node
    for _ in range(100):
        s = rng.choice(nodes)
        p = parent[s]
        # forbid regrafting into own subtree
        sub = set(); stack = [s]
        while stack:
            u = stack.pop(); sub.add(u)
            if u in ch: stack.extend(ch[u])
        sib = ch[p][0] if ch[p][1] == s else ch[p][1]
        targets = [v for v in nodes if v not in sub and v != p and v != sib] + ([root] if root not in sub and root != p else [])
        if not targets:
            continue
        t = rng.choice(targets)
        # prune: remove p, connect sib to p's parent
        if p == root:
            root = sib; del parent[sib]
        else:
            g = parent[p]
            a, b = ch[g]; ch[g] = (sib, b) if a == p else (a, sib); parent[sib] = g
        del ch[p]
        # regraft: new node p above t
        if t == root:
            ch[p] = (t, s); parent[t] = p; parent[s] = p; root = p
        else:
            g = parent[t]
            a, b = ch[g]; ch[g] = (p, b) if a == t else (a, p)
            ch[p] = (t, s); parent[p] = g; parent[t] = p; parent[s] = p
        return ch, root
    return ch, root


def mmd_plus(G):
    """MMD+ (min-d) treewidth lower bound: repeatedly contract a min-degree vertex into its min-degree neighbour."""
    H = nx.Graph(G)
    lb = 0
    while H.number_of_nodes() > 1:
        v = min(H.nodes, key=H.degree)
        dv = H.degree(v)
        lb = max(lb, dv)
        if dv == 0:
            H.remove_node(v); continue
        u = min(H.neighbors(v), key=H.degree)
        H = nx.contracted_nodes(H, u, v, self_loops=False)
    return lb


def union_tw(ch1, ch2):
    G = nx.Graph()
    G.add_edges_from(tree_edges(ch1, 'A')); G.add_edges_from(tree_edges(ch2, 'B'))
    ub, _ = treewidth_min_fill_in(G)
    return mmd_plus(G), ub


def part_a(quick):
    rng = random.Random(7)
    out = {}
    Ns = [16, 32, 64, 128, 256] if not quick else [16, 32]
    for N in Ns:
        leaves = [('x', i) for i in range(N)]
        rec = {'copy': [], 'indep': []}
        ks = [1, 2, 4, 8, 16, 32]
        for k in ks: rec['spr%d' % k] = []
        reps = 5 if not quick else 2
        for rep in range(reps):
            ch1, r1 = random_tree(leaves, rng)
            rec['copy'].append(union_tw(ch1, ch1))
            ch2, r2 = random_tree(leaves, rng)
            rec['indep'].append(union_tw(ch1, ch2))
            chk, rk = ch1, r1
            done = 0
            for k in ks:
                while done < k:
                    chk, rk = spr(chk, rk, rng); done += 1
                rec['spr%d' % k].append(union_tw(chk, ch1))
        out[N] = {key: {'lb_mean': float(np.mean([a for a, b in v])), 'ub_mean': float(np.mean([b for a, b in v]))}
                  for key, v in rec.items()}
        print('N', N, {k: (round(v['lb_mean'], 1), round(v['ub_mean'], 1)) for k, v in out[N].items()}, flush=True)
    return out


def gf2_rank(M):
    M = M.copy() % 2
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = [i for i in range(r, rows) if M[i, c]]
        if not piv: continue
        M[[r, piv[0]]] = M[[piv[0], r]]
        for i in range(rows):
            if i != r and M[i, c]: M[i] ^= M[r]
        r += 1
        if r == rows: break
    return r


def part_b():
    n = 12; S = list(range(6)); Sc = list(range(6, 12))
    rng = np.random.default_rng(3)
    graphs = {'complete': np.triu(np.ones((n, n), dtype=np.int64), 1),
              'gnp_half': np.triu((rng.random((n, n)) < 0.5).astype(np.int64), 1),
              'sparse_3reg_like': None}
    Gs = nx.random_regular_graph(3, n, seed=5)
    U = np.zeros((n, n), dtype=np.int64)
    for a, b in Gs.edges(): U[min(a, b), max(a, b)] = 1
    graphs['sparse_3reg_like'] = U
    X = np.array(list(itertools.product([0, 1], repeat=6)), dtype=np.int64)
    out = {}
    for name, U in graphs.items():
        classes = set()
        for xs in X:
            resid = []
            for z in X:
                x = np.concatenate([xs, z])
                resid.append(int(x @ U @ x) % 2)
            classes.add(tuple(resid))
        Bsym = (U + U.T)[np.ix_(S, Sc)]
        rk = gf2_rank(Bsym)
        tw = treewidth_min_fill_in(nx.from_numpy_array(U + U.T))[0]
        out[name] = {'nerode_classes': len(classes), 'cut_rank_gf2': rk, 'bound_2^(rank+1)': 2 ** (rk + 1),
                     'treewidth_ub_min_fill': tw}
        print(name, out[name], flush=True)
    return out


if __name__ == '__main__':
    quick = len(sys.argv) > 1
    t0 = time.time()
    res = {'b_gf2': part_b()}
    res['a_tanglegram'] = part_a(quick)
    res['seconds'] = time.time() - t0
    json.dump(res, open('/home/agent/workspaces/mats/work/scale-free-theory/n3_tanglegram_gf2%s.json'
                        % ('_quick' if quick else ''), 'w'), indent=1, default=str)
