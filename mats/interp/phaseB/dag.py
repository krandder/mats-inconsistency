"""Lateral arm (PREREG-B amendment 6): sparse surrogate DAGs over tokens / sources for the noisy tiny PFN.

A DAG variant keeps every map real except the attention sites, where each query token's keys are restricted to a sparse
parent set (softmax renormalised over the allowed keys). KL over the DAG = sum over scored nodes of E||f - g||^2/(2 sigma^2)
(scored = ancestors of the output: all tokens before the last attention site, only the query token at and after it).
Deviation prediction (R^2 vs the network's own noise-marginalised deviation) and scenario profiles as in devpred.py.
Treewidth of the moralised token-level DAG (networkx min-degree heuristic) and the largest clique's state dimension.
usage: python dag.py RUN_DIR -> RUN_DIR/dag.json
"""
import json, math, os, sys, time
import numpy as np
import torch
import networkx as nx
from networkx.algorithms.approximation import treewidth_min_degree
import nnoise as N
import world as W
from mirun import trace
from sur_pfn import Sur, episodes
import devpred as DP

torch.set_num_threads(1)
T = 16; QQ = [4, 9, 14]; QT = 15
SRC = [list(range(5 * s, 5 * s + 5)) for s in range(3)]


def base_allow():
    A = np.zeros((T, T), bool)
    for i in range(T - 1): A[i, :T - 1] = True                      # context tokens see all context tokens
    A[QT, :] = True                                                  # query sees context + itself
    return A


def masks(variant, depth):
    """Per-layer allowed-key matrices (T x T bool) for a variant; None = real (base) mask."""
    full = base_allow(); out = {l: full.copy() for l in range(1, depth + 1)}
    per_src = np.zeros((T, T), bool)
    for s in range(3):
        for i in SRC[s]: per_src[i, SRC[s]] = True
    per_src[QT, :] = True
    lat = per_src.copy()
    for i in QQ: lat[i, QQ] = True
    qt_only = per_src.copy(); qt_only[QT, :] = False; qt_only[QT, QQ + [QT]] = True
    qt_only_lat = qt_only.copy()
    for i in QQ: qt_only_lat[i, QQ] = True
    if variant == 'D1': out[1] = per_src
    elif variant == 'D1+lat': out[1] = lat
    elif variant == 'D2': out[1] = qt_only
    elif variant == 'D2+lat': out[1] = qt_only_lat
    elif variant == 'D3':
        out[1] = qt_only; m = full.copy(); m[QT, :] = False; m[QT, QQ + [QT]] = True; out[depth] = m
    elif variant == 'D3+lat':
        out[1] = qt_only_lat; m = full.copy(); m[QT, :] = False; m[QT, QQ + [QT]] = True; out[depth] = m
    elif variant == 'D1-L2src':                                       # 3L: per-source in layers 1 and 2
        out[1] = per_src; out[2] = per_src
    elif variant == 'D1-L2lat':
        out[1] = per_src; out[2] = lat
    elif variant == 'D1+lat-L2lat':
        out[1] = lat; out[2] = lat
    return out


def make_sur(model, M):
    sur = {}
    for l, A in M.items():
        if np.array_equal(A, base_allow()): continue
        At = torch.as_tensor(A); b = model.blocks[l - 1]
        sur[f'a{l}'] = Sur(lambda x, al, b=b, At=At: N.sphere(x + b.attn_out(x, At[None].expand(len(x), T, T))), 0, f'mask{l}')
    return sur


def kl_dag(model, X, valid, st, mu, sur, sigma, depth):
    out = {}
    for name, g in sur.items():
        l = int(name[1:]); xin = torch.as_tensor(st['emb'] if l == 1 else st[f'r{l - 1}'])
        with torch.no_grad():
            gv = g.fn(xin, None).numpy()
        err = (mu[name] - gv) ** 2 / (2 * sigma ** 2)                    # (n, T, d)
        scored = list(range(T)) if l < depth else [QT]
        out[name] = float(err[:, scored].sum((1, 2)).mean())
    return out


def treewidth(M, depth):
    """Token-level DAG over sites emb, a_l, m_l, r_l (ancestors of the output only); returns (tw, max clique state dim)."""
    G = nx.DiGraph(); dims = {}
    sites = ['emb'] + [f'{k}{l}' for l in range(1, depth + 1) for k in ('a', 'm', 'r')]
    def scored(site):
        if site == 'emb': return range(T)
        l = int(site[1:]); return range(T) if l < depth else [QT]
    for s in sites:
        for t in scored(s): G.add_node((s, t)); dims[(s, t)] = 64 if s[0] == 'm' else 32
    for l in range(1, depth + 1):
        prev = 'emb' if l == 1 else f'r{l - 1}'
        for t in scored(f'a{l}'):
            for k in np.flatnonzero(M[l][t]): G.add_edge((prev, int(k)), (f'a{l}', t))
            G.add_edge((prev, t), (f'a{l}', t))
            G.add_edge((f'a{l}', t), (f'm{l}', t)); G.add_edge((f'a{l}', t), (f'r{l}', t)); G.add_edge((f'm{l}', t), (f'r{l}', t))
    U = G.to_undirected()
    for v in G.nodes:
        ps = list(G.predecessors(v))
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)): U.add_edge(ps[i], ps[j])
    tw, decomp = treewidth_min_degree(U)
    maxdim = max(sum(dims[v] for v in bag) for bag in decomp.nodes)
    edges = sum(int(M[l].sum()) for l in M)
    return int(tw), int(maxdim), edges


def run(run_dir, n_test=4000, K=8):
    t0 = time.time()
    model, meta = N.load_run(run_dir); sg = meta['sigma']; depth = meta['depth']
    Tz = np.load(os.path.join(DP.TP, 'runs/testset.npz'))
    Tt = {k: Tz[k][:n_test] for k in ('comp_means', 'w_bayes', 'mean_bayes', 'var_bayes')}; Tt['comp_vars'] = Tz['comp_vars']
    X, valid = Tz['X'][:n_test], Tz['valid'][:n_test]
    h1, m1, s1 = DP.marginal(model, X, valid, sg, {}, K, 1); dev = DP.deviations(h1, m1, s1, Tt)
    h2, m2, s2 = DP.marginal(model, X, valid, sg, {}, K, 7); dev2 = DP.deviations(h2, m2, s2, Tt)
    nf = {k: DP.r2(dev[k], dev2[k]) for k in dev}
    dE, XE, vE = episodes(2 ** 13, 82_000); muE, stE = trace(model, XE, vE, sg, torch.Generator().manual_seed(83_000))
    rng = np.random.default_rng(95_000); d = W.sample_np(2 ** 15, rng); Xs, vs = W.tokens_np(d, rng, meta['ds'])
    ref, _, mom, wb = W.references(d)
    Ts = dict(comp_means=ref['bayes'][0], comp_vars=ref['bayes'][1], w_bayes=wb, mean_bayes=mom['bayes'][0], var_bayes=mom['bayes'][1])
    F = DP.scenario_features(d, Xs, Ts)
    hh, mm, ss = DP.marginal(model, Xs, vs, sg, {}, K, 21); net_sc = DP.score_scenarios(F, DP.deviations(hh, mm, ss, Ts))
    out = dict(sigma=sg, depth=depth, noise_floor=nf, network_scenarios=net_sc, rows=[])
    variants = ['exact', 'D1', 'D1+lat', 'D2', 'D2+lat', 'D3', 'D3+lat'] if depth == 2 else ['exact', 'D1', 'D1+lat', 'D1-L2src', 'D1-L2lat', 'D1+lat-L2lat']
    for v in variants:
        M = masks(v if v != 'exact' else 'none', depth); sur = make_sur(model, M)
        kl = kl_dag(model, XE, vE, stE, muE, sur, sg, depth); tw, maxdim, edges = treewidth(M, depth)
        row = dict(variant=v, KL=float(sum(kl.values())), KL_sites=kl, treewidth=tw, max_clique_dim=maxdim, attention_edges=edges)
        if v != 'exact':
            hv, mv, sv = DP.marginal(model, X, valid, sg, sur, K, 11); dh = DP.deviations(hv, mv, sv, Tt)
            row.update({f'R2_{k}': DP.r2(dev[k], dh[k]) for k in dev})
            hs_, ms_, ss_ = DP.marginal(model, Xs, vs, sg, sur, K, 21); sc = DP.score_scenarios(F, DP.deviations(hs_, ms_, ss_, Ts))
            row['scenarios'] = sc; row['scenario_hit_agreement'] = float(np.mean([sc[k]['hit'] == net_sc[k]['hit'] for k in sc]))
        out['rows'].append(row)
        print(v, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in row.items() if k not in ('scenarios', 'KL_sites')}, flush=True)
        json.dump(out, open(os.path.join(run_dir, 'dag.json'), 'w'), indent=1)
    out['seconds'] = time.time() - t0
    json.dump(out, open(os.path.join(run_dir, 'dag.json'), 'w'), indent=1)
    print('done', out['seconds'], flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
