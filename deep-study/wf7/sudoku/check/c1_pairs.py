"""c1_pairs.py (checker) -- the pair check, exhaustively.
The pair score of a board reads the network ONLY at one-cell contexts. There are N*K of them (216 at 6x6, 64 at 4x4),
so the whole table Q[i,u,j,v] = log q(x_j = v | x_i = u alone) can be written down. From the table:
  L_bad  = the largest entry over peer pairs with equal digits (an invalid board contains at least one such pair);
  L_good = the smallest entry over every other pair (a valid board contains only those).
If L_bad < theta <= L_good then "pair score >= theta" is EXACTLY "the board is valid", for every one of the K^N boards.
Also: self-test of my exact counter, the exact pair table, and the rule graph read out of the network."""
import json, math
import numpy as np, torch
import ck

OUT = {}
for K, name in ((6, "mA6"), (4, "mA4")):
    net = ck.M(name, K); N = K * K; P = ck.peers(K); Q = ck.pair_table(net).numpy()          # [N, K, N, K]
    same = np.zeros((N, K, N, K), bool); diag = np.zeros((N, K, N, K), bool)
    for i in range(N):
        diag[i, :, i, :] = True
        for u in range(K):
            same[i, u, P[i], u] = True                                                        # peer, equal digit: impossible
    good = ~same & ~diag
    L_bad, L_good = float(Q[same].max()), float(Q[good].min())
    # exact pair table from my counter
    ex = ck.Exact(K); PX = np.zeros((N, K, N, K))
    for i in range(N):
        for u in range(K):
            s = -np.ones(N, dtype=np.int64); s[i] = u; tot, Pc = ex.cond(s); PX[i, u] = Pc
    assert np.allclose(PX.sum(3)[~np.eye(N, dtype=bool)[:, None, :].repeat(K, 1)], 1.0)
    assert (PX[same] == 0).all() and (PX[good] > 0).all(), "exact table: impossible = peer with equal digit, nothing else"
    q = np.exp(Q); kl = np.where(PX > 0, PX * (np.log(np.where(PX > 0, PX, 1)) - Q), 0.0).sum(3)   # [N, K, N]
    kl = kl[~np.eye(N, dtype=bool)[:, None, :].repeat(K, 1)]
    # the rule graph as the network states it: j is a peer of i iff q(x_j = u | x_i = u) is tiny for every u
    eq = np.stack([Q[:, u, :, u] for u in range(K)], 0)                                       # [K, N, N]
    theta6 = json.load(open(f"{ck.W}/out/pair_meta.json"))["theta"] if K == 6 else None
    thr = theta6 if K == 6 else (L_bad + L_good) / 2
    graph = (eq.max(0) < thr); np.fill_diagonal(graph, False)
    r = dict(contexts=N * K, n_impossible_atoms=int(same.sum()), n_possible_atoms=int(good.sum()),
             L_bad=L_bad, L_good=L_good, gap=L_good - L_bad, exact_min_possible=float(np.log(PX[good].min())),
             exact_distinct_values=sorted(set(np.round(PX[good], 6).tolist())),
             mass_on_impossible_mean=float(q[same].mean()), mass_on_impossible_max=float(q[same].max()),
             kl_per_atom_mean=float(kl.mean()), kl_per_atom_max=float(kl.max()),
             max_abs_err_prob=float(np.abs(q - PX)[good | same].max()),
             graph_equals_peer_relation=bool((graph == P).all()), graph_edges=int(graph.sum()), peer_edges=int(P.sum()))
    if K == 6:
        r["theta_worker"] = theta6; r["certificate"] = bool(L_bad < theta6 <= L_good)
    else:
        r["certificate_any_theta_between"] = bool(L_bad < L_good)
    OUT[name] = r
    print(f"== {name} ({K}x{K}): {N*K} one-cell contexts, {int(same.sum())} impossible atoms (peer, equal digit), {int(good.sum())} possible atoms")
    print(f"   largest log q on an impossible atom  L_bad  = {L_bad:.3f}  (probability {math.exp(L_bad):.2e})")
    print(f"   smallest log q on a possible atom    L_good = {L_good:.3f}  (exact smallest possible conditional: {r['exact_min_possible']:.4f} = ln {math.exp(r['exact_min_possible']):.4f})")
    print(f"   gap {L_good - L_bad:.2f} nats" + (f"; worker's theta = {theta6:.3f}; L_bad < theta <= L_good: {r['certificate']}" if K == 6 else ""))
    print(f"   so 'pair score >= theta' is exactly 'valid' on all {K}^{N} boards: {r.get('certificate', r.get('certificate_any_theta_between'))}")
    print(f"   exact one-cell conditionals take the values {r['exact_distinct_values']}")
    print(f"   network against exact on one-cell contexts: KL per (context, cell) mean {kl.mean():.2e}, max {kl.max():.2e}; largest |q - p| {r['max_abs_err_prob']:.4f}; "
          f"mean mass on an impossible atom {q[same].mean():.2e}")
    print(f"   rule graph read from the network (pairs it rules out for every digit): {int(graph.sum())} ordered pairs; equals the peer relation ({int(P.sum())}): {r['graph_equals_peer_relation']}")
    # valid boards: is the score a constant?
    if K == 6:
        X = ck.sample_grids(6, 2, 3000, 4242); sc = ck.pair_score(torch.from_numpy(Q), X).numpy()
    else:
        G = np.load(f"{ck.W}/data/k4.npz")["G"].astype(np.int64); assert ck.is_valid(G, 4).all() and len(np.unique(G, axis=0)) == 288
        sc = ck.pair_score(torch.from_numpy(Q), torch.from_numpy(G)).numpy()
    r["valid_score_min"], r["valid_score_max"] = float(sc.min()), float(sc.max())
    print(f"   pair score of {len(sc)} valid grids: min {sc.min():.4f}, max {sc.max():.4f} (spread {sc.max()-sc.min():.4f})")

# ---- self-test of my counter against backtracking and against known totals
ex6 = ck.Exact(6); ex4 = ck.Exact(4)
assert ex6.cond(-np.ones(36, dtype=np.int64))[0] == 28200960 and ex4.cond(-np.ones(16, dtype=np.int64))[0] == 288
assert len(ex6.B) == 39168
rng = np.random.default_rng(99); bad = 0; n = 0
X = ck.sample_grids(6, 2, 60, 7).numpy()
for a in range(60):
    k = int(rng.integers(15, 30)); s = -np.ones(36, dtype=np.int64); idx = rng.choice(36, k, replace=False); s[idx] = X[a, idx]
    tot, Pc = ex6.cond(s); bad += int(tot != ck.brute_count(s, 6)); n += 1
    e = np.where(s < 0)[0]; assert np.allclose(Pc[e].sum(1), 1.0)
    i = int(e[0])
    for v in range(6):                                                     # conditional = ratio of counts, by backtracking
        s2 = s.copy(); s2[i] = v; assert abs(ck.brute_count(s2, 6) / tot - Pc[i, v]) < 1e-12
    # a state that is NOT a partial filling of a valid grid must count 0 when backtracking says 0
    s3 = s.copy(); j = int(idx[0]); s3[j] = (s3[j] + 1) % 6; bad += int(ex6.cond(s3)[0] != ck.brute_count(s3, 6)); n += 1
print(f"\nmy exact counter: empty 6x6 board {28200960} completions, 39,168 base grids all valid and distinct; "
      f"{n} dense states against backtracking: {bad} mismatches")
OUT["counter_selftest"] = dict(n=n, mismatches=bad)
json.dump(OUT, open("out/c1_pairs.json", "w"), indent=1)
