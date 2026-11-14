"""P2 unit tests (all must pass before any GPU run). Run: python tests.py [--trap-data]"""
import itertools
import json
import math
import os
import sys
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

import jm

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = {}


def check(name, ok, detail=""):
    RES[name] = {"pass": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name, detail)


# ---------------------------------------------------------------- small mazes
def small_mazes(num, n, p, seed):
    import gen
    rng = gen.seed_all(seed)
    out = []
    while len(out) < num:
        r = gen.maze_record(n, p, False, rng)
        out.append(r)
    return out


def batch_from(recs):
    L = recs[0]["open"].shape[0]
    op = torch.tensor(np.stack([r["open"] for r in recs]))
    ds = torch.tensor(np.stack([r["ds"] for r in recs])).long()
    dg = torch.tensor(np.stack([r["dg"] for r in recs])).long()
    D = torch.tensor([r["D"] for r in recs])
    dag = torch.tensor(np.stack([r["dag"] for r in recs]))
    s = torch.tensor([r["s"][0] * L + r["s"][1] for r in recs])
    g = torch.tensor([r["g"][0] * L + r["g"][1] for r in recs])
    return op, ds, dg, D, dag, s, g


def enum_paths(r):
    """All shortest paths (as lists of flat indices) by DFS over the DAG."""
    L = r["open"].shape[0]
    ds, dag = r["ds"], r["dag"]
    s = r["s"]
    paths = []

    def rec(cur, acc):
        if cur == r["g"]:
            paths.append(acc)
            return
        for di, dj in jm.MOVES:
            a, b = cur[0] + di, cur[1] + dj
            if 0 <= a < L and 0 <= b < L and dag[a, b] and ds[a, b] == ds[cur] + 1:
                rec((a, b), acc + [a * L + b])
    rec(s, [s[0] * L + s[1]])
    return paths


def vsp_ref(S, op, s, g, D):
    """Spec definition: start and goal in the set, |set| = D + 1, connected, open pixels only."""
    L = op.shape[0]
    if not (S[s] and S[g]) or S.sum() != D + 1 or (S & ~op).any():
        return False
    seen = {s}
    q = deque([s])
    while q:
        i, j = q.popleft()
        for di, dj in jm.MOVES:
            a, b = i + di, j + dj
            if 0 <= a < L and 0 <= b < L and S[a, b] and (a, b) not in seen:
                seen.add((a, b))
                q.append((a, b))
    return len(seen) == S.sum()


# ---------------------------------------------------------------- tests
def test_bfs_and_dag():
    import gen
    recs = small_mazes(20, 5, 0.3, 1)
    op, ds, dg, D, dag, s, g = batch_from(recs)
    d1 = jm.bfs_dist(op, s)
    ok = torch.equal(d1, ds)
    cm = jm.contested_mask(ds, dag, D)
    ok2 = all(int(cm[i].sum()) == recs[i]["contested"] for i in range(len(recs)))
    check("bfs_batched_matches_numpy", ok)
    check("contested_mask_matches_gen", ok2)


def test_dag_dps():
    recs = [r for r in small_mazes(60, 5, 0.3, 2) if r["tie"]][:25]
    recs += small_mazes(10, 5, 0.0, 3)
    op, ds, dg, D, dag, s, g = batch_from(recs)
    torch.manual_seed(0)
    m = torch.randn(ds.shape) * 2
    Fl = jm.dag_forward(m, ds, dag, "lse").flatten(1).gather(1, g[:, None]).squeeze(1)
    Fm = jm.dag_forward(m, ds, dag, "max").flatten(1).gather(1, g[:, None]).squeeze(1)
    vp = jm.viterbi_path(m, ds, dag, g)
    e1 = e2 = e3 = 0.0
    npaths = []
    for i, r in enumerate(recs):
        P = enum_paths(r)
        npaths.append(len(P))
        mf = m[i].flatten()
        sums = torch.stack([mf[p].sum() for p in P])
        e1 = max(e1, abs(float(torch.logsumexp(sums, 0) - Fl[i])))
        e2 = max(e2, abs(float(sums.max() - Fm[i])))
        e3 = max(e3, abs(float(mf[vp[i].flatten()].sum() - sums.max())))
    check("U-any_forward_sum_vs_enumeration", e1 < 1e-8, f"max err {e1:.2e}, paths per maze up to {max(npaths)}")
    check("U-min_viterbi_value_vs_enumeration", e2 < 1e-8 and e3 < 1e-8, f"value err {e2:.2e}, path err {e3:.2e}")
    # U-any score = log sum_paths P(y = path) under the factorised belief, by enumeration
    sa = jm.score_any(m[:, None], op.double(), ds, dag, g)
    e4 = 0.0
    for i, r in enumerate(recs):
        lp = []
        for P in enum_paths(r):
            y = torch.zeros(op.shape[1] ** 2, dtype=torch.float64)
            y[P] = 1
            lp.append(jm.score_unary(m[i:i + 1, None], y.view(1, *op.shape[1:]), op[i:i + 1].double()))
        e4 = max(e4, abs(float(torch.logsumexp(torch.cat(lp), 0) - sa[i])))
    check("U-any_score_vs_enumeration", e4 < 1e-8, f"max err {e4:.2e}")
    # uniform sampler: chi-square over the paths of one maze with several paths
    i = int(np.argmax(npaths))
    P = enum_paths(recs[i])
    k = len(P)
    N = 4000
    rep = lambda t: t[i:i + 1].expand(N, *t.shape[1:]).contiguous()
    gen_ = torch.Generator().manual_seed(0)
    smp = jm.sample_uniform_path(rep(ds), rep(dag), g[i:i + 1].expand(N).contiguous(), gen=gen_)
    keys = {tuple(sorted(p)): j for j, p in enumerate(P)}
    cnt = np.zeros(k)
    bad = 0
    for row in smp.flatten(1):
        key = tuple(sorted(torch.nonzero(row).flatten().tolist()))
        if key in keys:
            cnt[keys[key]] += 1
        else:
            bad += 1
    chi2 = float(((cnt - N / k) ** 2 / (N / k)).sum())
    # 99.9% quantile of chi2 with k-1 dof, Wilson-Hilferty
    q = (k - 1) * (1 - 2 / (9 * (k - 1)) + 3.09 * math.sqrt(2 / (9 * (k - 1)))) ** 3
    check("uniform_path_sampler", bad == 0 and chi2 < q, f"{k} paths, chi2 {chi2:.1f} < {q:.1f}, invalid {bad}")
    return recs


def test_fast_dag_dps():
    """Fast DAG DPs (padded views, neighbour table) equal the plain reference forms on bigger mazes."""
    ev = ew = 0.0
    same, nval, ntot, nm, grad_ok = True, 0, 0, 0, True
    for recs in (small_mazes(40, 13, 0.1, 7), small_mazes(30, 9, 0.05, 8) + small_mazes(10, 9, 0.0, 9)):
        nm += len(recs)
        op, ds, dg, D, dag, s, g = batch_from(recs)
        torch.manual_seed(1)
        m = torch.randn(ds.shape) * 2
        for w in (m, (m > 0).double()):  # continuous weights and binary (near-miss style, many ties)
            for mode in ("lse", "max"):
                a = jm.dag_forward(w, ds, dag, mode)
                b = jm.dag_forward_ref(w, ds, dag, mode)
                ev = max(ev, float((a - b).abs().max()))
        vp = jm.viterbi_path(m, ds, dag, g)
        vr = jm.dag_backtrack_ref(jm.dag_forward_ref(m, ds, dag, "max"), ds, dag, g)
        same &= bool(torch.equal(vp, vr))
        wb = (m > 0).double()  # ties: compare path values, not paths
        pv = jm.viterbi_path(wb, ds, dag, g)
        fb = jm.dag_forward_ref(wb, ds, dag, "max").flatten(1).gather(1, g[:, None]).squeeze(1)
        ew = max(ew, float(((pv.double() * wb).flatten(1).sum(1) - fb).abs().max()))
        gen_ = torch.Generator().manual_seed(3)
        smp = torch.cat([jm.sample_uniform_path(ds, dag, g, gen=gen_) for _ in range(20)])
        rep = lambda t: t.repeat(20, *([1] * (t.dim() - 1)))
        valid = jm.vsp(smp, rep(ds), rep(dg), rep(D), rep(g))
        nval, ntot = nval + int(valid.sum()), ntot + len(valid)
        wq = m.clone().requires_grad_(True)  # U-any trains through the lse DP: backward must work
        jm.dag_forward(wq, ds, dag, "lse").flatten(1).gather(1, g[:, None]).sum().backward()
        grad_ok &= bool(torch.isfinite(wq.grad).all())
    check("fast_dag_dp_matches_reference", ev < 1e-9 and same and ew < 1e-9 and nval == ntot and grad_ok,
          f"{nm} mazes n in (9,13): forward max err {ev:.1e}, viterbi paths equal {same}, tie-path value err {ew:.1e}, "
          f"sampled paths valid {nval}/{ntot}, lse backward finite {grad_ok}")


def test_vsp_nearmiss(recs):
    op, ds, dg, D, dag, s, g = batch_from(recs)
    rng = np.random.default_rng(0)
    cands, idx = [], []
    for i, r in enumerate(recs):
        P = enum_paths(r)
        L = r["open"].shape[0]
        for p in P[:3]:
            S = np.zeros(L * L, bool)
            S[p] = True
            cands.append(S.copy())
            idx.append(i)
            T = S.copy()  # remove a pixel
            T[p[len(p) // 2]] = False
            cands.append(T)
            idx.append(i)
            T = S.copy()  # add a random open pixel
            T[rng.choice(np.flatnonzero(r["open"].ravel()))] = True
            cands.append(T)
            idx.append(i)
        for _ in range(4):  # random DAG subsets and random open subsets
            cands.append(r["dag"].ravel() & (rng.random(L * L) < 0.7))
            idx.append(i)
            cands.append(r["open"].ravel() & (rng.random(L * L) < 0.3))
            idx.append(i)
        if len(P) > 1:  # union of two paths
            S = np.zeros(L * L, bool)
            S[P[0]] = True
            S[P[1]] = True
            cands.append(S)
            idx.append(i)
    idx = np.array(idx)
    Lq = recs[0]["open"].shape[0]
    Sb = torch.tensor(np.stack(cands)).view(-1, Lq, Lq)
    sel = lambda t: t[torch.tensor(idx)]
    fast = jm.vsp(Sb, sel(ds), sel(dg), sel(D), sel(g))
    ref = torch.tensor([vsp_ref(Sb[j].numpy(), recs[i]["open"], recs[i]["s"], recs[i]["g"], recs[i]["D"])
                        for j, i in enumerate(idx)])
    check("vsp_checker_vs_reference", torch.equal(fast, ref),
          f"{len(idx)} candidates, {int(ref.sum())} valid, mismatches {int((fast != ref).sum())}")
    nm = jm.near_miss(Sb, sel(ds), sel(dag), sel(D), sel(g))
    err = 0
    for j, i in enumerate(idx):
        S = Sb[j].flatten().numpy()
        best = min(int((S ^ np.isin(np.arange(S.size), p)).sum()) for p in enum_paths(recs[i]))
        err = max(err, abs(best - float(nm[j])))
    check("near_miss_dp_vs_enumeration", err < 1e-9, f"max err {err}")
    coh = jm.coherent(Sb, sel(op), sel(s), sel(g))
    check("coherence_true_on_valid_paths", bool(coh[ref].all()), f"{int(coh.sum())} coherent")


def rand_tree_state(model, S, B, K, gen):
    x = torch.zeros(B, 1, *S)
    st = model.init_state(torch.zeros(B, model.trunk.proj.in_channels, *S))
    st["theta"] = torch.randn(st["theta"].shape, generator=gen) * 2
    if "trans" in st:
        st["trans"] = [torch.randn(t.shape, generator=gen) for t in st["trans"]]
    st["root"] = torch.randn(st["root"].shape, generator=gen)
    return st


def brute_tree(st, y, mask, g, K):
    """Enumerate all code assignments of a small tree. Returns log P(y), dict leaf-code tuple ->
    max over the other codes of log P(z) + sum log max_y P(y|z), and leaf prior marginals."""
    nodes = []  # (level, coord)
    for l in range(g.Lv + 1):
        G = g.G0 // 2 ** l
        for c in itertools.product(range(G), repeat=g.dim):
            nodes.append((l, c))
    leaves = [nd for nd in nodes if nd[0] == 0]
    b = 0
    lps, bymap = [], {}
    leafm = {}
    for z in itertools.product(range(K), repeat=len(nodes)):
        a = dict(zip(nodes, z))
        if g.kind == "tree":
            lp = torch.log_softmax(st["root"][b].flatten(), 0)[a[(g.Lv, (0,) * g.dim)]]
            for (l, c), zc in a.items():
                if l == g.Lv:
                    continue
                par = tuple(ci // 2 for ci in c)
                lp = lp + torch.log_softmax(st["trans"][l][(b, a[(l + 1, par)], slice(None)) + c], 0)[zc]
        else:
            lp = 0.0
            for (l, c), zc in a.items():
                lp = lp + torch.log_softmax(st["root"][(b, slice(None)) + c], 0)[zc]
        lk = 0.0
        lkm = 0.0
        for pos in itertools.product(*[range(s_) for s_ in g.S]):
            if mask[(b,) + pos] == 0:
                continue
            c = tuple(pi // g.leaf for pi in pos) if g.kind != "global" else (0,) * g.dim
            th = st["theta"][(b, a[(0, c)]) + pos]
            yy = y[(b,) + pos]
            lk = lk + yy * F.logsigmoid(th) + (1 - yy) * F.logsigmoid(-th)
            lkm = lkm + F.logsigmoid(th.abs())
        lps.append(lp + lk)
        key = tuple(a[nd] for nd in leaves)
        bymap[key] = max(bymap.get(key, -1e30), float(lp + lkm))
        for (l, c), zc in a.items():
            if l == 0:
                leafm.setdefault((c, zc), []).append(lp)
    return torch.logsumexp(torch.stack(lps), 0), bymap, leafm


def test_tree_exact():
    gen = torch.Generator().manual_seed(0)
    errs, merrs, perrs, serrs, cases = [], [], [], [], []
    for dim, S, leaf, kind, K in [(1, (4,), 2, "tree", 3), (1, (3,), 2, "tree", 3), (2, (4, 4), 2, "tree", 2),
                                  (2, (3, 4), 2, "tree", 2), (1, (6,), 2, "tile", 3), (1, (5,), 5, "global", 3)]:
        arm = {"tree": "M-tree", "tile": "M-tile", "global": "M-global"}[kind]
        for rep in range(3):
            model = jm.Model(arm, dim=dim, cin=1, w=4, K=K, leaf=leaf, fdim=2).double()
            g = model.geom(S)
            st = rand_tree_state(model, S, 1, K, gen)
            y = (torch.rand(1, *S, generator=gen) < 0.5).double()
            mask = (torch.rand(1, *S, generator=gen) < 0.8).double()
            nn_ = sum((g.G0 // 2 ** l) ** dim for l in range(g.Lv + 1))
            lp, bymap, leafm = brute_tree(st, y, mask, g, K)
            errs.append(abs(float(lp - jm.score_m(st, y, mask, g)[0])))
            dec, z = jm.tree_map(st, mask, g)
            key = tuple(int(v) for v in z[0].flatten())
            best = max(bymap.values())
            merrs.append(abs(bymap[key] - best))                      # backtracked codes are a MAP
            merrs.append(abs(jm_tree_map_value(st, mask, g) - best))  # forward max pass value
            th = jm.pixels_from_codes(st["theta"], z, g)
            merrs.append(float(((th > 0) != dec).double().sum()))
            pi = model.leaf_prior(st, g)
            for (c, zc), v in leafm.items():
                perrs.append(abs(float(torch.logsumexp(torch.stack(v), 0).exp() - pi[(0, zc) + c])))
            # ancestral sampler: leaf-code frequencies vs prior marginals
            codes, pix = jm.tree_sample(st, g, 20000, gen)
            for zc in range(K):
                serrs.append(float(((codes[0] == zc).double().mean(0) - pi[:, zc]).abs().max()))
            cases.append(f"{kind}{S}:{nn_}n")
    check("exact_tree_score_vs_enumeration", max(errs) < 1e-5, f"max err {max(errs):.2e} on {sorted(set(cases))}")
    check("tree_MAP_vs_enumeration", max(merrs) < 1e-5, f"max err {max(merrs):.2e}")
    check("leaf_prior_marginals_vs_enumeration", max(perrs) < 1e-8, f"max err {max(perrs):.2e}")
    check("ancestral_sampler_leaf_marginals", max(serrs) < 0.02, f"max freq err {max(serrs):.3f} (20000 samples)")


def jm_tree_map_value(st, mask, g):
    beta = jm.leaf_loglik(st["theta"], None, mask, g, maxmode=True)
    lp = torch.log_softmax(st["root"], 1)
    if g.kind != "tree":
        return float((lp + beta).max(1).values.sum())
    for l in range(g.Lv):
        logT = torch.log_softmax(st["trans"][l], 2)
        beta = g.up((logT + beta.unsqueeze(1)).max(2).values)
    return float((lp + beta).max(1).values.sum())


def test_ledger_and_masking():
    torch.manual_seed(0)
    x = torch.zeros(3, 3, 19, 19)
    x[:, 0] = (torch.rand(3, 19, 19) < 0.6).double()
    y = (torch.rand(3, 19, 19) < 0.3).double() * x[:, 0]
    mask = x[:, 0]
    worst = 0.0
    start_ok = True
    for arm in ("U", "M-tree", "M-tile"):
        model = jm.Model(arm).double()
        with torch.no_grad():
            st = model.init_state(x)
            g = model.geom((19, 19))
            sc = (lambda s: jm.score_unary(s["m"], y, mask)) if arm == "U" else (lambda s: jm.score_m(s, y, mask, g))
            S0 = sc(st)
            pays = []
            prev = S0
            for t in range(8):
                st = model.step(st, x)
                cur = sc(st)
                pays.append(cur - prev)
                prev = cur
            worst = max(worst, float((torch.stack(pays).sum(0) - (prev - S0)).abs().max()))
            start_ok = start_ok and bool(torch.allclose(S0, -mask.flatten(1).sum(1) * math.log(2)))
    check("ledger_telescopes", worst < 1e-4, f"max |sum pays - (S_T - S_0)| = {worst:.2e} nats")
    check("uniform_start_bound_n_open_ln2", start_ok)
    # masking: an all-masked lattice earns 0 for every arm; changing y or theta at masked pixels changes nothing
    ok = True
    for arm in ("M-tree", "M-tile"):
        model = jm.Model(arm).double()
        with torch.no_grad():
            st = model.run(x, 3)
            g = model.geom((19, 19))
            z0 = jm.score_m(st, y, torch.zeros_like(mask), g)
            ok &= bool(z0.abs().max() < 1e-9)
            s1 = jm.score_m(st, y, mask, g)
            y2 = torch.where(mask > 0, y, 1 - y)
            st2 = dict(st)
            st2["theta"] = torch.where(mask[:, None] > 0, st["theta"], st["theta"] + 5)
            ok &= bool((jm.score_m(st2, y2, mask, g) - s1).abs().max() < 1e-9)
    check("masking_padding_earns_0", ok)


def test_canonical_and_pool():
    import gen
    recs = small_mazes(30, 9, 0.1, 5)
    op, ds, dg, D, dag, s, g = batch_from(recs)
    can = torch.tensor(np.stack([r["canon"] for r in recs]))
    check("canonical_path_is_VSP", bool(jm.vsp(can, ds, dg, D, g).all()))
    # canonical move order: first move from start is the first of D,R,U,L that lies on a shortest path
    ok = True
    for r in recs:
        L = r["open"].shape[0]
        for di, dj in jm.MOVES:
            a, b = r["s"][0] + di, r["s"][1] + dj
            if 0 <= a < L and 0 <= b < L and r["dag"][a, b] and r["ds"][a, b] == 1:
                ok &= bool(r["canon"][a, b])
                break
    check("canonical_first_move_order_DRUL", ok)


def test_decoder_trap(split="dev", ns=(9, 13), T=30):
    """Every decoder on constant beliefs and on random-init nets must score VSP <= 1%."""
    import gen
    torch.set_default_dtype(torch.float32)
    out = {}
    for n in ns:
        for p in (0.0, 0.05, 0.1):
            c = gen.load_cell(split, n, p, True)
            op = torch.tensor(c["open"])
            B, L = op.shape[0], c["L"]
            s, g, D = torch.tensor(c["s"]), torch.tensor(c["g"]), torch.tensor(c["D"])
            ds, dg = jm.bfs_dist(op, s), jm.bfs_dist(op, g)
            x = torch.zeros(B, 3, L, L)
            x[:, 0] = op.float()
            x.view(B, 3, -1)[torch.arange(B), 1, s] = 1
            x.view(B, 3, -1)[torch.arange(B), 2, g] = 1
            decs = {}
            for cval in (0.0, 0.1, -0.1, 1.0):
                m = torch.full((B, 1, L, L), cval)
                decs[f"const{cval}_thr"] = jm.decode_thr(m)
                decs[f"const{cval}_cl"] = jm.decode_cl(jm.decode_thr(m), op, s, g)
                mt = jm.Model("M-tree")
                with torch.no_grad():
                    st = mt.init_state(x)
                    st["theta"] = torch.full_like(st["theta"], cval)
                    decs[f"const{cval}_tv"] = jm.tree_map(st, op.float(), mt.geom((L, L)))[0]
            for arm in ("U", "M-tree", "M-tile"):
                torch.manual_seed(1)
                model = jm.Model(arm)
                with torch.no_grad():
                    st = model.run(x, T)
                    m, _ = model.pixel_marg(st)
                    decs[f"rand_{arm}_thr"] = jm.decode_thr(m)
                    decs[f"rand_{arm}_cl"] = jm.decode_cl(jm.decode_thr(m), op, s, g)
                    if arm != "U":
                        decs[f"rand_{arm}_tv"] = jm.tree_map(st, op.float(), model.geom((L, L)))[0]
            for k, S in decs.items():
                out.setdefault(k, []).append(float(jm.vsp(S & op, ds, dg, D, g).float().mean()))
    worst = {k: max(v) for k, v in out.items()}
    bad = {k: v for k, v in worst.items() if v > 0.01}
    check("decoder_trap_test", not bad, f"worst VSP {max(worst.values()):.3f}; failing {bad}")
    torch.set_default_dtype(torch.float64)
    return worst


if __name__ == "__main__":
    test_bfs_and_dag()
    recs = test_dag_dps()
    test_fast_dag_dps()
    test_vsp_nearmiss(recs)
    test_tree_exact()
    test_ledger_and_masking()
    test_canonical_and_pool()
    if "--trap-data" in sys.argv:
        RES["decoder_trap_detail"] = {"pass": True, "detail": test_decoder_trap()}
    ov = os.path.join(HERE, "results", "overlap.json")
    if os.path.exists(ov):
        o = json.load(open(ov))
        check("train_test_hash_overlap_0", o["overlap"] == 0, json.dumps(o))
    json.dump(RES, open(os.path.join(HERE, "results", "unit_tests.json"), "w"), indent=1)
    fails = [k for k, v in RES.items() if not v["pass"]]
    print("ALL PASS" if not fails else f"FAILURES: {fails}")
    sys.exit(1 if fails else 0)
