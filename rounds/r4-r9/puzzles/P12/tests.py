"""P12 unit tests (CPU). Run: nice python tests.py  -> prints PASS/FAIL lines, writes results/tests.json.

1. flow BP vs brute force (3^m enumeration of the 3-state directed-flow model) on 3x3 and 4x4 mazes:
   trees: max-sum = exact max-marginals, sum-product (tau 1, 0.5) = exact marginals;
   loopy: max-sum MAP with negative theta = brute-force MAP.
2. A0-flow: max-sum with theta = -true edge cost decodes the Dijkstra path on 20 W mazes at n = 9.
3. semantic loss = brute-force -log P_indep(valid local configuration).
4. true arbitrage ~ 0 at all-0.5 on an open ordinary cell; > 0 for an incoherent cell.
5. validity checker: path valid; path + stray cycle invalid; broken path invalid.
6. shared decode symmetric; GPU Bellman-Ford shortest path = Dijkstra label on W mazes.
"""
import itertools, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maze, flow, nets

torch.set_num_threads(4)
RES = {}


def check(name, ok, info=""):
    RES[name] = {"ok": bool(ok), "info": info}
    print(("PASS " if ok else "FAIL ") + name + ("  " + str(info) if info else ""), flush=True)


def edges_of(op):
    """list of (u, v, d) with d in {0,1} (E or S) for open edges, u the west/north cell."""
    n = op.shape[0]; E = []
    for i in range(n):
        for j in range(n):
            for d in (0, 1):
                if op[i, j, d]:
                    a, b = maze.DIRS[d]; E.append(((i, j), (i + a, j + b), d))
    return E


def brute(op, S, G, theta_e, tau=None):
    """Enumerate 3^m states. theta_e (m,). Returns per edge: exact on/off log-odds (tau>0) or
    max-marginal difference (tau None), plus the MAP on-set."""
    E = edges_of(op); m = len(E); n = op.shape[0]
    st = np.array(list(itertools.product([0, 1, 2], repeat=m)), np.int8)   # 0 off, 1 u->v, 2 v->u
    net = np.zeros((len(st), n * n), np.int16)
    for k, (u, v, d) in enumerate(E):
        iu, iv = u[0] * n + u[1], v[0] * n + v[1]
        s1 = (st[:, k] == 1).astype(np.int16); s2 = (st[:, k] == 2).astype(np.int16)
        net[:, iu] += s1 - s2; net[:, iv] += s2 - s1
    want = np.zeros(n * n, np.int16); want[S[0] * n + S[1]] = 1; want[G[0] * n + G[1]] = -1
    ok = (net == want).all(1)
    st = st[ok]; on = (st > 0).astype(np.float64)
    score = on @ theta_e
    out = np.zeros(m)
    for k in range(m):
        a = score[on[:, k] == 1]; b = score[on[:, k] == 0]
        if tau is None:
            out[k] = (a.max() if len(a) else -np.inf) - (b.max() if len(b) else -np.inf)
        else:
            la = tau * np.logaddexp.reduce(a / tau) if len(a) else -np.inf
            lb = tau * np.logaddexp.reduce(b / tau) if len(b) else -np.inf
            out[k] = (la - lb) / tau
    mapon = on[score.argmax()]
    return out, mapon, E


def run_bp(op, S, G, theta_slots, tau, damp, iters):
    o = torch.tensor(op[None]); Sx = torch.tensor([S]); Gx = torch.tensor([G])
    bp = flow.FlowBP(o, Sx, Gx, tau=tau, damp=damp)
    th = torch.tensor(theta_slots[None], dtype=torch.float32)
    st = bp.init_state()
    for _ in range(iters):
        st = bp.step(st, th)
    return (th + bp.delta(st))[0].double().numpy()


def slots_from_edges(op, E, val):
    n = op.shape[0]; x = np.zeros((n, n, 4))
    for k, (u, v, d) in enumerate(E):
        x[u[0], u[1], d] = val[k]; x[v[0], v[1], maze.OPP[d]] = val[k]
    return x


def small_maze(n, rng, loopy, max_edges):
    while True:
        oE, oS = maze.dfs_tree(n, rng)
        if loopy:
            oE = oE | (rng.random(oE.shape) < 0.5); oS = oS | (rng.random(oS.shape) < 0.5)
        op = maze.slots(oE, oS)
        E = edges_of(op)
        # drop random edges until <= max_edges (keeps a forest / loopy subgraph; S-G connectivity checked)
        while len(E) > max_edges:
            u, v, d = E[rng.integers(len(E))]
            op[u[0], u[1], d] = False; op[v[0], v[1], maze.OPP[d]] = False
            E = edges_of(op)
        S = (int(rng.integers(n)), int(rng.integers(n))); dS = maze.bfs(op, S)
        far = np.argwhere((dS >= 2) & (dS < 10 ** 6))
        if len(far) == 0:
            continue
        G = tuple(int(x) for x in far[rng.integers(len(far))])
        return op, S, G


def is_tree_forest(op):
    n = op.shape[0]; m = len(edges_of(op))
    # components via BFS
    seen = np.zeros((n, n), bool); comp = 0
    for i in range(n):
        for j in range(n):
            if not seen[i, j]:
                comp += 1; d = maze.bfs(op, (i, j)); seen |= d < 10 ** 6
    return m == n * n - comp


def test_bp_bruteforce():
    rng = np.random.default_rng(1)
    worst = {"tree_max": 0.0, "tree_sp1": 0.0, "tree_sp05": 0.0}; map_ok = 0; map_n = 0
    for trial in range(12):
        n = 3 if trial < 6 else 4
        op, S, G = small_maze(n, rng, loopy=False, max_edges=8 if n == 3 else 12)
        assert is_tree_forest(op)
        E = edges_of(op)
        th = rng.normal(0, 1.5, size=len(E))
        ts = slots_from_edges(op, E, th)
        for key, tau in (("tree_max", None), ("tree_sp1", 1.0), ("tree_sp05", 0.5)):
            ex, _, _ = brute(op, S, G, th, tau)
            b = run_bp(op, S, G, ts, 0.0 if tau is None else tau, 0.0, 40)
            if tau is None:
                got = np.array([b[u[0], u[1], d] for u, v, d in E])
            else:
                got = np.array([b[u[0], u[1], d] / tau + math.log(2) for u, v, d in E])
            fin = np.isfinite(ex) & (np.abs(ex) < 50)
            err = float(np.abs(got[fin] - ex[fin]).max()) if fin.any() else 0.0
            # on a forest the flow constraints leave exactly one feasible state (the S-G path), so every
            # exact marginal is +inf (forced on) or -inf (forced off): BP must give the right sign, |b| > 100
            neg = ex == -np.inf; pos = ex == np.inf
            if neg.any() and not (got[neg] < -100).all():
                err = 1e9
            if pos.any() and not (got[pos] > 100).all():
                err = 1e9
            worst[key] = max(worst[key], err)
    for k, v in worst.items():
        check(f"bp_{k}_forced_edges_forest_3x3_4x4", v < 1e-3, f"max abs err {v:.2e}")
    # loopy: max-sum MAP with negative theta
    for trial in range(12):
        n = 3 if trial < 6 else 4
        op, S, G = small_maze(n, rng, loopy=True, max_edges=12 if n == 3 else 13)
        E = edges_of(op)
        th = -rng.uniform(1, 9, size=len(E))
        _, mapon, _ = brute(op, S, G, th, None)
        for damp in (0.0, 0.5):
            b = run_bp(op, S, G, slots_from_edges(op, E, th), 0.0, damp, 200)
            got = np.array([b[u[0], u[1], d] > 0 for u, v, d in E]).astype(float)
            map_ok += int((got == mapon).all()); map_n += 1
    check("bp_maxsum_MAP_equals_bruteforce_loopy_negtheta", map_ok == map_n, f"{map_ok}/{map_n}")
    # non-degenerate marginals exist only on loopy graphs (stray cycles); record loopy sum-product error
    # (tau = 1) against brute force as a diagnostic (loopy BP is not exact; no pass/fail)
    errs = []
    for trial in range(6):
        op, S, G = small_maze(3, rng, loopy=True, max_edges=12)
        E = edges_of(op); th = rng.normal(-1, 1.5, size=len(E))
        ex, _, _ = brute(op, S, G, th, 1.0)
        b = run_bp(op, S, G, slots_from_edges(op, E, th), 1.0, 0.5, 300)
        got = np.array([b[u[0], u[1], d] + math.log(2) for u, v, d in E])
        fin = np.isfinite(ex) & (np.abs(ex) < 50)
        if fin.any():
            errs.append(float(np.abs(got[fin] - ex[fin]).mean()))
    RES["bp_sp1_loopy_mean_abs_logodds_err_diag"] = errs
    RES["bp_bruteforce_detail"] = worst


def test_a0_flow():
    d, meta = maze.make_set(9, 0.3, 20, seed=777, weighted=True)
    o = torch.tensor(d["open"]); S = torch.tensor(d["S"]); G = torch.tensor(d["G"])
    ec = flow.edge_cost_tensor(torch.tensor(d["cost"]), o)
    path = torch.tensor(d["path"]); endm = flow.endpoints(S, G, 20, 9, "cpu")
    out = {}
    for damp in (0.0, 0.5):
        bp = flow.FlowBP(o, S, G, tau=0.0, damp=damp); st = bp.init_state(); th = -ec
        first = torch.full((20,), -1)
        for t in range(1, 401):
            st = bp.step(st, th)
            on = flow.decode(th + bp.delta(st), o)
            ex = flow.exact(on, path)
            first = torch.where((first < 0) & ex, torch.tensor(t), first)
        out[damp] = {"exact": int(ex.sum()), "valid": int(flow.valid_path(on, endm).sum()),
                     "first_exact_median": float(first[first > 0].float().median()) if (first > 0).any() else None,
                     "first_exact_max": int(first.max())}
    check("a0_flow_true_costs_exact_20_W_n9_damp0", out[0.0]["exact"] == 20, out[0.0])
    check("a0_flow_true_costs_exact_20_W_n9_damp0.5", out[0.5]["exact"] == 20, out[0.5])
    RES["a0_flow_n9"] = out


def test_semantic_loss():
    rng = np.random.default_rng(3); worst = 0.0
    V = np.array(list(itertools.product([0, 1], repeat=4)))
    for _ in range(200):
        p = rng.uniform(0.01, 0.99, size=4); o = rng.random(4) < 0.8; end = rng.random() < 0.3
        pp = np.where(o, p, 0.0)
        pr = np.prod(np.where(V == 1, pp, 1 - pp), 1)
        deg = V.sum(1)
        okv = (deg == 1) if end else ((deg == 0) | (deg == 2))
        ref = -math.log(max(pr[okv].sum(), 1e-30))
        pt = torch.tensor(p, dtype=torch.float64).reshape(1, 1, 1, 4)
        got = float(flow.semantic_loss(pt, torch.tensor(o).reshape(1, 1, 1, 4), torch.tensor([[[end]]]))[0, 0, 0])
        if math.isfinite(ref) and ref < 30:
            worst = max(worst, abs(got - ref))
    check("semantic_loss_equals_bruteforce", worst < 1e-5, f"max abs err {worst:.2e}")


def test_arbitrage():
    p = torch.full((1, 3, 3, 4), 0.5); o = torch.ones(1, 3, 3, 4, dtype=torch.bool)
    endm = torch.zeros(1, 3, 3, dtype=torch.bool)
    a = flow.true_arbitrage(p, o, endm, iters=400)[0, 1, 1]
    check("true_arbitrage_zero_at_half_open_ordinary_cell", float(a) < 2e-3, f"{float(a):.2e}")
    p2 = p.clone(); p2[0, 1, 1] = torch.tensor([0.9, 0.05, 0.05, 0.05])   # one edge on, rest off: degree 1 at an ordinary cell
    a2 = flow.true_arbitrage(p2, o, endm, iters=400)[0, 1, 1]
    check("true_arbitrage_positive_for_incoherent_cell", float(a2) > 0.3, f"{float(a2):.3f}")
    # semantic loss >= arbitrage? not a theorem; record both on the incoherent cell
    RES["arb_vs_sem_example"] = {"arb": float(a2), "sem": float(flow.semantic_loss(p2, o, endm)[0, 1, 1])}


def test_validity():
    d, _ = maze.make_set(9, 0.3, 30, seed=778, weighted=True)
    o = torch.tensor(d["open"]); S = torch.tensor(d["S"]); G = torch.tensor(d["G"]); path = torch.tensor(d["path"])
    endm = flow.endpoints(S, G, 30, 9, "cpu")
    check("valid_path_on_labels", bool(flow.valid_path(path, endm).all()))
    # add a stray 4-cycle where all 4 edges of a 2x2 block are open and untouched by the path
    added = 0; bad = []
    for b in range(30):
        pb = path[b].clone(); ob = o[b]
        for i in range(8):
            for j in range(8):
                if ob[i, j, 0] and ob[i, j, 1] and ob[i + 1, j, 0] and ob[i, j + 1, 1] and \
                        pb[i, j].sum() == 0 and pb[i + 1, j].sum() == 0 and pb[i, j + 1].sum() == 0 and pb[i + 1, j + 1].sum() == 0:
                    for (a, c, dd) in ((i, j, 0), (i, j, 1), (i + 1, j, 0), (i, j + 1, 1)):
                        pb[a, c, dd] = True; aa, bb = maze.DIRS[dd]; pb[a + aa, c + bb, maze.OPP[dd]] = True
                    bad.append((b, pb)); break
            else:
                continue
            break
    if bad:
        idx = torch.tensor([b for b, _ in bad]); pbs = torch.stack([p for _, p in bad])
        v = flow.valid_path(pbs, endm[idx]); s, _ = flow.syndrome_ok(pbs, endm[idx])
        check("stray_cycle_is_degree_valid_but_invalid_path", bool(s.all()) and not bool(v.any()), f"{len(bad)} cases")
    # broken path: drop one edge
    pb = path.clone()
    for b in range(30):
        i, j, dd = [int(x) for x in torch.nonzero(pb[b])[0]]
        pb[b, i, j, dd] = False; aa, bb = maze.DIRS[dd]; pb[b, i + aa, j + bb, maze.OPP[dd]] = False
    check("broken_path_invalid", not bool(flow.valid_path(pb, endm).any()))


def test_decode_and_bf():
    torch.manual_seed(0)
    d, _ = maze.make_set(15, 0.3, 20, seed=779, weighted=True)
    o = torch.tensor(d["open"]); S = torch.tensor(d["S"]); G = torch.tensor(d["G"]); path = torch.tensor(d["path"])
    u = torch.randn(20, 15, 15); v = torch.randn(20, 15, 15)
    on = flow.decode(flow.es_to_slots(u, v), o)
    check("decode_symmetric", bool((on == flow.nb(on.float()).bool()).all()))
    ec = flow.edge_cost_tensor(torch.tensor(d["cost"]), o)
    y, ok = nets.shortest_path(ec, o, S, G)
    ex = flow.exact(y, path)
    check("bellman_ford_equals_dijkstra_W_n15", bool(ex.all()) and bool(ok.all()), f"{int(ex.sum())}/20")
    # dihedral maps preserve the label (path) as the Dijkstra solution
    allok = True
    for k in range(8):
        dk = maze.dihedral(d, k)
        o2 = torch.tensor(dk["open"]); ec2 = flow.edge_cost_tensor(torch.tensor(dk["cost"]), o2)
        y2, _ = nets.shortest_path(ec2, o2, torch.tensor(dk["S"]), torch.tensor(dk["G"]))
        allok &= bool(flow.exact(y2, torch.tensor(dk["path"])).all())
        allok &= bool((o2 == flow.nb(o2.float()).bool()).all())
    check("dihedral_maps_consistent", allok)


def test_bb_grad():
    d, _ = maze.make_set(9, 0.3, 8, seed=780, weighted=True)
    o = torch.tensor(d["open"]); S = torch.tensor(d["S"]); G = torch.tensor(d["G"]); path = torch.tensor(d["path"])
    net = nets.CostNet(16)
    loss, y = nets.costnet_loss(net, o, S, G, torch.tensor(d["code"].astype(np.int64)), path)
    loss.backward()
    g = sum(float(p.grad.abs().sum()) for p in net.parameters())
    check("costnet_bb_gradient_nonzero", g > 0, f"loss {float(loss):.2f} gradsum {g:.3g}")


def test_loop_smoke():
    d, _ = maze.make_set(9, 0.3, 8, seed=781, weighted=True)
    o = torch.tensor(d["open"]); S = torch.tensor(d["S"]); G = torch.tensor(d["G"]); path = torch.tensor(d["path"])
    code = torch.tensor(d["code"].astype(np.int64))
    ok = True; info = {}
    for name, cfg in [("DT", {}), ("DT+fb", {"fb": 1, "extra": 8}), ("DT+pen", {"pen": 1.0}), ("DT-TD", {"td": 1}),
                      ("DTxARB", {"fb": 1, "extra": 8, "arb": "sg"}), ("DTxARB-bp", {"fb": 1, "extra": 8, "arb": "bp"})]:
        m = nets.DTNet(16, fb=bool(cfg.get("fb")), extra=cfg.get("extra", 0))
        lp = nets.Loop(m, cfg, o, S, G, code)
        loss, ex = nets.train_loss(lp, path, 6, 0.5, 2, 3)
        loss.backward()
        gn = sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
        ok &= math.isfinite(float(loss)) and gn > 0
        info[name] = round(float(loss), 3)
    check("loop_smoke_all_arms_finite_grad", ok, info)


if __name__ == "__main__":
    t0 = time.time()
    ALL = (test_semantic_loss, test_arbitrage, test_validity, test_decode_and_bf, test_bb_grad, test_loop_smoke,
           test_bp_bruteforce, test_a0_flow)
    only = sys.argv[1:]
    prev = json.load(open("results/tests.json")) if only and os.path.exists("results/tests.json") else {}
    RES.update({k: v for k, v in prev.items() if k not in ("all_ok", "seconds")})
    for f in [f for f in ALL if not only or f.__name__ in only]:
        t = time.time(); f(); print(f"  [{f.__name__} {time.time() - t:.1f}s]", flush=True)
    os.makedirs("results", exist_ok=True)
    RES["all_ok"] = all(v["ok"] for k, v in RES.items() if isinstance(v, dict) and "ok" in v)
    RES["seconds"] = round(time.time() - t0, 1)
    json.dump(RES, open("results/tests.json", "w"), indent=1, default=str)
    print("ALL OK" if RES["all_ok"] else "SOME FAILED", RES["seconds"], "s")
