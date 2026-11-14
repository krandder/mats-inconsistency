"""c3_six.py (checker) -- 6x6 re-runs with my own code and fresh seeds. The worker's experiment scripts, oracle.py
and eval_lib.py are not imported; the worker's checkpoints and data arrays are read.

  python -u c3_six.py splits            are test puzzles disjoint from training, up to every symmetry? (my own canonical form)
  python -u c3_six.py routes            E2: excess of T_pi on fresh grids for every order policy; mixture of routes
  python -u c3_six.py draws             E3: mass of T on invalid boards (fresh draws), route threshold, pair check
  python -u c3_six.py stiff             E1 / stiffness by context size / confidently wrong atoms, with my exact counter
  python -u c3_six.py search ARM N      E4: search on the first N HARD puzzles, fresh seed, my own loop
  python -u c3_six.py base              one-pass baselines on HARD, first-error confidence for M_B and for M_A
  python -u c3_six.py fresh N           new hard puzzles from the test orbits (my generator, my seed), then S1 / K0 / K1 on them
"""
import sys, time, math, json, itertools
import numpy as np, torch
import ck

MODE = sys.argv[1]; K, N = 6, 36; T0 = time.time(); HREAL = math.log(28200960)
META = json.load(open(f"{ck.W}/out/trust_meta_mA6.json")); TAU = META["tau"]
THETA = json.load(open(f"{ck.W}/out/pair_meta.json"))["theta"]


def log(*a):
    print(*a, flush=True)


def canon_keys(X):
    """my canonical form of 6x6 grids under the 3,456 cell symmetries (rows in bands, bands, columns in stacks, stacks)
    and digit relabelling: the lexicographically smallest relabel-normalised image, as bytes."""
    rows = []
    for bp in itertools.permutations(range(3)):
        for inner in itertools.product([(0, 1), (1, 0)], repeat=3):
            rows.append([bp[b] * 2 + inner[b][k] for b in range(3) for k in range(2)])
    cols = []
    for sp in itertools.permutations(range(2)):
        for inner in itertools.product(list(itertools.permutations(range(3))), repeat=2):
            cols.append([sp[b] * 3 + inner[b][k] for b in range(2) for k in range(3)])
    syms = np.array([[rp[r] * 6 + cp[c] for r in range(6) for c in range(6)] for rp in rows for cp in cols])
    assert len(syms) == 3456
    X = np.asarray(X, dtype=np.int64); best = None; w = (6 ** np.arange(18, dtype=np.int64))[::-1]
    for t in syms:
        Y = X[:, t]; inv = np.argsort(Y[:, :6], axis=1); Yn = np.take_along_axis(inv, Y, 1)
        hi = (Yn[:, :18] * w).sum(1); lo = (Yn[:, 18:] * w).sum(1)
        if best is None:
            best = np.stack([hi, lo], 1)
        else:
            better = (hi < best[:, 0]) | ((hi == best[:, 0]) & (lo < best[:, 1]))
            best[better] = np.stack([hi, lo], 1)[better]
    return [tuple(r) for r in best.tolist()]


if MODE == "splits":
    d = np.load(f"{ck.W}/data/k6_base.npz"); B = d["B"].astype(np.int64); bs = d["base_split"]; bo = d["base_orbit"]
    look = {r.tobytes(): j for j, r in enumerate(B.astype(np.int8))}

    def split_of(sol):
        inv = np.argsort(sol[:, :6], axis=1); n = np.take_along_axis(inv, sol, 1).astype(np.int8)
        idx = np.array([look[r.tobytes()] for r in n]); return bs[idx], bo[idx]
    rng = np.random.default_rng(5)
    files = {"train_min (M_B training)": "k6_train_min", "train_12p (D6 training)": "k6_train_12p", "val_min": "k6_val_min",
             "test_min (ALL, HARD)": "k6_test_min", "test_12p": "k6_test_12p", "test_multi": "k6_test_multi"}
    SOL = {}
    for lab, f in files.items():
        z = np.load(f"{ck.W}/data/{f}.npz"); sol = z["sol"].astype(np.int64); puz = z["puz"].astype(np.int64)
        assert ck.is_valid(sol[:3000], 6).all() and ((puz < 0) | (puz == sol)).all()
        sp, orb = split_of(sol); SOL[lab] = (sol, sp, orb)
        log(f"   {lab:28s}: {len(sol):6d} puzzles; solutions by the worker's split (0 train, 1 val, 2 test): {np.bincount(sp, minlength=3).tolist()}; orbits used {len(set(orb.tolist()))}")
    tr_orb = set(SOL["train_min (M_B training)"][2].tolist()) | set(SOL["train_12p (D6 training)"][2].tolist()) | set(bo[bs == 0].tolist())
    te_orb = set(SOL["test_min (ALL, HARD)"][2].tolist()) | set(SOL["test_12p"][2].tolist()) | set(SOL["test_multi"][2].tolist())
    log(f"   orbits shared between any training set (incl. the grid stream of M_A and of the trust learner) and any test set: {len(tr_orb & te_orb)}")
    # independent canonical form on a sample: do my orbits agree with the worker's ids, and are test and train disjoint under MY form?
    z = np.load(f"{ck.W}/data/k6_test_min.npz"); hard = np.where(z["rating"] == 2)[0][:300]
    te = SOL["test_min (ALL, HARD)"][0][hard]; teo = SOL["test_min (ALL, HARD)"][2][hard]
    j = rng.choice(12000, 400, replace=False); tr = SOL["train_min (M_B training)"][0][j]; tro = SOL["train_min (M_B training)"][2][j]
    kt = canon_keys(te); kr = canon_keys(tr)
    mine = {}; ok = True
    for k_, o in list(zip(kt, teo)) + list(zip(kr, tro)):
        ok &= mine.setdefault(k_, o) == o
    back = {}
    for k_, o in mine.items():
        ok &= back.setdefault(o, k_) == k_
    log(f"   my canonical form on 300 HARD solutions and 400 training solutions: {len(set(kt))} test classes, {len(set(kr))} training classes, "
        f"shared {len(set(kt) & set(kr))}; one-to-one with the worker's orbit ids: {ok}")
    # exact duplicates of puzzles
    a = set(map(bytes, np.load(f"{ck.W}/data/k6_test_min.npz")["puz"].astype(np.int8))); b = set(map(bytes, np.load(f"{ck.W}/data/k6_train_min.npz")["puz"].astype(np.int8)))
    log(f"   identical puzzles in test_min and train_min: {len(a & b)}")
    # HARD puzzles: unique solution by my counter? clue counts
    ex = ck.Exact(6); z = np.load(f"{ck.W}/data/k6_test_min.npz"); bad = 0
    for p in hard[:120]:
        bad += int(ex.cond(z["puz"][p].astype(np.int64))[0] != 1)
    log(f"   120 HARD puzzles: solution count by my counter != 1 for {bad}; clues per HARD puzzle {np.bincount((z['puz'][hard] >= 0).sum(1)).tolist()[8:]} for 8, 9, ...")
    log(f"total {time.time()-T0:.0f}s")

if MODE == "routes":
    net = ck.M("mA6", 6); trust = ck.Trust(f"{ck.W}/ckpt/trust_mA6.pt", 6)
    XT = ck.sample_grids(6, 2, 800, 9001); XV = ck.sample_grids(6, 1, 256, 9002); XF = ck.sample_grids(6, 0, 400, 9003)
    assert ck.is_valid(XT.numpy(), 6).all()
    POL = {"own confidence: lowest entropy": ck.o_entropy, "own confidence: top probability": ck.o_maxprob,
           "own confidence: top-1 minus top-2": ck.o_margin, "least confident first": ck.o_anti, "raster order": ck.o_raster, "pi_trust": trust}
    gen = torch.Generator().manual_seed(123)
    rnd = lambda s, lp: torch.rand(s.shape, generator=gen)

    def nll(order, X):
        return -ck.walk(net, order, torch.full_like(X, -1), X=X)["logp"].numpy()
    NT, NV, R = {}, {}, {}
    for lab, p in POL.items():
        NT[lab] = nll(p, XT); NV[lab] = nll(p, XV)
        log(f"   {lab:36s} test excess {NT[lab].mean()-HREAL:8.4f}   validation excess {NV[lab].mean()-HREAL:8.4f} ({time.time()-T0:.0f}s)")
    NT["one random order"] = nll(rnd, XT)
    base = NT["pi_trust"]
    log(f"\nE2 on {len(XT)} fresh held-out test grids (my seed). excess, 95% interval, and paired difference to pi_trust")
    for lab, v in NT.items():
        m, lo, hi = ck.boot(v - HREAL); dm, dlo, dhi = ck.boot(v - base, seed=1)
        R[lab] = dict(excess=m, lo=lo, hi=hi, d=dm, dlo=dlo, dhi=dhi)
        log(f"   {lab:36s} {m:8.4f} [{lo:7.4f},{hi:7.4f}]   {dm:+8.4f} [{dlo:+7.4f},{dhi:+7.4f}]")
    labs = list(POL); LV = -np.stack([NV[l] for l in labs], 1); w = np.full(len(labs), 1 / len(labs))
    for _ in range(500):
        a = LV + np.log(w + 1e-300); a -= a.max(1, keepdims=True); r_ = np.exp(a); r_ /= r_.sum(1, keepdims=True); w = r_.mean(0)
    LTm = -np.stack([NT[l] for l in labs], 1); mixnll = -np.log((np.exp(LTm - LTm.max(1, keepdims=True)) * w).sum(1)) - LTm.max(1)
    m, lo, hi = ck.boot(mixnll - HREAL); dm, dlo, dhi = ck.boot(mixnll - base, seed=1); rm, rlo, rhi = ck.boot(mixnll - NT["raster order"], seed=2)
    R["mixture"] = dict(excess=m, lo=lo, hi=hi, d=dm, dlo=dlo, dhi=dhi, vs_raster=rm, vs_raster_lo=rlo, vs_raster_hi=rhi, weights=dict(zip(labs, w.tolist())))
    log(f"   {'learned mixture (EM, 256 fresh val grids)':36s} {m:8.4f} [{lo:7.4f},{hi:7.4f}]   {dm:+8.4f} [{dlo:+7.4f},{dhi:+7.4f}]   minus raster {rm:+.4f} [{rlo:+.4f},{rhi:+.4f}]")
    log("      weights: " + ", ".join(f"{l.split(':')[-1].strip()[:20]} {x:.3f}" for l, x in zip(labs, w)))
    dm, dlo, dhi = ck.boot(NT["own confidence: lowest entropy"] - base, seed=2)
    log(f"   headline comparison, lowest entropy minus pi_trust: {dm:+.4f} [{dlo:+.4f}, {dhi:+.4f}]")
    log(f"\nin distribution: {len(XF)} fresh grids of the training orbits")
    F = {l: nll(POL[l], XF) for l in ("own confidence: lowest entropy", "raster order", "pi_trust")}
    for l, v in F.items():
        m, lo, hi = ck.boot(v - HREAL); dm, dlo, dhi = ck.boot(v - F["pi_trust"], seed=1)
        R["indist " + l] = dict(excess=m, lo=lo, hi=hi, d=dm, dlo=dlo, dhi=dhi)
        log(f"   {l:36s} {m:8.4f} [{lo:7.4f},{hi:7.4f}]   {dm:+8.4f} [{dlo:+7.4f},{dhi:+7.4f}]")
    # the worker's own saved per-grid numbers: mixture against raster, paired (not reported in the note)
    z = np.load(f"{ck.W}/out/e_trust6_mA6_nll.npz"); keys = list(z.keys()); log(f"\nworker's saved per-grid NLL arrays: {keys}")
    json.dump(R, open("out/c3_routes.json", "w"), indent=1); np.savez("out/c3_routes_nll.npz", **{k.replace(" ", "_").replace(":", ""): v for k, v in NT.items()})
    log(f"calls of M {net.calls}; total {time.time()-T0:.0f}s")

if MODE == "draws":
    net = ck.M("mA6", 6); trust = ck.Trust(f"{ck.W}/ckpt/trust_mA6.pt", 6); Q = ck.pair_table(net)
    R = {}
    for lab, pol, n, seed, tau in (("pi_trust", trust, 2000, 31337, TAU), ("lowest entropy", ck.o_entropy, 1000, 31338, META["tau_conf"]),
                                   ("raster", ck.o_raster, 1000, 31339, None)):
        g = torch.Generator().manual_seed(seed)
        o = ck.walk(net, pol, torch.full((n, N), -1, dtype=torch.long), gen=g)
        X = o["s"].numpy(); val = ck.is_valid(X, 6); lt = o["logp"].numpy(); ps = ck.pair_score(Q, o["s"]).numpy()
        nv = ck.n_violations(X[~val], 6) if (~val).any() else np.array([0])
        r = dict(n=n, invalid=int((~val).sum()), frac_invalid=float((~val).mean()),
                 route_accepts_invalid=int((lt[~val] >= tau).sum()) if tau is not None else None,
                 max_logT_invalid=float(lt[~val].max()) if (~val).any() else None, median_logT_valid=float(np.median(lt[val])),
                 pair_accepts_invalid=int((ps[~val] >= THETA).sum()), pair_max_invalid=float(ps[~val].max()) if (~val).any() else None,
                 pair_rejects_valid=int((ps[val] < THETA).sum()), pair_min_valid=float(ps[val].min()),
                 violations_median=float(np.median(nv)), violations_min=int(nv.min()))
        R[lab] = r
        se = math.sqrt(r["frac_invalid"] * (1 - r["frac_invalid"]) / n)
        log(f"   T by {lab}: {n} fresh draws, invalid {r['invalid']} ({100*r['frac_invalid']:.2f}% +- {100*1.96*se:.2f}); "
            f"accepted by the route threshold: {r['route_accepts_invalid']}; largest log T of an invalid draw {r['max_logT_invalid']}; median log T of valid draws {r['median_logT_valid']:.2f}")
        log(f"      pair check: invalid draws accepted {r['pair_accepts_invalid']} (largest score {r['pair_max_invalid']}); valid draws rejected {r['pair_rejects_valid']} (smallest score {r['pair_min_valid']:.3f}); "
            f"violations per invalid draw: median {r['violations_median']:.0f}, min {r['violations_min']} ({time.time()-T0:.0f}s)")
    # near-miss boards and uniformly random boards against both tests
    XT = ck.sample_grids(6, 2, 600, 555); rng = np.random.default_rng(556); Xc = XT.numpy().copy(); Xs = XT.numpy().copy()
    for a in range(len(Xc)):
        i = rng.integers(N); Xc[a, i] = (Xc[a, i] + rng.integers(1, K)) % K
        r_ = rng.integers(K); c1, c2 = rng.choice(K, 2, replace=False); Xs[a, r_ * K + c1], Xs[a, r_ * K + c2] = Xs[a, r_ * K + c2], Xs[a, r_ * K + c1]
    for lab, Xn in (("one cell changed", Xc), ("two cells of a row swapped", Xs)):
        assert not ck.is_valid(Xn, 6).any()
        lt = ck.walk(net, trust, torch.full((len(Xn), N), -1, dtype=torch.long), X=torch.from_numpy(Xn))["logp"].numpy(); ps = ck.pair_score(Q, Xn).numpy()
        R[lab] = dict(route_accepts=int((lt >= TAU).sum()), route_max=float(lt.max()), pair_accepts=int((ps >= THETA).sum()), pair_max=float(ps.max()))
        log(f"   {len(Xn)} near-miss boards ({lab}): route threshold accepts {R[lab]['route_accepts']} (largest log T {lt.max():.2f}); pair check accepts {R[lab]['pair_accepts']} (largest score {ps.max():.2f})")
    lt = ck.walk(net, trust, torch.full_like(XT, -1), X=XT)["logp"].numpy()
    R["valid"] = dict(route_rejects=int((lt < TAU).sum()), route_min=float(lt.min()), n=len(XT))
    log(f"   {len(XT)} fresh valid test grids: route threshold rejects {R['valid']['route_rejects']} (smallest log T {lt.min():.2f}; tau {TAU:.2f})")
    json.dump(R, open("out/c3_draws.json", "w"), indent=1); log(f"calls of M {net.calls}; total {time.time()-T0:.0f}s")

if MODE == "stiff":
    net = ck.M("mA6", 6); ex = ck.Exact(6); PEER = ck.peers(6); TH = math.log(1e-3); R = {"rows": []}
    rng = np.random.default_rng(4711)
    log(f"{'cells revealed':>14s} {'ruled out':>10s} {'real value ruled out':>21s} {'worst log q(real)':>18s} | exact, 120 states: {'ruled out & possible':>21s} {'rule-breaking caught':>21s} {'dead ends caught':>17s} {'(dead ends)':>12s}")
    leak_all = []; conf_tot = 0; conf_wrong = 0; conf_p = []; conf_q = []
    for depth in (1, 2, 3, 6, 9, 12, 15, 18, 24, 30):
        n = 600; X = ck.sample_grids(6, 2, n, 5000 + depth).numpy()
        S = -np.ones((n, N), dtype=np.int64)
        for a in range(n):
            idx = rng.choice(N, depth, replace=False); S[a, idx] = X[a, idx]
        LP = net.logp(torch.from_numpy(S)).numpy().astype(np.float64); emp = S < 0
        real = np.take_along_axis(LP, X[:, :, None], 2)[:, :, 0][emp]; veto = (LP < TH) & emp[:, :, None]
        m = 120; fp = 0; nd = nde = cd = cde = 0
        for a in range(m):
            tot, P = ex.cond(S[a]); e = emp[a]; imp = (P == 0) & e[:, None]; v = veto[a]
            used = np.zeros((N, K), bool)
            for i in np.where(S[a] >= 0)[0]:
                used[PEER[i], S[a, i]] = True
            direct = imp & used; dead = imp & ~used
            assert not (used & e[:, None] & (P > 0)).any()
            fp += int((v & ~imp).sum()); nd += int(direct.sum()); nde += int(dead.sum()); cd += int((direct & v).sum()); cde += int((dead & v).sum())
            q = np.exp(LP[a]); leak_all.append((q * used)[e].sum(1))
            top = q.argmax(1); qt = q[np.arange(N), top]; pt = P[np.arange(N), top]; c = e & (qt >= 0.9)
            conf_tot += int(c.sum()); conf_wrong += int((c & (pt <= 0.1)).sum()); conf_p += pt[c].tolist(); conf_q += qt[c].tolist()
        row = dict(depth=depth, ruled_out=float(veto.sum() / (emp.sum() * K)), real_ruled_out=float((real < TH).mean()), worst_real=float(real.min()),
                   ruled_out_but_possible=fp, direct_recall=cd / max(1, nd), dead_recall=(cde / nde) if nde else None, n_dead=nde)
        R["rows"].append(row)
        log(f"{depth:14d} {row['ruled_out']:10.4f} {row['real_ruled_out']:21.2e} {row['worst_real']:18.2f} | {'':18s} {fp:21d} {row['direct_recall']:21.4f} "
            f"{(row['dead_recall'] if nde else float('nan')):17.4f} {nde:12d} ({time.time()-T0:.0f}s)")
    leak = np.concatenate(leak_all)
    R["leak_direct_mean"] = float(leak.mean()); R["leak_direct_p99"] = float(np.quantile(leak, .99)); R["leak_direct_max"] = float(leak.max())
    R["conf"] = dict(n=conf_tot, wrong=conf_wrong, mean_exact=float(np.mean(conf_p)), mean_M=float(np.mean(conf_q)))
    log(f"mass on directly rule-breaking values per atom (all depths above, {len(leak)} atoms): mean {leak.mean():.2e}, 99th percentile {np.quantile(leak, .99):.2e}, max {leak.max():.2e}")
    log(f"confident atoms (top probability >= 0.9) on these real contexts: {conf_tot}; confidently wrong (exact probability <= 0.1): {conf_wrong}; "
        f"mean exact probability of the top value {np.mean(conf_p):.4f} (M says {np.mean(conf_q):.4f})")
    json.dump(R, open("out/c3_stiff.json", "w"), indent=1); log(f"total {time.time()-T0:.0f}s")


def load_hard(n):
    z = np.load(f"{ck.W}/data/k6_test_min.npz"); idx = np.where(z["rating"] == 2)[0][:300][:n]
    return torch.from_numpy(z["puz"][idx].astype(np.int64)), torch.from_numpy(z["sol"][idx].astype(np.int64))


def run_search(prop, order, puz, flag, Q, gen, floor=0.0, greedy_first=True, cap=256, route=None, batch=(1, 7, 24, 96, 128)):
    """my own search loop. Candidates are generated in order; for each one the pair score comes from the table Q
    (no network call) and, if `route` = (net, trust, tau) is given, the full route log T as well. A puzzle stops at the
    first candidate that passes the pair check. Returns per puzzle the list of (board, pair score, route logT, route steps
    before the running sum falls below tau)."""
    P = len(puz); cands = [[] for _ in range(P)]; done = torch.zeros(P, dtype=torch.bool); tried = 0
    for r in batch:
        r = min(r, cap - tried)
        if r <= 0 or done.all():
            break
        todo = torch.where(~done)[0]; nt = len(todo); fl = None if flag is None else flag[todo]; Xs = []
        nd = r
        if greedy_first and tried == 0:
            Xs.append(ck.walk(prop, order, puz[todo], flag=fl, greedy=True)["s"][:, None]); nd -= 1
        if nd > 0:
            rep = torch.arange(nt).repeat_interleave(nd)
            Xs.append(ck.walk(prop, order, puz[todo][rep], flag=None if fl is None else fl[rep], floor=floor, gen=gen)["s"].view(nt, nd, N))
        X = torch.cat(Xs, 1); ps = ck.pair_score(Q, X.reshape(-1, N)).view(nt, r)
        ok = ps >= THETA; hit = ok.any(1); j = ok.float().argmax(1); ncons = torch.where(hit, j + 1, torch.full_like(j, r))
        for a, p in enumerate(todo.tolist()):
            cands[p].append((X[a, :ncons[a]], ps[a, :ncons[a]]))
        done[todo[hit]] = True; tried += r
    out = []
    for p in range(P):
        X = torch.cat([c[0] for c in cands[p]]); ps = torch.cat([c[1] for c in cands[p]]); out.append([X, ps, None, None])
    if route is not None:
        rnet, rtrust, tau = route; allX = torch.cat([o[0] for o in out])
        w = ck.walk(rnet, rtrust, torch.full_like(allX, -1), X=allX, keep=True)
        run = torch.zeros(len(allX), dtype=torch.float64); steps = torch.full((len(allX),), N, dtype=torch.long); alive = torch.ones(len(allX), dtype=torch.bool)
        for t, (ia, sa, lp, cell, v) in enumerate(w["trace"]):
            run += lp[torch.arange(len(ia)), cell, v].double(); newly = alive & (run < tau); steps[newly] = t + 1; alive &= ~newly
        a = 0
        for o in out:
            n = len(o[0]); o[2] = w["logp"][a:a + n]; o[3] = steps[a:a + n]; a += n
    return out


if MODE == "search":
    arm = sys.argv[2]; n = int(sys.argv[3]); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 20261110
    MA = ck.M("mA6", 6); Q = ck.pair_table(MA); puz, sol = load_hard(n); gen = torch.Generator().manual_seed(seed + ord(arm[1]))
    cfgs = {"S1": ("mA6", ck.o_margin, 0.0, True, False), "S2": ("mA6", ck.o_anti, 0.05, False, False),
            "S3": ("D6", ck.o_margin, 0.0, True, True), "S4": ("mB6", ck.o_margin, 0.0, True, True)}
    pname, order, floor, gfirst, useflag = cfgs[arm]; prop = MA if pname == "mA6" else ck.M(pname, 6)
    route = (MA, ck.Trust(f"{ck.W}/ckpt/trust_mA6.pt", 6), TAU) if arm in ("S1", "S2") else None
    res = []
    for a in range(0, n, 50):
        res += run_search(prop, order, puz[a:a + 50], (puz[a:a + 50] >= 0) if useflag else None, Q, gen, floor=floor, greedy_first=gfirst, route=route)
        log(f"   {min(a+50, n)}/{n} ({time.time()-T0:.0f}s)")
    nfree = (puz < 0).sum(1).numpy(); FLM = MA.flops; FLP = prop.flops; FLT = route[1].flops if route else 0
    # pair check (table): first candidate that passes
    acc_i = np.array([int((o[1] >= THETA).float().argmax()) if (o[1] >= THETA).any() else -1 for o in res])
    ans_ok = np.array([bool((o[0][i] == sol[p]).all()) if i >= 0 else False for p, (o, i) in enumerate(zip(res, acc_i))])
    ncand = np.array([len(o[0]) for o in res]); first_ok = np.array([bool((o[0][0] == sol[p]).all()) for p, o in enumerate(res)])
    wrong = sum(int(((o[0] != sol[p]).any(1)).sum()) for p, o in enumerate(res)); wrong_acc = sum(int((((o[0] != sol[p]).any(1)) & (o[1] >= THETA)).sum()) for p, o in enumerate(res))
    right_rej = sum(int((((o[0] == sol[p]).all(1)) & (o[1] < THETA)).sum()) for p, o in enumerate(res))
    solved = acc_i >= 0
    # compute: (i) table check costs nothing per candidate; (ii) the worker's accounting: one call of M_A per one-cell context read, with early stop
    used = [ck.pair_score(Q, o[0], stop_below=THETA)[1].numpy() for o in res]
    fl_table = np.array([(i + 1) * nfree[p] * FLP if i >= 0 else np.inf for p, i in enumerate(acc_i)])
    fl_calls = np.array([((i + 1) * nfree[p] * FLP + used[p][:i + 1].sum() * FLM) if i >= 0 else np.inf for p, i in enumerate(acc_i)])
    R = dict(arm=arm, n=n, accepted=float(solved.mean()), accepted_correct=float(ans_ok[solved].mean()), wrong_answers=int((solved & ~ans_ok).sum()),
             first_correct=float(first_ok.mean()), cands_mean=float((acc_i[solved] + 1).mean()), candidates=int(ncand.sum()), wrong_candidates=wrong,
             wrong_accepted_pair=wrong_acc, right_rejected_pair=right_rej, flops_table_mean=float(fl_table[solved].mean()), flops_table_median=float(np.median(fl_table[solved])),
             flops_calls_mean=float(fl_calls[solved].mean()), flops_calls_median=float(np.median(fl_calls[solved])), pair_calls_per_wrong=float(np.mean(np.concatenate([u[:-1] if solved[p] else u for p, u in enumerate(used)])) if wrong else 0))
    log(f"== {arm} on the first {n} HARD puzzles, fresh seed, pair check from the table (theta {THETA:.3f})")
    log(f"   accepted within 256 candidates {R['accepted']:.4f} ({int(solved.sum())} of {n}); accepted answers that are the solution {R['accepted_correct']:.4f} (wrong {R['wrong_answers']}); "
        f"candidate 0 correct {R['first_correct']:.4f}; candidates to accept {R['cands_mean']:.2f}")
    log(f"   candidates {R['candidates']}, wrong {wrong}, wrong accepted by the pair check {wrong_acc}; solution candidates rejected {right_rej}")
    log(f"   FLOPs to accept, proposals only (check read from the 216-context table): mean {R['flops_table_mean']:.3e} median {R['flops_table_median']:.3e}; "
        f"counting one call per context read: mean {R['flops_calls_mean']:.3e} median {R['flops_calls_median']:.3e} (contexts read per wrong candidate {R['pair_calls_per_wrong']:.1f})")
    marks = [1e9, 3e9, 6.011e9, 1e10, 3e10, 1e11]
    f0 = nfree * FLP
    any_t = [float((((b >= fl_table) & ans_ok) | ((b >= f0) & (b < fl_table) & first_ok)).mean()) for b in marks]
    any_c = [float((((b >= fl_calls) & ans_ok) | ((b >= f0) & (b < fl_calls) & first_ok)).mean()) for b in marks]
    log("   anytime accuracy at budgets " + " ".join(f"{m:.2e}" for m in marks))
    log("      table check:              " + " ".join(f"{v:8.4f}" for v in any_t)); log("      one call per context read: " + " ".join(f"{v:8.4f}" for v in any_c))
    R["anytime_marks"] = marks; R["anytime_table"] = any_t; R["anytime_calls"] = any_c
    if route is not None:                                                    # the pre-registered route test on the same candidate stream
        ri = np.array([int((o[2] >= TAU).float().argmax()) if (o[2] >= TAU).any() else -1 for o in res])
        r_ok = np.array([bool((o[0][i] == sol[p]).all()) if i >= 0 else False for p, (o, i) in enumerate(zip(res, ri))]); rs = ri >= 0
        wa = sum(int((((o[0] != sol[p]).any(1)) & (o[2] >= TAU)).sum()) for p, o in enumerate(res)); rr = sum(int((((o[0] == sol[p]).all(1)) & (o[2] < TAU)).sum()) for p, o in enumerate(res))
        fl_r = np.array([((i + 1) * nfree[p] * FLP + res[p][3][:i + 1].sum().item() * (FLM + FLT)) if i >= 0 else np.inf for p, i in enumerate(ri)])
        any_r = [float((((b >= fl_r) & r_ok) | ((b >= f0) & (b < fl_r) & first_ok)).mean()) for b in marks]
        R.update(route_accepted=float(rs.mean()), route_wrong_answers=int((rs & ~r_ok).sum()), route_wrong_accepted=wa, route_right_rejected=rr,
                 route_flops_mean=float(fl_r[rs].mean()), route_flops_median=float(np.median(fl_r[rs])), anytime_route=any_r)
        log(f"   same candidates, pre-registered route threshold (tau {TAU:.2f}): accepted {rs.mean():.4f}; puzzles ending on a wrong accepted answer {R['route_wrong_answers']} of {n}; "
            f"wrong candidates above tau {wa} of {wrong} (seen before the pair check stopped the puzzle); solution candidates below tau {rr}")
        log(f"      FLOPs to accept mean {R['route_flops_mean']:.3e} median {R['route_flops_median']:.3e}; anytime accuracy: " + " ".join(f"{v:8.4f}" for v in any_r))
    torch.save(dict(acc_i=acc_i, ans_ok=ans_ok, first_ok=first_ok, fl_table=fl_table, fl_calls=fl_calls, answers=[o[0][i] if i >= 0 else None for o, i in zip(res, acc_i)]), f"out/c3_search_{arm}.pt")
    json.dump(R, open(f"out/c3_search_{arm}.json", "w"), indent=1); log(f"calls: proposal {prop.calls}, M_A {MA.calls}; total {time.time()-T0:.0f}s")


def first_error(net, order, puz, sol, flag):
    o = ck.walk(net, order, puz, flag=flag, greedy=True, keep=True); P = len(puz)
    ok = (o["s"] == sol).all(1).numpy(); pw = np.full(P, np.nan); st = np.full(P, np.nan); nemp = np.full(P, np.nan); fw = np.full(P, np.nan); found = np.zeros(P, bool)
    for t, (ia, sa, lp, cell, v) in enumerate(o["trace"]):
        ar = torch.arange(len(ia)); wrong = (v != sol[ia, cell]).numpy()
        for b in np.where(wrong)[0]:
            p = int(ia[b])
            if not found[p]:
                found[p] = True; pw[p] = float(lp[b, cell[b], v[b]].exp()); st[p] = float(-lp[b, cell[b], sol[p, cell[b]]])
                e = sa[b] < 0; nemp[p] = int(e.sum()); fw[p] = float(((lp[b].argmax(-1) != sol[p]) & e).sum() / e.sum())
    return ok, pw, st, nemp, fw


if MODE == "base":
    puz, sol = load_hard(300); clue = puz >= 0; MA = ck.M("mA6", 6); MB = ck.M("mB6", 6); R = {}
    for lab, net, flag in (("M_A (grids only)", MA, None), ("M_B (puzzle-trained)", MB, clue)):
        for rule, order in (("margin", ck.o_margin), ("maxprob", ck.o_maxprob), ("entropy", ck.o_entropy)):
            ok, pw, st, nemp, fw = first_error(net, order, puz, sol, flag); f = ~ok
            R[f"{lab} {rule}"] = dict(acc=float(ok.mean()), fails=int(f.sum()), p_wrong_mean=float(np.nanmean(pw)), p_wrong_median=float(np.nanmedian(pw)),
                                      share_above_09=float((pw[f] >= 0.9).mean()), share_below_06=float((pw[f] < 0.6).mean()), surprise_true=float(np.nanmean(st)),
                                      empty=float(np.nanmean(nemp)), frac_top_wrong=float(np.nanmean(fw)))
            r = R[f"{lab} {rule}"]
            log(f"   one pass of {lab:22s} own confidence ({rule:7s}): correct {r['acc']:.4f}; fails on {r['fails']}. At its first wrong step: probability on the wrong value mean {r['p_wrong_mean']:.3f} "
                f"median {r['p_wrong_median']:.3f} (>= 0.9 on {r['share_above_09']:.2f} of failures, < 0.6 on {r['share_below_06']:.2f}); surprise at the true value {r['surprise_true']:.2f}; "
                f"empty cells {r['empty']:.1f}, top value wrong at {r['frac_top_wrong']:.3f} of them")
    # K2 with a fresh seed, and E4(e)
    gen = torch.Generator().manual_seed(8642); k = 32; rep = torch.arange(300).repeat_interleave(k - 1)
    g0 = ck.walk(MB, ck.o_margin, puz, flag=clue, greedy=True); gs = ck.walk(MB, ck.o_margin, puz[rep], flag=clue[rep], gen=gen)
    Xc = torch.cat([g0["s"][:, None], gs["s"].view(300, k - 1, N)], 1); lq = torch.cat([g0["logp"][:, None], gs["logp"].view(300, k - 1)], 1)
    right = (Xc == sol[:, None]).all(2)
    for kk in (1, 2, 4, 8, 16, 32):
        pick = lq[:, :kk].argmax(1); a = right[torch.arange(300), pick].float().mean().item()
        R[f"K2_{kk}"] = dict(acc=a, any=float(right[:, :kk].any(1).float().mean()))
        log(f"   K2 best of {kk:2d} passes of M_B by its own route log-probability (fresh draws): {a:.4f}; solution among them {R[f'K2_{kk}']['any']:.4f}; distinct boards per puzzle {np.mean([len(torch.unique(Xc[p, :kk], dim=0)) for p in range(300)]):.2f}")
    tB = ck.Trust(f"{ck.W}/ckpt/trust_mB6.pt", 6); g = ck.walk(MB, tB, puz, flag=clue, greedy=True); kt = (g["s"] == sol).all(1).numpy(); acc = (g["logp"] > math.log(0.5)).numpy()
    R["KT"] = dict(acc=float(kt.mean()), rule_accepts=int(acc.sum()), rule_wrong=int((acc & ~kt).sum()))
    log(f"   KT one pass of M_B in its trust order: {kt.mean():.4f}; rule 'T_B > 1/2' accepts {int(acc.sum())} of 300, of which wrong {int((acc & ~kt).sum())}")
    json.dump(R, open("out/c3_base.json", "w"), indent=1); log(f"total {time.time()-T0:.0f}s")

if MODE == "fresh":
    want = int(sys.argv[2]); ex = ck.Exact(6); PEER = ck.peers(6); rng = np.random.default_rng(424242)
    d = np.load(f"{ck.W}/data/k6_base.npz"); Bt = d["B"][d["base_split"] == 2].astype(np.int64)
    N_, r_, c_, b_ = ck.geom(6); units = [np.where(g == u)[0] for g in (r_, c_, b_) for u in range(6)]

    def count(s):
        return ck.count_fast(ex, s)
    _r = np.random.default_rng(1); _X = ck.sample_grids(6, 2, 40, 3).numpy()
    for _a in range(40):                                                    # the fast count against my full counter
        _s = -np.ones(N, dtype=np.int64); _i = _r.choice(N, int(_r.integers(6, 30)), replace=False); _s[_i] = _X[_a, _i]
        assert count(_s) == ex.cond(_s)[0]
        _s[_i[0]] = (_s[_i[0]] + 1) % 6; assert count(_s) == ex.cond(_s)[0]

    def singles_finish(p):
        s = p.copy()
        while True:
            cand = np.ones((N, K), bool)
            for i in np.where(s >= 0)[0]:
                cand[PEER[i], s[i]] = False
            prog = False
            for i in np.where(s < 0)[0]:
                if cand[i].sum() == 1:
                    s[i] = int(cand[i].argmax()); prog = True
            if not prog:
                for u in units:
                    for v in range(K):
                        if (s[u] == v).any():
                            continue
                        pos = [i for i in u if s[i] < 0 and cand[i, v]]
                        if len(pos) == 1:
                            s[pos[0]] = v; prog = True; break
                    if prog:
                        break
            if not prog:
                return bool((s >= 0).all())
    PZ, SL = [], []; made = 0
    while len(PZ) < want and made < 4000:
        g = rng.permutation(6)[Bt[rng.integers(len(Bt))]]; s = g.copy()
        for i in rng.permutation(N):
            v = s[i]; s[i] = -1
            if count(s) != 1:
                s[i] = v
        made += 1
        if not singles_finish(s):
            PZ.append(s.copy()); SL.append(g.copy())
        if made % 200 == 0:
            log(f"   generated {made} minimal puzzles, hard {len(PZ)} ({time.time()-T0:.0f}s)")
    puz = torch.from_numpy(np.array(PZ)); sol = torch.from_numpy(np.array(SL)); n = len(puz)
    old = set(map(bytes, np.load(f"{ck.W}/data/k6_test_min.npz")["puz"].astype(np.int8)))
    log(f"== {n} new hard puzzles from {made} minimal ones ({100*n/made:.1f}% hard; the worker's rate 5.2%); clues {np.bincount((puz >= 0).sum(1).numpy()).tolist()[8:]}; "
        f"also in the worker's test file: {len(old & set(map(bytes, np.array(PZ).astype(np.int8))))}")
    np.savez("out/c3_fresh_puzzles.npz", puz=np.array(PZ), sol=np.array(SL))
    MA = ck.M("mA6", 6); MB = ck.M("mB6", 6); Q = ck.pair_table(MA); clue = puz >= 0
    k0 = (ck.walk(MA, ck.o_margin, puz, greedy=True)["s"] == sol).all(1).float().mean().item()
    k1 = {r: (ck.walk(MB, o, puz, flag=clue, greedy=True)["s"] == sol).all(1).float().mean().item() for r, o in (("margin", ck.o_margin), ("entropy", ck.o_entropy), ("maxprob", ck.o_maxprob))}
    log(f"   one pass, no check: M_A {k0:.4f}; M_B margin {k1['margin']:.4f}, entropy {k1['entropy']:.4f}, top probability {k1['maxprob']:.4f}")
    gen = torch.Generator().manual_seed(11235); res = run_search(MA, ck.o_margin, puz, None, Q, gen)
    acc_i = np.array([int((o[1] >= THETA).float().argmax()) if (o[1] >= THETA).any() else -1 for o in res])
    ok = np.array([bool((o[0][i] == sol[p]).all()) if i >= 0 else False for p, (o, i) in enumerate(zip(res, acc_i))]); solved = acc_i >= 0
    wrong = sum(int(((o[0] != sol[p]).any(1)).sum()) for p, o in enumerate(res)); wa = sum(int((((o[0] != sol[p]).any(1)) & (o[1] >= THETA)).sum()) for p, o in enumerate(res))
    log(f"   S1 with the pair check: accepted {int(solved.sum())} of {n}; accepted answers that are the solution {int(ok[solved].sum())} of {int(solved.sum())}; "
        f"candidates to accept {np.mean(acc_i[solved]+1):.2f}; wrong candidates {wrong}, accepted {wa}")
    json.dump(dict(n=n, made=made, k0=k0, k1=k1, accepted=int(solved.sum()), correct=int(ok[solved].sum()), wrong=wrong, wrong_accepted=wa, cands=float(np.mean(acc_i[solved] + 1))),
              open("out/c3_fresh.json", "w"), indent=1); log(f"total {time.time()-T0:.0f}s")

if MODE == "conf":
    # per-atom calibration on PUZZLE contexts (clue state plus correct fills, i.e. real contexts), for both networks.
    # every such state has exactly one completion, so the exact probability of any value other than the solution's is 0.
    z = np.load(f"{ck.W}/data/k6_test_min.npz"); R = {}
    sets = {"HARD (300)": np.where(z["rating"] == 2)[0][:300], "ALL (500 minimal)": np.arange(500)}
    MA = ck.M("mA6", 6); MB = ck.M("mB6", 6)
    for sname, idx in sets.items():
        puz = torch.from_numpy(z["puz"][idx].astype(np.int64)); sol = torch.from_numpy(z["sol"][idx].astype(np.int64)); clue = puz >= 0
        for lab, net, flag in (("M_A (grids only)", MA, None), ("M_B (puzzle-trained)", MB, clue)):
            o = ck.walk(net, ck.o_margin, puz, X=sol, flag=flag, keep=True); qs, oks = [], []
            for (ia, sa, lp, cell, v) in o["trace"]:
                e = sa < 0; top = lp.max(-1); qs.append(top.values.exp()[e].numpy()); oks.append((top.indices == sol[ia])[e].numpy())
            q = np.concatenate(qs); ok = np.concatenate(oks); c9 = q >= 0.9; c99 = q >= 0.99
            bins = [(0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01)]
            cal = [(lo, hi, int(((q >= lo) & (q < hi)).sum()), float(ok[(q >= lo) & (q < hi)].mean()) if ((q >= lo) & (q < hi)).any() else None) for lo, hi in bins]
            R[f"{sname} {lab}"] = dict(atoms=int(len(q)), conf=int(c9.sum()), conf_wrong=int((c9 & ~ok).sum()), conf99=int(c99.sum()), conf99_wrong=int((c99 & ~ok).sum()), cal=cal)
            log(f"   {sname:18s} {lab:22s}: atoms {len(q)}; top probability >= 0.9: {int(c9.sum())}, of which the top value is wrong (exact probability 0): {int((c9 & ~ok).sum())} "
                f"({(c9 & ~ok).sum()/max(1,c9.sum()):.2e}); >= 0.99: {int(c99.sum())}, wrong {int((c99 & ~ok).sum())} ({(c99 & ~ok).sum()/max(1,c99.sum()):.2e})")
            log("      share of top values that are right, by M's top probability: " + "; ".join(f"{lo:.2f}-{min(hi,1):.2f}: {a:.3f} (n={n})" if a is not None else f"{lo:.2f}-{min(hi,1):.2f}: none" for lo, hi, n, a in cal))
    json.dump(R, open("out/c3_conf.json", "w"), indent=1); log(f"total {time.time()-T0:.0f}s")

if MODE == "valorbits":
    # the same three routes on fresh grids of the VALIDATION orbits and of the TRAINING orbits, with orbit ids kept
    net = ck.M("mA6", 6); trust = ck.Trust(f"{ck.W}/ckpt/trust_mA6.pt", 6)
    d = np.load(f"{ck.W}/data/k6_base.npz"); bs = d["base_split"]; bo = d["base_orbit"]; OUTV = {}
    for sp, nm, seed, n in ((1, "validation orbits", 7101, 600), (0, "training orbits", 7102, 600)):
        rng = np.random.default_rng(seed); idx = rng.integers(int((bs == sp).sum()), size=n); orb = bo[bs == sp][idx]
        X = ck.sample_grids(6, sp, n, seed)                          # same generator calls, so the same grids as idx
        assert (np.sort(np.unique(orb)) == np.sort(np.unique(orb))).all()
        V = {l: -ck.walk(net, p, torch.full_like(X, -1), X=X)["logp"].numpy() for l, p in (("entropy", ck.o_entropy), ("raster", ck.o_raster), ("pi_trust", trust))}
        np.savez(f"out/c3_valorbits_{sp}.npz", orb=orb, **V)
        log(f"== {n} fresh grids of the {nm} ({len(np.unique(orb))} orbits)")
        for l, v in V.items():
            m, lo, hi = ck.boot(v - HREAL); log(f"   {l:10s} excess {m:.4f} [{lo:.4f}, {hi:.4f}] (interval resampling grids)")
        for a, b in (("entropy", "pi_trust"), ("raster", "pi_trust")):
            m, lo, hi = ck.boot(V[a] - V[b], seed=1); log(f"   {a} minus {b}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        log("   per orbit (n; entropy, raster, pi_trust): " + "; ".join(f"{o}: n={int((orb == o).sum())}; " + ", ".join(f"{V[l][orb == o].mean() - HREAL:.2f}" for l in V) for o in np.unique(orb)))
    log(f"total {time.time()-T0:.0f}s")

if MODE == "population":
    # KL(reality || T_pi) is an average over ALL of reality (49 orbits), not over the 9 test orbits. Uniform sample of all grids.
    net = ck.M("mA6", 6); trust = ck.Trust(f"{ck.W}/ckpt/trust_mA6.pt", 6)
    d = np.load(f"{ck.W}/data/k6_base.npz"); B = d["B"].astype(np.int64); bs = d["base_split"]; bo = d["base_orbit"]
    rng = np.random.default_rng(31415); n = 900; idx = rng.integers(len(B), size=n)
    X = torch.from_numpy(np.take_along_axis(np.stack([rng.permutation(6) for _ in range(n)]), B[idx], 1)); sp = bs[idx]; orb = bo[idx]
    assert ck.is_valid(X.numpy(), 6).all()
    POL = {"own confidence: lowest entropy": ck.o_entropy, "own confidence: top probability": ck.o_maxprob, "own confidence: top-1 minus top-2": ck.o_margin,
           "least confident first": ck.o_anti, "raster order": ck.o_raster, "pi_trust": trust}
    V = {}
    for l, p in POL.items():
        V[l] = -ck.walk(net, p, torch.full_like(X, -1), X=X)["logp"].numpy(); log(f"   {l:36s} done ({time.time()-T0:.0f}s)")
    fit = np.arange(n) < 300; ev = ~fit; labs = list(POL)                                 # mixture weights on the first 300, everything scored on the other 600
    LV = -np.stack([V[l][fit] for l in labs], 1); w = np.full(len(labs), 1 / len(labs))
    for _ in range(500):
        a = LV + np.log(w + 1e-300); a -= a.max(1, keepdims=True); r_ = np.exp(a); r_ /= r_.sum(1, keepdims=True); w = r_.mean(0)
    LT = -np.stack([V[l] for l in labs], 1); V["learned mixture (EM on 300 of these grids)"] = -(np.log((np.exp(LT - LT.max(1, keepdims=True)) * w).sum(1)) + LT.max(1))
    log(f"== {int(ev.sum())} grids drawn uniformly from ALL 28,200,960 grids (training / validation / test orbits: {np.bincount(sp[ev], minlength=3).tolist()}); "
        f"this mean is the KL(reality || T) of the chain rule")
    base = V["pi_trust"][ev]; R = {}
    for l, v in V.items():
        m, lo, hi = ck.boot(v[ev] - HREAL); dm, dlo, dhi = ck.boot(v[ev] - base, seed=1)
        R[l] = dict(kl=m, lo=lo, hi=hi, d=dm, dlo=dlo, dhi=dhi, by_split=[float(v[ev][sp[ev] == k].mean() - HREAL) for k in range(3)])
        log(f"   {l:44s} {m:7.4f} [{lo:7.4f},{hi:7.4f}]  minus pi_trust {dm:+.4f} [{dlo:+.4f},{dhi:+.4f}]  by orbit set (train, val, test): " + ", ".join(f"{x:.3f}" for x in R[l]["by_split"]))
    log("   mixture weights: " + ", ".join(f"{l.split(':')[-1].strip()[:20]} {x:.3f}" for l, x in zip(labs, w)))
    np.savez("out/c3_population.npz", sp=sp, orb=orb, **{k.replace(" ", "_").replace(":", "").replace("(", "").replace(")", ""): v for k, v in V.items()})
    json.dump(R, open("out/c3_population.json", "w"), indent=1); log(f"total {time.time()-T0:.0f}s")


def my_hard(p, PEER, units):
    s = p.copy()
    while True:
        cand = np.ones((N, K), bool)
        for i in np.where(s >= 0)[0]:
            cand[PEER[i], s[i]] = False
        prog = False
        for i in np.where(s < 0)[0]:
            if cand[i].sum() == 1:
                s[i] = int(cand[i].argmax()); prog = True
        if not prog:
            for u in units:
                for v in range(K):
                    if (s[u] == v).any():
                        continue
                    pos = [i for i in u if s[i] < 0 and cand[i, v]]
                    if len(pos) == 1:
                        s[pos[0]] = v; prog = True; break
                if prog:
                    break
        if not prog:
            return not bool((s >= 0).all())


if MODE == "rater":
    # is my "hard" (naked and hidden singles do not finish) the worker's rating 2, on the worker's own file?
    PEER = ck.peers(6); N_, r_, c_, b_ = ck.geom(6); units = [np.where(g == u)[0] for g in (r_, c_, b_) for u in range(6)]
    z = np.load(f"{ck.W}/data/k6_test_min.npz"); puz = z["puz"].astype(np.int64)[:2500]; rt = z["rating"][:2500]
    mine = np.array([my_hard(p, PEER, units) for p in puz])
    log(f"   first 2,500 puzzles of the worker's test file: worker says hard {int((rt == 2).sum())}; I say hard {int(mine.sum())}; disagreements {int((mine != (rt == 2)).sum())}")
    f = np.load("out/c3_fresh_puzzles.npz"); ex = ck.Exact(6)
    log(f"   my 100 fresh puzzles: unique solution by my full counter {sum(int(ex.cond(p)[0] == 1) for p in f['puz'])} of {len(f['puz'])}; "
        f"minimal (removing any clue gives more than one solution) checked on 20: {sum(all(ex.cond(np.where(np.arange(N) == i, -1, p))[0] > 1 for i in np.where(p >= 0)[0]) for p in f['puz'][:20])} of 20")

if MODE == "fresh2":
    want = int(sys.argv[2]); ex = ck.Exact(6); PEER = ck.peers(6); rng = np.random.default_rng(424243)
    d = np.load(f"{ck.W}/data/k6_base.npz"); Bt = d["B"][d["base_split"] == 2].astype(np.int64)
    N_, r_, c_, b_ = ck.geom(6); units = [np.where(g == u)[0] for g in (r_, c_, b_) for u in range(6)]
    PZ, SL = [], []; made = 0
    while len(PZ) < want and made < 6000:
        g = rng.permutation(6)[Bt[rng.integers(len(Bt))]]; s = g.copy()
        for i in rng.permutation(N):
            v = s[i]; s[i] = -1
            if ck.count_fast(ex, s) != 1:
                s[i] = v
        made += 1
        if my_hard(s, PEER, units):
            PZ.append(s.copy()); SL.append(g.copy())
        if made % 400 == 0:
            log(f"   generated {made} minimal puzzles, hard {len(PZ)} ({time.time()-T0:.0f}s)")
    f = np.load("out/c3_fresh_puzzles.npz"); P2 = np.concatenate([f["puz"], np.array(PZ)]); S2 = np.concatenate([f["sol"], np.array(SL)])
    np.savez("out/c3_fresh_puzzles_all.npz", puz=P2, sol=S2)
    puz = torch.from_numpy(P2); sol = torch.from_numpy(S2); n = len(puz); clue = puz >= 0; nfree = (puz < 0).sum(1).numpy()
    log(f"== {n} fresh hard puzzles in all ({len(PZ)} new from {made} minimal ones, {100*len(PZ)/made:.1f}% hard); test orbits, my generator")
    MA = ck.M("mA6", 6); MB = ck.M("mB6", 6); Q = ck.pair_table(MA); R = {"n": n}
    k0 = {r: (ck.walk(MA, o, puz, greedy=True)["s"] == sol).all(1).numpy() for r, o in (("margin", ck.o_margin), ("entropy", ck.o_entropy), ("maxprob", ck.o_maxprob))}
    k1 = {r: (ck.walk(MB, o, puz, flag=clue, greedy=True)["s"] == sol).all(1).numpy() for r, o in (("margin", ck.o_margin), ("entropy", ck.o_entropy), ("maxprob", ck.o_maxprob))}
    for r in k0:
        a, b = k0[r], k1[r]; ao, bo_ = int((a & ~b).sum()), int((b & ~a).sum()); m = ao + bo_
        pval = min(1.0, 2 * sum(math.comb(m, i) for i in range(0, min(ao, bo_) + 1)) / 2 ** m) if m else 1.0
        R[r] = dict(MA=float(a.mean()), MB=float(b.mean()), only_MA=ao, only_MB=bo_, p=pval)
        log(f"   one pass, rule {r:8s}: M_A {a.mean():.4f}, M_B {b.mean():.4f}; only M_A right {ao}, only M_B right {bo_}, exact sign test p = {pval:.3f}")
    gen = torch.Generator().manual_seed(2718); res = []
    for a in range(0, n, 50):
        res += run_search(MA, ck.o_margin, puz[a:a + 50], None, Q, gen)
    acc_i = np.array([int((o[1] >= THETA).float().argmax()) if (o[1] >= THETA).any() else -1 for o in res]); solved = acc_i >= 0
    ok = np.array([bool((o[0][i] == sol[p]).all()) if i >= 0 else False for p, (o, i) in enumerate(zip(res, acc_i))])
    first_ok = np.array([bool((o[0][0] == sol[p]).all()) for p, o in enumerate(res)])
    used = [ck.pair_score(Q, o[0], stop_below=THETA)[1].numpy() for o in res]; FLM = MA.flops
    fl_c = np.array([((i + 1) * nfree[p] * FLM + used[p][:i + 1].sum() * FLM) if i >= 0 else np.inf for p, i in enumerate(acc_i)])
    fl_t = np.array([(i + 1) * nfree[p] * FLM if i >= 0 else np.inf for p, i in enumerate(acc_i)]); f0 = nfree * FLM
    marks = [6.011e9, 1.202e10, 3e10, 1e11]
    any_c = [float((((b >= fl_c) & ok) | ((b >= f0) & (b < fl_c) & first_ok)).mean()) for b in marks]
    any_t = [float((((b >= fl_t) & ok) | ((b >= f0) & (b < fl_t) & first_ok)).mean()) for b in marks]
    wrong = sum(int(((o[0] != sol[p]).any(1)).sum()) for p, o in enumerate(res)); wa = sum(int((((o[0] != sol[p]).any(1)) & (o[1] >= THETA)).sum()) for p, o in enumerate(res))
    log(f"   S1 with the pair check: accepted {int(solved.sum())} of {n}; all accepted answers are the solution: {bool(ok[solved].all())}; candidates to accept {np.mean(acc_i[solved]+1):.2f}; wrong candidates {wrong}, accepted {wa}")
    log(f"   anytime accuracy at 6.0e9 / 1.2e10 / 3e10 / 1e11 FLOPs: one call per context read {any_c}; table check {any_t}")
    R.update(accepted=int(solved.sum()), all_correct=bool(ok[solved].all()), cands=float(np.mean(acc_i[solved] + 1)), any_calls=any_c, any_table=any_t, wrong=wrong, wrong_accepted=wa)
    json.dump(R, open("out/c3_fresh2.json", "w"), indent=1); log(f"total {time.time()-T0:.0f}s")
