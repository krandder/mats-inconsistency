"""c2_4x4.py (checker) -- 4x4, exact, my own code. The worker's e_4x4.py is not imported.

  python -u c2_4x4.py sym     tables for every order policy; trust MDP solved with the symmetry reduction (reality = all 288,
                              the training orbit, the held-out orbit); tree walk of T; what the optimal route looks like
  python -u c2_4x4.py full    trust MDP solved on EVERY reachable state with no symmetry argument (15.8M network calls),
                              for the same three realities and for finite real samples drawn with replacement
Trust MDP: state = partial filling of a real grid; action = the cell reality reveals next; cost = M's surprise.
"""
import sys, time, math, json
import numpy as np, torch
import ck

MODE = sys.argv[1]; K, N = 4, 16; T0 = time.time(); LN288 = math.log(288)
d = np.load(f"{ck.W}/data/k4.npz"); G = d["G"].astype(np.int64); orb_file = d["orbit"]
assert G.shape == (288, 16) and ck.is_valid(G, 4).all() and len(np.unique(G, axis=0)) == 288
net = ck.M("mA4", 4); Gt = torch.from_numpy(G)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- my own orbits (cell symmetries incl. transpose, x relabelling)
def line_perms():
    out = []
    for swap in (0, 1):
        for a in (0, 1):
            for b in (0, 1):
                bands = [[0, 1], [2, 3]]
                if a: bands[0] = bands[0][::-1]
                if b: bands[1] = bands[1][::-1]
                if swap: bands = bands[::-1]
                out.append(bands[0] + bands[1])
    return out


LP = line_perms()
SYM = np.array([[rp[r] * 4 + cp[c] for r in range(4) for c in range(4)] for rp in LP for cp in LP])          # 64, keep rows rows
SYMT = np.array([[cp[c] * 4 + rp[r] for r in range(4) for c in range(4)] for rp in LP for cp in LP])         # with transposition
gkey = (G << (2 * np.arange(N))[None, :]).sum(1); gsort = np.argsort(gkey)


def gindex(X):
    k = (np.asarray(X) << (2 * np.arange(N))[None, :]).sum(1); j = np.searchsorted(gkey[gsort], k)
    assert (gkey[gsort][j] == k).all(); return gsort[j]


import itertools
lab = -np.ones(288, dtype=int); n_orb = 0
for g0 in range(288):
    if lab[g0] >= 0:
        continue
    members = set()
    for t in np.concatenate([SYM, SYMT]):
        x = G[g0][t]
        for p in itertools.permutations(range(4)):
            members.add(int(gindex(np.array(p)[x][None])[0]))
    lab[list(members)] = n_orb; n_orb += 1
sizes = np.bincount(lab).tolist()
train = np.where(orb_file == 1)[0]; held = np.where(orb_file == 0)[0]
assert n_orb == 2 and sorted(sizes) == [96, 192]
assert len(set(lab[train])) == 1 and len(set(lab[held])) == 1 and lab[train[0]] != lab[held[0]]
log(f"== 4x4: 288 valid distinct grids; my orbit computation: {n_orb} orbits of sizes {sizes}; the worker's training set is one whole orbit "
    f"({len(train)} grids) and the held-out set is the other ({len(held)}); network {net.name} has {net.params} parameters")
SETS = {"all288": np.arange(288), "heldout96": held, "train192": train}

# ---------------------------------------------------------------- order policies and exact tables
def fixed_order(perm):
    rank = torch.empty(N); rank[torch.as_tensor(perm)] = torch.arange(N).float()
    return lambda s, lp: rank.expand(s.shape[0], -1).clone()


def logT_all(order):
    return ck.walk(net, order, torch.full_like(Gt, -1), X=Gt)["logp"].numpy()


def row(lt):
    return dict(valid_mass=float(np.exp(lt).sum()), **{k: float(-lt[v].mean() - LN288) for k, v in SETS.items()})


trustpol = ck.Trust(f"{ck.W}/ckpt/trust_mA4.pt", 4)
POL = {"own confidence: lowest entropy": ck.o_entropy, "own confidence: top probability": ck.o_maxprob,
       "own confidence: top-1 minus top-2": ck.o_margin, "least confident first": ck.o_anti, "raster order": ck.o_raster,
       "pi_trust (the worker's learned net)": trustpol}
RES = {}; LT = {}


def show(label):
    r = RES[label]; log(f"   {label:74s} {r['all288']:9.4f} {r['heldout96']:9.4f} {r['train192']:9.4f} {r['valid_mass']:10.6f}")


# ---------------------------------------------------------------- table policy from a DP
class TablePol:
    """order policy read from a DP table; a state that matches no grid of the table's reality falls back to M's
    lowest-entropy cell. full: pol[mask, grid]. sym: pol[canonical mask -> row][grid], looked up through the symmetry."""

    def __init__(self, pol, sym=None):
        self.pol = pol; self.sym = sym; self.on = 0; self.off = 0; self.w = (1 << torch.arange(N))

    def __call__(self, s, lp):
        g = torch.full((len(s),), -1, dtype=torch.long)
        for a in range(0, len(s), 8192):
            sc = s[a:a + 8192]; match = ((sc[:, None, :] == Gt[None]) | (sc[:, None, :] < 0)).all(2)
            g[a:a + 8192] = torch.where(match.any(1), match.float().argmax(1), torch.full((len(sc),), -1))
        m = ((s >= 0).long() * self.w).sum(1).numpy(); gn = g.numpy(); has = gn >= 0
        cell = -np.ones(len(s), dtype=np.int64)
        if self.sym is None:
            cell[has] = self.pol[m[has], gn[has]]
        else:
            CM, TI, PERM, rowof = self.sym; t = TI[m[has]]; ic = self.pol[rowof[CM[m[has]]], PERM[t, gn[has]]]
            cell[has] = np.where(ic >= 0, SYM[t, np.maximum(ic, 0)], -1)
        ok = cell >= 0; self.on += int(ok.sum()); self.off += int((~ok).sum())
        score = ck.o_entropy(s, lp); okt = torch.from_numpy(ok)
        score[okt] = 1e6; score[torch.where(okt)[0], torch.from_numpy(cell[ok])] = -1e6
        return score


def dp(wsets, sym):
    """exact DP of the trust MDP for several realities at once (rows of wsets: weight of each of the 288 grids).
    Returns V at the empty state per reality, and the policy table."""
    Wt = np.stack(list(wsets.values())).astype(np.float64); nw = len(Wt)
    allm = np.arange(1 << N, dtype=np.int64); pc = np.array([bin(m).count("1") for m in range(1 << N)])
    ck3 = (G + 1) << (3 * np.arange(N))[None, :]
    if sym:
        TM = np.stack([sum((((allm >> int(t[i])) & 1) << i) for i in range(N)) for t in SYM])
        CM = TM.min(0); TI = TM.argmin(0); PERM = np.stack([gindex(G[:, t]) for t in SYM])
        for a, t in enumerate(SYM):
            assert (G[PERM[a]] == G[:, t]).all()
        for w in Wt:                                         # the reduction needs a reality that the symmetries preserve
            assert all(np.allclose(w[PERM[a]], w) for a in range(len(SYM)))
        masks = np.unique(CM)
    else:
        masks = allm
    rowof = -np.ones(1 << N, dtype=np.int64); rowof[masks] = np.arange(len(masks))
    pol = -np.ones((nw, len(masks), 288), dtype=np.int8)
    Vnext = None; nstates = 0; V0 = None
    for k in range(N, -1, -1):
        ms = masks[pc[masks] == k]; Vcur = np.zeros((len(ms), nw, 288)); pos = {int(m): j for j, m in enumerate(ms)}
        if k < N:
            for a in range(0, len(ms), 96):
                sub = ms[a:a + 96]; S, INV, OFF = [], [], [0]
                for m in sub:
                    bits = [i for i in range(N) if (int(m) >> i) & 1]
                    key = ck3[:, bits].sum(1) if bits else np.zeros(288, dtype=np.int64)
                    u, first, inv = np.unique(key, return_index=True, return_inverse=True)
                    st = -np.ones((len(u), N), dtype=np.int64)
                    if bits:
                        st[:, bits] = G[first][:, bits]
                    S.append(st); INV.append(inv); OFF.append(OFF[-1] + len(u))
                lp = net.logp(torch.from_numpy(np.concatenate(S))).numpy().astype(np.float64); nstates += OFF[-1]
                for j, m in enumerate(sub):
                    m = int(m); inv = INV[j]; nu = OFF[j + 1] - OFF[j]
                    cost = -lp[OFF[j]:OFF[j + 1]][inv[:, None], np.arange(N)[None, :], G]          # [288, 16]
                    order = np.argsort(inv, kind="stable"); starts = np.searchsorted(inv[order], np.arange(nu))
                    cnt = np.add.reduceat(Wt[:, order], starts, axis=1)                             # [nw, nu]
                    best = np.full((nw, nu), np.inf); bi = -np.ones((nw, nu), dtype=np.int64)
                    for i in range(N):
                        if (m >> i) & 1:
                            continue
                        m2 = m | (1 << i)
                        if sym:
                            vch = Vnext[posn[int(CM[m2])]][:, PERM[TI[m2]]]                         # [nw, 288]
                        else:
                            vch = Vnext[posn[m2]]
                        tot = (cost[None, :, i] + vch) * Wt
                        q = np.add.reduceat(tot[:, order], starts, axis=1) / np.maximum(cnt, 1e-300)
                        better = (q < best - 1e-12) & (cnt > 0)
                        best = np.where(better, q, best); bi = np.where(better, i, bi)
                    best = np.where(cnt > 0, best, 0.0)
                    Vcur[pos[m]] = best[:, inv]; pol[:, rowof[m]] = bi[:, inv]
        Vnext = Vcur; posn = pos
        if k % 2 == 0 or k < 3:
            log(f"      DP level {k:2d}: {len(ms)} masks; states evaluated so far {nstates} ({time.time()-T0:.0f}s)")
    V0 = Vcur[0]                                                                                      # [nw, 288] at the empty state
    extra = (CM, TI, PERM, rowof) if sym else None
    return V0, pol, nstates, extra


def equivariance(n=600, seed=0):
    g = torch.Generator().manual_seed(seed); x = Gt[torch.randint(0, 288, (n,), generator=g)]
    x[n // 2:] = torch.randint(0, 4, (n - n // 2, N), generator=g)                                    # half the states are off the real support
    s = torch.where(torch.rand(n, N, generator=g) < 0.5, x, torch.full_like(x, -1))
    t = torch.from_numpy(SYM[np.random.default_rng(seed).integers(len(SYM), size=n)])
    a = net.logp(torch.gather(s, 1, t)); b = torch.gather(net.logp(s), 1, t[:, :, None].expand(-1, -1, a.shape[-1]))
    return float((a - b).abs().max())


def tree_walk(order, thr=1e-9):
    S = torch.full((1, N), -1, dtype=torch.long); L = torch.zeros(1, dtype=torch.float64); pruned = 0.0; maxfront = 1
    for depth in range(N):
        lp = net.logp(S); sc = order(S, lp).masked_fill(S >= 0, float("inf")); cell = sc.argmin(1)
        lpc = lp[torch.arange(len(S)), cell][:, :K].double()
        newl = (L[:, None] + lpc).reshape(-1); S2 = S.repeat_interleave(K, 0)
        S2[torch.arange(len(S2)), cell.repeat_interleave(K)] = torch.arange(K).repeat(len(S))
        keep = newl >= math.log(thr); pruned += float(newl[~keep].exp().sum()); S, L = S2[keep], newl[keep]
        maxfront = max(maxfront, len(S))
    return S.numpy(), L.numpy(), pruned, maxfront


def exact_cond(s):
    """exact conditionals at a 4x4 state by listing the 288 grids."""
    match = ((G == s[None]) | (s[None] < 0)).all(1); P = np.zeros((N, K))
    for v in range(K):
        P[:, v] = (G[match] == v).sum(0)
    return P / max(1, match.sum()), int(match.sum())


def describe(order, label):
    """route of every grid under `order`: type of each chosen cell and the exact excess paid there."""
    PEER = ck.peers(4); out = ck.walk(net, order, torch.full_like(Gt, -1), X=Gt, keep=True)
    typ, exc, step, forced_avail = [], [], [], []
    for t, (ia, sa, lp, cell, v) in enumerate(out["trace"]):
        sa = sa.numpy(); lpn = lp.numpy().astype(np.float64); cell = cell.numpy()
        for b in range(len(sa)):
            s = sa[b]; P, _ = exact_cond(s); emp = np.where(s < 0)[0]
            loc = np.array([[not any(s[j] == vv for j in np.where(PEER[i])[0]) for vv in range(K)] for i in emp])     # [e, K]
            nloc = dict(zip(emp.tolist(), loc.sum(1).tolist())); c = int(cell[b]); p = P[c]
            ty = 0 if nloc[c] == 1 else (1 if p.max() == 1.0 else 2)
            typ.append(ty); step.append(t); forced_avail.append(any(n == 1 for n in nloc.values()))
            exc.append(float(np.where(p > 0, p * (np.log(np.where(p > 0, p, 1)) - lpn[b, c, :K]), 0.0).sum()))
    typ, exc, step, fa = map(np.array, (typ, exc, step, forced_avail))
    r = dict(free_per_route=float((typ == 2).sum() / 288), mean_step_free=float(step[typ == 2].mean()),
             forced_taken_when_avail=float((typ[fa] == 0).mean()), excess_total=float(exc.sum() / 288),
             excess_free=float(exc[typ == 2].sum() / 288), excess_forced_nonlocal=float(exc[typ == 1].sum() / 288),
             excess_forced_local=float(exc[typ == 0].sum() / 288), route_minus_ln288=float(-out["logp"].mean() - LN288))
    log(f"   {label}: free choices per route {r['free_per_route']:.2f}, at mean step {r['mean_step_free']:.2f}; a locally forced cell is taken "
        f"{r['forced_taken_when_avail']:.3f} of the times one exists; exact excess: free {r['excess_free']:.4f}, forced-not-locally "
        f"{r['excess_forced_nonlocal']:.4f}, locally forced {r['excess_forced_local']:.4f}, total {r['excess_total']:.4f} "
        f"(route surprise minus ln 288: {r['route_minus_ln288']:.4f})")
    return r


if MODE == "sym":
    log("\n[1] every order policy as a definition of T, exact on all 288 grids. excess = mean_x[-log T(x)] - ln 288 (nats)")
    log(f"   {'order policy':74s} {'all 288':>9s} {'held-out':>9s} {'train':>9s} {'valid mass':>10s}")
    for lab_, p in POL.items():
        LT[lab_] = logT_all(p); RES[lab_] = row(LT[lab_]); show(lab_)
    R = 256; rng = np.random.default_rng(20261110); lts = np.array([logT_all(fixed_order(rng.permutation(N))) for _ in range(R)])
    mix = np.log(np.exp(lts).mean(0)); best = int(np.argmin(-lts.mean(1)))
    RES[f"one random fixed order (mean over {R}, my seeds)"] = dict(valid_mass=float(np.exp(lts).sum(1).mean()), **{k: float(-lts[:, v].mean() - LN288) for k, v in SETS.items()})
    RES[f"uniform mixture over {R} random orders (my seeds)"] = row(mix)
    RES[f"best of {R} random fixed orders (picked with the true law)"] = row(lts[best])
    for l in list(RES)[-3:]:
        show(l)
    log(f"   M's inconsistency: std over {R} random orders of log q_sigma(x), mean over grids: {lts.std(0).mean():.4f} nats")
    RES["inconsistency_std"] = dict(v=float(lts.std(0).mean()))
    log(f"\n[2] trust MDP, symmetry-reduced DP. equivariance of the network under the 64 cell symmetries, 600 states (half off the real support): "
        f"largest |difference of log-probabilities| = {equivariance():.2e}")
    wsets = {"all": np.ones(288), "train": (orb_file == 1).astype(float), "held": (orb_file == 0).astype(float)}
    V0, pol, nstates, extra = dp(wsets, sym=True)
    log(f"   states evaluated {nstates}; V*(empty) - ln(support): true law {V0[0][0]-LN288:.6f}; training orbit as reality {V0[1][train[0]]-math.log(192):.6f}; "
        f"held-out orbit as reality {V0[2][held[0]]-math.log(96):.6f}")
    TP = {nm: TablePol(pol[j], extra) for j, nm in enumerate(wsets)}
    for nm, lab_ in (("all", "exact optimum of the trust MDP, true law (DP)"), ("train", "DP with the TRAINING orbit as reality (entropy order off the table)"),
                     ("held", "DP with the HELD-OUT orbit as reality (entropy order off the table)")):
        LT[lab_] = logT_all(TP[nm]); RES[lab_] = row(LT[lab_]); show(lab_)
    r = RES["exact optimum of the trust MDP, true law (DP)"]
    log(f"   chain rule / DP consistency: V*(empty) - ln 288 = {V0[0][0]-LN288:.6f}; walking the table policy on all 288 grids: {r['all288']:.6f}")
    RES["V0"] = dict(all=float(V0[0][0] - LN288), train=float(V0[1][train[0]] - math.log(192)), held=float(V0[2][held[0]] - math.log(96)))
    np.save("out/c2_pol_sym.npy", pol)
    log("\n[3] T on all 4^16 boards (every board with T >= 1e-9 listed; the rest bounded by the pruned mass)")
    WALK = {}
    for lab_, p in (("pi_trust (the worker's learned net)", trustpol), ("own confidence: lowest entropy", ck.o_entropy),
                    ("exact optimum of the trust MDP, true law (DP)", TP["all"])):
        S, L, pruned, mf = tree_walk(p); val = ck.is_valid(S, 4); lt = LT[lab_]; tau = float(lt[train].min() - 2.0)
        assert val.sum() == 288
        w = dict(n=len(S), mass_valid=float(np.exp(L[val]).sum()), mass_invalid=float(np.exp(L[~val]).sum()), pruned=pruned,
                 max_invalid=float(np.exp(L[~val].max())), min_valid=float(np.exp(lt.min())), tau=tau, exp_tau=math.exp(tau),
                 invalid_accepted=int((L[~val] >= tau).sum()), valid_rejected=int((lt < tau).sum()), heldout_rejected=int((lt[held] < tau).sum()),
                 pruned_below_tau=bool(pruned < math.exp(tau)), max_frontier=mf)
        WALK[lab_] = w
        log(f"   {lab_}: {w['n']} boards with T >= 1e-9; mass on valid {w['mass_valid']:.6f}, on listed invalid {w['mass_invalid']:.6f}, pruned {w['pruned']:.2e}")
        log(f"      smallest T on a valid grid {w['min_valid']:.3e}; largest T on an invalid board {w['max_invalid']:.3e}; tau = min over training grids - 2.0 = {tau:.3f} (T = {w['exp_tau']:.2e})")
        log(f"      invalid boards accepted {w['invalid_accepted']} (pruned mass {w['pruned']:.1e} < e^tau: {w['pruned_below_tau']}, so no unlisted board can pass); "
            f"valid grids rejected {w['valid_rejected']} of 288 (held-out orbit {w['heldout_rejected']} of 96)")
    log("\n[4] what the routes look like (all 288 grids, exact conditionals by listing the grids)")
    DESC = {}
    for lab_, p in (("DP optimum (true law)", TP["all"]), ("pi_trust (worker's net)", trustpol), ("lowest entropy", ck.o_entropy), ("top probability", ck.o_maxprob)):
        DESC[lab_] = describe(p, lab_)
    # learned mixture (EM on the training grids), my own EM
    labs = [l for l in POL]; Lm = np.stack([LT[l] for l in labs], 1); w = np.full(len(labs), 1 / len(labs))
    for _ in range(500):
        a = Lm[train] + np.log(w + 1e-300); a -= a.max(1, keepdims=True); r_ = np.exp(a); r_ /= r_.sum(1, keepdims=True); w = r_.mean(0)
    lmix = np.log((np.exp(Lm) * w).sum(1)); RES["learned mixture of the six routes (EM on training grids)"] = row(lmix)
    log("\n[5] learned mixture, weights " + ", ".join(f"{l.split(':')[-1].strip()[:22]} {x:.3f}" for l, x in zip(labs, w)))
    show("learned mixture of the six routes (EM on training grids)")
    json.dump(dict(RES=RES, WALK=WALK, DESC=DESC, calls=net.calls), open("out/c2_4x4_sym.json", "w"), indent=1)
    log(f"\ncalls of M {net.calls}; total {time.time()-T0:.0f}s")

if MODE == "full":
    rng = np.random.default_rng(777)
    wsets = {"all": np.ones(288), "train": (orb_file == 1).astype(float), "held": (orb_file == 0).astype(float)}
    for n, reps in ((36, 3), (72, 3), (144, 3), (288, 3), (576, 2), (1152, 2), (4608, 1)):
        for r in range(reps):
            wsets[f"iid{n}_{r}"] = np.bincount(rng.integers(288, size=n), minlength=288).astype(float)
    log(f"[full] trust MDP on every reachable state, no symmetry; {len(wsets)} realities: {list(wsets)}")
    V0, pol, nstates, _ = dp(wsets, sym=False)
    log(f"   states evaluated {nstates} (the worker states 15,802,049 reachable); calls of M {net.calls} ({time.time()-T0:.0f}s)")
    OUTF = {"nstates": int(nstates)}
    log(f"   {'reality the trust MDP was solved on':38s} {'grids in it':>11s} {'fit: V0 - H(sample)':>20s} {'KL(true || T)':>14s} {'held-out orbit':>15s} {'train orbit':>12s} {'grids not in sample':>20s} {'off table':>10s}")
    for j, (nm, w) in enumerate(wsets.items()):
        tp = TablePol(pol[j]); lt = logT_all(tp); p = w / w.sum(); H = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        v0 = float(V0[j][np.argmax(w > 0)]); notin = w == 0
        r = dict(n_distinct=int((w > 0).sum()), fit_excess=v0 - H, fit_walk=float(-(p * lt).sum() - H), kl_true=float(-lt.mean() - LN288),
                 heldout=float(-lt[held].mean() - LN288), train=float(-lt[train].mean() - LN288),
                 unseen=float(-lt[notin].mean() - LN288) if notin.any() else float("nan"), valid_mass=float(np.exp(lt).sum()), off=tp.off, on=tp.on)
        OUTF[nm] = r
        log(f"   {nm:38s} {r['n_distinct']:11d} {r['fit_excess']:20.4f} {r['kl_true']:14.4f} {r['heldout']:15.4f} {r['train']:12.4f} {r['unseen']:20.4f} {r['off']:10d}")
        assert abs(r["fit_excess"] - r["fit_walk"]) < 1e-6, (r["fit_excess"], r["fit_walk"])
    np.save("out/c2_pol_full_all.npy", pol[0])
    json.dump(OUTF, open("out/c2_4x4_full.json", "w"), indent=1)
    log(f"total {time.time()-T0:.0f}s")
