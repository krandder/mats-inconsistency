"""e_4x4.py -- the whole design on 4x4 Sudoku, where everything can be enumerated.

  [1] M_A (trained on the 192-grid orbit only, or on all 288 with 'full'): held-out route surprise.
  [2] Exact dynamic programming for the trust MDP over all 15.8M reachable states (reference: uses the true law),
      and the same DP on the training grids only (a tabular trust policy learned from real training boards).
  [3] The learned trust policy (core.fit_trust, training orbit only).
  [4] For every order policy: exact KL(reality || T_pi), on all 288 grids and on the held-out orbit; exact mass of
      T_pi on valid grids; exhaustive walk of T's tree for the invalid boards.
  [5] Search on held-out puzzles, corrected against T.
Usage: python -u e_4x4.py MODEL_NAME [skipdp]
Scoring uses oracle.py (validity, enumeration). The arms (M, trust learner, searchers) are in core.py.
"""
import sys, time, math, json
import numpy as np, torch
import core
import oracle as O            # scoring and the exact reference only

name = sys.argv[1]; skipdp = len(sys.argv) > 2 and sys.argv[2] == "skipdp"   # third argument "order": also describe the routes
import os
DPONLY = (not skipdp) and os.path.exists(f"ckpt/trust_{name}.pt")   # second run: reuse the trust policy of the first run,
#                                                                     skip the random orders and the puzzle search (already in *_nodp.out)
torch.set_num_threads(1); torch.manual_seed(0)
K, N = 4, 16; T0 = time.time(); LOG288 = math.log(288)
d = np.load("data/k4.npz"); G = d["G"].astype(np.int64); orb = d["orbit"]
train_rows = np.where(orb == 1)[0] if name == "mA4" else np.arange(288)
test_rows = np.where(orb == 0)[0]
model, meta = core.load_model(f"ckpt/{name}.pt"); M = core.Net(model, K, name)
Gt = torch.from_numpy(G)
RES = {}


def log(*a):
    print(*a, flush=True)


log(f"== {name}: {M.params} parameters; trained on {len(train_rows)} grids; held-out orbit has {len(test_rows)} grids")

# ------------------------------------------------------------------ [2] exact DP over all reachable states
# The network has no positional embedding, so it is exactly equivariant under the 64 cell permutations that keep rows
# rows, columns columns and boxes boxes. The optimal value is invariant under them, so the DP is run on one mask per
# symmetry class (the 15.8M reachable states fall into about 64 times fewer classes). Checked below: (i) numerically,
# that M's atoms are equivariant; (ii) that V*(empty) equals the route surprise obtained by walking the policy.
ck = (G + 1) << (3 * np.arange(N))[None, :]
bitlist = [[i for i in range(N) if m >> i & 1] for m in range(1 << N)]
wsets = {"all": np.ones(288), "train": np.isin(np.arange(288), train_rows).astype(np.float64)}
RM = core.relmask(K).numpy()
SYM = np.array([t for t in np.load("data/k4_syms.npy") if all((RM[h][t][:, t] == RM[h]).all() for h in range(3))])
gkey = (G << (2 * np.arange(N))[None, :]).sum(1); gorder = np.argsort(gkey)
PERM = np.stack([gorder[np.searchsorted(gkey[gorder], (G[:, t] << (2 * np.arange(N))[None, :]).sum(1))] for t in SYM])  # grid g o t
assert all((G[PERM[a]] == G[:, t]).all() for a, t in enumerate(SYM))
allm = np.arange(1 << N, dtype=np.int64)
TM = np.stack([sum((((allm >> int(t[i])) & 1) << i) for i in range(N)) for t in SYM])       # mask of (state o t)
CM = TM.min(0); TI = TM.argmin(0)                                                           # canonical mask, a symmetry reaching it


def equivariance_error(n=200, seed=0):
    g = torch.Generator().manual_seed(seed); x = Gt[torch.randint(0, 288, (n,), generator=g)]
    s = torch.where(torch.rand(n, N, generator=g) < 0.5, x, torch.full_like(x, -1))
    t = torch.from_numpy(SYM[np.random.default_rng(seed).integers(len(SYM), size=n)])
    lp = M.logp(s); lpt = M.logp(torch.gather(s, 1, t))                       # atoms at s and at s o t
    return float((lpt - torch.gather(lp, 1, t[:, :, None].expand(-1, -1, K))).abs().max())


def dp_trust(chunk=64):
    """Exact trust-MDP dynamic programme. V[r][c][g]: optimal expected remaining surprise at the state (canonical
    mask c, values of grid g on c), for reality r (uniform on the grids with weight 1). POLC[r][c][g]: the optimal
    cell there, or -1 if no grid of r matches the state."""
    V = {r: {(1 << N) - 1: np.zeros(288)} for r in wsets}; POLC = {r: {} for r in wsets}
    levels = [[] for _ in range(N + 1)]
    for m in np.unique(CM).tolist():
        levels[len(bitlist[m])].append(m)
    nstates = 0; ar = np.arange(N)[None, :]
    for k in range(N - 1, -1, -1):
        ms = levels[k]
        for a in range(0, len(ms), chunk):
            sub = ms[a:a + chunk]; S, INV, OFF = [], [], [0]
            for m in sub:
                bits = bitlist[m]
                key = ck[:, bits].sum(1) if bits else np.zeros(288, dtype=np.int64)
                u, first, inv = np.unique(key, return_index=True, return_inverse=True)
                st = -np.ones((len(u), N), dtype=np.int64)
                if bits:
                    st[:, bits] = G[first][:, bits]
                S.append(st); INV.append(inv); OFF.append(OFF[-1] + len(u))
            lp = M.logp(torch.from_numpy(np.concatenate(S))).numpy().astype(np.float64)
            nstates += OFF[-1]
            for j, m in enumerate(sub):
                inv = INV[j]; nc = OFF[j + 1] - OFF[j]
                cost = -lp[OFF[j]:OFF[j + 1]][inv[:, None], ar, G]                 # [288, 16]: surprise at the real value
                for r, w in wsets.items():
                    cnt = np.bincount(inv, weights=w, minlength=nc)
                    best = np.full(nc, np.inf); bi = -np.ones(nc, dtype=np.int64)
                    for i in range(N):
                        if m >> i & 1:
                            continue
                        m2 = m | (1 << i)
                        vch = V[r][int(CM[m2])][PERM[TI[m2]]]                      # V(m2, g) through the canonical mask
                        tot = (cost[:, i] + vch) * w
                        q = np.bincount(inv, weights=tot, minlength=nc) / np.maximum(cnt, 1e-12)
                        better = (q < best - 1e-12) & (cnt > 0)
                        best = np.where(better, q, best); bi = np.where(better, i, bi)
                    best = np.where(cnt > 0, best, 0.0)
                    V[r][m] = best[inv]; POLC[r][m] = bi[inv]
        log(f"   DP level {k}: {len(ms)} canonical masks, states evaluated so far {nstates} ({time.time()-T0:.0f}s)")
    return V, POLC, nstates


class TablePolicy:
    """Order policy read from the DP table. A state is looked up through any grid of the table's reality that matches
    it and through the symmetry that takes its mask to the canonical one; if no grid matches (the state is off the
    table), the cell with M's lowest entropy is taken."""

    def __init__(self, polc, rows):
        cs = sorted(polc); self.table = torch.from_numpy(np.stack([polc[c] for c in cs]))        # [n canonical, 288]
        self.cpos = torch.full((1 << N,), -1, dtype=torch.long); self.cpos[torch.tensor(cs)] = torch.arange(len(cs))
        self.cm = torch.from_numpy(CM); self.ti = torch.from_numpy(TI); self.perm = torch.from_numpy(PERM)
        self.sym = torch.from_numpy(SYM); self.rows = torch.from_numpy(rows); self.G = Gt[self.rows]
        self.w = (1 << torch.arange(N)); self.off = 0; self.on = 0

    def __call__(self, s, lp, emp):
        has, gi = [], []
        for a in range(0, len(s), 4096):                                             # chunked: [B, n, N] is large
            sc = s[a:a + 4096]
            match = ((sc[:, None, :] == self.G[None]) | (sc[:, None, :] < 0)).all(2)   # [b, n]
            has.append(match.any(1)); gi.append(match.float().argmax(1))
        has = torch.cat(has); g = self.rows[torch.cat(gi)]
        m = ((s >= 0).long() * self.w).sum(1)
        t = self.ti[m]; cpos = self.cpos[self.cm[m]].clamp(min=0)
        ic = self.table[cpos, self.perm[t, g]]                                       # cell in canonical coordinates
        ok = has & (ic >= 0)
        cell = self.sym[t, ic.clamp(min=0)]                                          # back to the state's coordinates
        score = core.pol_entropy(s, lp, emp)
        self.on += int(ok.sum()); self.off += int((~ok).sum())
        score[ok] = 1e6
        score[torch.where(ok)[0], cell[ok]] = -1e6
        return score


# ------------------------------------------------------------------ helpers (scoring)
def eval_policy(policy, label, rows_sets):
    out = core.route_logT(M, policy, Gt)
    lt = out["logp"].numpy().astype(np.float64)
    res = dict(valid_mass=float(np.exp(lt).sum()))
    for nm, rows in rows_sets.items():
        res[f"excess_{nm}"] = float((-lt[rows]).mean() - LOG288)
    res["min_logT_train"] = float(lt[train_rows].min()); res["min_logT_test"] = float(lt[test_rows].min())
    res["logT"] = lt
    return res


def tree_walk(policy, thr=1e-9, cap=600000):
    """Exhaustive walk of T's tree from the empty board: every complete board with T >= thr, and the mass pruned."""
    S = torch.full((1, N), -1, dtype=torch.long); LP = torch.zeros(1); pruned = 0.0
    for depth in range(N):
        lp = M.logp(S); emp = S < 0
        score = policy(S, lp, emp).masked_fill(~emp, float("inf")); cell = score.argmin(1)
        lpc = lp[torch.arange(len(S)), cell]                                         # [B, 4]
        newlp = (LP[:, None] + lpc).reshape(-1)
        Srep = S.repeat_interleave(K, 0); Srep[torch.arange(len(Srep)), cell.repeat_interleave(K)] = torch.arange(K).repeat(len(S))
        keep = newlp >= math.log(thr)
        if keep.sum() > cap:
            thr_eff = torch.sort(newlp, descending=True).values[cap].item(); keep = newlp > thr_eff
        pruned += float(newlp[~keep].exp().sum())
        S, LP = Srep[keep], newlp[keep]
    return S, LP, pruned


rows_sets = {"all288": np.arange(288), "heldout96": test_rows, "train": train_rows}

# ------------------------------------------------------------------ [1] own-confidence and fixed orders
log("\n[1] order policies that need no learning (exact, all 288 grids)")
POL = {"own confidence: lowest entropy": core.pol_entropy, "own confidence: top probability": core.pol_maxprob,
       "own confidence: top-1 minus top-2": core.pol_margin, "least confident first": core.pol_anti,
       "raster order": core.pol_raster}
for lab, p in POL.items():
    RES[lab] = eval_policy(p, lab, rows_sets)
# random orders: the mean over orders, the uniform mixture, the best sampled fixed order
R = 8 if DPONLY else 256; lts = []
for r in range(R):
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(1000 + r)).float()
    inv = torch.empty(N); inv[perm.long()] = torch.arange(N).float()
    lts.append(core.route_logT(M, lambda s, lp, emp, inv=inv: inv.expand(s.shape[0], -1), Gt)["logp"].numpy().astype(np.float64))
lts = np.array(lts)                                                    # [R, 288]
mix = np.log(np.exp(lts).mean(0))
RES[f"one random fixed order (mean over {R} orders)"] = dict(
    valid_mass=float(np.exp(lts).sum(1).mean()), **{f"excess_{nm}": float((-lts[:, rows]).mean() - LOG288) for nm, rows in rows_sets.items()})
RES[f"uniform mixture over {R} random orders"] = dict(
    valid_mass=float(np.exp(mix).sum()), **{f"excess_{nm}": float((-mix[rows]).mean() - LOG288) for nm, rows in rows_sets.items()})
bestr = int(np.argmin((-lts).mean(1)))
RES[f"best of {R} random fixed orders (picked on all 288: oracle pick)"] = dict(
    valid_mass=float(np.exp(lts[bestr]).sum()), **{f"excess_{nm}": float((-lts[bestr, rows]).mean() - LOG288) for nm, rows in rows_sets.items()})
spread = lts.std(0)
log(f"   M's inconsistency: std over {R} random orders of log q_sigma(x): mean over grids {spread.mean():.4f} nats "
    f"(train orbit {spread[train_rows].mean():.4f}, held-out orbit {spread[test_rows].mean():.4f})")

# ------------------------------------------------------------------ [3] learned trust policy (training orbit only)
log("\n[3] trust policy learned by paired routes on the training grids only")
gen = torch.Generator().manual_seed(5)


def sample_real(n):
    x = Gt[torch.from_numpy(train_rows)[torch.randint(0, len(train_rows), (n,), generator=gen)]]
    start = torch.full_like(x, -1)
    sub = torch.rand(n, generator=gen) < 0.4                          # 40%: a random subset of the real grid as start
    r = torch.floor(torch.rand(n, generator=gen) * (N - 1)).long()
    rank = torch.rand(n, N, generator=gen).argsort(1).argsort(1)
    rev = (rank < r[:, None]) & sub[:, None]
    start = torch.where(rev, x, start)
    return x, start, None


if DPONLY:
    _ck = torch.load(f"ckpt/trust_{name}.pt", map_location="cpu"); _m = core.RelFormer(**_ck["cfg"]); _m.load_state_dict(_ck["state"])
    trust = core.TrustPolicy(_m, K); log("   (loaded the trust policy fitted by the first run)")
else:
    policies, hist = core.fit_trust(M, K, sample_real, core.pol_entropy, iters=4, boards_per_iter=1536, n_dev=6, seed=3,
                                    log=log, d=64, L=2)
    # selection on real boards the learner may use: 4x4 has no validation orbit, so the training grids (see addendum)
    sel = [float(-core.route_logT(M, p, Gt[torch.from_numpy(train_rows)])["logp"].mean()) for p in policies]
    best = int(np.argmin(sel)); trust = policies[best]
    log(f"   route surprise of each iterate on the training grids: {[round(v, 4) for v in sel]} -> pi_trust = iterate {best+1}")
    torch.save(dict(cfg=trust.m.cfg, state=trust.m.state_dict()), f"ckpt/trust_{name}.pt")
RES["pi_trust (learned, paired routes)"] = eval_policy(trust, "trust", rows_sets)

# ------------------------------------------------------------------ [2] exact DP
if not skipdp:
    try:
        log("\n[2] exact dynamic programming over every reachable state")
        log(f"   equivariance of M under the {len(SYM)} cell symmetries: largest |difference of log-probabilities| on 200 random states = {equivariance_error():.2e}")
        V, pol, nstates = dp_trust()
        log(f"   states evaluated (one per symmetry class of masks): {nstates} of 15,802,049 reachable; V*(empty) with the true law = {V['all'][0][0]:.6f} nats "
            f"(entropy of reality {LOG288:.6f}); with the training grids as reality = {V['train'][0][train_rows[0]]:.6f} "
            f"(entropy {math.log(len(train_rows)):.6f})")
        tp_all = TablePolicy(pol["all"], np.arange(288)); tp_tr = TablePolicy(pol["train"], train_rows)
        RES["exact optimum of the trust MDP (DP with the true law; reference)"] = eval_policy(tp_all, "dp", rows_sets)
        RES["tabular trust policy (DP on the training grids; entropy order off the table)"] = eval_policy(tp_tr, "dptr", rows_sets)
        log(f"   tabular policy: steps on the table {tp_tr.on}, off the table {tp_tr.off}")
        # chain rule check: V*(empty) = H + KL(p || T*)
        r = RES["exact optimum of the trust MDP (DP with the true law; reference)"]
        log(f"   chain rule: V*(empty) - ln 288 = {V['all'][0][0]-LOG288:.6f}; mean_x[-log T*(x)] - ln 288 = {r['excess_all288']:.6f}")
    except Exception:
        import traceback; traceback.print_exc(file=sys.stdout); log("   DP FAILED; continuing without it"); skipdp = True

# ------------------------------------------------------------------ [2c] what the optimal route looks like (exact)
if not skipdp and len(sys.argv) > 2 and sys.argv[2] == "order":
    import eval_lib as E2_
    C4 = O.Counter(d["B"], K); c_ = E2_.COLS
    log("\n[2c] the order each policy takes on all 288 grids (exact conditionals of the chosen cell)")
    for lab, p in (("exact optimum (DP, true law)", tp_all), ("tabular (DP on training grids)", tp_tr), ("pi_trust (learned)", trust),
                   ("own confidence: lowest entropy", core.pol_entropy), ("own confidence: top probability", core.pol_maxprob),
                   ("random order", core.RandomOrder(9))):
        out = core.rollout(M, p, torch.full_like(Gt, -1), X=Gt, record="full")
        St = out["states"].reshape(-1, N); At = out["atoms"].reshape(-1, N, K); Xr = Gt.repeat_interleave(N, 0)
        Aall = E2_.atom_stats(St.numpy(), Xr.numpy(), At.numpy(), C4, K); ty = E2_.ctx_type(Aall)
        chosen = out["cells"].reshape(-1).numpy(); stt = Aall[:, c_["state"]].astype(int); cel = Aall[:, c_["cell"]].astype(int)
        is_ch = cel == chosen[stt]; ns = len(St)
        av = np.zeros((ns, 3), bool); ch_t = np.zeros(ns, int); ch_ex = np.zeros(ns); ch_H = np.zeros(ns); ch_sur = np.zeros(ns)
        order_ = np.argsort(stt, kind="stable"); bounds = np.searchsorted(stt[order_], np.arange(ns + 1))
        for si in range(ns):
            ix = order_[bounds[si]:bounds[si + 1]]
            for t_ in range(3):
                av[si, t_] = (ty[ix] == t_).any()
            j = ix[is_ch[ix]][0]; ch_t[si] = ty[j]; ch_ex[si] = Aall[j, c_["excess"]]; ch_H[si] = Aall[j, c_["H"]]; ch_sur[si] = Aall[j, c_["surprise"]]
        step = np.tile(np.arange(N), 288); orbit_rep = np.repeat(orb, N)
        log(f"   {lab}")
        log(f"      steps by type of the chosen cell: locally forced {np.mean(ch_t == 0):.3f}, forced but not locally {np.mean(ch_t == 1):.3f}, free {np.mean(ch_t == 2):.3f}; "
            f"a locally forced cell exists at {av[:, 0].mean():.3f} of states and is then chosen {np.mean(ch_t[av[:, 0]] == 0):.3f} of the time")
        log(f"      exact excess summed along the route {ch_ex.sum() / 288:.4f} (free choices {ch_ex[ch_t == 2].sum() / 288:.4f}; forced-not-locally "
            f"{ch_ex[ch_t == 1].sum() / 288:.4f}; locally forced {ch_ex[ch_t == 0].sum() / 288:.4f}); entropy paid at free choices {ch_H[ch_t == 2].sum() / 288:.4f} of {LOG288:.4f}")
        log(f"      mean step index of free choices {step[ch_t == 2].mean():.2f} (of 0..15); number of free choices per route {np.mean(ch_t == 2) * N:.2f}; "
            f"excess on the held-out orbit {ch_ex[orbit_rep == 0].sum() / 96:.4f}, on the training orbit {ch_ex[orbit_rep == 1].sum() / 192:.4f}")
        fc = np.bincount(out["cells"][:, 0].numpy(), minlength=N); sc = out["cells"][:, 1].numpy(); f0 = out["cells"][:, 0].numpy()
        same_row = np.mean(f0 // K == sc // K); same_col = np.mean(f0 % K == sc % K); same_box = np.mean((f0 // 8 * 2 + f0 % K // 2) == (sc // 8 * 2 + sc % K // 2))
        log(f"      first cell: {int(fc.argmax())} (always the same: {bool(fc.max() == 288)}); second cell shares a row with it {same_row:.2f}, a column {same_col:.2f}, a box {same_box:.2f}")

# ------------------------------------------------------------------ [3b] learned mixture over routes (addendum A6)
import eval_lib as E
mix_labs = [l for l in RES if "logT" in RES[l] and "DP" not in l]
Lall = np.stack([RES[l]["logT"] for l in mix_labs], 1)                      # [288, k]
wmix = E.fit_mixture(Lall[train_rows]); lmix = E.mixture_logT(Lall, wmix)
RES["learned mixture over the routes above (weights by EM on the training grids)"] = dict(
    valid_mass=float(np.exp(lmix).sum()), logT=lmix, **{f"excess_{nm}": float((-lmix[rows]).mean() - LOG288) for nm, rows in rows_sets.items()})
log("\n[3b] learned mixture over routes: weights " + ", ".join(f"{l.split(':')[-1].strip()[:28]} {w:.3f}" for l, w in zip(mix_labs, wmix)))

# ------------------------------------------------------------------ [4] table
log("\n[4] every order policy as a definition of T (exact). excess = mean_x[-log T(x)] - ln 288, in nats")
log(f"   {'order policy':78s} {'all 288':>9s} {'held-out':>9s} {'train':>9s} {'mass on valid':>14s}")
for lab, r in RES.items():
    log(f"   {lab:78s} {r['excess_all288']:9.4f} {r['excess_heldout96']:9.4f} {r['excess_train']:9.4f} {r['valid_mass']:14.6f}")

# ------------------------------------------------------------------ [4b] T on invalid boards: exhaustive tree walk
log("\n[4b] exhaustive walk of T's tree (all complete boards with T >= 1e-9)")
tr_lt = RES["pi_trust (learned, paired routes)"]["logT"]
tau = float(tr_lt[train_rows].min() - 2.0)
log(f"   acceptance rule: log T >= tau = {tau:.3f} (minimum over the training grids minus 2.0)")
walk_pols = {"pi_trust (learned, paired routes)": trust, "own confidence: lowest entropy": core.pol_entropy}
if not skipdp:
    walk_pols["exact optimum of the trust MDP (DP with the true law; reference)"] = tp_all
WALK = {}
for lab, p in walk_pols.items():
    S, LP, pruned = tree_walk(p)
    valid = O.is_valid_batch(S.numpy(), K)
    lt = RES[lab]["logT"]; tau_p = float(lt[train_rows].min() - 2.0)
    inv_lp = LP[torch.from_numpy(~valid)]
    WALK[lab] = dict(n_boards=len(S), n_valid=int(valid.sum()), mass_valid=float(LP[torch.from_numpy(valid)].exp().sum()),
                     mass_invalid_enumerated=float(inv_lp.exp().sum()), pruned=pruned,
                     max_invalid_T=float(inv_lp.max().exp()) if len(inv_lp) else 0.0,
                     min_valid_T=float(np.exp(lt.min())), tau=tau_p,
                     n_invalid_accepted=int((inv_lp >= tau_p).sum()),
                     n_valid_rejected_heldout=int((lt[test_rows] < tau_p).sum()),
                     n_valid_rejected_all=int((lt < tau_p).sum()))
    w = WALK[lab]
    log(f"   {lab}: boards with T >= 1e-9: {w['n_boards']} ({w['n_valid']} valid); mass on valid {w['mass_valid']:.6f}; "
        f"on enumerated invalid {w['mass_invalid_enumerated']:.6f}; pruned {w['pruned']:.2e}")
    log(f"      smallest T on a valid grid {w['min_valid_T']:.3e}; largest T on an invalid board {w['max_invalid_T']:.3e}; "
        f"tau {tau_p:.3f}: invalid boards accepted {w['n_invalid_accepted']} of 4^16-288; "
        f"valid grids rejected {w['n_valid_rejected_all']} of 288 (held-out orbit: {w['n_valid_rejected_heldout']} of {len(test_rows)})")

json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'logT'} for k, v in RES.items()}, open(f"out/e_4x4_{name}{'_nodp' if skipdp else ''}.json", "w"), indent=1)
json.dump(WALK, open(f"out/e_4x4_{name}{'_nodp' if skipdp else ''}_walk.json", "w"), indent=1)
if DPONLY:
    log(f"\ncalls of M in this script: {M.calls}; total time {time.time()-T0:.0f}s"); sys.exit(0)
# ------------------------------------------------------------------ [5] search on puzzles, corrected against T
log("\n[5] search on minimal puzzles, corrected against T (pi_trust, tau as above)")
pz = np.load("data/k4_puzzles.npz"); PZ = torch.from_numpy(pz["puz"]); SOL = torch.from_numpy(pz["sol"]); porb = pz["orbit"]
checker = core.make_checker(M, trust, tau)
fl_M, fl_T = M.flops_per_call, trust.flops_per_call
ARMS = {"S1 own confidence (margin)": dict(policy=core.pol_margin, floor=0.0, greedy_first=True),
        "S2 least confident first, 5% floor": dict(policy=core.pol_anti, floor=0.05, greedy_first=False),
        "S5 trust order": dict(policy=trust, floor=0.0, greedy_first=True)}
for split, o in (("held-out orbit", 0), ("training orbit", 1)):
    idx = torch.from_numpy(np.where(porb == o)[0]); puz, sol = PZ[idx], SOL[idx]
    k0 = core.rollout(M, core.pol_margin, puz, greedy=True)["s"]
    log(f"   {split}: {len(idx)} puzzles, clues {np.bincount((puz >= 0).sum(1).numpy()).tolist()[4:]} for 4,5,6; "
        f"one pass of M, own confidence, no check: accuracy {(k0 == sol).all(1).float().mean():.4f}")
    answers = {}
    for lab, cfg in ARMS.items():
        g = torch.Generator().manual_seed(77)
        r = core.search_first_accept(M, cfg["policy"], puz, None, checker, 256, g, floor=cfg["floor"], greedy_first=cfg["greedy_first"])
        solved = r["acc_idx"] >= 0; correct = (r["answer"] == sol).all(1)
        uses_t = 1 if cfg["policy"] is trust else 0
        flops = r["prop_steps"] * (fl_M + uses_t * fl_T) + r["chk_steps"] * (fl_M + fl_T)
        answers[lab] = r["answer"]
        log(f"      {lab:38s} accepted {solved.float().mean():.4f}; accepted answers correct {correct[solved].float().mean():.4f}; "
            f"candidates to accept: mean {(r['acc_idx'][solved]+1).float().mean():.2f}, max {int(r['acc_idx'].max())+1}; "
            f"FLOPs mean {flops[solved].float().mean():.3e} median {flops[solved].float().median():.3e}")
    labs = list(answers); agree = []
    for a in range(len(labs)):
        for b in range(a + 1, len(labs)):
            both = (answers[labs[a]] >= 0).all(1) & (answers[labs[b]] >= 0).all(1)
            agree.append(float((answers[labs[a]][both] == answers[labs[b]][both]).all(1).float().mean()))
    log(f"      identical accepted answers across pairs of policies: {agree}")

log(f"\ncalls of M in this script: {M.calls}; total time {time.time()-T0:.0f}s")
