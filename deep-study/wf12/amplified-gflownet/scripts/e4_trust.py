"""E4 (registered in PREREG.md): where the reality-trained trust head sits.
F1 trust in KINDS OF DEMONSTRATION: weights c(order) inside the score; reads only the submitted demonstration.
F2 free head on search statistics; F3 search-effort trust that fades with the effective sample size.
Output: out/e4_trust.out / .json."""
import json, time
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from world import A, D, N, XARR, XIDX, PIDX, PERMS, children, make_world, noisy_policy, traj_law, tv, kl, vmarg
from queries import Amp, KEYS, QUERIES, cond, match, sig, DXF
from e3b_helpers import draw_queries

t0 = time.time()
lines, out = [], {}
NREAL = 2000


def selective(Q):
    Q = {k: v.copy() for k, v in Q.items()}
    for (a, pre), q in Q.items():
        for ci, c in enumerate(children(pre)):
            x = dict(pre + (c,))
            if x.get(0) == 1 and x.get(1) == 1:
                q[ci] = -np.inf
    return Q


def em(G, iters=300):
    """Maximise sum_n log sum_s c(s) G[n, s] over the simplex (mixture weights)."""
    c = np.full(G.shape[1], 1.0 / G.shape[1])
    for _ in range(iters):
        r = c[None, :] * G
        r /= r.sum(1, keepdims=True)
        c = r.mean(0)
    return c


def limits(amp, Rc, form, th):
    """Exact K -> infinity limit of the final prediction for every query (None where no rollout can match)."""
    res = {}
    Rf = np.tile(Rc, A)
    for (i, key) in QUERIES:
        P = amp.PC[key]
        if not np.isfinite(P).all() or P.sum() == 0:
            res[(i, key)] = None
            continue
        sup = P > 0
        hit = DXF[:, i] == 1
        u = amp.lg(Rf[sup & hit].sum() / Rf[sup].sum())
        v = amp.lg(P[hit].sum())
        res[(i, key)] = {"raw": sig(u), "F2": sig(th[0] * u + th[1] * v), "F3": sig(u)}[form]
    return res


for wseed in (1, 2, 3):
    Wd = make_world(seed=wseed, sM=1.5, goal=0.0, stiff_order=(0, 1, 2))
    R, px = Wd["R"], Wd["pstar"]
    rng = np.random.default_rng(50 + wseed)
    Xr = rng.choice(8, NREAL, p=px)
    Mord = R.reshape(6, 8)                               # M's product along each order, for each value vector
    c_exact = em(Mord[:, Xr].T)
    Rc = (c_exact[:, None] * Mord).ravel()
    Tc_x, T_x = vmarg(Rc / Rc.sum()), vmarg(Wd["T"])
    lines.append(f"\n== world {wseed}: excess KL(reality || target): unweighted score {kl(px, T_x):.4f}; with fitted order "
                 f"weights {kl(px, Tc_x):.4f}; weights {np.round(c_exact, 3).tolist()} (orders {PERMS})")
    pol = {"A": noisy_policy(R, 0.5, 0.0, 21), "B": noisy_policy(R, 1.5, 2.0, 22)}
    pol["S"] = (selective(pol["B"][0]), pol["B"][1])
    amps = {k: Amp(Q, Qr, Rc) for k, (Q, Qr) in pol.items()}
    # F1(b): the same weights fitted THROUGH K rollouts masked to the real values, under A and under B
    row = {}
    ampsR = {n_: Amp(*pol[n_], R) for n_ in ("A", "B")}   # rollouts scored by M alone; c multiplies inside the fit
    xkeys = [tuple(int(v) for v in XARR[xi]) for xi in range(8)]
    freq = {n_: np.array([sum(ampsR[n_].PC[xkeys[xi]].reshape(A, D)[:, PIDX == s].sum() * np.mean(Xr == xi)
                              for xi in range(8)) for s in range(6)]) for n_ in ("A", "B")}
    for K in (1, 4, 16, 64, 256):
        cs = {}
        for name in ("A", "B"):
            G = np.zeros((NREAL, 6))
            for xi in range(8):
                n = np.where(Xr == xi)[0]
                idx = rng.choice(A * D, size=(len(n), K), p=ampsR[name].PC[xkeys[xi]])
                g = ampsR[name].wC[xkeys[xi]][idx] / K
                for s in range(6):
                    G[n, s] = (g * (PIDX[idx % D] == s)).sum(1)
            cs[name] = em(G)
        row[K] = dict(tvAB=tv(cs["A"], cs["B"]), tvA=tv(cs["A"], c_exact), tvB=tv(cs["B"], c_exact),
                      own=max(tv(cs["A"], freq["A"]), tv(cs["B"], freq["B"])))
        tx = {n_: cs[n_] @ Mord for n_ in ("A", "B")}       # POST HOC: value marginal of the implied target
        row[K].update(ptvAB=tv(tx["A"], tx["B"]), pklA=kl(px, tx["A"]), pklB=kl(px, tx["B"]))
    lines.append("F1 fitted through K rollouts: K            " + "".join(f"{k:9d}" for k in row))
    lines.append("  TV(weights under A, weights under B)     " + "".join(f"{row[k]['tvAB']:9.4f}" for k in row))
    lines.append("  TV(weights under A, exact fit)           " + "".join(f"{row[k]['tvA']:9.4f}" for k in row))
    lines.append("  TV(weights under B, exact fit)           " + "".join(f"{row[k]['tvB']:9.4f}" for k in row))
    lines.append("  POST HOC TV(target under A, target under B)" + "".join(f"{row[k]['ptvAB']:9.4f}" for k in row))
    lines.append("  POST HOC excess KL(reality || target), A  " + "".join(f"{row[k]['pklA']:9.4f}" for k in row))
    lines.append("  POST HOC excess KL(reality || target), B  " + "".join(f"{row[k]['pklB']:9.4f}" for k in row))
    lines.append(f"  at K = 1: TV(fitted weights, the policy's own order frequencies on the real values) = {row[1]['own']:.4f}")
    # F2 and F3 fitted on real data at K = 8 under policy A (score = the fitted F1 score)
    Xq = XARR[Xr]
    qi, keys, y = draw_queries(Xq, rng)
    pw, pu, ess = amps["A"].stats(qi, keys, 8, rng)
    u, v = amps["A"].lg(pw), amps["A"].lg(pu)
    ll = lambda z: np.mean(np.logaddexp(0, np.where(y == 1, -z, z)))
    th2 = minimize(lambda th: ll(th[0] * u + th[1] * v), [1.0, 0.0], method="Nelder-Mead").x
    f3 = lambda lk, u, v, ess: (ess / (ess + np.exp(lk))) * u + (1 - ess / (ess + np.exp(lk))) * v
    lk = minimize_scalar(lambda lk: ll(f3(lk, u, v, ess)), bounds=(-6, 8), method="bounded").x
    js = [amps["A"].stats(qi, keys, K_, rng) for K_ in (2, 8, 32, 128)]
    lkj = minimize_scalar(lambda lk: sum(ll(f3(lk, amps["A"].lg(a_), amps["A"].lg(b_), e_)) for a_, b_, e_ in js),
                          bounds=(-6, 8), method="bounded").x
    lines.append(f"POST HOC F3j: kappa fitted jointly at K = 2, 8, 32, 128 under A: {np.exp(lkj):.2f}")
    lines.append(f"F2 fitted (alpha, beta) = ({th2[0]:.2f}, {th2[1]:.2f});  F3 fitted kappa = {np.exp(lk):.2f}")
    # finite-K accuracy against reality (4,000 fresh real test queries), then the exact limits
    rt = np.random.default_rng(900 + wseed)
    Xt = XARR[rt.choice(8, 4000, p=px)]
    ti, tk, _ = draw_queries(Xt, rt)
    pt = np.array([cond(px, i, k) for i, k in zip(ti, tk)])
    ex = lambda f: float(np.mean(pt * np.log(pt / np.clip(f, 1e-9, 1)) + (1 - pt) * np.log((1 - pt) / np.clip(1 - f, 1e-9, 1))))
    for name in ("A", "B"):
        for K in (8, 512):
            pw, pu, ess = amps[name].stats(ti, tk, K, rt)
            u, v = amps[name].lg(pw), amps[name].lg(pu)
            lines.append(f"excess vs reality, policy {name}, K = {K:<4d}: raw {ex(sig(u)):.4f}   F2 {ex(sig(th2[0] * u + th2[1] * v)):.4f}"
                         f"   F3 {ex(sig(f3(lk, u, v, ess))):.4f}   F3j {ex(sig(f3(lkj, u, v, ess))):.4f}")
    lim = {(f, n): limits(amps[n], Rc, f, th2) for f in ("raw", "F2", "F3") for n in ("A", "B", "S")}
    tgt = {q: cond(Tc_x, *q) for q in QUERIES}
    for f in ("F2", "F3"):
        dAB = max(abs(lim[(f, "A")][q] - lim[(f, "B")][q]) for q in QUERIES)
        dT = {n: max(abs(lim[(f, n)][q] - tgt[q]) for q in QUERIES if lim[(f, n)][q] is not None) for n in ("A", "B", "S")}
        nS = sum(lim[(f, "S")][q] is None for q in QUERIES)
        lines.append(f"limit of compute, {f}: max |limit under A - under B| = {dAB:.4f};  max |limit - target|: A {dT['A']:.4f}  "
                     f"B {dT['B']:.4f}  S {dT['S']:.4f} ({nS} of 27 queries unanswerable under S)")
        out[f"w{wseed}_{f}"] = dict(dAB=dAB, dT=dT)
    out[f"w{wseed}"] = dict(kl_T=kl(px, T_x), kl_Tc=kl(px, Tc_x), c=c_exact.tolist(), F1b={k: v for k, v in row.items()},
                            th2=th2.tolist(), kappa=float(np.exp(lk)))
lines.append(f"\nruntime {time.time() - t0:.0f}s")
open("../out/e4_trust.out", "w").write("\n".join(lines) + "\n")
json.dump(out, open("../out/e4_trust.json", "w"), indent=1, default=float)
print("\n".join(lines))
