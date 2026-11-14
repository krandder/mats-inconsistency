"""M5 item 2: coherence arbitrageurs (AND / OR / ex3 / D-prove / twin) as class-C shadow traders.

Traders (one per constraint instance, see PREREG.md):
  K  Kelly projection trader: q = Euclidean projection of the tuple's prices onto the coherent polytope, clipped to
     [DELTA, 1-DELTA]; trades every leg; E_T = sum log q(y)/p(y), booked at trade price, settled at reveal time.
  B  bounded bundle arbitrageur: <= 1 unit of the dominating bundle per step when the violation > TOL; riskless.
Member test: per question, D+1 interleaved 2-member Kelly pools (market 1-eps, trader eps), fixed-point clearing.

Constraint spec: list of dicts {kind, legs: [q...], fixed: target value for 1-leg D-prove kinds}.
Prices P[n,Q] (probabilities) at trade times t[n]; mask[n,Q] live; y[n,Q] outcomes; delay[Q].
"""
import math
import numpy as np

DELTA, TOL, EPS = 1e-3, 0.01, 1e-3


# ------------------------------------------------------------------ projections (pure python, per step)
def _halfspaces(kind):
    # rows (coef, rhs) meaning coef . x <= rhs, x = (a, b, c)
    if kind == "and":
        return [((-1, 0, 1), 0.0), ((0, -1, 1), 0.0), ((1, 1, -1), 1.0)]
    if kind == "or":
        return [((1, 0, -1), 0.0), ((0, 1, -1), 0.0), ((-1, -1, 1), 0.0)]
    raise ValueError(kind)


def _feasible(x, H):
    return all(sum(c * v for c, v in zip(co, x)) <= r + 1e-12 for co, r in H) and all(0 <= v <= 1 for v in x)


def proj_poly(x, kind, iters=200):
    """Dykstra projection of x=(a,b,c) onto {halfspaces} cap [0,1]^3."""
    H = _halfspaces(kind)
    x = list(x)
    if _feasible(x, H):
        return x
    sets = len(H) + 1
    inc = [[0.0] * 3 for _ in range(sets)]
    for _ in range(iters):
        for k in range(sets):
            yv = [x[i] + inc[k][i] for i in range(3)]
            if k < len(H):
                co, r = H[k]
                s = sum(c * v for c, v in zip(co, yv)) - r
                nn = sum(c * c for c in co)
                z = [v - max(0.0, s) * c / nn for v, c in zip(yv, co)]
            else:
                z = [min(1.0, max(0.0, v)) for v in yv]
            inc[k] = [yv[i] - z[i] for i in range(3)]
            x = z
        if _feasible(x, H):
            break
    return x


def proj_simplex(x):
    u = sorted(x, reverse=True); css = 0.0; rho = 0; th = 0.0
    for i, ui in enumerate(u):
        css += ui
        t = (css - 1) / (i + 1)
        if ui - t > 0:
            rho, th = i, t
    return [max(0.0, v - th) for v in x]


def belief(kind, p, fixed=None):
    """Coherent belief for the legs' prices p (list)."""
    if kind in ("and", "or"):
        q = proj_poly(p, kind)
    elif kind == "ex3":
        q = proj_simplex(p)
    elif kind in ("taut_or", "impl", "contra"):
        q = [float(fixed)]
    elif kind == "twin":
        m = 0.5 * (p[0] + p[1]); q = [m, m]
    else:
        raise ValueError(kind)
    return [min(1 - DELTA, max(DELTA, v)) for v in q]


def bundle_pnl(kind, p, y, fixed=None):
    """Bounded bundle: return (traded, pnl) for one step (1 unit), realised on outcomes y."""
    if kind == "and":
        a, b, c = p; ya, yb, yc = y
        v1 = c - min(a, b); v2 = a + b - 1 - c
        if v1 > TOL:
            if a <= b: return True, (ya - yc) - (a - c)
            return True, (yb - yc) - (b - c)
        if v2 > TOL:
            return True, (yc - ya - yb) - (c - a - b)
        return False, 0.0
    if kind == "or":
        a, b, c = p; ya, yb, yc = y
        v1 = max(a, b) - c; v2 = c - a - b
        if v1 > TOL:
            if a >= b: return True, (yc - ya) - (c - a)
            return True, (yc - yb) - (c - b)
        if v2 > TOL:
            return True, (ya + yb - yc) - (a + b - c)
        return False, 0.0
    if kind == "ex3":
        s = sum(p); ys = sum(y)
        if 1 - s > TOL: return True, ys - s
        if s - 1 > TOL: return True, s - ys
        return False, 0.0
    if kind in ("taut_or", "impl", "contra"):
        (pp,), (yy,) = p, y
        if fixed == 1 and 1 - pp > TOL: return True, yy - pp
        if fixed == 0 and pp > TOL: return True, pp - yy
        return False, 0.0
    if kind == "twin":
        (pb, pt), (yb, yt) = p, y
        if abs(pb - pt) > TOL:
            return True, (yt - yb) - (pt - pb) if pt < pb else (yb - yt) - (pb - pt)
        return False, 0.0
    raise ValueError(kind)


def _ls(q, y):
    return math.log(q if y == 1 else 1 - q)


# ------------------------------------------------------------------ accounting
def shadow(cons, P, y, mask, t, delay, T_end):
    """Shadow K and B traders. Returns per-trader event arrays (settle_time, dE_K) and (settle_time, dE_B),
    plus traded-violation stats. Only steps where all legs are live and all reveals <= T_end are booked."""
    out = []
    for c in cons:
        legs, kind, fx = c["legs"], c["kind"], c.get("fixed")
        live = mask[:, legs].all(1)
        rev = t[:, None] + delay[None, legs]
        live &= (rev <= T_end).all(1)
        idx = np.where(live)[0]
        sK, eK, sB, eB, eC, tI, eI = [], [], [], [], [], [], []
        for i in idx:
            p = [float(v) for v in P[i, legs]]; yy = [int(v) for v in y[i, legs]]
            q = belief(kind, p, fx)
            for j, qj in enumerate(q):
                pj = min(1 - 1e-7, max(1e-7, p[j]))
                sK.append(rev[i, j]); eK.append(_ls(qj, yy[j]) - _ls(pj, yy[j]))
                cj = min(1 - DELTA, max(DELTA, pj))       # control: clip-only trader on the same legs
                eC.append(_ls(cj, yy[j]) - _ls(pj, yy[j]))
                # invasion at share EPS (worker-1 convention, exploit_core.gain_matrix): log q - log((1-eps)p + eps q)
                qy = qj if yy[j] else 1 - qj; py = pj if yy[j] else 1 - pj
                tI.append(t[i]); eI.append(math.log(qy) - math.log((1 - EPS) * py + EPS * qy))
            ok, pnl = bundle_pnl(kind, p, yy, fx)
            if ok:
                sB.append(rev[i].max()); eB.append(pnl)
        out.append(dict(kind=kind, legs=legs, sK=np.array(sK), eK=np.array(eK), sB=np.array(sB), eB=np.array(eB),
                        sC=np.array(sK), eC=np.array(eC), sA=np.array(sK), eA=np.array(eK) - np.array(eC),
                        tI=np.array(tI), eI=np.array(eI),
                        n_live=len(idx)))
    return out


def curve(s, e, grid):
    """Cumulative E at each grid time (events settled at time <= grid)."""
    if len(s) == 0:
        return np.zeros(len(grid))
    o = np.argsort(s, kind="stable"); cs = np.cumsum(e[o]); ss = s[o]
    k = np.searchsorted(ss, grid, side="right")
    return np.where(k > 0, cs[np.maximum(k - 1, 0)], 0.0)


def member(c, P, y, mask, t, delay, T_end, eps=EPS, fp_iters=300):
    """Trader joins each leg's pool with share eps (D+1 interleaved pools per leg, T1 5.3), fixed-point clearing
    p* = (1-w) p + w q(p*) (T1 5.2). Pool update at reveal: w <- w q(y) / p*(y).
    Returns max per-pool member E (bound ln(1/eps)), final shares."""
    import heapq
    legs, kind, fx = c["legs"], c["kind"], c.get("fixed")
    L = len(legs); Hs = [int(delay[q]) + 1 for q in legs]
    w = [np.full(h, eps) for h in Hs]
    Esub = [np.zeros(h) for h in Hs]
    live = mask[:, legs].all(1) & ((t[:, None] + delay[None, legs]) <= T_end).all(1)
    pend = []; Emax = 0.0; n = 0; fp_gap = 0.0
    def apply(r):
        nonlocal Emax
        _, _, j, sl, qy, py = r
        if not (qy > 0 and py > 0): raise RuntimeError(f"bad settle {r} w={w[j][sl]}")
        w[j][sl] = w[j][sl] * qy / py; Esub[j][sl] += math.log(qy / py); Emax = max(Emax, Esub[j][sl])
    for i in range(len(t)):
        ti = int(t[i])
        if live[i]:
            p = [float(v) for v in P[i, legs]]
            slots = [ti % h for h in Hs]
            ww = [float(w[j][slots[j]]) for j in range(L)]
            pc = list(p)
            for _ in range(fp_iters):
                q = belief(kind, pc, fx)
                new = [(1 - ww[j]) * p[j] + ww[j] * q[j] for j in range(L)]
                d = max(abs(a - b) for a, b in zip(new, pc)); pc = new
                if d < 1e-13:
                    break
            q = belief(kind, pc, fx)
            cons_pc = [(1 - ww[j]) * p[j] + ww[j] * q[j] for j in range(L)]
            fp_gap = max(fp_gap, max(abs(pc[j] - cons_pc[j]) for j in range(L)))
            pc = cons_pc          # clear at the pool mix of the final belief, so p*(y) >= w q(y) exactly
            for j in range(L):
                yy = int(y[i, legs[j]])
                qy = q[j] if yy else 1 - q[j]; py = pc[j] if yy else 1 - pc[j]
                heapq.heappush(pend, (ti + int(delay[legs[j]]), n, j, slots[j], qy, py)); n += 1
        while pend and pend[0][0] <= ti:
            apply(heapq.heappop(pend))
    while pend:
        apply(heapq.heappop(pend))
    return dict(max_member_E=float(Emax), share_leg0_mean=float(np.mean(w[0])),
                share_min=float(min(x.min() for x in w)), share_max=float(max(x.max() for x in w)),
                fp_residual=float(fp_gap))
