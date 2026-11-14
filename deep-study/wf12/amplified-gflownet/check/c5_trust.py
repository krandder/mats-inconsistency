"""Check of claims 10 and 11 (note section 4, E4): own implementation, fresh worlds (501-503) and seeds.
F1: weights c(order) inside the score, fitted on real samples exactly and through K rollouts masked to the real
values, under policies A and B.  F2 free head, F3 fading head: limits of compute.  Output: out/c5_trust.out"""
import itertools
import numpy as np
from scipy.optimize import minimize
from cw import *

L = []
KEYS = list(itertools.product([-1, 0, 1], repeat=N))
DXF = np.tile(DXV, (2, 1))
ORDF = np.tile(ORD, 2)


def em(G, it=500):
    c = np.full(G.shape[1], 1 / G.shape[1])
    for _ in range(it):
        r = c[None, :] * G
        r /= r.sum(1, keepdims=True)
        c = r.mean(0)
    return c


def kl(p, q):
    return float((p * np.log(p / q)).sum())


def cond(px, i, key):
    m = np.all([(XA[:, j] == key[j]) | (key[j] == -1) for j in range(N)], axis=0)
    return px[m & (XA[:, i] == 1)].sum() / px[m].sum()


lg = lambda p: np.log(np.clip(p, 0.02, 0.98) / (1 - np.clip(p, 0.02, 0.98)))
sig = lambda z: 1 / (1 + np.exp(-z))
for ws in (501, 502, 503):
    W = make_world(ws, noise=1.5, goal=0.0, exact_order=(0, 1, 2))
    R, px = W["R"], W["pstar"]
    rng = np.random.default_rng(900 + ws)
    Xr = rng.choice(8, 2000, p=px)
    Mord = np.zeros((6, 8))
    np.add.at(Mord, (ORD, VAL), R)
    c_ex = em(Mord[:, Xr].T)
    Rc = c_ex[ORD] * R
    Tc = vmarg(Rc / Rc.sum())
    L.append(f"\n== world {ws}: excess KL(reality || target) with all routes equal {kl(px, vmarg(W['T'])):.4f}; with order weights "
             f"fitted exactly {kl(px, Tc):.4f}; weight on the reliable route {c_ex[PERMS.index((0, 1, 2))]:.3f}")
    pol = {"A": noisy_policy(R, 0.5, 60 + ws), "B": noisy_policy(R, 1.5, 61 + ws, lean=2.0)}
    PX = {n_: [law(*pol[n_], dict(enumerate(XS[xi]))).ravel() for xi in range(8)] for n_ in pol}
    for n_ in pol:
        for xi in range(8):
            PX[n_][xi] = PX[n_][xi] / PX[n_][xi].sum()
    freq = {n_: np.array([sum(PX[n_][xi][ORDF == s].sum() * np.mean(Xr == xi) for xi in range(8)) for s in range(6)]) for n_ in pol}
    for K in (1, 16, 256, 4096):
        cs = {}
        for n_ in pol:
            Gm = np.zeros((len(Xr), 6))
            for xi in range(8):
                rows = np.where(Xr == xi)[0]
                p = PX[n_][xi]
                idx = rng.choice(2 * D, size=(len(rows), K), p=p)
                w = (0.5 * np.tile(R, 2) / np.where(p > 0, p, 1))[idx] / K
                for s in range(6):
                    Gm[rows, s] = (w * (ORDF[idx] == s)).sum(1)
            cs[n_] = em(Gm)
        tA, tB = cs["A"] @ Mord, cs["B"] @ Mord
        L.append(f"F1 through K = {K:4d} rollouts: TV(weights A, weights B) {tv(cs['A'], cs['B']):.3f}; TV to exact fit A {tv(cs['A'], c_ex):.3f} "
                 f"B {tv(cs['B'], c_ex):.3f}; TV to the policy's own order frequencies A {tv(cs['A'], freq['A']):.3f} B {tv(cs['B'], freq['B']):.3f};"
                 f" excess KL of implied target A {kl(px, tA / tA.sum()):.4f} B {kl(px, tB / tB.sum()):.4f}")
    # F2 / F3 limits (score = the exactly fitted F1 score Rc)
    hide = lambda x: x.get(0) == 1 and x.get(1) == 1
    pol["S"] = noisy_policy(R, 1.5, 61 + ws, lean=2.0, hide=hide)
    PK = {n_: {k: law(*pol[n_], {j: k[j] for j in range(N) if k[j] != -1}).ravel() for k in KEYS} for n_ in pol}
    Rcf = np.tile(Rc, 2)
    qs = [(i, k) for k in KEYS for i in range(N) if k[i] == -1]
    # fit F2 on real data at K = 8 under A
    Xq = XA[Xr]
    qi = rng.integers(0, N, len(Xr)); inc = rng.random((len(Xr), N)) < 0.5
    u, v, y = np.zeros(len(Xr)), np.zeros(len(Xr)), Xq[np.arange(len(Xr)), qi]
    for m in range(len(Xr)):
        k = tuple(int(Xq[m, j]) if (inc[m, j] and j != qi[m]) else -1 for j in range(N))
        p = PK["A"][k] / PK["A"][k].sum()
        idx = rng.choice(2 * D, size=8, p=p)
        w = Rcf[idx] / p[idx]
        one = DXF[idx, qi[m]] == 1
        u[m], v[m] = lg((w * one).sum() / w.sum()), lg(one.mean())
    th = minimize(lambda t: np.mean(np.logaddexp(0, np.where(y == 1, -(t[0] * u + t[1] * v), t[0] * u + t[1] * v))),
                  [1.0, 0.0], method="Nelder-Mead").x
    lim = {}
    for n_ in pol:
        for (i, k) in qs:
            p = PK[n_][k]
            if p.sum() == 0:
                continue
            p = p / p.sum()
            sup, hit = p > 0, DXF[:, i] == 1
            uu, vv = lg(Rcf[sup & hit].sum() / Rcf[sup].sum()), lg(p[hit].sum())
            lim[(n_, i, k)] = (sig(uu), sig(th[0] * uu + th[1] * vv), cond(Tc, i, k))
    dAB = max(abs(lim[("A",) + q][1] - lim[("B",) + q][1]) for q in qs)
    dT = {n_: max(abs(lim[(n_,) + q][1] - lim[(n_,) + q][2]) for q in qs if (n_,) + q in lim) for n_ in pol}
    dR = {n_: max(abs(lim[(n_,) + q][0] - lim[(n_,) + q][2]) for q in qs if (n_,) + q in lim) for n_ in pol}
    L.append(f"F2 fitted at K = 8 under A: (alpha, beta) = ({th[0]:.2f}, {th[1]:.2f}); limit differs between A and B by up to {dAB:.3f}; "
             f"from the target: A {dT['A']:.3f} B {dT['B']:.3f} S {dT['S']:.3f}")
    L.append(f"limit of the weighted share itself (what any head that fades to it returns): from the target A {dR['A']:.4f} B {dR['B']:.4f} S {dR['S']:.3f}"
             f"  (0.02-0.98 clipping of the logit included)")
    # F3: does the effective sample size grow without bound, also under S?  (weight ESS / (ESS + kappa) -> 1)
    for n_ in ("A", "B", "S"):
        p = PK[n_][(-1, -1, -1)]
        e = []
        for K in (64, 4096):
            idx = rng.choice(2 * D, size=(200, K), p=p)
            w = (Rcf / np.where(p > 0, p, 1))[idx]
            e.append(np.mean(w.sum(1) ** 2 / (w ** 2).sum(1)) / K)
        L.append(f"F3 policy {n_}: effective sample size / K at K = 64, 4096: {e[0]:.3f}, {e[1]:.3f}")
open("out/c5_trust.out", "w").write("\n".join(L) + "\n")
print("\n".join(L))
