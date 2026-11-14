"""Skeptic stress tests for frequency-and-prior (numpy/scipy only).
Run: OMP_NUM_THREADS=4 python3 skeptic.py > skeptic-output.txt
Reuses the solvers from check.py by import (check.py's module-level demo runs on import, so we exec only its defs)."""
import itertools, numpy as np
from scipy.special import xlogy, expit, logit
from scipy.optimize import minimize

src = open(__file__.replace("skeptic.py", "check.py")).read()
defs = src.split('out = []')[0]            # helper defs only (generator, bkl, kl, project, project_slsqp)
g = {}; exec(defs, g)
bkl, kl, project_lbfgs, clause_val, W, K = g["bkl"], g["kl"], g["project"], g["clause_val"], g["W"], g["K"]
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver_md import solve as project   # accurate mirror-descent solver; check.py's L-BFGS kept for comparison

def setup(seed, sig):
    g["rng"] = np.random.default_rng(seed)
    rng = g["rng"]
    cls, mod = g["gen_theory"](); Q, _ = g["gen_queries"](cls)
    th = rng.uniform(0.05, 0.95, K)
    pi = np.prod(np.where(W == 1, th, 1 - th), axis=1)
    piT = pi * mod / (pi * mod).sum()
    A = Q[:, mod]; base = piT[mod]; f = Q @ piT
    P = expit(logit(np.clip(f, 1e-3, 1 - 1e-3)) + sig * rng.standard_normal(len(f)))
    P = np.clip(P, 1e-9, 1 - 1e-9)
    return A, base, f, P, rng

print("=== S1: R3 sandwich + tilt, R4 Pythagoras + KKT, R6, across 40 random (theory, prior, price) draws ===")
worst = dict(pyth=np.inf, kkt=np.inf, sand=np.inf, tilt=0.0, r6=np.inf); lb_bad = 0; lb_gap = 0.0
for seed in range(40):
    sig = [0.5, 1.0, 3.0][seed % 3]
    A, base, f, P, rng = setup(1000 + seed, sig)
    mu_c, x_c, coh = project(A, P)
    c_lb = project_lbfgs(A, P)[2]; lb_gap = max(lb_gap, c_lb - coh); lb_bad += int(c_lb - coh > 1e-4)
    ce = bkl(f, P).sum()
    worst["r6"] = min(worst["r6"], ce - coh)
    for _ in range(50):
        y = A @ rng.dirichlet(np.full(A.shape[1], 0.2))
        y = np.where(np.abs(y - np.round(y)) < 1e-12, np.round(y), y)
        worst["pyth"] = min(worst["pyth"], bkl(y, P).sum() - bkl(y, x_c).sum() - coh)
    free = np.ptp(A, axis=1) > 0                      # non-forced rows
    gv = logit(np.clip(x_c[free], 1e-12, 1 - 1e-12)) - logit(P[free])
    worst["kkt"] = min(worst["kkt"], ((A[free].T - x_c[free]) @ gv).min())
    for b0 in [0.03, 1.0, 30.0]:
        mu, x, v = project(A, P, base=base, b0=b0, bP=1.0)
        worst["sand"] = min(worst["sand"], v - coh, ce - v)
        lt = np.log(base) - (1 / b0) * (A[free].T @ (logit(np.clip(x[free], 1e-15, 1 - 1e-15)) - logit(P[free])))
        t = np.exp(lt - lt.max()); t /= t.sum(); worst["tilt"] = max(worst["tilt"], np.abs(t - mu).max())
print("  min Pythagorean margin D(y,P)-D(y,x*)-Coh   :", f"{worst['pyth']:.2e}")
print("  min KKT <grad h(x*)-grad h(P), v-x*> (free rows):", f"{worst['kkt']:.2e}")
print("  min sandwich slack (value-lo, hi-value)       :", f"{worst['sand']:.2e}")
print("  max tilt-form error (forced rows dropped)     :", f"{worst['tilt']:.2e}")
print("  min D(f,P)-Coh (R6)                           :", f"{worst['r6']:.2e}")
print("  check.py L-BFGS-on-softmax Coh above the accurate value by >1e-4 in", lb_bad, "of 40 draws; worst excess", f"{lb_gap:.2e}")

print("\n=== S2: R7 block map across seeds and step sizes: monotone KL, Chen-Teboulle bound, empirical rate ===")
for seed, b0 in [(2001, 0.01), (2002, 0.1), (2003, 1.0), (2004, 0.1), (2005, 0.03)]:
    A, base, f, P, rng = setup(seed, 2.0)
    Pt = P.copy(); mu = None; ks = []; D1 = None; okb = True
    T = 120
    for t in range(1, T + 1):
        mu, Pt, _ = project(A, np.clip(Pt, 1e-12, 1 - 1e-12), base=base, b0=b0, bP=1.0, mu0=mu, tol=1e-15, iters=50000)
        if t == 1: D1 = bkl(f, Pt).sum()
        k = kl(mu, base); ks.append(k)
        if t > 1 and b0 * k > D1 / (t - 1) + 1e-9: okb = False
    ks = np.array(ks)
    mono = bool(np.all(np.diff(ks) <= 1e-9))
    tail = ks[60:] ; good = tail > 1e-11
    rate = np.exp(np.polyfit(np.arange(60, T)[good], np.log(tail[good]), 1)[0]) if good.sum() > 5 else float('nan')
    print(f"  seed={seed} b0/bP={b0:5.2f}: KL t=1 {ks[0]:.2e}, t=20 {ks[19]:.2e}, t=120 {ks[-1]:.2e}; monotone={mono}; bound holds={okb}; per-step ratio (t>=60) ~ {rate:.4f}")

print("\n=== S3: R5 'or, equivalently' check: PDG-with-data objective vs CE + lambda*Coh ===")
A, base, f, P0, rng = setup(7, 1.0)
w = rng.choice(A.shape[1], p=base); y = A[:, w]
def ce(P): return float(-(xlogy(y, P) + xlogy(1 - y, 1 - P)).sum())
# two price tables with the same CE on this label but different Coh
Pa = np.clip(f, 1e-6, 1 - 1e-6)                                       # coherent
s = np.where(y == 1, 1, -1); free = np.ptp(A, axis=1) > 0
# move two free coordinates in opposite directions on the CE scale, keeping total CE fixed
i, j = np.where(free)[0][:2]
def mk(d):
    Pb = Pa.copy()
    Pb[i] = expit(logit(Pa[i]) + s[i] * d)
    target = ce(Pa)
    from scipy.optimize import brentq
    def gap(e):
        Pc = Pb.copy(); Pc[j] = expit(logit(Pa[j]) - s[j] * e); return ce(Pc) - target
    e = brentq(gap, 0, 50); Pb[j] = expit(logit(Pa[j]) - s[j] * e); return Pb
Pb = mk(2.0)
for nm, Px in [("coherent f", Pa), ("perturbed, same CE", Pb)]:
    coh = project(A, Px)[2]
    print(f"  {nm:20s}: CE(y,P)={ce(Px):.5f}  Coh_T(P)={coh:.5f}  "
          f"PDG objective - prior NLL (= bP*CE, R5) = {ce(Px):.5f}   CE + 1.0*Coh = {ce(Px) + coh:.5f}")
print("  -> the PDG-with-data objective cannot see Coh beyond CE; it is not equivalent to CE + lambda*Coh.")
print("  Lemma 1 bound check: bP*CE(y,P) >= bP*Coh_T(P) for 200 random P:",
      all(ce(Px) >= project(A, Px)[2] - 1e-8 for Px in rng.uniform(0.02, 0.98, (200, len(f)))))

print("\n=== S4: Ideas 'self-distillation unbiased if deeper prices are calibrated' ===")
r = np.random.default_rng(3); n = 200000
z = r.integers(0, 2, n)                        # feature visible at budget N
yy = (r.random(n) < np.where(z == 1, 0.9, 0.1)).astype(float)
PM = np.full(n, yy.mean())                     # deeper predictor: constant base rate, perfectly calibrated
print("  calibration of P_M: E[y | P_M] =", round(yy.mean(), 4), " P_M =", round(PM[0], 4))
print("  distillation target E[P_M | z] =", [round(PM[z == k].mean(), 4) for k in (0, 1)],
      " vs Bayes E[y | z] =", [round(yy[z == k].mean(), 4) for k in (0, 1)])
print("  -> calibration alone is not enough; need P_M = E[y | G] with G containing F_N.")

print("\n=== S5: R1b with the generator's mixed widths (2 or 3) and without-replacement draws ===")
allc = {k: [list(zip(vs, sg)) for vs in itertools.combinations(range(K), k) for sg in itertools.product([0, 1], repeat=k)] for k in (2, 3)}
cv = {k: np.array([clause_val([(int(v), int(s_)) for v, s_ in c]) for c in allc[k]]) for k in (2, 3)}
th = r.uniform(0.1, 0.9, K); pi = np.prod(np.where(W == 1, th, 1 - th), axis=1)
worst = 0.0
for _ in range(50):
    widths = r.choice([2, 3], 5)
    lik = np.ones(len(W)); modT = np.ones(len(W), bool)
    for k in (2, 3):
        m = int((widths == k).sum())
        if m == 0: continue
        idx = r.choice(len(allc[k]), m, replace=False)
        sat = np.all(cv[k][idx], axis=0); modT &= sat
        Nsat = cv[k].sum(0)                                  # same for all worlds
        from math import comb
        lik *= sat / np.array([comb(int(s_), m) for s_ in Nsat])   # uniform m-subset of satisfied clauses
    post = pi * lik / (pi * lik).sum()
    worst = max(worst, np.abs(post - pi * modT / (pi * modT).sum()).max())
print("  #2-clauses satisfied per world:", set(cv[2].sum(0).tolist()), " #3-clauses:", set(cv[3].sum(0).tolist()))
print("  max |p(w|T) - pi(w|Mod T)| over 50 mixed-width theories:", f"{worst:.1e}")
