"""Skeptic checks for no-history-traders.md.
S1  R4(a) uniqueness including clipped (boundary) fixed points: iterate from many starts incl. corners.
S2  R5: with few particles the empirical mean sits on a proper face of C -> a weak book exists (R3), although no strong book.
S3  R6 proof gap: the case 'Sigma r, Sigma s finite => worth bounded' fails for sign-changing rates. A market that applies the
    stopping rule to trader A (const 0.2) only, where trader B has bounded all-day partial sums (so is classed 'finite'),
    is exploited by B, which buys 1 on the last low day and sells 1 on the last high day of each phase.
S4  R12: blocks to converge vs the stated local formula log(1/tol)/(eta(1+1/delta)) and vs the far-field term.
S5  R10(iv): persistence analogue via Doob maximal inequality on the part-3 Bayes martingale."""
import itertools, numpy as np
from scipy.optimize import minimize, linprog
rng = np.random.default_rng(7)
imp = lambda x, y: (not x) or y
S = {'a': lambda a,b,c: a, 'b': lambda a,b,c: b, 'c': lambda a,b,c: c, 'a->c': lambda a,b,c: imp(a,c),
     'a&c': lambda a,b,c: a and c, '~b': lambda a,b,c: not b, 'b|c': lambda a,b,c: b or c,
     'a&~c': lambda a,b,c: a and not c, 'c->a': lambda a,b,c: imp(c,a)}
names = list(S); allw = list(itertools.product([0,1], repeat=3))
gamma = lambda a,b,c: imp(a,b) and imp(b,c)
models = [w for w in allw if gamma(*w)]
M = np.array([[float(S[s](*w)) for s in names] for w in models])
def project(P):
    k = M.shape[0]
    r = minimize(lambda l: 0.5*np.sum((l@M-P)**2), np.ones(k)/k, jac=lambda l: (l@M-P)@M.T, method='SLSQP',
                 bounds=[(0,1)]*k, constraints=[{'type':'eq','fun':lambda l: l.sum()-1}], options={'ftol':1e-15,'maxiter':500})
    return r.x@M
q = np.array([0.3, 0.6, 0.5])
mu = np.array([np.prod([q[i] if w[i] else 1-q[i] for i in range(3)]) for w in models]); pm = mu/mu.sum(); mstar = pm@M
# S1
worst = 0
starts = [np.array(c, float) for c in itertools.product([0,1], repeat=len(names))][::64] + [rng.random(len(names)) for _ in range(6)]
for V in starts:
    for t in range(120): V = np.clip(V + 0.3*((project(V)-V) + (mstar-V)), 0, 1)
    worst = max(worst, np.linalg.norm(V-mstar))
print(f'S1 R4(a): {len(starts)} starts (incl. cube corners), max |V_final - m*| = {worst:.2e}')
# S2
def weak_book(P):
    D = M - P; r = linprog(-D.sum(0), A_ub=-D, b_ub=np.zeros(len(models)), bounds=[(-1,1)]*len(names)); return -r.fun
def strong_book(P):  # max t s.t. x.(w-P) >= t all w, |x|<=1
    D = M - P; n = len(names)
    r = linprog(np.r_[np.zeros(n), -1], A_ub=np.c_[-D, np.ones(len(models))], b_ub=np.zeros(len(models)),
                bounds=[(-1,1)]*n + [(None, None)]); return -r.fun
miss = 0; wb = []
for trial in range(400):
    idx = rng.choice(len(models), size=10, p=pm)
    if len(set(idx)) < len(models):
        miss += 1
        if len(wb) < 5: V = M[idx].mean(0); wb.append((round(weak_book(V), 3), round(strong_book(V), 9)))
print(f'S2 R5: K=10 particles miss at least one model in {miss/400:.3f} of draws; (weak-book value, strong-book margin) there: {wb}')
# S3
LO, HI = 0.3, 0.7
# schedule from trader A alone (r=s=0.2), doc's rule: low phase until wA0 < -j, high until wA1 < -j
wA0 = wA1 = 0.0; n = 0; phases = []
for j in range(1, 13):
    for x, world in ((LO, 0), (HI, 1)):
        g = 0.2*(1 if x == LO else -1); step = g*((0 if world == 0 else 1) - x)   # per-day change of watched worth (<0)
        cur = wA0 if world == 0 else wA1
        L = int(np.floor((-j - cur)/step)) + 1 if cur >= -j else 1
        wA0 += L*g*(0-x); wA1 += L*g*(1-x); phases.append((n+1, n+L, x)); n += L
# trader B: r = +1 on last low day, -1 on first high day (r unused on high days); s = +1 on last high day, -1 on first low day.
# all-day partial sums of r and s therefore stay in {0,1} -> the doc's case analysis classes B as 'both finite'.
evR = sorted([(e, 1) for (st, e, x) in phases if x == LO] + [(st, -1) for (st, e, x) in phases if x == HI])
evS = sorted([(e, 1) for (st, e, x) in phases if x == HI] + [(st, -1) for (st, e, x) in phases if x == LO])
cR = np.cumsum([v for _, v in evR]); cS = np.cumsum([v for _, v in evS])
cash = h = 0.0; lo = hi = 0.0
for (st, e, x) in phases:            # B trades only on the last day of each phase
    sh = 1.0 if x == LO else -1.0
    cash -= sh*x; h += sh; lo = min(lo, cash, cash+h); hi = max(hi, cash, cash+h)
print(f'S3 R6: A-only schedule, {len(phases)//2} cycles, {n} days. Trader B all-day partial sums of r in [{cR.min()},{cR.max()}], of s in [{cS.min()},{cS.max()}] (classed finite);')
print(f'       B worth range [{lo:.2f}, {hi:.2f}], gaining 0.4 per cycle -> B exploits a market that ignores it; case "both finite => bounded" is false.')
# S4
p, d = 0.8, 0.01; xs = p/(1+d); T = lambda x: np.clip((p-x)/d, 0, 1) - x
for eta in [0.005, 0.015]:
    x = 0.3
    for t in range(1, 5000):
        x = x + eta*T(x)
        if abs(x-xs) < 1e-6:
            # check it stays
            break
    loc = np.log(1/1e-6)/(eta*(1+1/d)); far = np.log((1-0.3)/(1-(p-d)))/eta
    print(f'S4 R12: eta={eta}: first within 1e-6 at block {t}; stated formula {loc:.1f}; far-field (1/eta)log((1-x0)/(1-p+delta)) = {far:.1f}; exact local log(1/tol)/-log|slope| = {np.log(1e6)/-np.log(abs(1-eta*(1+1/d))):.1f}')
# S5
n_, qq, N = 40, 0.03, 100000
bits = rng.random((N, n_)) < qq; anyone = bits.any(1); first = np.where(anyone, bits.argmax(1)+1, n_+1)
P = np.empty((N, n_+1))
for t in range(n_+1):
    P[:, t] = np.where(first <= t, 1.0, 1-(1-qq)**(n_-t))
P[~anyone, n_] = 0.0
for t, eps in [(10, 0.8), (20, 0.8), (30, 0.5)]:
    lhs = (np.abs(P[:, t:] - P[:, [t]]).max(1) >= eps).mean(); rhs = np.mean((P[:, -1]-P[:, t])**2)/eps**2
    print(f'S5 R10(iv): t={t} eps={eps}: Pr(sup_s>=t |P_s-P_t| >= eps) = {lhs:.4f} <= E|P_H-P_t|^2/eps^2 = {rhs:.4f}')
