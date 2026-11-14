"""Part 4: paradoxical sentence chi <-> (P(chi) < p) with history-free traders at a fixed point.
Truth of chi at price x: W_x(chi) = 1[x < p].  Trade value of share count s at price x: s*(1[x<p] - x).
Trader T_delta(x) = Ind_delta(x<p) - x, Ind_delta(x<p) = clip((p-x)/delta, 0, 1)  (continuous, resolution delta).
Market-maker map fix(x) = clip(x + T(x), 0, 1)."""
import numpy as np
from scipy.optimize import brentq
p = 0.8
ind = lambda x, d: np.clip((p - x)/d, 0, 1)
T = lambda x, d: ind(x, d) - x
print('(4a) fixed point of fix_delta vs formula p/(1+delta):')
for d in [0.1, 0.01, 0.001, 1e-6]:
    xs = brentq(lambda x: T(x, d), 0, 1, xtol=1e-15)
    print(f'   delta={d:<7g} x*={xs:.10f}  p/(1+delta)={p/(1+d):.10f}  |x*-p|={abs(xs-p):.2e}  chi true at x*: {xs < p}')
# (4b) discontinuous trader (delta=0): no fixed point
grid = np.linspace(0, 1, 2000001)
Td0 = (grid < p).astype(float) - grid
fix0 = np.clip(grid + Td0, 0, 1)
print(f'(4b) delta=0: min_x |fix(x)-x| over grid = {np.min(np.abs(fix0-grid)):.4f} (>0: no fixed point); '
      f'inf_x |T(x)| = {np.min(np.abs(Td0)):.4f}')
x = 0.5; orbit = []
for t in range(8):
    x = float(np.clip(x + ((x < p) - x), 0, 1)); orbit.append(x)
print('     delta=0 block iteration orbit:', orbit)
# (4c) block iteration x <- clip(x + eta*T_delta(x)): local slope 1-eta(1+1/delta); stable iff eta < 2*delta/(1+delta)
d = 0.01; xs = p/(1+d)
print('(4c) block dynamics, delta=0.01, threshold eta_c = 2d/(1+d) =', round(2*d/(1+d), 6))
for eta in [0.005, 0.015, 0.025, 0.05]:
    x = 0.3; hist = []
    for t in range(5000):
        x = float(np.clip(x + eta*T(x, d), 0, 1)); hist.append(x)
    hist = np.array(hist); err = np.abs(hist - xs)
    stay = next((i for i in range(len(err)) if np.all(err[i:] < 1e-6)), None)
    print(f'   eta={eta:<6g} slope={1-eta*(1+1/d):+.3f}  final |x-x*|={err[-1]:.2e}  last-100 spread={np.ptp(hist[-100:]):.2e}  blocks until it stays within 1e-6: {stay}')
# (4d) static mixed-resolution population (weights fixed): fixed point is set by the COARSE traders
deltas = np.array([0.5, 0.1, 0.01, 0.001]); wts = np.ones(4)/4
Tpop = lambda x, w: float(np.sum(w*(np.clip((p-x)/deltas, 0, 1) - x)))
xs = brentq(lambda x: Tpop(x, wts), 0, 1, xtol=1e-15)
print(f'(4d) static equal weights over deltas {deltas}: x* = {xs:.5f} (p - x* = {p-xs:.3f}); every delta-truth-seeking pop has x* in [p - max delta, p)')
# (4e') wealth kept in the block STATE (Markov, no attention to past prices): fixed point each day vs wealth-weighted traders,
#       wealth updated by realised value (truth of chi is fixed by the day's price). Coarse traders lose -> x* -> p.
w = wts.copy(); alpha = 0.5
for day in range(1, 2001):
    xs = brentq(lambda x: Tpop(x, w), 0, 1, xtol=1e-15)
    truth = float(xs < p)
    s_k = np.clip((p-xs)/deltas, 0, 1) - xs                     # shares per unit wealth
    w = w*(1 + alpha*s_k*(truth - xs)); w = w/w.sum()
    if day in (1, 10, 100, 1000, 2000):
        print(f'   day {day:>4}: x* = {xs:.6f}  p-x* = {p-xs:.2e}  wealth shares = {w.round(4)}')
# (4f) any continuous trader: value v(x)=T(x)(1[x<p]-x) cannot be >0 both at p and just below p
rng = np.random.default_rng(2); bad = 0
for _ in range(10000):
    knots = np.sort(rng.random(6)); vals = rng.normal(size=6)
    Tr = lambda x: np.interp(x, knots, vals)
    v_at_p = Tr(p)*(0 - p); v_below = Tr(p - 1e-9)*(1 - (p - 1e-9))
    if v_at_p > 1e-6 and v_below > 1e-6: bad += 1
print(f'(4f) random continuous traders with positive value both at p and at p-1e-9: {bad} / 10000 (lemma: 0)')
