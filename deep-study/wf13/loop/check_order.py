"""check_order.py -- exact check of NOTE section 1 on a tiny world (4 binary variables, a random positive score R that
does not depend on the order). A rollout policy picks the next variable with probability pi(i | state) and its value
with probability q(v | state, i); both are arbitrary random tables with full support. By enumerating every route:
 (a) weight R(x) / prod q                counts every FILLING once:  E[w 1(x)] = R(x)               (order cancels)
 (b) weight R(x) / (prod pi * prod q)    counts every DEMONSTRATION (order + filling) once: E[w 1(order, x)] = R(x)
 (c) unweighted counting of rollouts gives the policy's own law, and counting visits (weight R(x), no division)
     over-weights fillings the policy likes.
Both (a) and (b) give the same answers about fillings: sum over the 4! orders of (b) = 24 R(x). -> out/check_order.out"""
import itertools
import numpy as np

rng = np.random.default_rng(0)
n = 4
out = []
worst = dict(a=0.0, b=0.0, ab=0.0, c=0.0)
for trial in range(200):
    R = rng.gamma(0.5, size=2 ** n) + 1e-3
    pi, q = {}, {}

    def tables(state):
        if state not in pi:
            op = [i for i in range(n) if state[i] < 0]
            p = rng.dirichlet(np.ones(len(op)) * 0.5)
            pi[state] = dict(zip(op, p))
            q[state] = {i: rng.beta(0.5, 0.5) * 0.98 + 0.01 for i in op}
        return pi[state], q[state]

    Ea = np.zeros(2 ** n)                       # expected weight (a) per filling
    Eb = {}                                     # expected weight (b) per (order, filling)
    Pf = np.zeros(2 ** n)                       # the policy's own law over fillings
    Ev = np.zeros(2 ** n)                       # expected "visit count x R" per filling

    def rec(state, order, ppi, pq):
        if all(v >= 0 for v in state):
            x = sum(v << i for i, v in enumerate(state))
            Ea[x] += ppi * pq * R[x] / pq
            Eb[(order, x)] = Eb.get((order, x), 0.0) + ppi * pq * R[x] / (ppi * pq)
            Pf[x] += ppi * pq
            Ev[x] += ppi * pq * R[x]
            return
        p, qq = tables(state)
        for i, pi_i in p.items():
            for v in (0, 1):
                s2 = list(state)
                s2[i] = v
                rec(tuple(s2), order + (i,), ppi * pi_i, pq * (qq[i] if v else 1 - qq[i]))
    rec(tuple([-1] * n), (), 1.0, 1.0)
    T = R / R.sum()
    worst['a'] = max(worst['a'], np.abs(Ea - R).max())
    worst['b'] = max(worst['b'], max(abs(v - R[x]) for (o, x), v in Eb.items()))
    sb = np.zeros(2 ** n)
    for (o, x), v in Eb.items():
        sb[x] += v
    worst['ab'] = max(worst['ab'], np.abs(sb / sb.sum() - Ea / Ea.sum()).max())
    worst['c'] = max(worst['c'], 0.5 * np.abs(Ev / Ev.sum() - T).sum())
    assert len(Eb) == 24 * 16
s = ('200 random worlds, every route enumerated (4 variables, 24 orders x 16 fillings):\n'
     ' (a) max |E[w 1(x)] - R(x)| with weight R / prod q (order probabilities NOT divided out): %.2e\n'
     ' (b) max |E[w 1(order, x)] - R(x)| with weight R / (prod pi prod q): %.2e\n'
     ' (a) and (b) give the same law over fillings: max difference %.2e\n'
     ' (c) counting visits with weight R and no division: total variation to T up to %.3f\n'
     % (worst['a'], worst['b'], worst['ab'], worst['c']))
print(s)
open('/home/agent/workspaces/mats/work/deep-study/wf13/loop/out/check_order.out', 'w').write(s)
