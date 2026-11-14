"""c1_order.py -- checker's own re-derivation of claim 1 (the search's order cancels in weighted counting).
Own code, own random worlds: n variables with 2 or 3 values, a state-dependent order policy pi(i | s) and value policy
q(v | s, i), both arbitrary random tables; score R(x) arbitrary >= 0 (some zeros). Enumerates EVERY route exactly.
 (a) weight R / prod q                -> E[w 1(filling = x)] should equal R(x) for every x
 (b) q with a zero where R > 0        -> the estimate must fail (support condition is needed)
 (c) R that depends on the route order (soft M) -> (a) no longer returns a function of x alone: reports what it returns
"""
import itertools
import numpy as np

rng = np.random.default_rng(20261110)


def run(n, kv, zero_support=False, order_dep=False):
    vals = list(itertools.product(range(kv), repeat=n))
    R = {x: (rng.random() if rng.random() > 0.2 else 0.0) for x in vals}
    Rord = {}
    pi_tab, q_tab = {}, {}

    def pi(s):
        if s not in pi_tab:
            op = [i for i in range(n) if s[i] < 0]
            p = rng.dirichlet(np.ones(len(op)) * 0.5)
            pi_tab[s] = dict(zip(op, p))
        return pi_tab[s]

    def q(s, i):
        if (s, i) not in q_tab:
            p = rng.dirichlet(np.ones(kv) * 0.5)
            if zero_support and i == 0:
                p[0] = 0.0
                p = p / p.sum()
            q_tab[(s, i)] = p
        return q_tab[(s, i)]
    est = {x: 0.0 for x in vals}
    tot_prob = 0.0

    def rec(s, prob, qprod, route):
        nonlocal tot_prob
        if all(v >= 0 for v in s):
            tot_prob += prob
            r = R[s]
            if order_dep:
                key = (s, tuple(route))
                if key not in Rord:
                    Rord[key] = r * (0.5 + rng.random())
                r = Rord[key]
            est[s] += prob * r / qprod
            return
        for i, p_i in pi(s).items():
            qq = q(s, i)
            for v in range(kv):
                if qq[v] == 0:
                    continue
                t = list(s)
                t[i] = v
                rec(tuple(t), prob * p_i * qq[v], qprod * qq[v], route + [i])
    rec(tuple([-1] * n), 1.0, 1.0, [])
    err = max(abs(est[x] - R[x]) for x in vals)
    Z = sum(R.values())
    Ze = sum(est.values())
    tv = 0.5 * sum(abs(est[x] / Ze - R[x] / Z) for x in vals)
    return err, tv, tot_prob


for name, kw in [('(a) full support', {}), ('(b) one value never proposed', dict(zero_support=True)),
                 ('(c) score depends on the route order', dict(order_dep=True))]:
    E, TV = [], []
    for w in range(60):
        n, kv = (4, 2) if w % 2 == 0 else (3, 3)
        e, tv, tp = run(n, kv, **kw)
        assert abs(tp - 1) < 1e-12
        E.append(e)
        TV.append(tv)
    print('%-40s 60 worlds: max |E[w 1(x)] - R(x)| = %.3g ; total variation to T: max %.3g mean %.3g' % (name, max(E), max(TV), np.mean(TV)))
