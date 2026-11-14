"""Compare deterministic estimators (rows with cost_paths, nmse) with RQMC/MC at equal FLOPs.
usage: python compare.py SAMPLERSJSON ESTJSON [ESTJSON ...]
RQMC's nMSE at an arbitrary cost N (in paths) is interpolated log-log between measured budgets (extrapolated with the
end slopes); MC is 1/N. Prints, per row: cost, nMSE, RQMC(N), MC(N), ratio RQMC/est (>1 means the estimator wins).
"""
import sys, json, math
import numpy as np


def rqmc_curve(S):
    xs, ys = [], []
    for lg, d in sorted(S['est'].items(), key=lambda kv: int(kv[0])):
        if 'RQMC' in d and d['RQMC']['nmse'] > 0:
            xs.append(math.log(2 ** int(lg))); ys.append(math.log(d['RQMC']['nmse']))
    xs, ys = np.array(xs), np.array(ys)

    def f(N):
        x = math.log(max(N, 1e-9))
        if x <= xs[0]:
            s = (ys[1] - ys[0]) / (xs[1] - xs[0]); return math.exp(ys[0] + s * (x - xs[0]))
        if x >= xs[-1]:
            s = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]); return math.exp(ys[-1] + s * (x - xs[-1]))
        return math.exp(float(np.interp(x, xs, ys)))
    return f


if __name__ == '__main__':
    S = json.load(open(sys.argv[1]))
    f = rqmc_curve(S)
    for p in sys.argv[2:]:
        E = json.load(open(p))
        for r in E['rows']:
            N = r['cost_paths']
            rq = f(N)
            print(f"{p.split('/')[-1]} {r.get('F','')} c={r.get('c')} tau={r.get('tau','')} k={r.get('k')} G={r.get('G')}: "
                  f"cost {N:.0f} nMSE {r['nmse']:.3g} | RQMC {rq:.3g} MC {1/N:.3g} | RQMC/est {rq/r['nmse']:.3g}")
