"""Exact finite-n radius law (Nystrom solver) against the 1/n and 1/n^2 Kramers-Moyal expansion, Kesten constant
and left tail. Output: results/radius_corrections.json.
usage: radius_corrections.py n1,n2,... --out file"""
import sys, os, math, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
import numpy as np
from scipy.special import gammaln
from radius_chain import RadiusLaw, kesten_C, E_AlogA

ap = argparse.ArgumentParser()
ap.add_argument('ns')
ap.add_argument('--out', required=True)
a = ap.parse_args()

XS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 2 / (5 * math.log(2)), 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0]


def F0(x):
    return math.exp(-0.4 / x)


def F1(x):
    return -(65 * x - 47) * math.exp(-0.4 / x) / (375 * x ** 2)


def F2(x):
    return -(198375 * x ** 3 - 18825 * x ** 2 + 63640 * x - 6627) * math.exp(-0.4 / x) / (843750 * x ** 4)


res = {}
if os.path.exists(a.out):
    res = json.load(open(a.out))
for n in [int(v) for v in a.ns.split(',')]:
    t0 = time.time()
    R = RadiusLaw(n)
    d = dict(n=n, N=int(R.u.size), h=R.h, umin=R.umin, umax=R.umax, lam_minus_1=R.lam - 1, pi0=R.pi0)
    d['quantiles'] = {str(q): R.Q_quantile(q) for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)}
    d['median_over_n'] = d['quantiles']['0.5'] / n
    rows = []
    for x in XS:
        Fe = R.Q_cdf(x * n) if x < 1 else 1.0 - R.Q_sf(x * n)
        sfe = R.Q_sf(x * n)
        rows.append(dict(x=x, F=Fe, sf=sfe, F0=F0(x), F1=F1(x), F2=F2(x),
                         R1=n * (Fe - F0(x)) - F1(x), R2=n * n * (Fe - F0(x) - F1(x) / n) - F2(x)))
    d['cdf_rows'] = rows
    C = kesten_C(n)
    d['C_n'] = C
    d['C_series'] = 0.4 * n + 13 / 75 + 529 / (2250 * n)
    d['m_n_times_n3_resid'] = (E_AlogA(n) - 2.5 / n + 13 / (12 * n * n)) * n ** 3
    d['tail_xP'] = {str(f): f * n * R.Q_sf(f * n) for f in (1e2, 1e4, 1e6, 1e8)}
    # left tail
    lt = {}
    for q in (1.0, 2.0, 4.0):
        lt['P(Q<=%g)' % q] = R.Q_cdf(q)
        lt['P(r<=%g)' % (q + 1)] = R.r_cdf(q + 1)
    for x in (0.02, 0.05):
        lt['P(Q/n<=%g)' % x] = R.Q_cdf(x * n)
    lt['pi0'] = R.pi0
    d['left_tail'] = lt
    # moments
    elogS = math.log(2.0) + float(__import__('scipy.special', fromlist=['digamma']).digamma(n / 2.0)) - math.log(n)
    d['E_log_Q_over_n'] = R.Q_logmoment() - math.log(n)
    d['E_log_Q_over_n_limit'] = math.log(0.4) + 0.5772156649015329  # E log(2/(5Z)) = log(2/5) + gamma
    Esq_S = math.exp(0.5 * math.log(2.0 / n) + gammaln((n + 1) / 2.0) - gammaln(n / 2.0))
    Esq_r = R.h * float(np.dot(np.sqrt(1.0 + np.exp(R.u)), R.p)) + R.pi0
    d['E_sqrt_Q_over_n'] = Esq_S * Esq_r / math.sqrt(n)
    d['E_sqrt_limit'] = math.sqrt(2 * math.pi / 5)
    d['sec'] = time.time() - t0
    res[str(n)] = d
    json.dump(res, open(a.out, 'w'), indent=1)
    print(n, 'N', d['N'], 'median/n %.10f' % d['median_over_n'], 'C_n %.8f series %.8f' % (C, d['C_series']),
          'R1@median %.3e R2@median %.3e' % (rows[6]['R1'], rows[6]['R2']), 'P(r<=2) %.3e' % lt['P(r<=2)'],
          'P(Q<=1) %.3e' % lt['P(Q<=1)'], '%.1fs' % d['sec'], flush=True)
