import sys, time, math, json
sys.path.insert(0, 'lib')
import numpy as np
from radius_chain import RadiusLaw, kesten_C, E_AlogA
for n in [8, 64]:
    t0 = time.time()
    R = RadiusLaw(n)
    t1 = time.time()
    qs = [R.Q_quantile(q) for q in (0.1, 0.5, 0.9, 0.99)]
    print(n, 'N', R.u.size, 'h', round(R.h,4), 'lam-1', R.lam - 1, 'pi0', R.pi0, 'colmass dev max (interior)', np.max(np.abs(R.colmass[R.u < R.umax - 5] - 1)), 'build+solve s', round(t1-t0,1))
    print('  quantiles', [round(x, 5) for x in qs], 'median/n', qs[1]/n)
    print('  C_n', kesten_C(n), ' x*P(Q>x) at x=1e4n,1e6n:', [x*R.Q_sf(x) for x in (1e4*n, 1e6*n)])
    print('  CDF at 0.5 n, n:', R.Q_cdf(0.5*n), R.Q_cdf(n), ' sf at n', R.Q_sf(n), ' sum', R.Q_cdf(n)+R.Q_sf(n))
