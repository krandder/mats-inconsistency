"""Post hoc (not registered): test of astra's M-design via kernel values. For each net, with f = a_L at unit inputs:
  k(-1) = E_u <f(u), f(-u)>,  k(0) = E <f(u), f(v)> (v uniform on u's orthogonal complement),
  q = |E_u f(u)|^2 (split-half unbiased),  a_2 = degree-2 harmonic energy (split-half unbiased),
  b_2 ~ a_2 n/(n-1) (t^2 = P_0/n + (n-1)/n P_2; the O(b_4/n) correction is ignored).
Then even mass above degree 2: E4 = (1 + k(-1))/2 - k(0) - b_2 (requires k(1) = 1, checked: kk1 = E|f(u)|^2), and the
bound of astra's (2.5): positive-weight designs with N <= n(n+2)/6 gain at most 2 (1 - q) / E4 over RB-MC.
Ensemble values are ratios of averages over nets. Usage: kernel_even.py n L seeds(comma) [--m 16]
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from nets import Net, sphere_from_gauss  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('n', type=int); ap.add_argument('L', type=int); ap.add_argument('seeds')
ap.add_argument('--m', type=int, default=16)
a = ap.parse_args()
n, L = a.n, a.L
per = []
for s in [int(x) for x in a.seeds.split(',')]:
    t0 = time.time()
    net = Net(s, n, L)
    rng = np.random.default_rng([777, s, n, L])
    N = 1 << a.m
    acc = dict(km1=0.0, k0=0.0, k1=0.0)
    halves = []
    J = min(n, 16)
    for h in range(2):
        sm = np.zeros(n); s2 = np.zeros((J, n * n))
        for c in range(0, N // 2, 4096):
            U = sphere_from_gauss(rng.standard_normal((4096, n)))
            G = rng.standard_normal((4096, n))
            G -= np.sum(G * U, axis=1, keepdims=True) * U
            Vv = sphere_from_gauss(G)
            fu = net.forward(U, noise=False); fm = net.forward(-U, noise=False); fv = net.forward(Vv, noise=False)
            acc['km1'] += float(np.sum(fu * fm)); acc['k0'] += float(np.sum(fu * fv)); acc['k1'] += float(np.sum(fu * fu))
            sm += fu.sum(0)
            Q = (U[:, :, None] * U[:, None, :]).reshape(len(U), n * n); Q[:, ::n + 1] -= 1.0 / n
            s2 += fu[:, :J].T @ Q
        halves.append((sm / (N // 2), s2 / (N // 2)))
    (m0, b0), (m1, b1) = halves
    km1, k0, k1 = acc['km1'] / N, acc['k0'] / N, acc['k1'] / N
    q = float(np.sum(m0 * m1))
    a2J = float(n * (n + 2) / 2 * np.sum(b0 * b1))
    # scale the degree-2 energy of the first J outputs to all n outputs by their share of E|f|^2
    a2 = a2J * n / J
    per.append(dict(seed=s, km1=km1, k0=k0, k1=k1, q=q, a2=a2, seconds=time.time() - t0))
    print(f'n={n} L={L} seed={s}: k(1)={k1:.3f} k(-1)={km1:.3f} k(0)={k0:.3f} q={q:.3f} a2={a2:.4f} {time.time() - t0:.0f}s', flush=True)
E = {k: float(np.mean([p[k] for p in per])) for k in ('km1', 'k0', 'k1', 'q', 'a2')}
b2 = E['a2'] * n / (n - 1)
E4 = (E['k1'] + E['km1']) / 2 - E['k0'] - b2
res = dict(n=n, L=L, per=per, ens=E, b2=b2, even_mass_ge4=E4, ang=E['k1'] - E['q'],
           even_ge4_over_ang=E4 / (E['k1'] - E['q']), design_gain_bound=2 * (E['k1'] - E['q']) / E4 if E4 > 0 else None)
print(f'n={n} L={L}: even mass >= 4 = {E4:.4f}, angular {E["k1"] - E["q"]:.4f}, ratio {E4 / (E["k1"] - E["q"]):.3f}, '
      f'positive-design gain bound (N <= n(n+2)/6) {res["design_gain_bound"]}', flush=True)
json.dump(res, open(os.path.join(os.path.dirname(__file__), '..', 'runs', f'keven_n{n}_L{L}.json'), 'w'))
