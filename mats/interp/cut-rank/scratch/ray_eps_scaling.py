"""Exact eps-rank for ray fronts and one random ReLU back layer on the full sphere S^{n-1} (Theorem E check).

Fronts: h = R u, u uniform on S^{n-1}, R independent with E R = 1, Var R = vR. Back: one row w ~ N(0, I_n), output
(w.h)_+ (exact at any width: one layer has the arc-cosine kernel exactly). Squared singular values: n eta_l^2 with
multiplicity N(n, l) (astra's closed-form eta_l; odd l >= 3 vanish); total = 1/2; Vbar = vR / 2.
r(eps) = smallest s with tail <= eps^2 Vbar. Prediction: r ~ C_n (eps^2 vR)^{-(n-1)/3} as eps -> 0 (fixed n), so
r grows faster than sampling's 1/eps^2 iff (n-1)/3 > 1, i.e. n >= 5.
usage: python3 scratch/ray_eps_scaling.py
"""
import math
import numpy as np


def lg_eta2(n, l):
    """log of eta_l^2 for (u.v)_+ on S^{n-1} (closed form), or None if eta_l = 0."""
    if l == 0:
        return 2 * (math.lgamma(n / 2) - math.log(2 * math.sqrt(math.pi)) - math.lgamma((n + 1) / 2))
    if l == 1:
        return 2 * math.log(1 / (2 * n))
    if l % 2 == 1:
        return None
    m = l // 2
    return 2 * (math.lgamma(n / 2) + math.lgamma(m - 0.5) - math.log(4 * math.pi) - math.lgamma(m + (n + 1) / 2))


def lg_N(n, l):
    if l == 0:
        return 0.0
    # N(n,l) = (2l+n-2)/l * C(l+n-3, n-2)
    return math.log(2 * l + n - 2) - math.log(l) + math.lgamma(l + n - 2) - math.lgamma(n - 1) - math.lgamma(l)


def eps_rank(n, eps, vR=1.0, lmax=200000):
    thr = eps * eps * vR / 2
    ev, mult = [], []
    for l in range(lmax + 1):
        a = lg_eta2(n, l)
        if a is None:
            continue
        ev.append(math.log(n) + a); mult.append(lg_N(n, l))
    ev = np.array(ev); mult = np.array(mult)
    order = np.argsort(-ev)
    ev, mult = ev[order], mult[order]
    E = np.exp(ev + mult)                     # energy per degree
    tail_after = 0.5 - np.cumsum(E)          # tail after taking all harmonics of the first j degrees
    cnt = np.cumsum(np.exp(mult))
    j = int(np.argmax(tail_after <= thr))
    if tail_after[j] > thr:
        return float('inf')
    before = 0.5 - (np.cumsum(E)[j - 1] if j > 0 else 0.0)
    need = before - thr
    return (cnt[j - 1] if j > 0 else 0.0) + math.ceil(need / math.exp(ev[j]))


if __name__ == '__main__':
    print('total energy check (should be 0.5):', end=' ')
    for n in (3, 6, 12):
        s = sum(math.exp(math.log(n) + lg_eta2(n, l) + lg_N(n, l)) for l in range(0, 20000) if lg_eta2(n, l) is not None)
        print(f'n={n}: {s:.6f}', end='  ')
    print()
    epss = [0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3]
    print('n | r(eps) at eps = ' + ' / '.join(map(str, epss)) + ' | local exponent dlog r / dlog(1/eps) at the two smallest eps | predicted 2(n-1)/3')
    for n in (3, 4, 5, 6, 8, 10, 12, 16):
        rs = [eps_rank(n, e) for e in epss]
        ex = math.log(rs[-1] / rs[-2]) / math.log(epss[-2] / epss[-1])
        print(f'{n} | ' + ' / '.join(f'{r:.3g}' for r in rs) + f' | {ex:.2f} | {2 * (n - 1) / 3:.2f}', flush=True)
