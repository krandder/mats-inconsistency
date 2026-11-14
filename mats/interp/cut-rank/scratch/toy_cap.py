"""Lemma C (cap toy), exact spectrum: one random ReLU layer back, von Mises-Fisher fronts with uniform centres on S^{k-1}.

kappa(t) = E relu(w.u) relu(w.u'), w ~ N(0, I_k): (sqrt(1 - t^2) + (pi - arccos t) t) / (2 pi)  (Cho-Saul degree 1).
Funk-Hecke: kappa(t) = sum_d E_d P_d(t), P_d the dimension-k Legendre polynomial with P_d(1) = 1, E_d = N(k,d) mu_d.
vMF front of concentration K: multiplier lambda_d = I_{d+nu}(K) / I_nu(K), nu = k/2 - 1 (exact).
Isotropic centres => C is diagonal: eigenvalue mu_d lambda_d^2 with multiplicity N(k, d).
Vbar = sum_d E_d (1 - lambda_d^2) (variance within fronts; radius fixed at 1).
r(eps) = smallest s with tail <= eps^2 Vbar. Also checks E_d by direct quadrature and the d^-4 decay.
usage: python3 scratch/toy_cap.py
"""
import math
import numpy as np
from scipy import special

DMAX = 400


def N_harm(k, d):
    if d == 0:
        return 1
    return math.comb(d + k - 1, k - 1) - math.comb(d + k - 3, k - 1)


def kappa(t):
    t = np.clip(t, -1, 1)
    return (np.sqrt(1 - t * t) + (np.pi - np.arccos(t)) * t) / (2 * np.pi)


def energies(k, dmax=DMAX, nq=None):
    """Exact energies E_d = N(k,d) mu_d of kappa(t) = E relu(w.u) relu(w.u'), w ~ N(0, I_k) (closed form, gpt-6-astra
    round 2; replaces the earlier quadrature, which was unstable for larger k): mu_0 = k eta_0^2, mu_1 = 1/(4k),
    mu_{2j} = k [Gamma(k/2) Gamma(j-1/2) / (4 pi Gamma(j+(k+1)/2))]^2, odd d >= 3 zero."""
    E = np.zeros(dmax + 1)
    for d in range(dmax + 1):
        if d == 0:
            eta = math.gamma(k / 2) / (2 * math.sqrt(math.pi) * math.gamma((k + 1) / 2)) if k < 300 else None
            lmu = math.log(k) + 2 * (math.lgamma(k / 2) - math.log(2 * math.sqrt(math.pi)) - math.lgamma((k + 1) / 2))
        elif d == 1:
            lmu = math.log(1 / (4 * k))
        elif d % 2 == 1:
            continue
        else:
            j = d // 2
            lmu = math.log(k) + 2 * (math.lgamma(k / 2) + math.lgamma(j - 0.5) - math.log(4 * math.pi) - math.lgamma(j + (k + 1) / 2))
        E[d] = math.exp(lmu) * N_harm(k, d)
    return E


def vmf_mult(k, K, dmax=DMAX):
    nu = k / 2 - 1
    d = np.arange(dmax + 1)
    return special.ive(d + nu, K) / special.ive(nu, K)


def toy_rank(k, s_ang, eps_list=(0.3, 0.1, 0.03, 0.01), E=None):
    E = energies(k) if E is None else E
    lam = vmf_mult(k, 1.0 / s_ang ** 2)
    Nd = np.array([N_harm(k, d) for d in range(len(E))], dtype=float)
    ev = np.where(Nd > 0, E / np.maximum(Nd, 1), 0) * lam ** 2   # eigenvalue per harmonic of degree d
    Vbar = float(0.5 - np.sum(E * lam ** 2))
    order = np.argsort(-ev)
    evs, mult = ev[order], Nd[order]
    tot = float(np.sum(E * lam ** 2))
    out = {}
    for eps in eps_list:
        thr = eps * eps * Vbar
        cum_mass, cum_n = 0.0, 0.0
        r = None
        if tot <= thr:
            r = 0
        else:
            for e, m in zip(evs, mult):
                if e <= 0:
                    continue
                need = tot - cum_mass - thr          # energy still to capture
                if e * m >= need:
                    r = cum_n + math.ceil(need / e)
                    break
                cum_mass += e * m; cum_n += m
        out[eps] = r
    return out, Vbar, tot


if __name__ == '__main__':
    E3 = energies(3)
    print('E_d (k=3), d = 0..12:', np.round(E3[:13], 6))
    print('check sum E_d vs kappa(1):', E3.sum(), kappa(1.0))
    dd = np.arange(20, 200, 2)
    sl = np.polyfit(np.log(dd), np.log(E3[dd]), 1)[0]
    print('E_d decay exponent for even d (k=3):', round(sl, 3), ' odd d>=3 max |E_d|:', float(np.max(np.abs(E3[3::2]))))
    for s_ang in (0.3, 0.1, 0.03):
        print(f'angular spread per direction s = {s_ang}')
        for k in (2, 3, 4, 6, 8, 12, 16, 24):
            E = energies(k)
            if abs(E.sum() - 0.5) > 2e-3:
                print(f'  k={k}: energy check failed, sum E_d = {E.sum():.6f}'); continue
            r, Vbar, tot = toy_rank(k, s_ang, E=E)
            print(f'  k={k:2d}  r(eps) for eps=0.3/0.1/0.03/0.01: {[int(r[e]) if r[e] is not None else None for e in (0.3, 0.1, 0.03, 0.01)]}  '
                  f'Vbar={Vbar:.3e} EM2/V={tot / Vbar:.1f}', flush=True)
