"""Exact eps-ranks for ray fronts and one random back layer, for a family of activations (theory numbers; no networks).

Generalises work/cut-rank/scratch/ray_eps_scaling.py (Theorem E) from ReLU to any activation phi.
Model: fronts h = s R0 u_f, u_f uniform on S^{n-1} (one direction per front), R0 ~ Gamma(1/vR, vR) (E R0 = 1,
Var R0 = vR) independent of u_f; back = one unit phi(w.h), w ~ N(0, I_n) (GLU: phi(w1.h) (w2.h)). The pre-activation
scale is s R0 (s = sqrt 2 is He-ReLU's). With Phibar(x) = E_R phi(s R0 x), the cut Gram is G(f, f') = kt(u_f.u_f'),
kt(t) = sum_k b_k t^k, b_k = (E_R c_k(s R0))^2 (Mehler), and the per-net variance is
Vbar = sum_k [E_R c_k(s R0)^2 - (E_R c_k(s R0))^2]. The squared singular values are the Funk-Hecke eigenvalues of kt
on S^{n-1}: mu_l = E_l / N(n, l), E_l = sum_{k >= l, k = l mod 2} b_k e_{k,l}, e_{k,l} = share of t^k's energy at
degree l (Gegenbauer expansion of x^k). r(eps) = min{s : tail <= eps^2 Vbar}; fractional degrees handled as in the
lane's code. ReLU uses the lane's closed-form eigenvalues (series would need K ~ l^2 terms).
Outputs: logs/ray_family.log (ranks), runs/ray_family.json.
usage: python3 scratch/ray_family.py [K]
"""
import json, math, os, sys
import numpy as np
from scipy import special as sp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import acts as A

K = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
EPS = [0.3, 0.1, 0.03, 0.01, 1e-3, 1e-4, 1e-6]
NS = [4, 8, 16, 32, 64]
S0 = math.sqrt(2.0)


def radial_nodes(vR, m=24):
    a = 1.0 / vR - 1.0
    x, w = sp.roots_genlaguerre(m, a)
    return vR * x, w / w.sum()


def cstats(name, p, s, vR, glu=False):
    """(b_k, Vbar, missing) for k = 0..K with radial averaging; missing = energy beyond K (from Parseval)."""
    R, wR = radial_nodes(vR)
    C = np.stack([A.coef(name, p, K, s * r) for r in R])          # (m, K+1)
    if glu:
        C = C * (s * R)[:, None]
    m1 = wR @ C; m2 = wR @ (C * C)
    b = m1 ** 2
    if glu:
        b = np.concatenate([[0.0], b])                             # extra factor t: b_{k+1}
    # Parseval check: E_R E_X phi(sR X)^2 (x (sR)^2 for GLU) against sum m2 -> missing energy beyond K
    e2 = np.array([A.moments(name, p, s * r)[1] for r in R])
    if glu:
        e2 = e2 * (s * R) ** 2
    tot2 = float(wR @ e2)
    miss = max(tot2 - float(m2.sum()), 0.0)
    Vbar = float(m2.sum() - m1 @ m1)
    return b, Vbar, miss


def lN(n, l):
    l = np.asarray(l, dtype=np.float64)
    out = np.zeros_like(l)
    pos = l > 0
    lp = l[pos]
    out[pos] = np.log(2 * lp + n - 2) - np.log(lp) + sp.gammaln(lp + n - 2) - sp.gammaln(n - 1) - sp.gammaln(lp)
    return out


def degree_energy(b, n, Lmax):
    """E_l = sum_k b_k e_{k,l}, l = 0..Lmax."""
    lam = (n - 2) / 2.0
    Kb = len(b) - 1
    E = np.zeros(Lmax + 1)
    for l in range(Lmax + 1):
        k = np.arange(l, Kb + 1, 2)
        if k.size == 0:
            continue
        j = (k - l) / 2
        le = (sp.gammaln(k + 1) + math.log(lam + l) - k * math.log(2) - sp.gammaln(j + 1)
              - (sp.gammaln(lam + (k + l) / 2 + 1) - sp.gammaln(lam))
              + (sp.gammaln(2 * lam + l) - sp.gammaln(2 * lam)) - sp.gammaln(l + 1))
        bb = b[k]
        nz = bb > 0
        E[l] = np.sum(bb[nz] * np.exp(le[nz]))
    return E


def rank_from_degrees(E, n, total, thr):
    """Exact eps-rank from per-degree energies E_l (multiplicity N(n,l)); total = full energy (incl. beyond Lmax)."""
    l = np.arange(len(E))
    lm = lN(n, l)
    with np.errstate(divide='ignore'):
        lmu = np.where(E > 0, np.log(np.maximum(E, 1e-320)) - lm, -np.inf)
    order = np.argsort(-lmu)
    tail = total; cnt = 0.0
    for i in order:
        if not np.isfinite(lmu[i]):
            break
        if tail <= thr:
            return cnt, True
        e = E[i]
        if tail - e <= thr:
            return cnt + math.ceil((tail - thr) / math.exp(lmu[i])), True
        tail -= e; cnt += math.exp(lm[i])
    return (cnt, tail <= thr)


# --- ReLU closed form (the lane's Theorem E code, eta_l for (u.v)_+ with w ~ N(0, I)) --------------------------
def relu_rank(n, eps, vR):
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'cut-rank', 'scratch'))
    import ray_eps_scaling as RE
    return RE.eps_rank(n, eps, vR=vR)


ARMS = [('relu', None, False), ('softplus', 32.0, False), ('softplus', 16.0, False), ('softplus', 8.0, False),
        ('softplus', 4.0, False), ('softplus', 2.0, False), ('softplus', 1.0, False), ('gelu', None, False),
        ('silu', None, False), ('tanh', None, False), ('gsrelu', 0.25, False), ('gsrelu', 0.5, False),
        ('gsrelu', 1.0, False), ('smoothstep', 1.0, False), ('relu2', None, False),
        ('silu', None, True), ('gelu', None, True), ('linear', None, True)]

if __name__ == '__main__':
    out = {}
    TAG = os.environ.get('RAY_TAG', '')
    ONLY = [x for x in os.environ.get('RAY_ARMS', '').split(',') if x]
    VRS = [float(x) for x in os.environ.get('RAY_VR', '1.0,0.1').split(',')]
    if ONLY:
        ARMS = [a for a in ARMS if (('glu-' if a[2] else '') + A.label(a[0], a[1])) in ONLY]
    log = open(os.path.join(HERE, '..', 'logs', f'ray_family{TAG}.log'), 'w')

    def pr(*a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True); log.write(s + '\n'); log.flush()

    pr(f'# ray fronts, one back layer, s = sqrt 2, K = {K}; ranks at eps = {EPS}; "unres" = energy beyond the '
       f'computed degrees exceeds the threshold (rank not resolved); 1/eps^2 = {[round(1 / e ** 2) for e in EPS]}')
    for vR in VRS:
        pr(f'## vR = {vR}')
        for name, p, glu in ARMS:
            lab = ('glu-' if glu else '') + A.label(name, p)
            kk = K
            if name == 'relu' and not glu:
                rows = {n: [relu_rank(n, e, vR) for e in EPS] for n in NS}
                for n in NS:
                    pr(f'{lab:14s} n={n:3d} ' + ' '.join(f'{r:9.3g}' for r in rows[n]))
                out[f'{lab}|{vR}'] = {str(n): rows[n] for n in NS}
                continue
            Ksave = K
            globals()['K'] = kk
            b, Vbar, miss = cstats(name, p, S0, vR, glu)
            globals()['K'] = Ksave
            res = {}
            for n in NS:
                Lmax = min(len(b) - 1, 600)
                E = degree_energy(b, n, Lmax)
                total = float(b.sum()) + miss          # missing energy beyond K is assigned to degrees > Lmax
                beyond = total - E.sum()
                row = []
                for e in EPS:
                    thr = e * e * Vbar
                    if beyond > thr:
                        row.append(None)
                        continue
                    r, ok = rank_from_degrees(E, n, total, thr)
                    row.append(r if ok else None)
                res[str(n)] = row
                pr(f'{lab:14s} n={n:3d} ' + ' '.join(('    unres' if r is None else f'{r:9.3g}') for r in row)
                   + f'   Vbar {Vbar:.3g} beyond {beyond:.2e} miss {miss:.1e}')
            out[f'{lab}|{vR}'] = res
    json.dump(out, open(os.path.join(HERE, '..', 'runs', f'ray_family{TAG}.json'), 'w'), indent=1)
    pr('DONE')
