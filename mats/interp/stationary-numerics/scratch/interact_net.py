"""Full-network interaction order (Owen's mean dimension over layer blocks) with ONE unbiased estimator, and causal
gate-freezing variants. Model S (sigma = 1), nets = stationary lane's st.W_layer(seed, n, l) (same objects).
Window: layers -J..0 (J = 4n, J+1 blocks), deep past = 12n layers from h = 0 (stationary lane's convention).
Estimator (per net): D_f = (J+1) * mean_s |f_s - f_s^(j_s)|^2 / mean_s |f_s - f'_s|^2, where j_s is uniform over the
window (single-layer refresh) and f' refreshes the whole window (V = E Var(f | deep past)). Unbiased numerator and
denominator; bootstrap SE over samples. f in: h (h_0), logR (log|h_0|), u (h_0/|h_0|), a (relu(W_1 h_0)), g (gates).
Variants (all use the same noises: common random numbers):
  true        : the network.
  frozen      : every gate frozen to the noiseless continuation of the deep past (reference path, independent of the
                window noise): h_0 is then affine in the window noise, so D_h = 1 exactly (code check).
  free_k      : a unit keeps its own gate only if the reference's relative margin |z_i|/rms(z) < k/sqrt(0.577 n)
                (within k noise SDs of the boundary at the typical radius); other units are frozen to the reference.
  recent_a    : own gates only in layers of age < a (a = n/4, n); older layers frozen to the reference.
  clamp       : the stationary lane's radius clamp, h <- Rbar h/|h| after every layer, Rbar = sqrt(2/(5 ln 2)) n.
usage: interact_net.py n seed --P P --variants true,frozen,... --out file"""
import sys, os, math, json, time, argparse
sys.path.insert(0, '/home/agent/workspaces/mats/work/stationary/lib')
import numpy as np
import st

ap = argparse.ArgumentParser()
ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--P', type=int, default=512); ap.add_argument('--variants', default='true')
ap.add_argument('--out', required=True)
a = ap.parse_args()
n, seed, P = a.n, a.seed, a.P
J = 4 * n
NB = J + 1
RBAR = math.sqrt(2 / (5 * math.log(2))) * n
TQ = 1.0 / math.sqrt(0.5770780 * n)
variants = a.variants.split(',')
V = len(variants)
t0 = time.time()
rng = np.random.default_rng([6161, seed, n])
Wf = lambda l: st.W_layer(seed, n, l)

# deep past
h = np.zeros((P, n))
for l in range(-J - 12 * n, -J):
    h = np.maximum(h @ Wf(l).T, 0.0) + rng.standard_normal((P, n))
hdeep = h.copy()
href = hdeep.copy()
jsel = rng.integers(-J, 1, P)  # refreshed layer of each sample
H = np.repeat(hdeep[None, None], V, 0).repeat(3, 1)  # (V, 3 copies, P, n): base, single, all
clampv = np.array([v == 'clamp' for v in variants])


def free_mask(v, l, zref):
    """(P, n) boolean: True where the unit uses its own gate."""
    if v in ('true', 'clamp'):
        return None
    if v == 'frozen':
        return np.zeros(zref.shape, bool)
    if v.startswith('free_'):
        k = float(v[5:])
        rms = np.sqrt(np.mean(zref * zref, 1, keepdims=True))
        return np.abs(zref) < k * TQ * np.maximum(rms, 1e-300)
    if v.startswith('recent_'):
        frac = v[7:]
        aa = {'q': n // 4, 'n': n}[frac]
        return np.full(zref.shape, (-l) < aa)
    raise ValueError(v)


frac_free = {v: [] for v in variants}
for l in range(-J, 1):
    W = Wf(l)
    zref = href @ W.T
    gref = zref > 0
    xi = rng.standard_normal((P, n))
    x1 = rng.standard_normal((P, n))
    x2 = rng.standard_normal((P, n))
    m1 = (jsel == l)[:, None]
    noise = np.stack([xi, np.where(m1, x1, xi), x2])  # (3, P, n)
    Z = H.reshape(-1, n) @ W.T
    Z = Z.reshape(V, 3, P, n)
    for vi, v in enumerate(variants):
        fm = free_mask(v, l, zref)
        if fm is None:
            A = np.maximum(Z[vi], 0.0)
        else:
            g = np.where(fm[None], Z[vi] > 0, gref[None])
            A = Z[vi] * g
            frac_free[v].append(float(fm.mean()))
        Hn = A + noise
        if clampv[vi]:
            Hn = Hn * (RBAR / np.maximum(np.linalg.norm(Hn, axis=2, keepdims=True), 1e-300))
        H[vi] = Hn
    href = np.maximum(zref, 0.0)

W1 = Wf(1)


def observe(h):
    R = np.linalg.norm(h, axis=1)
    z1 = h @ W1.T
    return dict(h=h, logR=np.log(R)[:, None], u=h / R[:, None], a=np.maximum(z1, 0.0), g=(z1 > 0).astype(float))


res = dict(n=n, seed=seed, P=P, J=J, variants=variants, results={})
B = 300
bidx = rng.integers(0, P, (B, P))
for vi, v in enumerate(variants):
    f0, f1, f2 = observe(H[vi, 0]), observe(H[vi, 1]), observe(H[vi, 2])
    out = {}
    for k in f0:
        d1 = np.sum((f0[k] - f1[k]) ** 2, 1)
        d2 = np.sum((f0[k] - f2[k]) ** 2, 1)
        D = NB * d1.mean() / d2.mean()
        Db = NB * d1[bidx].mean(1) / d2[bidx].mean(1)
        # age profile in octaves (T_j = E|f - f^(j)|^2 / (2V), V = E|f - f'|^2/2)
        age = -jsel
        prof = []
        for lo, hi in zip([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512], [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1025]):
            m = (age >= lo) & (age < hi)
            if m.any():
                prof.append([lo, hi, int(m.sum()), float(d1[m].mean() / d2.mean())])
        out[k] = dict(D=float(D), D_se=float(Db.std()), V=float(d2.mean() / 2), age_profile=prof)
    out['frac_free'] = float(np.mean(frac_free[v])) if frac_free[v] else 1.0
    out['Q0_median'] = float(np.median(np.sum(H[vi, 0] ** 2, 1) / n))
    res['results'][v] = out
res['sec'] = time.time() - t0
json.dump(res, open(a.out, 'w'))
print(json.dumps({v: {k: (round(x['D'], 3), round(x['D_se'], 3)) for k, x in r.items() if isinstance(x, dict)}
                  for v, r in res['results'].items()}), 'sec %.0f' % res['sec'], flush=True)
