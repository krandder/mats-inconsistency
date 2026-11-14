"""Part C: input bits below the memory time. Model S with an input: x ~ N(0, I_n) at layer 0, absolute noise sigma after
each hidden layer, layers 1..L, output a_L = relu(W_L h_{L-1}) (no noise after the output layer). Weights W_l =
st.W_layer(seed, n, l), l = 1..L. Cost unit: one matrix-vector product (mv); MC costs L mv per sample.
usage: partC.py n L sigma seed [--K 128] [--M 8] [--out file]
Measures (K inputs x M noise draws, plus coupled noiseless and antithetic partners):
- input share S_x = Var(E[a|x]) / Var(a) (nested ANOVA, unbiased within-term correction);
- noise effect on the mean |E a_sigma - E a_0|^2 / trVar and Var(a_sigma - a_0)/trVar (coupled two-level estimator);
- relative noise eps_L = sigma sqrt(n) / |pre-noise state| at layer L-1 (median over samples), births per layer at L-1
  (flips under resampling one layer's noise), input gate entropy at layer L-1 over inputs (noiseless);
- antithetic pairs: (x, xi) with (x, -xi), and (x, xi) with (-x, -xi): pair variance / (trVar/2);
- final-noise Rao-Blackwellisation variance ratio; RQMC over x (scrambled Sobol, MC noise): variance ratio at N = 256.
"""
import argparse, json, math, os, sys, time
import numpy as np
from scipy.special import ndtr, ndtri
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
import st

ap = argparse.ArgumentParser(); ap.add_argument('n', type=int); ap.add_argument('L', type=int); ap.add_argument('sigma', type=float)
ap.add_argument('seed', type=int); ap.add_argument('--K', type=int, default=128); ap.add_argument('--M', type=int, default=8)
ap.add_argument('--R', type=int, default=8); ap.add_argument('--out')
a = ap.parse_args(); n, L, sig, seed, K, M = a.n, a.L, a.sigma, a.seed, a.K, a.M
t0 = time.time()
Wf = st.WCache(seed, n, max_bytes=400e6)
INV2PI = 1.0 / math.sqrt(2 * math.pi)


def relu_mean(mu, s):
    t = mu / s
    return mu * ndtr(t) + s * np.exp(-0.5 * t * t) * INV2PI


def fwd(X, XI_fn, stats=False):
    """X (P, n) inputs; XI_fn(l, shape) -> noise for hidden layer l (or None). Returns output, RB'd output, extras."""
    h = X
    ex = {}
    for l in range(1, L + 1):
        W = Wf(l)
        z = h @ W.T
        apre = np.maximum(z, 0.0)
        if l < L:
            xi = XI_fn(l, apre.shape)
            if stats and l == L - 1:
                Wn = Wf(L)
                xi2 = XI_fn(-l, apre.shape)
                z1 = (apre + sig * xi) @ Wn.T; z2 = (apre + sig * xi2) @ Wn.T
                ex['births'] = float(np.mean(np.sum((z1 > 0) != (z2 > 0), 1)))
                nr = np.linalg.norm(apre, axis=1)
                ex['eps_med'] = float(np.median(sig * math.sqrt(n) / np.maximum(nr, 1e-300)))
                ex['eps_q90'] = float(np.quantile(sig * math.sqrt(n) / np.maximum(nr, 1e-300), 0.9))
            h_prev_pre = apre
            h = apre + sig * xi
        else:
            out = apre
    W_L = Wf(L)
    rb = relu_mean(h_prev_pre @ W_L.T, sig * np.linalg.norm(W_L, axis=1)[None, :]) if sig > 0 else out
    return out, rb, ex


rng = np.random.default_rng([31337, seed, n, L, int(round(sig * 1e6))])
X = rng.standard_normal((K, n))
XX = np.repeat(X, M, 0)                                  # (K M, n)
SIGC = int(round(sig * 1e6))


def noise_tab(sign=1.0):
    # seeded per layer, regenerated on demand (the antithetic pass sees the same draws); key -l = resampled copy
    def f(l, shape):
        return sign * np.random.default_rng([5151, seed, n, L, SIGC, l + st.OFF]).standard_normal(shape)
    return f


A, ARB, ex = fwd(XX, noise_tab(1.0), stats=True)                 # noisy
A0, _, _ = fwd(X, lambda l, s: np.zeros(s))                      # noiseless, one per input
Am, _, _ = fwd(XX, noise_tab(-1.0))                              # antithetic in noise (same x)
Axm, _, _ = fwd(-XX, noise_tab(-1.0))                            # antithetic in both
Ak = A.reshape(K, M, n)
trv = float(np.sum(A.var(0)))
within = float(np.sum(Ak.var(1, ddof=1).mean(0)))
between = float(np.sum(Ak.mean(1).var(0, ddof=1))) - within / M
D = A - np.repeat(A0, M, 0)
res = dict(n=n, L=L, sigma=sig, seed=seed, K=K, M=M, trvar=trv, input_share=between / trv, noise_share=within / trv,
           noise_effect_nmse=float(np.sum(D.mean(0) ** 2) / trv), coupled_diff_var=float(np.sum(D.var(0)) / trv),
           anti_noise_pair=float(np.sum((0.5 * (A + Am)).var(0)) / (trv / 2)),
           anti_both_pair=float(np.sum((0.5 * (A + Axm)).var(0)) / (trv / 2)),
           rb_ratio=float(np.sum(ARB.var(0)) / trv), **ex)
# input gate entropy at layer L-1 over inputs (noiseless)
h = X
for l in range(1, L):
    z = h @ Wf(l).T; h = np.maximum(z, 0.0)
p = np.mean(z > 0, 0)
Hb = -(p * np.log2(np.clip(p, 1e-12, 1)) + (1 - p) * np.log2(np.clip(1 - p, 1e-12, 1)))
res['input_gate_entropy_bits'] = float(Hb.sum())
res['distinct_patterns_frac'] = float(len(set(map(bytes, (z > 0).astype(np.uint8)))) / K)
# RQMC over x only (noise MC), N = 256, R replicates, against MC variance trVar/N
from scipy.stats import qmc
N = 256; ests = []
for r in range(a.R):
    U = qmc.Sobol(d=n, scramble=True, seed=np.random.default_rng([4141, seed, n, L, r])).random_base2(8)
    Xq = ndtri(np.clip(U, 1e-16, 1 - 1e-16))
    rq = np.random.default_rng([4242, seed, n, L, r])
    Aq, _, _ = fwd(Xq, lambda l, s: rq.standard_normal(s))
    ests.append(Aq.mean(0))
res['rqmc_x_ratio_N256'] = float((trv / N) / max(np.sum(np.array(ests).var(0, ddof=1)), 1e-300))
res['T_mem_heuristic'] = (4 * n / 5) * math.log(max(1.0 / (sig * math.sqrt(0.4 * n)), 1.0))
# Amendment C1: crossed design (Kc inputs x Mc noise sequences, every input with every noise sequence) and the
# two-way random-effects ANOVA: input main, noise main, input x noise interaction (shares of their sum).
Kc = Mc = 32
Xc = np.random.default_rng([6161, seed, n, L]).standard_normal((Kc, n))
XC = np.repeat(Xc, Mc, 0)


def crossed_noise(l, shape):
    blk = np.random.default_rng([6262, seed, n, L, SIGC, l + st.OFF]).standard_normal((Mc, n))
    return np.tile(blk, (Kc, 1))


AC, _, _ = fwd(XC, crossed_noise)
Y = AC.reshape(Kc, Mc, n)
gm = Y.mean((0, 1)); rm = Y.mean(1); cm = Y.mean(0)
ms_r = float(np.sum(Mc * ((rm - gm) ** 2).sum(0)) / (Kc - 1))
ms_c = float(np.sum(Kc * ((cm - gm) ** 2).sum(0)) / (Mc - 1))
resid = Y - rm[:, None, :] - cm[None, :, :] + gm
ms_e = float(np.sum(resid ** 2) / ((Kc - 1) * (Mc - 1)))
va = max((ms_r - ms_e) / Mc, 0.0); vb = max((ms_c - ms_e) / Kc, 0.0); vg = ms_e
tot = va + vb + vg
res.update(anova_input=va / tot if tot > 0 else None, anova_noise=vb / tot if tot > 0 else None,
           anova_interaction=vg / tot if tot > 0 else None)
res['sec'] = time.time() - t0
s = json.dumps(res)
if a.out:
    open(a.out, 'w').write(s)
print(s, flush=True)
