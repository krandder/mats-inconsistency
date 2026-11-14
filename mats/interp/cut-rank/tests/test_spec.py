"""Tests for lib/spec.py on synthetic matrices with known spectra and split noise (cut-rank lane).

Run: python3 tests/test_spec.py  (from the lane directory; prints PASS/FAIL lines, exit 1 on failure)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import spec

ok = True


def check(name, cond, info=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name + (' | ' + info if info else ''))
    ok &= bool(cond)


def make(F, C, svals, noise, rng, rowcorr=0.0):
    U = np.linalg.qr(rng.standard_normal((F, len(svals))))[0]
    V = np.linalg.qr(rng.standard_normal((C, len(svals))))[0]
    M = (U * svals) @ V.T * np.sqrt(F * C)  # per-entry energy of mode i = svals_i^2
    Ms = []
    for _ in range(8):
        E = rng.standard_normal((F, C))
        if rowcorr > 0:  # noise shared across the columns of a row, like common inputs per front
            E = np.sqrt(1 - rowcorr) * E + np.sqrt(rowcorr) * rng.standard_normal((F, 1))
        Ms.append(M + noise * E)
    return M, Ms


rng = np.random.default_rng(1)
# 1. exact low rank, no noise: tails exact
sv = np.array([3.0, 1.0, 0.3, 0.1])
M, Ms = make(64, 256, sv, 0.0, rng)
ti, lam = spec.tails_in(Ms)
check('noiseless tail_in matches spectrum', np.allclose(ti[:5], [np.sum(sv[i:] ** 2) for i in range(5)], rtol=1e-8),
      str(np.round(ti[:5], 6)))
tc = spec.tails_cf(Ms)
check('noiseless tail_cf matches spectrum', np.allclose(tc[:5], ti[:5], rtol=1e-6, atol=1e-10))

# 2. rank 1 + noise: cross-split removes the noise bias in the total; tail beyond 1 small vs noise^2
M, Ms = make(128, 1024, np.array([2.0]), 0.5, rng, rowcorr=0.3)
ti, lam = spec.tails_in(Ms)
tc = spec.tails_cf(Ms)
raw = np.mean(Ms[0] ** 2)
check('total energy unbiased (pair Gram)', abs(ti[0] - 4.0) < 0.05, f'tail_in[0]={ti[0]:.4f} raw={raw:.4f}')
check('in-sample tail beyond true rank is small vs noise var', abs(ti[1]) < 0.02 * 0.25, f'tail_in[1]={ti[1]:.2e}')
check('cross-fitted tail beyond true rank >= in-sample', tc[1] >= ti[1] - 1e-12, f'tc[1]={tc[1]:.2e}')

# 3. bracket on a power-law spectrum with noise: r_in <= r_true <= r_cf (typically)
F, C = 256, 2048
sv = 1.0 / np.arange(1, 129) ** 1.0
M, Ms = make(F, C, sv, 0.05, rng, rowcorr=0.2)
true_tail = np.concatenate([[np.sum(sv ** 2)], np.sum(sv ** 2) - np.cumsum(sv ** 2)])
true_tail = np.concatenate([true_tail, np.zeros(F + 1 - len(true_tail))])
ti, _ = spec.tails_in(Ms)
tc = spec.tails_cf(Ms)
for thr in (1e-1, 3e-2, 1e-2):
    rt = spec.eps_rank(true_tail, thr); ri = spec.eps_rank(ti, thr); rc = spec.eps_rank(tc, thr)
    check(f'bracket at tail<={thr}', ri <= rt + 1 and rc >= rt - 1, f'r_in={ri} r_true={rt} r_cf={rc}')

# 4. probe protocol on exact rank 3: held-out error ~ 0 at s = 3
M, Ms = make(200, 400, np.array([2.0, 1.0, 0.5]), 0.0, rng)
pe = spec.probe_protocol(Ms, [0, 1, 2, 3, 5], np.random.default_rng(0))
check('probe protocol exact at true rank', pe[3] < 1e-10 and pe[0] > 1.0, str({k: f'{v:.2e}' for k, v in pe.items()}))

# 5. per-net errors shape and unbiasedness at full rank with noise
M, Ms = make(64, 128, np.array([1.0, 0.5]), 0.1, rng)
e = spec.per_net_errors(Ms, 2, 8)
check('per_net_errors shape', e.shape == (64, 16))
check('per_net_errors ~0 at full rank (noise only in subspace)', abs(e.mean()) < 0.01, f'{e.mean():.2e}')

print('ALL PASS' if ok else 'SOME FAILED')
sys.exit(0 if ok else 1)
