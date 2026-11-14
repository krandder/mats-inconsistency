"""Tests for lib/ib.py (run: python3 tests/test_ib.py; exit 0 = all pass). Each test states what it proves."""
import math
import sys

import numpy as np

sys.path.insert(0, '/home/agent/workspaces/mats/work/input-bits/lib')
import ib  # noqa: E402

fails = []


def check(name, ok, msg=''):
    print(('PASS ' if ok else 'FAIL ') + name + (': ' + msg if msg else ''), flush=True)
    if not ok:
        fails.append(name)


rng = np.random.default_rng(123)

# 1. positive homogeneity (sigma = 0): a(x) = |x| g(x/|x|)  [assumption A2]
net = ib.Net(9100, 32, 12)
X = rng.standard_normal((256, 32))
a = net.forward(X)
g = net.g(ib.sphere(X))
err = np.max(np.abs(a - np.linalg.norm(X, axis=1, keepdims=True) * g)) / np.max(np.abs(a))
check('homogeneity', err < 1e-12, f'max rel err {err:.1e}')

# 2. the weights are the stationary lane's stream: layer l of Net == st.W_layer(seed, n, l)
import st  # noqa: E402
check('weights = st.W_layer', np.array_equal(net.W[4], st.W_layer(9100, 32, 5)))

# 3. var_split: V_rad + V_amp + V_dir equals tr Var_x(a) estimated directly on independent Gaussian inputs (3 SE)
net = ib.Net(9101, 16, 6)
U = ib.sphere(rng.standard_normal((200000, 16)))
vs = ib.var_split(net.g(U), 16)
Xs = rng.standard_normal((200000, 16))
A = net.forward(Xs)
reps = [float(np.sum(A[i::20].var(0, ddof=1))) for i in range(20)]
direct, se = np.mean(reps), np.std(reps, ddof=1) / math.sqrt(20)
check('var_split total', abs(vs['V'] - direct) < 3 * se + 1e-3 * direct, f'split {vs["V"]:.5f} direct {direct:.5f} se {se:.5f}')

# 4. gate_layer_stats: unbiased Hamming on synthetic Bernoulli gates with known p
p = rng.uniform(0, 1, 64)
hams = []
for r in range(200):
    Z = (rng.uniform(size=(16, 64)) < p).astype(float) - 0.5
    hams.append(ib.gate_layer_stats(Z)['hamming'])
true = float(np.sum(2 * p * (1 - p)))
check('hamming unbiased', abs(np.mean(hams) - true) < 3 * np.std(hams) / math.sqrt(200), f'{np.mean(hams):.3f} vs {true:.3f}')

# 5. saltelli on f = (x0 + x1 x2, 0): V = 2, S = (1/2, 0, 0), T = (1/2, 1/2, 1/2)
f = lambda Y: np.stack([Y[:, 0] + Y[:, 1] * Y[:, 2], np.zeros(len(Y))], 1)
sl = ib.saltelli(f, 5, 200000, rng, [0, 1, 2, 3])
okS = np.all(np.abs(sl['S'] - [0.5, 0, 0, 0]) < 4 * sl['se_S'] + 1e-3)
okT = np.all(np.abs(sl['T'] - [0.5, 0.5, 0.5, 0]) < 4 * sl['se_T'] + 1e-3)
check('saltelli S', okS, str(np.round(sl['S'], 3)))
check('saltelli T', okT, str(np.round(sl['T'], 3)))

# 6. Hermite shares: f = x0 + (x1^2 - 1) + x2 x3  -> V = 1 + 2 + 1 = 4; W1 = 1/4; univariate deg2 = 2/4; W2 = 3/4 (diag 2/4, off 1/4)
Xh = rng.standard_normal((400000, 6))
F = (Xh[:, 0] + Xh[:, 1] ** 2 - 1 + Xh[:, 2] * Xh[:, 3])[:, None]
us, V = ib.univariate_shares(Xh, F)
check('univariate shares', abs(us[0] - 0.25) < 0.01 and abs(us[1] - 0.5) < 0.01 and abs(us[2]) < 0.01 and abs(us[3]) < 0.01,
      str(np.round(us, 4)))
d2 = ib.deg2_share(Xh, F)
check('degree-2 share', abs(d2['W2'] - 0.75) < 0.02 and abs(d2['W2_diag'] - 0.5) < 0.02 and abs(d2['W2_off'] - 0.25) < 0.02,
      str({k: round(v, 4) for k, v in d2.items()}))
ss = ib.stein_shares(Xh, F, deg2_outputs=1)
check('stein degree 1', abs(ss['s_lin'] - 0.25) < 0.01, f'{ss["s_lin"]:.4f}')

# 7. even share: f = x0 + x1^2 -> even part x1^2 (var 2) of V = 3 -> s_even = 2/3
se_, _ = ib.even_share(lambda Y: (Y[:, 0] + Y[:, 1] ** 2)[:, None], rng.standard_normal((200000, 3)))
check('even share', abs(se_ - 2 / 3) < 0.01, f'{se_:.4f}')

# 8. exact Gram chain vs nets (annealed): mean output cosine deficit and final-layer Hamming, n = 16, L = 8, sigma = 0,
#    independent Gaussian input pairs. Nets: 400 seeds x 64 pairs; chain: 25600 chains. Means within 3 combined SE.
n, L = 16, 8
cd_net, ham_net = [], []
for s in range(400):
    nt = ib.Net(20000 + s, n, L)
    X1 = rng.standard_normal((64, n)); X2 = rng.standard_normal((64, n))
    zs = {}
    def cb(l, z, h, key):
        if l == L:
            zs[key] = z
    a1 = nt.forward(X1, cb=lambda l, z, h: cb(l, z, h, 1)); a2 = nt.forward(X2, cb=lambda l, z, h: cb(l, z, h, 2))
    c = np.sum(a1 * a2, 1) / np.maximum(np.linalg.norm(a1, axis=1) * np.linalg.norm(a2, axis=1), 1e-300)
    ok = (np.linalg.norm(a1, axis=1) > 0) & (np.linalg.norm(a2, axis=1) > 0)
    cd_net.extend(list(1 - c[ok]))
    ham_net.extend(list(np.sum((zs[1] > 0) != (zs[2] > 0), 1)))
G0 = np.empty((25600, 2, 2))
Xa = rng.standard_normal((25600, n)); Xb = rng.standard_normal((25600, n))
G0[:, 0, 0] = np.sum(Xa * Xa, 1) / n; G0[:, 1, 1] = np.sum(Xb * Xb, 1) / n; G0[:, 0, 1] = G0[:, 1, 0] = np.sum(Xa * Xb, 1) / n
hams = []
def rec(l, Gp, Gn):
    if l == L:
        hams.append(ib.chain_hamming(Gp, n))
Gf = ib.gram_chain_inputs(G0, L, n, rng, record=rec)
okc = (Gf[:, 0, 0] > 0) & (Gf[:, 1, 1] > 0)
cch = 1 - Gf[okc, 0, 1] / np.sqrt(Gf[okc, 0, 0] * Gf[okc, 1, 1])
m1, s1 = np.mean(cd_net), np.std(cd_net) / math.sqrt(len(cd_net))
m2, s2 = np.mean(cch), np.std(cch) / math.sqrt(len(cch))
# nets' pairs within one net are correlated: inflate the net SE by a design factor estimated from per-net means
pern = np.array(cd_net[:len(cd_net) // 64 * 64]).reshape(-1, 64).mean(1) if len(cd_net) >= 64 else np.array(cd_net)
s1b = np.std(pern, ddof=1) / math.sqrt(len(pern))
check('chain vs nets: output cosine deficit', abs(m1 - m2) < 3 * math.hypot(max(s1, s1b), s2), f'nets {m1:.4f}+-{max(s1, s1b):.4f} chain {m2:.4f}+-{s2:.4f}')
hn = np.array(ham_net, float).reshape(400, 64).mean(1)
hc = np.concatenate(hams)
check('chain vs nets: layer-L Hamming', abs(hn.mean() - hc.mean()) < 3 * math.hypot(hn.std(ddof=1) / 20, hc.std() / math.sqrt(len(hc))),
      f'nets {hn.mean():.3f}+-{hn.std(ddof=1) / 20:.3f} chain {hc.mean():.3f}')

# 9. chain split identity: annealed V = V_rad + V_amp + V_dir (exact in expectation, noiseless); means within 3 SE
T = ib.chain_split_terms(Gf, n, np.stack([np.sum(Xa * Xa, 1), np.sum(Xb * Xb, 1)], 1))
lhs = T['V']; rhs = T['V_rad'] + T['V_amp'] + T['V_dir']
dd = lhs - rhs
check('chain split identity', abs(dd.mean()) < 3 * dd.std() / math.sqrt(len(dd)), f'mean diff {dd.mean():.4f} +- {dd.std() / math.sqrt(len(dd)):.4f} (V {lhs.mean():.3f})')

print('ALL PASS' if not fails else f'{len(fails)} FAILED: {fails}')
sys.exit(1 if fails else 0)
