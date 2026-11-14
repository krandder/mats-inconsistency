"""Brute-force checks of the worked examples in scale-free-explainer.html (29 October 2026).

Run: /home/agent/.venvs/arc-noisy/bin/python -I scale-free-explainer-checks.py
Covers: complete-graph classes across a cut, the Gaussian clique's rank-one precision block, coherence sums,
the black-box factor-4 ratio, sign-rank vectors for equality and order, the sharp-selection logit range,
parity and complete-graph +-1 ranks, binding classes, and two closed-form numbers from the hardness report.
"""
import itertools, math
import numpy as np

# 1. complete-graph GF(2) quadratic form f(x) = sum_{i<j} x_i x_j mod 2
def f_complete(x):
    s = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            s ^= (x[i] & x[j])
    return s

for n in (6, 12):
    S = n // 2
    rows = {}
    for xs in itertools.product((0, 1), repeat=S):
        row = tuple(f_complete(list(xs) + list(xr)) for xr in itertools.product((0, 1), repeat=n - S))
        rows.setdefault(row, set()).add(sum(xs) % 4)
    print("n", n, "classes across balanced cut:", len(rows),
          "| each class is one value of |x_S| mod 4:", sorted(tuple(sorted(v)) for v in rows.values()))
    ok = all(f_complete(list(x)) == (1 if sum(x) % 4 in (2, 3) else 0) for x in itertools.product((0, 1), repeat=n))
    print("  f = 1 iff |x| mod 4 in {2,3}:", ok)

# 2. Gaussian clique: x_i = theta + e_i, theta ~ N(0, 1/tau0), e_i ~ N(0, 1/I)
tau0, I, N = 1.0, 2.0, 6
Sigma = (1 / tau0) * np.ones((N, N)) + (1 / I) * np.eye(N)
Lam = np.linalg.inv(Sigma)
off = Lam[~np.eye(N, dtype=bool)]
print("precision off-diagonal min/max:", off.min(), off.max(), "predicted:", -I ** 2 / (tau0 + N * I))
print("rank of off-diagonal block [S, S^c]:", np.linalg.matrix_rank(Lam[:3, 3:]))

# 3. coherence: sum_{l<L'} r^(2l), r = 1 - d
def coh(d, Lp):
    r = 1 - d
    return sum(r ** (2 * l) for l in range(Lp))

for d in (1 / 64, 0.025, 0.003, 0.04, 0.06, 0.07, 0.08, 0.10):
    c = coh(d, 32)
    print(f"d={d:.4f}: coherent layers {c:.2f} of 32 ({c / 32:.0%}); 1/(2d) = {1 / (2 * d):.1f}")

# 4. black-box ratio limit 4(1 - 1/(2N)) and finite-w version
for Nq in (1, 4, 100):
    l = 1 / (2 * Nq)
    w = l / 100
    a = l - w
    b = l - 4 * w / 3
    print("N", Nq, "limit", 4 * (1 - 1 / (2 * Nq)), "finite w ratio", 2 * (b - a * a) / (Nq * a * a))

# 5. sign-rank vectors
Nl = 9
eq_ok = all(((2 * i) * j + (0.5 - i * i) * 1 + 1 * (-j * j) > 0) == (i == j) for i in range(Nl) for j in range(Nl))
ord_ok = all(((i + 0.5) * 1 + 1 * (-j) > 0) == (j <= i) for i in range(Nl) for j in range(Nl))
print("equality via 3-dim vectors:", eq_ok, " order via 2-dim vectors:", ord_ok)

# 6. sharp selection logit range
for Nn, delta in ((1024, 0.01), (10 ** 6, 0.01)):
    rng = math.log((Nn - 1) * (1 - delta) / delta)
    print("N", Nn, "delta", delta, "logit range >=", round(rng, 2), "B >=", round(math.sqrt(rng / 2), 2))

# 7. parity across a cut: rank of the +-1 matrix
M = np.array([[(-1) ** (sum(a) + sum(b)) for b in itertools.product((0, 1), repeat=3)]
              for a in itertools.product((0, 1), repeat=3)])
print("parity +-1 matrix rank across 3|3 cut:", np.linalg.matrix_rank(M))
M2 = np.array([[(-1) ** f_complete(list(a) + list(b)) for b in itertools.product((0, 1), repeat=3)]
               for a in itertools.product((0, 1), repeat=3)])
print("complete-graph (-1)^f real rank across 3|3 cut:", np.linalg.matrix_rank(M2), "(report: 2^{rank_GF2(B)} = 2)")

# 8. binding: positional cut 4|4, keys 1..4 each with one item per side; f = [A ∩ B nonempty]
classes = set()
for A in itertools.product((0, 1), repeat=4):
    row = tuple(int(any(A[k] and B[k] for k in range(4))) for B in itertools.product((0, 1), repeat=4))
    classes.add(row)
print("binding, 4 straddling keys: classes at the cut =", len(classes), "= 2^4")

# 9. Gaussian aggregation: posterior from sums equals posterior from all data
rng_ = np.random.default_rng(0)
theta = rng_.normal()
x = theta + rng_.normal(scale=1 / math.sqrt(I), size=8)
tau_post = tau0 + 8 * I
mu_post = I * x.sum() / tau_post
sA, sB = x[:5].sum(), x[5:].sum()
print("posterior mean from all data", mu_post, "from two sums", I * (sA + sB) / tau_post, "variance", 1 / tau_post)

# 10. two-replica overlap r ~ 1 - theta/pi at rms angle 0.285 rad
print("1 - 0.285/pi =", 1 - 0.285 / math.pi, "; 1 - 0.21/pi =", 1 - 0.21 / math.pi)
for ang in (0.285, 0.21, 0.08, 0.05):
    r = 1 - ang / math.pi
    print(f"angle {ang}: r = {r:.3f}, coherent layers of 32 = {sum(r ** (2 * l) for l in range(32)):.1f}")

# 11. ARC arithmetic: ratio (method MSE)/(MC MSE at equal FLOPs) for basic order K
for K in (2, 3):
    print(f"K={K}: ratio ∝ L^K / n ; at L = alpha n: ∝ alpha^K n^(K-1)")

# 12. Prop 7.2 fraction kept at first order: 5a/(e^{5a}-1)
for a in (0.25, 1.0):
    print("alpha", a, "first-order fraction", 5 * a / (math.exp(5 * a) - 1))

# 13. Lemma 6.10 example D=2 at alpha=1/4: 1 - exp(-5 r_D alpha), r_2 = 3/5 - 8/(5 pi)
r2 = 3 / 5 - 8 / (5 * math.pi)
print("r_2", r2, "nMSE", 1 - math.exp(-5 * r2 * 0.25))
