"""Final tidy of the report summary (line wrapping and one label), exact replacements."""
p = '/home/agent/workspaces/mats/outbox/stationary-numerics-2026-10-31.md'
s = open(p).read()
rep = [
("""   uses each net's measured radius and assumes a 1/k shape; the 4-replica participation ratio (0.79-0.99 of the 1/k
   value) is consistent with that shape but does not establish it, and I have not derived it. On 12 new nets registered before the runs (n = 64-256), measured/predicted is
   0.86-1.41 on 11 and 0.56 on one large-radius net at n = 64, so the registered test misses.""",
"""   uses each net's measured radius and assumes a 1/k shape; the 4-replica participation ratio (0.79-0.99 of the 1/k
   value) is consistent with that shape but does not establish it, and I have not derived it. On 12 new nets registered
   before the runs (n = 64-256), measured/predicted is 0.86-1.41 on 11 and 0.56 on one large-radius net at n = 64, so
   the registered test misses."""),
("""9. Response kernels at n >= 64 (MEASURED; covariant Lyapunov vectors from a finite window, not validated for
   convergence: they shift with the warm-up length). The tangent response along vector k
   decays like exp(-2 c (k + 1/4) t/n):""",
"""9. Response kernels at n >= 64 (MEASURED; covariant Lyapunov vectors from a finite window, not validated for
   convergence: they shift with the warm-up length). The tangent response along vector k decays like
   exp(-2 c (k + 1/4) t/n):"""),
("""   of paths at n >= 16 (20% at n = 8), so lifetimes scale with n. The exact cascade after
   resampling one layer's noise grows like log n""",
"""   of paths at n >= 16 (20% at n = 8), so lifetimes scale with n. The exact cascade after resampling one layer's noise
   grows like log n"""),
("""11. Exact small-width densities (NUMERICAL, no sampling; refinement diagnostics in total variation 5e-4 to 2e-3 at
   n = 2, 3; n = 4 is an exploratory 12-layer density with refinement 5e-2): at n = 2 the
   quenched stationary law is a mixture over the last layer's gate pattern.""",
"""11. Exact small-width densities (NUMERICAL, no sampling; refinement diagnostics in total variation 5e-4 to 2e-3 at
   n = 2, 3; n = 4 is an exploratory 12-layer density with refinement 5e-2). At n = 2 the quenched stationary law is a
   mixture over the last layer's gate pattern."""),
("""predicts its size, its excess over the Lyapunov picture (given the measured 1/k shape) and most of its interaction""",
 """predicts its size, its excess over the Lyapunov picture (given an assumed 1/k shape) and most of its interaction"""),
]
for o, nw in rep:
    c = s.count(o)
    assert c == 1, (c, o[:70])
    s = s.replace(o, nw)
open(p, 'w').write(s)
print('applied', len(rep))
