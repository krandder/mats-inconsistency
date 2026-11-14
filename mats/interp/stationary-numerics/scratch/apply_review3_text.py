"""Apply the round-3 review text corrections to the report (exact string replacements, each must match once)."""
p = '/home/agent/workspaces/mats/outbox/stationary-numerics-2026-10-31.md'
s = open(p).read()
rep = [
("9. Response kernels at n >= 64 (MEASURED; finite-window covariant Lyapunov vectors). The tangent response along vector k",
 "9. Response kernels at n >= 64 (MEASURED; covariant Lyapunov vectors from a finite window, not validated for\n   convergence: they shift with the warm-up length). The tangent response along vector k"),
("fast direction first falls like the tangent one, then levels off (median R ~ 0.004-0.01 at t = n, in units of its own\n   size squared) and decays slowly; along vector 8 it is about 1e9 above the tangent median by t = 1.5 n.",
 "fast direction first falls like the tangent one, then levels off (median R ~ 0.004-0.01 near t = n, in units of its\n   own size squared) and decays slowly; along vector 8 it is about 1e9 above the tangent median near t = 1.5 n."),
("   Its effect takes 1.0-1.3 n (lower quartile) and 1.4-1.9 n layers (median) to fall 100x, and more than 2n for over a\n   quarter of paths, so lifetimes scale with n.",
 "   Its effect takes 1.0-1.3 n (lower quartile) and 1.4-1.9 n layers (median) to fall 100x, and more than 2n for 27-44%\n   of paths at n >= 16 (20% at n = 8), so lifetimes scale with n."),
("5.2 Finite perturbations (empirical, n >= 64; `figs/response_collapse.png`, left, medians over paths throughout). Along\nfast directions the response first falls like the tangent one, then levels off. At t = n the median R over k >= 4 is\n0.004 for eps = 0.3/sqrt(n) and 0.010 for 3/sqrt(n) (the means are 0.039 and 0.063: the distribution over paths is\nskewed). At t = n this is 1.4-23 decades (median 5.7) above the tangent median for the same net and k. For CLV 8 at\nt = 1.5n the gap is 8.6-9.6 decades.",
 "5.2 Finite perturbations (empirical, n >= 64; `figs/response_collapse.png`, left). Populations: the same nets; the same\n64 paths at n = 128 and 256; at n = 64 the tangent runs used 64 paths with other noise than the 128 finite-size paths.\n\"Near t = n\" means the sampled layer closest to n, t/n = 0.95, 0.98, 1.05 at n = 64, 128, 256 (and 1.56, 1.51, 1.45 for\n\"near 1.5n\"). Along fast directions the response first falls like the tangent one, then levels off. Near t = n the\nmedian over nets of the per-net median R over paths, over k >= 4, is 0.004 for eps = 0.3/sqrt(n) and 0.010 for\n3/sqrt(n) (the means over paths are 0.039 and 0.063: the distribution is skewed). This is 1.4-23 decades (median 5.7)\nabove the tangent median for the same net and k. For CLV 8 near t = 1.5n the gap is 8.6-9.6 decades. (The figure shows\nthe mean over nets of the per-net log-medians.)"),
("Totals: 17 hits and 18 misses of 35.",
 "Totals: 17 hits, 17 misses and 1 unscorable (D1: its registered path-pooled statistic cannot be computed from the\nsaved outputs) of 35."),
("| D1 | finite-difference linear curves collapse within 0.5 decades | miss (mean of per-net medians; per-path pooling not computable from the saved outputs) |",
 "| D1 | finite-difference linear curves collapse within 0.5 decades | unscorable as registered (path pooling not computable); the mean of per-net medians misses (0.57-0.68 decades) |"),
("Pre-registered scorecard: 17 of 35 predictions hit (section 8, with every miss listed).",
 "Pre-registered scorecard: 17 of 35 predictions hit, 17 missed, 1 unscorable (section 8, every miss listed)."),
]
for o, nw in rep:
    c = s.count(o)
    assert c == 1, (c, o[:70])
    s = s.replace(o, nw)
open(p, 'w').write(s)
print('applied', len(rep))
