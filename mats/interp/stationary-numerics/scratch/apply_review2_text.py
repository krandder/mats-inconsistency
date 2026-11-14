"""Apply the round-2 review text corrections to the report (exact string replacements, each must match once)."""
p = '/home/agent/workspaces/mats/outbox/stationary-numerics-2026-10-31.md'
s = open(p).read()
rep = [
("uses each net's measured radius and assumes the 1/k shape, which I measured (participation ratio 0.79-0.99 of the\n   1/k value) but did not derive.",
 "uses each net's measured radius and assumes a 1/k shape; the 4-replica participation ratio (0.79-0.99 of the 1/k\n   value) is consistent with that shape but does not establish it, and I have not derived it."),
("recovers about 0.2, 0.3, 0.6 and 1.0 of the excess for k = 0.5, 1, 2, 4. The registered version on\n   the direction observable missed its bands (section 3.2).",
 "recovers about 0.2, 0.3, 0.6 and 1.0 of the excess for k = 0.5, 1, 2, 4. These are statements about\n   the intervened models, not an allocation of the network's interactions, and the registered version on the direction\n   observable missed its bands (section 3.2)."),
("11. Exact small-width densities (NUMERICAL, no sampling; grid error in total variation 5e-4 to 2e-3 at n = 2, 3 and\n   5e-2 at n = 4):",
 "11. Exact small-width densities (NUMERICAL, no sampling; refinement diagnostics in total variation 5e-4 to 2e-3 at\n   n = 2, 3; n = 4 is an exploratory 12-layer density with refinement 5e-2):"),
("The 0.56 is a net whose layer-0 radius is 5.7 times the median.",
 "The 0.56 is a net with Q/n = 5.65 at layer 0, 9.8 times the stationary median."),
("| G5 | births within 5% of L0; cascade within 25-30% | miss (births 6-7% above L0) |",
 "| G5 | births within 5% of L0; cascade within 25-30% | miss (births up to 10.2% above L0, n = 16) |"),
("| S2 | TV to Gaussian >= 0.05 on every net, mixture <= half | miss (one n = 2 net at 0.005; one ratio 0.55) |",
 "| S2 | TV to Gaussian >= 0.05 on every net, mixture <= half | miss (two n = 2 nets at 0.005, 0.040; mixture ratios 0.55, 0.57 on two nets) |"),
("| S4 | Gaussian boundary density off > 20% on half the units | miss (4 of 12; the mixture clause holds) |",
 "| S4 | Gaussian boundary density off > 20% on half the units | miss (4 of 12 with exact boundary densities; the mixture clause holds, 10 of 12) |"),
("| D1 | finite-difference linear curves collapse within 0.5 decades | miss |",
 "| D1 | finite-difference linear curves collapse within 0.5 decades | miss (mean of per-net medians; per-path pooling not computable from the saved outputs) |"),
("| D2 | slopes within 25% of QR exponents | miss (1.27 at n = 128, k = 1) |",
 "| D2 | slopes within 25% of QR exponents | miss (per run: 16 of 72 pairs outside; k = 5, 7 not simulated) |"),
("| D4 | flip cascade 0.08-0.4 n | hit |", "| D4 | flip cascade 0.08-0.4 n | hit (also on the registered population only) |"),
("| D5 | flip lifetimes scale with n (median within 1.5x) | hit |", "| D5 | flip lifetimes scale with n (median within 1.5x) | hit (also on the registered population only) |"),
]
for o, nw in rep:
    c = s.count(o)
    assert c == 1, (c, o[:60])
    s = s.replace(o, nw)
open(p, 'w').write(s)
print('applied', len(rep))
