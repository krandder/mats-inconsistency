"""Fourth batch: remaining "exact" wording in section 6, NC2 corrected-code wording, surviving fronts."""
p = '/home/agent/workspaces/mats/outbox/cut-rank-ext-2026-10-31.md'
s = open(p).read()
rep = [
    ("- ARC's K = 2 closure fed the exact mean and covariance is far worse on merged fronts. Fed the same mean with zero\n  covariance, the same code is mean propagation (`logs/arc_zero_cov_check.log`).",
     "- ARC's K = 2 closure fed the estimated mean and covariance is far worse on merged fronts. Fed the same mean with zero\n  covariance, the same code is mean propagation (`logs/arc_zero_cov_check.log`)."),
    ("**NC2: bits on the mean** (j bits per direction coordinate plus 16 for the norm, decoded by g_b). On (8,64,8) the error\nfalls like 2^{-2.1 j}. 48 bits reach 1% of Vbar; the rank-based lower bound log2 r_in(0.1) is 4.8 bits.",
     "**NC2: bits on the mean** (j bits per direction coordinate plus 16 for the norm, decoded by g_b). The registered code\ndecoded the norm with a front-dependent scale that was never transmitted, so it was not a true bit protocol (review\nfinding 9). With a global norm code (Amendment 2; no overflows) the numbers are unchanged: on (8,64,8) the error falls\nlike 2^{-2.1 j} and 48 bits reach 1% of Vbar. The rank-based lower bound log2 r_in(0.1) is 4.8 bits."),
    ("| (n, c, d) | MC floor | MF: g_b(mean), n numbers | best linear, n+1 numbers (in-sample SVD) | LIN1 (held out) | K2X: exact Gaussian closure | ARC2 (unit-scaled) | LIN2 (held out) | best linear, 1+n+n(n+1)/2 numbers |",
     "| (n, c, d) | MC floor | MF: g_b(mean), n numbers | best linear, n+1 numbers (in-sample SVD estimate) | LIN1 (held out) | K2X: Gaussian closure, exact ReLU moments | ARC2 (unit-scaled) | LIN2 (held out) | best linear, 1+n+n(n+1)/2 numbers (estimate) |"),
    ("| (8,64,8) merged | 6e-5 | 0.00018 [4e-5] |",
     "| (8,64,8) merged (211 of 256 test fronts alive) | 6e-5 | 0.00018 [4e-5] |"),
]
for a, b in rep:
    assert a in s, 'MISSING: ' + a[:70]
    s = s.replace(a, b)
open(p, 'w').write(s)
print('patched', len(rep))
