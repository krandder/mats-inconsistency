"""Apply the focused re-check's required changes to the outbox report and notes/coverage_map.md."""
p = '/home/agent/workspaces/mats/outbox/cut-rank-ext-2026-10-31.md'
s = open(p).read()
rep = [
    ("- `PREREG.md` (hash in `prereg_hashes.txt`, one amendment), `scratch/`, `runs/`, `logs/`, `review/`.",
     "- `PREREG.md` (hashes in `prereg_hashes.txt`, two amendments), `scratch/`, `runs/`, `logs/`, `review/`."),
    # convergence paragraph (re-check: error change vs prediction difference; 53 merged fronts)
    ("- Convergence (Amendment 2, `logs/conv_check.log`): with cut moments from 65,536 instead of 8,192 directions, the ARC2\n  and K2X errors on 64 test fronts move by at most 0.002 of Vbar ((8,8,8): ARC2 0.91 against 0.92; (8,64,8): 0.76\n  against 0.77). So the K = 2 closure errors are not moment-estimation noise.",
     "- Convergence (Amendment 2, `logs/conv_check.log`): cut moments from 65,536 instead of 8,192 directions, on 64 test\n  fronts of (8,8,8) and the 53 surviving ones of the first 64 of (8,64,8). The two sets of estimates differ by a mean\n  square of 0.0003-0.0015 Vbar, and the errors change by 0.01-0.015 (ARC2: 0.91 to 0.92 and 0.76 to 0.77). So the large\n  K = 2 closure errors are stable across these sample counts. That does not exclude every moment-estimation effect."),
    # finding 12: empirical log2 r_in
    ("like 2^{-2.1 j} and 48 bits reach 1% of Vbar. The rank-based lower bound log2 r_in(0.1) is 4.8 bits.",
     "like 2^{-2.1 j} and 48 bits reach 0.01 Vbar. For comparison, the rank-based bound evaluated at the empirical in-sample\nrank, log2 r_in(0.1), is 4.8 bits."),
    # finding 12/15: summary item 9 wording
    ("   - Merged fronts (n, c, d) = (8, 64, 8): the in-sample estimate of the best linear protocol with 9 numbers (optimistic)\n     leaves 9% of the per-net variance. Mean propagation with 8 numbers leaves 0.02% (Monte Carlo floor 0.006%). ARC's\n     K = 2 closure fed the estimated mean and covariance leaves 71%.",
     "   - Errors are in units of Vbar under the registered normalised loss. Merged fronts (n, c, d) = (8, 64, 8): the\n     in-sample estimate of the best linear protocol with 9 numbers (optimistic) leaves 0.09. Mean propagation with 8\n     numbers leaves 0.0002 (Monte Carlo floor 0.00006). ARC's K = 2 closure fed the estimated mean and covariance leaves\n     0.71."),
    ("   - Bits on the mean: 48 bits reach 1% of the variance on merged fronts. The error falls like 2^{-2j} with j bits per\n     coordinate. This uses a global norm code, after the review found an uncounted scale.",
     "   - A bit protocol on the mean (quantised mean, decoded by mean propagation): 48 bits reach 0.01 on merged fronts, and\n     the error falls like 2^{-2j} with j bits per coordinate. This uses a global norm code, after the review found an\n     uncounted scale. It is an upper-bound demonstration, unrelated to the hypotheses of the bit lower bound."),
    # finding 1: section 1 budgets paragraph
    ("**Budgets.** If a class fixes the decoder's amplitude, Lipschitz constant or derivative bounds independently of eps,\nevery message size leaves an error floor. The target is unbounded under Gaussian weights, and a bounded decoder cannot\nfollow it (gpt-6-astra, round 1 A3 and round 2 R2.1).",
     "**Budgets.** If a class fixes, independently of eps and uniformly over backs, the decoder's amplitude, or its Lipschitz\nconstant together with a bound on the message radius, or its derivative bounds on a bounded message domain, then every\nmessage size leaves an error floor. The target is unbounded under Gaussian weights, and such a decoder cannot follow it\n(gpt-6-astra, round 1 A3 and round 2 R2.1)."),
    # finding 1: table rows (decoder norm, uniformity, exceptional mass, C^k domain)
    ("| C^k, polylog budget on all derivatives up to order k | max(k a, n(n+1)/2)(1 - o(1)) |",
     "| C^k on a bounded message cube, polylog budget on all derivatives up to order k, uniform over backs | max(k a, n(n+1)/2)(1 - o(1)) |"),
    ("| Lipschitz: message in a ball of radius R outside a small exceptional mass, log(R x constant) = o(l), n >= 2 | (n(n+1)/2)(1 - o(1)) |",
     "| Lipschitz: decoder gamma-Lipschitz from R^s into the mean-square space over backs, message in a ball of radius R except on a mass below a fixed fraction of the chart event's probability, log(R gamma) = o(l), n >= 2 | (n(n+1)/2)(1 - o(1)) |"),
    # finding 8: section 9 case splitting
    ("- **Adaptive case splitting with unbounded case counts.** It falls outside every regularity class. The bit bound applies\n  to all bits it communicates and is weak.",
     "- **Adaptive case splitting with unbounded case counts.** Its updates are typically discontinuous (pruning, thresholds)\n  and its state grows, so the regularity classes need not apply. The bit bound applies to all bits it communicates and\n  is weak."),
    # review paragraph
    ("  - REVIEW2PLACEHOLDER",
     "  - A focused re-check (fresh session, 12:27-12:30Z; `review/review_gpt6_r2.md`) returned minor revision: nine\n    findings addressed and six partly (convergence wording, hypotheses in the Lipschitz and C^k rows, the coverage\n    notes, case splitting, empirical bounds, summary units). Those six were then completed; no further rerun was needed\n    and no third review was run."),
]
for a, b in rep:
    assert a in s, 'MISSING: ' + a[:70]
    s = s.replace(a, b)
s = s.replace("  - All findings were applied:\n", "  - All findings were applied, as follows:\n")
open(p, 'w').write(s)
print('report patched', len(rep))

q = '/home/agent/workspaces/mats/work/cut-rank-ext/notes/coverage_map.md'
t = open(q).read()
rep2 = [
    ("| An: readout holomorphic on a fixed neighbourhood (radius ratio q), amplitude <= eps^{-kappa} | (c(a, q, kappa) - o(1)) l | (p + o(1)) l / ln 2 when kappa >= p log2 3 (multilinear decoding of quantised weights) | lower PROVEN (Claude B1, corrected and strengthened by astra R2.2); upper PROVEN UNDER review (Claude Addendum 2, not co-prover-checked) |",
     "| An: readout holomorphic on a fixed polydisc (radius ratio rho), amplitude <= eps^{-kappa} | (c(a, rho, kappa) - o(1)) l | (p + o(1)) l / ln 2 at rho = 1/2, front and back clipped, kappa >= p log2 3 (multilinear decoding of quantised weights) | lower PROVEN (Claude B1, corrected and strengthened by astra R2.2); upper PROVEN (Claude Addendum 2, checked by the final review) |"),
    ("| P-multi: t turns, every message polynomial of degree <= D in earlier messages, no branching |",
     "| P-multi: t turns, every message AND the final output polynomial of degree <= D in earlier messages, no branching |"),
    ("| Stream: layer-by-layer state of size s | the class of the last update: degree q polynomial: s >= c eps^{-a/q}; C^k: as Ck; Lipschitz: n(n+1)/2 | | PROVEN UNDER the inherited hypotheses (Claude Sec. 5) |",
     "| Stream: layer-by-layer state of size s | the class of the composed readout from the cut state (hypotheses on the actual cut state and composed map): polynomial of degree D: s >= c eps^{-a/D}; C^k: as Ck; Lipschitz: n(n+1)/2 | | PROVEN UNDER the inherited hypotheses (Claude Sec. 5) |"),
    ("- ARC's actual ReLU cumulant propagation (fixed-size layer-by-layer state, nonlinear updates without uniform\n  regularity): only the dimension bounds apply, and its states satisfy them.",
     "- ARC's actual ReLU cumulant propagation (fixed-size layer-by-layer state, nonlinear updates without uniform\n  regularity): the dimension bounds apply only conditionally on regularity it lacks, and meeting a state-count\n  threshold would not by itself give arbitrary accuracy."),
]
for a, b in rep2:
    assert a in t, 'MISSING in coverage map: ' + a[:70]
    t = t.replace(a, b)
open(q, 'w').write(t)
print('coverage map patched', len(rep2))
