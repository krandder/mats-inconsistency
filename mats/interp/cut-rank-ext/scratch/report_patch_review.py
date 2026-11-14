"""Apply the final-review fixes to the outbox report (one-off editing helper; asserts every target string exists)."""
p = '/home/agent/workspaces/mats/outbox/cut-rank-ext-2026-10-31.md'
s = open(p).read()
rep = [
    # F1, F2, F3: table rows
    ("| analytic on a fixed polydisc (radius ratio rho < 1), amplitude <= eps^{-kappa} | c(a, rho, kappa) l | (p + o(1)) l / ln 2 (needs kappa >= p log2 3) | Claude B1, fixed by astra R2.2; upper bound Claude Addendum 2 |",
     "| analytic on a fixed polydisc (radius ratio rho < 1), amplitude <= eps^{-kappa} | c(a, rho, kappa) l | (p + o(1)) l / ln 2 at rho = 1/2, front and back clipped, kappa >= p log2 3 | Claude B1, fixed by astra R2.2; upper bound Claude Addendum 2, checked by the review |"),
    ("| C^k, polylog budget on all derivatives up to order k | max(k a, n(n+1)/2)(1 - o(1)) | (k+1) p | Claude B2; astra R2.3; Claude Addendum 1 |",
     "| C^k, polylog budget on all derivatives up to order k | max(k a, n(n+1)/2)(1 - o(1)) | (k+1) p | lower: Claude B2 and the Lipschitz row; upper: astra R2.3 (my Addendum 1 gives only C^{k-1,1}) |"),
    ("| Lipschitz, polylog constant | (n(n+1)/2)(1 - o(1)) | p | astra R2.4 with Claude C4; astra A2 |",
     "| Lipschitz: message in a ball of radius R outside a small exceptional mass, log(R x constant) = o(l), n >= 2 | (n(n+1)/2)(1 - o(1)) | p (clipped weights) | astra R2.4 with Claude C4; astra A2 |"),
    ("| continuous decoder, measurable encoder | none | 1, exact | Claude C3 |",
     "| continuous decoder, measurable encoder | none | 1, exact (continuous surjection R -> R^p) | Claude C3 |"),
    ("All lower bounds are PROVEN. The analytic upper bound (Addendum 2) was checked only by the final review, not by the\nco-prover.",
     "All entries are PROVEN. The analytic upper bound (Addendum 2) was checked by the final review, not by the co-prover. The\nn(n+1)/2 rows need the genuine ensemble: on ray fronts alone the family has dimension n."),
    # F2 bullet
    ("- *Analytic, upper bound.* Quantise the front weights to (p + o(1)) log2(1/eps) bits, send them as reals in {-1, 1}, and\n  decode with the multilinear extension of the clipped target. It is entire, with amplitude at most 3^B times the target\n  on the radius-2 polydisc.",
     "- *Analytic, upper bound.* Quantise the clipped front weights to (p + o(1)) log2(1/eps) bits, send them as reals in\n  {-1, 1}, and decode with the multilinear extension of the target (back weights clipped too). It is entire, with\n  amplitude at most 3^B times the target on the radius-2 polydisc, so radius ratio 1/2 (the review's sharper bound\n  sqrt(5)^B absorbs the polylog factors). For other ratios the amplitude threshold changes."),
    # F3 bullet
    ("- *C^k, upper bound.* Send a code of the clipped weights whose inverse is Holder of order k: a 1/k-snowflake\n  embedding (Assouad), or k + 1 interleaved Cantor-coordinate streams (astra). The clipped target, Lipschitz in the\n  weights, is then a Whitney jet with zero derivatives on the code's image. Whitney's extension gives a C^k decoder whose\n  budget grows only polylogarithmically.",
     "- *C^k, upper bound.* Send k + 1 interleaved Cantor-coordinate streams of the clipped weights (astra): the inverse of\n  this code is Holder of order strictly above k. The clipped target, Lipschitz in the weights, is then a Whitney jet with\n  zero derivatives on the code's image, and Whitney's extension gives a C^k decoder whose budget grows only\n  polylogarithmically. My 1/k-snowflake version (Assouad) has an inverse of order exactly k and gives only C^{k-1,1}."),
    # F4 C3 bullet
    ("- *Continuous decoder.* A space-filling curve onto the clipped weight ball gives s = 1 exactly. The encoder is only\n  measurable.",
     "- *Continuous decoder.* A continuous surjection from R onto R^p (space-filling curves on growing balls), with a\n  measurable selector as the encoder, gives s = 1, exact for every weight vector."),
    # F5 section 3 frame naming
    ("- **Frame naming (PROVEN, Claude D1 and D2).**\n  - If the back names any frame V and the front returns moments of degree <= K of the projected law, every feature is a\n    polynomial of degree <= K in h. Naming frames then never beats sending all moments of degree <= K one way.\n  - With B bits of naming, s forward reals and a degree-p readout: 2^B C(s+p, p) >= r(eps).",
     "- **Frame naming (PROVEN, Claude D1 and D2; both need the naming and the output coefficients to ignore the front's\n  replies).**\n  - If the back names any frame V and the front returns moments of degree <= K of the projected law, every feature is a\n    polynomial of degree <= K in h. Read linearly, naming frames then never beats sending all moments of degree <= K one\n    way.\n  - With B bits of naming, s forward reals and a degree-p readout: 2^B C(s+p, p) >= r(eps). An adaptive threshold on a\n    real reply already has infinite separation rank, so this needs non-adaptive naming."),
    # F1 streaming
    ("A layer-by-layer estimator keeps a state S_l in R^s with S_l = U_l(S_{l-1}, W_l). At any cut its state is a front\nmessage, and the rest of the computation is a decoder. So it inherits the one-way class of its last update (PROVEN,\nunder the inherited hypotheses):",
     "A layer-by-layer estimator keeps a state S_l in R^s with S_l = U_l(S_{l-1}, W_l). At any cut its state is a front\nmessage, and the rest of the computation is a decoder. So it inherits the one-way class of that composed decoder\n(PROVEN). The regularity and radius hypotheses must hold for the actual cut state and the composed readout, not only for\none update:"),
    # F6 section 5
    ("- At t = -1 the formal continuation is a Gaussian, where the closure is off by (w.v)/sqrt(2 pi) for a back row w. So its\n  error cannot vanish on any interval of t > 0.",
     "- Fix a back row w with w.v > 0. The physical target (w.v)(1 + t)/sqrt(2 pi) is analytic in t too. At t = -1 the formal\n  continuation is a standard Gaussian, for which the K >= 2 closure is exact and returns (w.v)/sqrt(2 pi), while the\n  continued target is 0. So closure minus target is a nonzero analytic function of t and cannot vanish on any interval\n  of t > 0. For odd K the extra trace of cumulant K + 1 must be supplied exactly."),
    ("- Quantitatively, the bias is at least c exp(-C K log(K+2)). The exact-input estimator therefore needs\n  K log K >~ log(1/eps). Its symmetric state then has Omega((l/log l)^n) entries at fixed n.",
     "- Quantitatively, the RMS error b_K over a fixed positive-probability neighbourhood of fronts and backs is at least\n  c exp(-C K log(K+2)). The exact-input one-step estimator therefore needs K log K >~ log(1/eps). Its symmetric state\n  then has Omega((l/log l)^n) entries at fixed n.\n- Numerically (post hoc, `scratch/arc_bias_alpha_t.py`, `logs/arc_bias_alpha_t.log`): ARC's own code fed the exact\n  cumulants of alpha_t v (n = 4, one back layer, t from 1/4 to 4, three back matrices) has relative bias 3-49% at K = 1,\n  3-19% at K = 2, 1-59% at K = 3 and 1-4% at K = 4."),
    # F11 / F13 design
    ("- Pilot: n = 8, c = d = 2, 16 fronts, timing only; never scored.",
     "- Pilot: n = 8, c = d = 2, 16 fronts. It was run once through the scorer to test the code\n  (`runs/pilot_n8_c2_d2_scores.json`); nothing was changed after it and it is not reported."),
    ("- Three independent splits of 8,192 input directions per front, radius integrated exactly.",
     "- Three independent splits of 8,192 input directions per front, radius integrated exactly. The front numbers (mean,\n  covariance, third cumulant) are estimates from one split, not exact moments.\n- Amendment 2 (12:10Z, after the final review): NC2 rerun with a global norm code, ARC3 rerun with the fourth cumulant,\n  and a moment-sample convergence check. All three are post hoc corrections."),
    # F12
    ("- On merged fronts mean propagation sits near the Monte Carlo floor. The best linear protocol with one more number\n  leaves 500 times as much.",
     "- On merged fronts mean propagation sits near the Monte Carlo floor. The in-sample estimate for the best linear protocol\n  with one more number leaves 500 times as much."),
    ("- The SVD columns are in-sample optima (biased low). They bound every linear protocol of that size, not only LIN1/LIN2.",
     "- The SVD columns are optimistic in-sample estimates (eigen-tails of a noisy cross-Gram) of the best linear protocol of\n  that size under the registered normalised loss. They are not certified bounds, and Err/Vbar is not a per-network\n  guarantee."),
    ("- LIN2 at width 16 fits 154 features on 256 training fronts",
     "- LIN2 at width 16 fits 153 features on 256 training fronts"),
    # F14 NC3
    ("- The rest of the spectrum falls smoothly over three to four decades, with no gap. Only 7-8 of 22 directions (n = 4,\n  c = 2), 3-6 of 34 (4, 3), 11-16 of 51 (6, 2) and 8-12 of 81 (6, 3) are resolved at 10x the noise.\n- Two fronts show exact extra zeros (1e-14). These fit the positive-probability extra symmetries astra pointed out\n  (always-on units).",
     "- The rest of the spectrum falls smoothly, spanning 3.4 to 8.2 decades, with no gap. Only 7-8 of 22 directions (n = 4,\n  c = 2), 3-6 of 34 (4, 3), 11-16 of 51 (6, 2) and 8-12 of 81 (6, 3) are resolved at 10x the noise.\n- Two fronts have numerical zeros (1e-14 relative), consistent with, but not certified as, the positive-probability\n  extra symmetries astra pointed out (always-on units).\n- Fronts rejected for dead units: 2, 6, 0 and 2. NC3 is inconclusive about the unresolved directions."),
]
for a, b in rep:
    assert a in s, 'MISSING: ' + a[:70]
    s = s.replace(a, b)
open(p, 'w').write(s)
print('patched', len(rep))
