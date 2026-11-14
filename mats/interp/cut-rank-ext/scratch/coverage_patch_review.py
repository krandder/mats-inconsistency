"""Apply final-review findings 7-8 to notes/coverage_map.md (one-off editing helper)."""
p = '/home/agent/workspaces/mats/work/cut-rank-ext/notes/coverage_map.md'
s = open(p).read()
rep = [
    ("# Coverage map: ARC's estimators against the cut-rank-ext bounds (revised after co-prover round 2, 11:46Z)",
     "# Coverage map: ARC's estimators against the cut-rank-ext bounds (revised after co-prover round 2, 11:46Z, and after the\n# final GPT-6 review findings 7-8, 12:16Z)"),
    ("| K = 1 cumulant propagation (Wu et al. 2026, mean plus one variance trace) | layer-by-layer, state n + 1, Hermite step | Stream, continuous on positive variance | only the dimension bounds; satisfied | nonlinear Hermite step |",
     "| K = 1 cumulant propagation (Wu et al. 2026, mean plus one variance trace) | layer-by-layer, state n + 1, Hermite step | Stream, continuous on positive variance | n + 1 < n(n+1)/2 for n >= 3: IF its readout were Lipschitz with subpolynomial constants, the Lipschitz bound would rule out arbitrary accuracy; it is not uniformly Lipschitz, so this is conditional | nonlinear Hermite step |"),
    ("| Covariance propagation, full Gaussian moments (CNX22 D.3; Xu 2024 Method 1; G. Wu thesis 2.1) | (mu, Sigma) per layer, exact Gaussian ReLU moments | Stream, continuous on PSD states | only dimension bounds and Bits (vacuous for its state) | nonlinear closure |",
     "| Covariance propagation for ReLU nets, full Gaussian moment matching (Xu 2024 Method 1; G. Wu thesis 2.1); CNX22 D.3 is the arithmetic-circuit version (product gates), a different model | (mu, Sigma) per layer, exact Gaussian ReLU moments | Stream, continuous on PSD states | dimension bounds only conditionally (no uniform regularity); Bits only for a finite-precision implementation | nonlinear closure |"),
    ("| Wu et al. K = 2 (Alg. 2: off-diagonal Sigma_ij c_i c_j, leading Hermite term) | (mu, Sigma) per layer | Stream, continuous on PSD states, not uniformly Lipschitz in Sigma | same; plus the oracle bias floor R2.5 at fixed K | as above |",
     "| Wu et al. K = 2 (Alg. 2: off-diagonal Sigma_ij c_i c_j, leading Hermite term); WhestBench covariance baseline (implementation not verified) | (mu, Sigma) per layer | Stream, continuous on PSD states, not uniformly Lipschitz in Sigma | same; plus the oracle bias floor R2.5 at fixed K | as above |"),
    ("| Random-walk Fourier sketching (Winer note) | backward random walks choose an instance-dependent support; forward linear propagation | back-and-forth, sampled, adaptive | BF-bits only once the whole tail-dependent naming transcript is counted | adaptive nonlinear support selection; different estimand |",
     "| Random-walk Fourier sketching (Winer note) | backward random walks choose an instance-dependent support; forward linear propagation | back-and-forth, sampled, adaptive; among the sources inspected the only ARC estimator with a backward component | BF-bits only once the whole tail-dependent naming transcript is counted | adaptive nonlinear support selection; different estimand |"),
    ("| Independent-gate, sparse-covariance, selected-term estimators (CNX22 Sec. 6.2, D.4, D.7) | Boolean circuits, linear circuits, permanents | other problems | n/a | n/a |",
     "| Independent-gate, sparse-covariance, selected-term estimators (CNX22 Sec. 6.2, D.4, D.7); arithmetic-circuit cumulant propagation (CNX22 D.5-D.6, Algorithm 1) | Boolean circuits, arithmetic circuits, linear circuits, permanents | other problems | n/a | n/a |\n| #SAT estimators, Methods 1-3 (Xu 2024 appendix) | independence, conditioning on the number of true variables, regression merge | other problems | n/a | n/a |"),
    ("| Region enumeration and Girard volumes (AlgZoo 2026; Hilton 2024-12; thesis 6.3) | exact per region, case counts grow | adaptive case splitting | none | different estimand; unbounded case counts |",
     "| Region enumeration and Girard volumes (AlgZoo 2026; Hilton 2024-12; thesis 6.3) | exact per region, case counts grow with depth | adaptive case splitting (enumeration itself need not make the estimator discontinuous) | none | different estimand; unbounded case counts |"),
    ("- Sampling and hybrids: outside by construction.",
     "- Sampling and hybrids: different estimands or access in ARC's uses; plain sampling is a Lipschitz readout of sampled\n  cut states and meets the dimension bounds rather than evading them."),
]
for a, b in rep:
    assert a in s, 'MISSING: ' + a[:70]
    s = s.replace(a, b)
open(p, 'w').write(s)
print('patched', len(rep))
