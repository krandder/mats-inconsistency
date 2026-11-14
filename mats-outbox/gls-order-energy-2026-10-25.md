# GLS energy over orderings of an inconsistent any-order model: exact n=3 test (2026-10-25)

Code: work/gls-energy/gls_energy.py; output results.txt / results.json. CPU, exact enumeration, 200 random worlds per row.
World: random joint p over 3 binary variables; the model's conditionals q(x_i|x_S) = p's conditionals + logit noise.
"forward-stiff": atoms whose conditioning variables all precede i get sigma_lo, others sigma_hi (one ordering is all-stiff).
"mixed": each atom independently stiff or fuzzy (no ordering is all-stiff).
Energies are computed per configuration, then normalised over all 8 configurations (so impossible ones get ~0 mass).
- gls_oracle: GLS weights from the true atom error variances (uses the truth: reference only)
- gls_est: GLS weights from atom variances fitted to the model's own ordering contrasts (no data, no truth)
- uniform: equal weights on the 6 orderings (= Shapley-weighted pseudo-energy)
- fwd/rev: one ordering; pseudo-Gibbs: exact stationary distribution of random scan and two systematic scans.

KL(p || mu), mean over 200 worlds (se in results.txt):

| stiff/fuzzy sigma | gls_oracle | gls_est | uniform | best single order | pseudo-Gibbs (rand / 123 / 321) | TV 123 vs 321 |
|---|---|---|---|---|---|---|
| 0.3/0.3 fwd | 0.0058 | 0.0094 | 0.0063 | 0.0237 | 0.016 / 0.019 / 0.019 | 0.045 |
| 0.1/0.5 fwd | 0.0018 | 0.0051 | 0.0065 | 0.0026 | 0.029 / 0.029 / 0.040 | 0.060 |
| 0.05/1.0 fwd | 0.0006 | 0.0138 | 0.0251 | 0.0007 | 0.102 / 0.099 / 0.148 | 0.111 |
| 0.05/2.0 fwd | 0.0008 | 0.0609 | 0.1020 | 0.0007 | 0.297 / 0.293 / 0.498 | 0.189 |
| 0.1/0.5 mixed | 0.0030 | 0.0077 | 0.0090 | 0.0338 | 0.019 / 0.023 / 0.023 | 0.046 |
| 0.05/1.0 mixed | 0.0074 | 0.0258 | 0.0344 | 0.1222 | 0.066 / 0.083 / 0.082 | 0.084 |
| 0.05/2.0 mixed | 0.0327 | 0.1021 | 0.1341 | 0.4167 | 0.193 / 0.271 / 0.262 | 0.141 |

Findings
1. Pseudo-Gibbs is scan-order dependent (TV 0.05-0.19 between the two sweeps, growing with inconsistency) and worse than the uniform order-energy in every row except one (mixed 0.05/2.0, where random scan is between uniform and the single orders).
2. With the right information structure (oracle C), the GLS energy is 4-120x better than uniform averaging whenever stiffness differs; when one ordering is all-stiff it matches that ordering, and in the mixed case (no stiff ordering) it is far better than any single ordering.
3. Estimating C from the model's own contrasts (no data) recovers part of it: 15-45% below uniform when stiffness differs; slightly worse than uniform when it doesn't (estimation noise from only 8 configurations, and a variance model that ignores the v-dependence).
Caveats: one noise model (independent logit noise per atom and conditioning value); the oracle uses the truth; the estimator is a crude moment fit from one model's 8 configurations. Not reviewed.

## Corrections after independent adversarial review (2026-10-25; scripts /tmp/gls-adv/)
- No code bug changes any number; gls_est uses no truth; (d) consistent model -> exact recovery holds for n=3 and n=4.
- My table text was wrong in two places: pseudo-Gibbs is worse than uniform in all 7 rows (not "all but one"); at equal precision gls_est is 49% worse than uniform (not "slightly"). The oracle/uniform ratios are 3.0-127x, not 4-120x.
- The largest gains come from the forward-stiff pattern, where one ordering is all-stiff by construction; an oracle that just picks the lowest-variance ordering does as well. In mixed worlds, the GLS oracle is only 1.3-1.5x better than the per-world best of all 6 orderings (4.7-6.3x better than the best fixed ordering chosen by sigma).
- Fails when the full conditionals are the stiff atoms: pseudo-Gibbs is then best (it uses exactly those atoms), the GLS oracle is only 1.1-1.3x better than uniform, gls_est is worse. Reason: every ordering contains one full conditional and two fuzzy atoms, so weighting whole orderings cannot isolate the stiff atoms.
- gls_est beats uniform only for precision ratios >= ~5x with independent noise; it loses at smaller ratios and under correlated noise, and can't tell from the model which regime it is in.
- Survives: policy-independent target; oracle gains under forward-stiff and mixed patterns for all noise models, concentrations, metrics and n=4.
Implication: aggregate at the level of atoms (conditionals), not orderings: estimate the joint by a precision-weighted projection of all atoms onto consistent joints. That subsumes both GLS over orderings and Gibbs' use of full conditionals.
