# Atom-level energy for an inconsistent any-order model, with learned precision (2026-10-25)

Kelvin: "run the atom-level test and see if you can train this somehow on synthetic world data". Not reviewed.
Code: work/gls-energy/atom/{atom_energy.py, run.py, summarize.py}; raw per-world KLs results.json; full tables summary.md.

Estimator: mu* = argmin_mu sum over atom (i,S) and context v_S of w * mu(v_S) * KL(mu(X_i|v_S) || q(X_i|v_S)) + 1e-4 * sum mu log mu,
exact over {0,1}^n (n = 3, 4), L-BFGS with analytic gradient. Consistent q gives KL(p||mu*) ~ 1e-10 for any positive w (checked).
World: Dirichlet(0.7) joint; each atom-context = true conditional + logit noise. Stiffness patterns: equal, forward, mixed,
fullstiff (full conditionals precise), graded (precision rises with |S|). Noise: gauss, t3 (heavy tails), tempered (fuzzy
atoms too flat: logit*tau, tau down to 0.5, plus small noise). Sigma (stiff/fuzzy) 0.25/0.35, 0.1/0.5, 0.05/1.0.
90 cells x 200 test worlds (seeds 5000-5199 per cell), n=3 and n=4.

Weight variants (w per atom-context):
- A-unif: w = 1.
- A-orc: 1 / expected squared logit error under the generator (uses the truth; reference only).
- A-self: per-atom residual variance vs a uniform-w fit, 2 rounds, shrunk toward uniform (lambda 0.75, chosen on training seeds 1000-1299).
- A-irls: A-self plus Student-t(3) reweighting by residual.
- A-d100 / A-d1k: per-atom variance from 100 / 1000 real samples (lambda 1).
- A-lrn: ridge regression (quadratic features computable without truth: |S|, n, is-full, entropy, |logit q|, residuals vs the
  uniform-w fit, disagreement with sub/super-context atoms, log mu(v_S)) predicting log squared logit error; trained on 700
  synthetic worlds (seeds 1300-1999, a mixture of all patterns, noises, sigmas, n); held-out R^2 0.32; lambda 1.
- A-lrnW: the same model trained separately per (pattern, noise).
Baselines: O-unif uniform order-averaging energy; O-orc / O-est GLS over orderings (oracle / contrast-estimated); Gibbs random scan.

## Summary: geometric-mean ratio of mean KL to plain uniform order-averaging (O-unif) over all 90 cells (lower is better)

| A-unif | A-orc | A-self | A-irls | A-d100 | A-d1k | A-lrn | A-lrnW | O-orc | O-est | Gibbs |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.898 | 0.114 | 0.685 | 0.713 | 0.620 | 0.260 | 0.576 | 0.346 | 0.407 | 0.949 | 0.510 |

Cells worse than O-unif (of 90; worst ratio): A-unif 31 (1.52x), A-orc 7 (1.48x, equal-tempered only), A-self 22 (2.02x),
A-irls 28 (2.82x), A-d100 26 (2.10x), A-d1k 0, A-lrn 14 (1.32x), A-lrnW 9 (4.13x, one heavy-tail cell), O-orc 2, O-est 49, Gibbs 62 (4.88x).

## Findings
1. Atom-level beats ordering-level, as the review predicted. With the true error structure the atom estimator is 0.114x
   uniform averaging overall (ordering-level GLS 0.407x), and it wins exactly where ordering-level failed: full-conditional-
   stiff (0.045 vs 0.686 of A-unif) and size-graded (0.099 vs 1.118). It also matches Gibbs where Gibbs is right
   (fullstiff/graded) and is far better where Gibbs is wrong (forward/mixed, where Gibbs is 2.2-2.8x worse than A-unif).
2. Learning C on synthetic worlds works partly. A model trained across all world types (A-lrn: no truth at test time, no data)
   gets 0.576x overall and is never worse than uniform averaging by more than 1.32x. Trained per world type (A-lrnW) it gets
   0.346x, including the too-flat (tempered) worlds where self-estimation cannot see the bias (0.246 vs 0.995 of A-unif).
3. Real data is the strongest route: 1000 samples (A-d1k) give 0.260x and are never worse than uniform averaging in any of
   the 90 cells; 100 samples are too few (worse than uniform in 26 cells, mostly the equal-precision ones).
4. Self-estimation from the model's own inconsistency (A-self, A-irls) helps modestly (0.69-0.71x) and fails under bias: in
   full-conditional-stiff tempered worlds it loses to A-unif in 95-100% of worlds. Bias is invisible to residuals.
5. Even the oracle weights hurt when all atoms are equally biased (equal-tempered, 1.1-1.5x worse): weighting cannot remove
   bias. Systematic flatness needs calibration, not weights.
6. Heavy tails produce rare catastrophic worlds for every method (KL > 1 in a few of 200 worlds; e.g. A-lrnW max 70 in
   n4 mixed t3 0.05/1.0, A-irls max 33 in n3 graded t3 0.05/1.0); the Student-t reweighting did not fix them.

## Caveats
- The oracle uses the generator's true error variances (reference only). One generator family (independent logit noise per
  atom-context, plus the tempered bias variant); n <= 4.
- Per-cell means in t3 cells are dominated by a few heavy-tail worlds; win rates (in the tables) are more robust.
- Learned-C train (seeds 1000-1999) and test (5000+) worlds are disjoint but come from the same generator family, so A-lrn and
  A-lrnW show in-distribution learning, not transfer to real models.
- Weights are normalised to mean 1 and clipped to [1e-3, 1e3]; lambdas are chosen on training seeds only; per-type models reuse
  the global lambda.
- The first run was killed by a timeout (BLAS thread oversubscription); the reported numbers are the full single-threaded rerun.

## Ratios to A-unif by stiffness pattern and by noise model (geometric mean over the matching cells)

### Geometric-mean ratio to A-unif by stiffness pattern (all n, noise, sigma)
| pattern | atom_oracle | atom_self | atom_irls | atom_data1000 | atom_learned | atom_learned_within | ord_gls_oracle | gibbs_random |
|---|---|---|---|---|---|---|---|---|
| equal | 1.115 | 0.947 | 0.980 | 0.702 | 0.847 | 1.144 | 1.026 | 2.000 |
| forward | 0.056 | 0.744 | 0.707 | 0.264 | 0.811 | 0.446 | 0.075 | 2.835 |
| mixed | 0.120 | 0.760 | 0.737 | 0.316 | 0.670 | 0.666 | 0.322 | 2.221 |
| fullstiff | 0.045 | 0.856 | 0.867 | 0.168 | 0.559 | 0.125 | 0.686 | 0.042 |
| graded | 0.099 | 0.564 | 0.715 | 0.206 | 0.422 | 0.200 | 1.118 | 0.111 |

### Same by noise model
| noise | atom_oracle | atom_self | atom_irls | atom_data1000 | atom_learned | atom_learned_within | ord_gls_oracle | gibbs_random |
|---|---|---|---|---|---|---|---|---|
| gauss | 0.231 | 0.782 | 0.831 | 0.369 | 0.676 | 0.551 | 0.664 | 1.042 |
| t3 | 0.234 | 0.570 | 0.588 | 0.198 | 0.455 | 0.424 | 0.714 | 0.850 |
| tempered | 0.038 | 0.995 | 1.025 | 0.332 | 0.858 | 0.246 | 0.196 | 0.207 |


## Full per-cell tables: KL(p||mu) means, [win rate vs A-unif]

### n=3: KL(p||mu), mean over test worlds (win rate vs A-unif in brackets for weighted atom methods)

| cell | A-unif | A-orc | A-self | A-irls | A-d100 | A-d1k | A-lrn | A-lrnW | O-unif | O-orc | O-est | Gibbs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| equal|gauss|0.25/0.35 | 0.0063 | 0.0063 | 0.0070 [0.41] | 0.0076 [0.34] | 0.0093 [0.40] | 0.0043 [0.80] | 0.0063 [0.52] | 0.0065 [0.48] | 0.0068 | 0.0066 | 0.0094 | 0.0150 |
| equal|gauss|0.1/0.5 | 0.0033 | 0.0033 | 0.0041 [0.37] | 0.0044 [0.28] | 0.0076 [0.28] | 0.0026 [0.65] | 0.0034 [0.49] | 0.0036 [0.38] | 0.0037 | 0.0035 | 0.0051 | 0.0093 |
| equal|gauss|0.05/1.0 | 0.0034 | 0.0034 | 0.0040 [0.34] | 0.0044 [0.27] | 0.0069 [0.28] | 0.0026 [0.70] | 0.0035 [0.49] | 0.0037 [0.46] | 0.0038 | 0.0036 | 0.0051 | 0.0078 |
| equal|t3|0.25/0.35 | 0.0188 | 0.0188 | 0.0149 [0.55] | 0.0143 [0.55] | 0.0137 [0.56] | 0.0075 [0.86] | 0.0119 [0.62] | 0.0137 [0.67] | 0.0201 | 0.0196 | 0.0176 | 0.0359 |
| equal|t3|0.1/0.5 | 0.0111 | 0.0111 | 0.0082 [0.57] | 0.0081 [0.55] | 0.0095 [0.54] | 0.0046 [0.90] | 0.0068 [0.71] | 0.0068 [0.76] | 0.0117 | 0.0117 | 0.0104 | 0.0199 |
| equal|t3|0.05/1.0 | 0.0107 | 0.0107 | 0.0085 [0.51] | 0.0087 [0.46] | 0.0085 [0.48] | 0.0042 [0.85] | 0.0064 [0.69] | 0.0097 [0.67] | 0.0119 | 0.0113 | 0.0102 | 0.0226 |
| equal|tempered|0.25/0.35 | 0.0195 | 0.0258 | 0.0196 [0.48] | 0.0199 [0.45] | 0.0193 [0.62] | 0.0189 [0.70] | 0.0197 [0.46] | 0.0277 [0.09] | 0.0197 | 0.0179 | 0.0204 | 0.0220 |
| equal|tempered|0.1/0.5 | 0.0193 | 0.0260 | 0.0192 [0.58] | 0.0193 [0.47] | 0.0196 [0.53] | 0.0190 [0.64] | 0.0194 [0.45] | 0.0269 [0.17] | 0.0194 | 0.0177 | 0.0197 | 0.0215 |
| equal|tempered|0.05/1.0 | 0.0193 | 0.0253 | 0.0193 [0.49] | 0.0195 [0.42] | 0.0192 [0.57] | 0.0190 [0.65] | 0.0193 [0.48] | 0.0262 [0.15] | 0.0193 | 0.0177 | 0.0198 | 0.0216 |
| forward|gauss|0.25/0.35 | 0.0060 | 0.0061 | 0.0071 [0.33] | 0.0076 [0.27] | 0.0091 [0.34] | 0.0043 [0.78] | 0.0061 [0.51] | 0.0066 [0.40] | 0.0062 | 0.0058 | 0.0085 | 0.0166 |
| forward|gauss|0.1/0.5 | 0.0074 | 0.0019 | 0.0056 [0.73] | 0.0056 [0.68] | 0.0057 [0.64] | 0.0034 [0.84] | 0.0061 [0.67] | 0.0050 [0.83] | 0.0067 | 0.0021 | 0.0060 | 0.0286 |
| forward|gauss|0.05/1.0 | 0.0248 | 0.0004 | 0.0129 [1.00] | 0.0115 [0.97] | 0.0064 [0.96] | 0.0036 [1.00] | 0.0174 [0.81] | 0.0114 [0.98] | 0.0225 | 0.0006 | 0.0166 | 0.0945 |
| forward|t3|0.25/0.35 | 0.0165 | 0.0158 | 0.0127 [0.61] | 0.0127 [0.57] | 0.0125 [0.62] | 0.0070 [0.92] | 0.0111 [0.69] | 0.0114 [0.69] | 0.0158 | 0.0156 | 0.0167 | 0.0384 |
| forward|t3|0.1/0.5 | 0.0219 | 0.0042 | 0.0116 [0.81] | 0.0101 [0.81] | 0.0086 [0.79] | 0.0052 [0.95] | 0.0101 [0.83] | 0.0080 [0.91] | 0.0210 | 0.0043 | 0.0118 | 0.0652 |
| forward|t3|0.05/1.0 | 0.0905 | 0.0011 | 0.0397 [0.98] | 0.0318 [0.96] | 0.1380 [0.96] | 0.0090 [0.99] | 0.0531 [0.92] | 0.0196 [0.97] | 0.0810 | 0.0019 | 0.0369 | 0.1786 |
| forward|tempered|0.25/0.35 | 0.0230 | 0.0010 | 0.0225 [0.61] | 0.0242 [0.48] | 0.0103 [0.91] | 0.0053 [1.00] | 0.0295 [0.18] | 0.0052 [0.98] | 0.0224 | 0.0012 | 0.0397 | 0.0580 |
| forward|tempered|0.1/0.5 | 0.0225 | 0.0002 | 0.0203 [0.71] | 0.0212 [0.60] | 0.0087 [0.94] | 0.0040 [1.00] | 0.0276 [0.24] | 0.0048 [0.99] | 0.0219 | 0.0003 | 0.0383 | 0.0551 |
| forward|tempered|0.05/1.0 | 0.0248 | 0.0000 | 0.0200 [0.89] | 0.0201 [0.78] | 0.0086 [0.96] | 0.0040 [1.00] | 0.0270 [0.43] | 0.0062 [0.97] | 0.0245 | 0.0001 | 0.0425 | 0.0671 |
| mixed|gauss|0.25/0.35 | 0.0061 | 0.0057 | 0.0068 [0.38] | 0.0073 [0.32] | 0.0099 [0.33] | 0.0042 [0.79] | 0.0058 [0.54] | 0.0060 [0.51] | 0.0069 | 0.0062 | 0.0089 | 0.0158 |
| mixed|gauss|0.1/0.5 | 0.0090 | 0.0026 | 0.0073 [0.74] | 0.0076 [0.70] | 0.0072 [0.67] | 0.0034 [0.91] | 0.0065 [0.78] | 0.0063 [0.80] | 0.0100 | 0.0046 | 0.0099 | 0.0251 |
| mixed|gauss|0.05/1.0 | 0.0356 | 0.0023 | 0.0247 [0.85] | 0.0231 [0.82] | 0.0129 [0.92] | 0.0084 [0.96] | 0.0209 [0.86] | 0.0346 [0.91] | 0.0380 | 0.0100 | 0.0280 | 0.0696 |
| mixed|t3|0.25/0.35 | 0.0152 | 0.0143 | 0.0137 [0.54] | 0.0136 [0.54] | 0.0121 [0.61] | 0.0070 [0.87] | 0.0109 [0.71] | 0.0111 [0.68] | 0.0167 | 0.0153 | 0.0171 | 0.0403 |
| mixed|t3|0.1/0.5 | 0.0273 | 0.0054 | 0.0166 [0.75] | 0.0199 [0.74] | 0.0125 [0.72] | 0.0056 [0.94] | 0.0114 [0.82] | 0.0098 [0.85] | 0.0310 | 0.0143 | 0.0143 | 0.0509 |
| mixed|t3|0.05/1.0 | 0.0924 | 0.0089 | 0.0491 [0.93] | 0.0477 [0.91] | 0.0200 [0.93] | 0.0148 [0.99] | 0.0375 [0.89] | 0.0383 [0.94] | 0.0938 | 0.0354 | 0.0526 | 0.1466 |
| mixed|tempered|0.25/0.35 | 0.0270 | 0.0052 | 0.0311 [0.38] | 0.0322 [0.35] | 0.0199 [0.75] | 0.0145 [0.97] | 0.0292 [0.49] | 0.0213 [0.83] | 0.0271 | 0.0083 | 0.0389 | 0.0471 |
| mixed|tempered|0.1/0.5 | 0.0267 | 0.0019 | 0.0303 [0.44] | 0.0317 [0.43] | 0.0195 [0.79] | 0.0124 [0.97] | 0.0273 [0.60] | 0.0190 [0.90] | 0.0270 | 0.0063 | 0.0394 | 0.0455 |
| mixed|tempered|0.05/1.0 | 0.0256 | 0.0025 | 0.0256 [0.62] | 0.0260 [0.60] | 0.0161 [0.85] | 0.0111 [0.97] | 0.0234 [0.70] | 0.0160 [0.94] | 0.0262 | 0.0065 | 0.0370 | 0.0459 |
| fullstiff|gauss|0.25/0.35 | 0.0073 | 0.0061 | 0.0077 [0.48] | 0.0084 [0.41] | 0.0094 [0.42] | 0.0044 [0.83] | 0.0066 [0.57] | 0.0066 [0.59] | 0.0077 | 0.0074 | 0.0097 | 0.0111 |
| fullstiff|gauss|0.1/0.5 | 0.0144 | 0.0018 | 0.0117 [0.77] | 0.0126 [0.71] | 0.0094 [0.70] | 0.0023 [0.94] | 0.0075 [0.94] | 0.0036 [0.97] | 0.0140 | 0.0114 | 0.0156 | 0.0016 |
| fullstiff|gauss|0.05/1.0 | 0.0546 | 0.0004 | 0.0409 [0.81] | 0.0434 [0.77] | 0.0138 [0.91] | 0.0028 [0.99] | 0.0207 [0.97] | 0.0034 [0.99] | 0.0545 | 0.0444 | 0.0665 | 0.0004 |
| fullstiff|t3|0.25/0.35 | 0.0190 | 0.0160 | 0.0153 [0.59] | 0.0150 [0.58] | 0.0130 [0.66] | 0.0073 [0.94] | 0.0111 [0.81] | 0.0145 [0.66] | 0.0188 | 0.0183 | 0.0195 | 0.0252 |
| fullstiff|t3|0.1/0.5 | 0.0398 | 0.0111 | 0.0229 [0.80] | 0.0258 [0.77] | 0.0110 [0.81] | 0.0043 [0.98] | 0.0132 [0.95] | 0.0077 [0.94] | 0.0370 | 0.0361 | 0.0255 | 0.0107 |
| fullstiff|t3|0.05/1.0 | 0.1467 | 0.0015 | 0.0825 [0.86] | 0.0813 [0.85] | 0.0148 [0.93] | 0.0034 [1.00] | 0.0410 [0.98] | 0.0053 [0.99] | 0.1375 | 0.1117 | 0.1106 | 0.0012 |
| fullstiff|tempered|0.25/0.35 | 0.0382 | 0.0014 | 0.0556 [0.02] | 0.0557 [0.04] | 0.0234 [0.78] | 0.0059 [0.99] | 0.0281 [0.97] | 0.0016 [1.00] | 0.0297 | 0.0149 | 0.0407 | 0.0010 |
| fullstiff|tempered|0.1/0.5 | 0.0399 | 0.0002 | 0.0567 [0.05] | 0.0574 [0.05] | 0.0239 [0.80] | 0.0045 [0.99] | 0.0282 [0.98] | 0.0011 [1.00] | 0.0315 | 0.0159 | 0.0421 | 0.0002 |
| fullstiff|tempered|0.05/1.0 | 0.0406 | 0.0000 | 0.0534 [0.22] | 0.0548 [0.23] | 0.0241 [0.76] | 0.0049 [1.00] | 0.0255 [0.98] | 0.0011 [1.00] | 0.0324 | 0.0189 | 0.0376 | 0.0000 |
| graded|gauss|0.25/0.35 | 0.0061 | 0.0051 | 0.0064 [0.48] | 0.0070 [0.40] | 0.0105 [0.45] | 0.0038 [0.81] | 0.0056 [0.56] | 0.0057 [0.54] | 0.0073 | 0.0068 | 0.0094 | 0.0113 |
| graded|gauss|0.1/0.5 | 0.0072 | 0.0014 | 0.0041 [0.90] | 0.0045 [0.84] | 0.0056 [0.65] | 0.0018 [0.91] | 0.0036 [0.84] | 0.0023 [0.93] | 0.0104 | 0.0077 | 0.0086 | 0.0018 |
| graded|gauss|0.05/1.0 | 0.0203 | 0.0005 | 0.0071 [0.98] | 0.0106 [0.95] | 0.0045 [0.88] | 0.0013 [0.99] | 0.0039 [0.97] | 0.0026 [1.00] | 0.0342 | 0.0237 | 0.0196 | 0.0004 |
| graded|t3|0.25/0.35 | 0.0214 | 0.0183 | 0.0130 [0.58] | 0.0135 [0.59] | 0.0118 [0.64] | 0.0062 [0.91] | 0.0100 [0.77] | 0.0095 [0.75] | 0.0218 | 0.0228 | 0.0162 | 0.0251 |
| graded|t3|0.1/0.5 | 0.0160 | 0.0041 | 0.0091 [0.88] | 0.0101 [0.85] | 0.0073 [0.74] | 0.0028 [0.95] | 0.0054 [0.93] | 0.0038 [0.94] | 0.0221 | 0.0171 | 0.0164 | 0.0047 |
| graded|t3|0.05/1.0 | 0.0779 | 0.0012 | 0.0288 [0.98] | 0.3468 [0.90] | 0.0057 [0.91] | 0.0014 [0.99] | 0.0096 [0.99] | 0.0069 [1.00] | 0.1232 | 0.0918 | 0.0460 | 0.0013 |
| graded|tempered|0.25/0.35 | 0.0143 | 0.0013 | 0.0146 [0.39] | 0.0151 [0.38] | 0.0118 [0.59] | 0.0044 [0.94] | 0.0094 [0.96] | 0.0019 [1.00] | 0.0152 | 0.0077 | 0.0147 | 0.0009 |
| graded|tempered|0.1/0.5 | 0.0103 | 0.0002 | 0.0074 [0.78] | 0.0077 [0.76] | 0.0080 [0.68] | 0.0024 [0.93] | 0.0053 [0.96] | 0.0008 [1.00] | 0.0126 | 0.0054 | 0.0092 | 0.0002 |
| graded|tempered|0.05/1.0 | 0.0090 | 0.0000 | 0.0037 [0.97] | 0.0040 [0.96] | 0.0086 [0.79] | 0.0013 [0.95] | 0.0028 [0.95] | 0.0003 [1.00] | 0.0132 | 0.0066 | 0.0078 | 0.0000 |

### n=4: KL(p||mu), mean over test worlds (win rate vs A-unif in brackets for weighted atom methods)

| cell | A-unif | A-orc | A-self | A-irls | A-d100 | A-d1k | A-lrn | A-lrnW | O-unif | O-orc | O-est | Gibbs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| equal|gauss|0.25/0.35 | 0.0044 | 0.0044 | 0.0049 [0.32] | 0.0054 [0.23] | 0.0092 [0.14] | 0.0036 [0.67] | 0.0043 [0.52] | 0.0045 [0.41] | 0.0055 | 0.0050 | 0.0075 | 0.0175 |
| equal|gauss|0.1/0.5 | 0.0024 | 0.0024 | 0.0027 [0.30] | 0.0030 [0.23] | 0.0066 [0.06] | 0.0023 [0.53] | 0.0024 [0.48] | 0.0026 [0.36] | 0.0031 | 0.0028 | 0.0041 | 0.0080 |
| equal|gauss|0.05/1.0 | 0.0024 | 0.0024 | 0.0029 [0.32] | 0.0032 [0.22] | 0.0061 [0.14] | 0.0024 [0.57] | 0.0025 [0.45] | 0.0027 [0.33] | 0.0032 | 0.0028 | 0.0040 | 0.0085 |
| equal|t3|0.25/0.35 | 0.0124 | 0.0124 | 0.0090 [0.70] | 0.0085 [0.70] | 0.0109 [0.51] | 0.0057 [0.91] | 0.0075 [0.90] | 0.0094 [0.88] | 0.0141 | 0.0130 | 0.0159 | 0.0368 |
| equal|t3|0.1/0.5 | 0.0073 | 0.0073 | 0.0052 [0.74] | 0.0052 [0.69] | 0.0082 [0.36] | 0.0036 [0.85] | 0.0044 [0.83] | 0.0058 [0.85] | 0.0085 | 0.0079 | 0.0096 | 0.0233 |
| equal|t3|0.05/1.0 | 0.0088 | 0.0088 | 0.0058 [0.76] | 0.0054 [0.74] | 0.0091 [0.47] | 0.0036 [0.91] | 0.0048 [0.91] | 0.0265 [0.89] | 0.0120 | 0.0108 | 0.0096 | 0.0227 |
| equal|tempered|0.25/0.35 | 0.0214 | 0.0303 | 0.0215 [0.47] | 0.0222 [0.32] | 0.0213 [0.55] | 0.0207 [0.66] | 0.0213 [0.53] | 0.0336 [0.03] | 0.0212 | 0.0189 | 0.0216 | 0.0248 |
| equal|tempered|0.1/0.5 | 0.0208 | 0.0303 | 0.0210 [0.39] | 0.0220 [0.18] | 0.0213 [0.51] | 0.0206 [0.62] | 0.0208 [0.52] | 0.0321 [0.03] | 0.0208 | 0.0184 | 0.0214 | 0.0234 |
| equal|tempered|0.05/1.0 | 0.0208 | 0.0307 | 0.0209 [0.43] | 0.0222 [0.23] | 0.0212 [0.46] | 0.0204 [0.68] | 0.0208 [0.49] | 0.0325 [0.02] | 0.0208 | 0.0183 | 0.0212 | 0.0236 |
| forward|gauss|0.25/0.35 | 0.0046 | 0.0047 | 0.0052 [0.34] | 0.0058 [0.21] | 0.0091 [0.13] | 0.0037 [0.77] | 0.0046 [0.50] | 0.0049 [0.35] | 0.0055 | 0.0048 | 0.0074 | 0.0201 |
| forward|gauss|0.1/0.5 | 0.0066 | 0.0019 | 0.0048 [0.81] | 0.0047 [0.78] | 0.0068 [0.50] | 0.0034 [0.91] | 0.0046 [0.93] | 0.0044 [0.94] | 0.0061 | 0.0019 | 0.0053 | 0.0300 |
| forward|gauss|0.05/1.0 | 0.0269 | 0.0006 | 0.0141 [0.99] | 0.0123 [0.99] | 0.0099 [0.95] | 0.0055 [1.00] | 0.0126 [0.98] | 0.0130 [0.97] | 0.0236 | 0.0009 | 0.0138 | 0.1080 |
| forward|t3|0.25/0.35 | 0.0153 | 0.0148 | 0.0112 [0.71] | 0.0106 [0.71] | 0.0125 [0.57] | 0.0066 [0.91] | 0.0093 [0.87] | 0.0099 [0.81] | 0.0175 | 0.0155 | 0.0178 | 0.0500 |
| forward|t3|0.1/0.5 | 0.0229 | 0.0075 | 0.0118 [0.90] | 0.0091 [0.92] | 0.0100 [0.86] | 0.0059 [0.99] | 0.0132 [0.97] | 0.0124 [0.98] | 0.0224 | 0.0092 | 0.0138 | 0.0734 |
| forward|t3|0.05/1.0 | 0.0823 | 0.0014 | 0.0349 [1.00] | 0.0226 [0.99] | 0.0142 [0.99] | 0.0103 [1.00] | 0.0630 [0.93] | 0.0513 [0.96] | 0.0750 | 0.0048 | 0.0427 | 0.1916 |
| forward|tempered|0.25/0.35 | 0.0395 | 0.0011 | 0.0468 [0.05] | 0.0506 [0.03] | 0.0224 [0.93] | 0.0102 [1.00] | 0.0481 [0.03] | 0.0120 [1.00] | 0.0386 | 0.0017 | 0.0585 | 0.0764 |
| forward|tempered|0.1/0.5 | 0.0399 | 0.0002 | 0.0443 [0.17] | 0.0471 [0.12] | 0.0200 [0.94] | 0.0084 [1.00] | 0.0462 [0.10] | 0.0117 [1.00] | 0.0394 | 0.0003 | 0.0587 | 0.0780 |
| forward|tempered|0.05/1.0 | 0.0420 | 0.0000 | 0.0386 [0.82] | 0.0391 [0.73] | 0.0185 [0.98] | 0.0083 [1.00] | 0.0425 [0.52] | 0.0131 [0.99] | 0.0410 | 0.0001 | 0.0599 | 0.0870 |
| mixed|gauss|0.25/0.35 | 0.0046 | 0.0042 | 0.0048 [0.51] | 0.0054 [0.33] | 0.0084 [0.23] | 0.0035 [0.77] | 0.0044 [0.57] | 0.0046 [0.51] | 0.0061 | 0.0051 | 0.0075 | 0.0165 |
| mixed|gauss|0.1/0.5 | 0.0063 | 0.0012 | 0.0035 [0.95] | 0.0032 [0.95] | 0.0073 [0.57] | 0.0022 [0.97] | 0.0034 [0.95] | 0.0034 [0.99] | 0.0080 | 0.0024 | 0.0050 | 0.0236 |
| mixed|gauss|0.05/1.0 | 0.0256 | 0.0004 | 0.0115 [0.99] | 0.0089 [0.99] | 0.0105 [0.94] | 0.0042 [1.00] | 0.0087 [0.99] | 0.0106 [0.98] | 0.0333 | 0.0038 | 0.0168 | 0.0762 |
| mixed|t3|0.25/0.35 | 0.0138 | 0.0127 | 0.0096 [0.78] | 0.0090 [0.79] | 0.0113 [0.56] | 0.0057 [0.95] | 0.0079 [0.93] | 0.0084 [0.92] | 0.0163 | 0.0140 | 0.0163 | 0.0353 |
| mixed|t3|0.1/0.5 | 0.0189 | 0.0045 | 0.0081 [0.97] | 0.0065 [0.98] | 0.0087 [0.81] | 0.0040 [0.98] | 0.0067 [0.97] | 0.0071 [0.98] | 0.0239 | 0.0088 | 0.0132 | 0.0443 |
| mixed|t3|0.05/1.0 | 0.0735 | 0.0011 | 0.0274 [0.99] | 0.0185 [1.00] | 0.0138 [0.97] | 0.0082 [1.00] | 0.0420 [0.96] | 0.3728 [0.99] | 0.0902 | 0.0208 | 0.0468 | 0.1182 |
| mixed|tempered|0.25/0.35 | 0.0257 | 0.0010 | 0.0287 [0.41] | 0.0306 [0.38] | 0.0168 [0.83] | 0.0080 [0.99] | 0.0261 [0.60] | 0.0119 [0.99] | 0.0264 | 0.0022 | 0.0381 | 0.0505 |
| mixed|tempered|0.1/0.5 | 0.0255 | 0.0002 | 0.0271 [0.48] | 0.0282 [0.47] | 0.0172 [0.81] | 0.0074 [1.00] | 0.0233 [0.67] | 0.0127 [0.99] | 0.0258 | 0.0014 | 0.0346 | 0.0464 |
| mixed|tempered|0.05/1.0 | 0.0276 | 0.0006 | 0.0240 [0.81] | 0.0237 [0.73] | 0.0164 [0.89] | 0.0072 [1.00] | 0.0222 [0.80] | 0.0121 [1.00] | 0.0291 | 0.0019 | 0.0369 | 0.0589 |
| fullstiff|gauss|0.25/0.35 | 0.0056 | 0.0046 | 0.0056 [0.54] | 0.0063 [0.34] | 0.0095 [0.19] | 0.0037 [0.84] | 0.0048 [0.72] | 0.0044 [0.78] | 0.0064 | 0.0060 | 0.0078 | 0.0109 |
| fullstiff|gauss|0.1/0.5 | 0.0104 | 0.0021 | 0.0065 [0.96] | 0.0071 [0.91] | 0.0123 [0.47] | 0.0035 [0.96] | 0.0061 [0.99] | 0.0035 [1.00] | 0.0118 | 0.0103 | 0.0130 | 0.0020 |
| fullstiff|gauss|0.05/1.0 | 0.0443 | 0.0005 | 0.0269 [0.98] | 0.0283 [0.94] | 0.0195 [0.90] | 0.0046 [1.00] | 0.0211 [0.99] | 0.0046 [1.00] | 0.0459 | 0.0405 | 0.0542 | 0.0005 |
| fullstiff|t3|0.25/0.35 | 0.0157 | 0.0134 | 0.0107 [0.82] | 0.0103 [0.81] | 0.0124 [0.62] | 0.0060 [0.98] | 0.0088 [0.90] | 0.0088 [0.84] | 0.0180 | 0.0169 | 0.0172 | 0.0283 |
| fullstiff|t3|0.1/0.5 | 0.0339 | 0.0070 | 0.0156 [0.98] | 0.0141 [0.97] | 0.0136 [0.84] | 0.0047 [1.00] | 0.0130 [0.98] | 0.0072 [0.99] | 0.0358 | 0.0319 | 0.0285 | 0.0044 |
| fullstiff|t3|0.05/1.0 | 0.1218 | 0.0016 | 0.0588 [0.98] | 0.0478 [0.98] | 0.0205 [0.98] | 0.0057 [1.00] | 0.0389 [0.99] | 0.0079 [1.00] | 0.1146 | 0.1085 | 0.1063 | 0.0014 |
| fullstiff|tempered|0.25/0.35 | 0.0562 | 0.0027 | 0.0755 [0.00] | 0.0747 [0.00] | 0.0484 [0.64] | 0.0187 [0.99] | 0.0482 [1.00] | 0.0032 [1.00] | 0.0377 | 0.0183 | 0.0410 | 0.0010 |
| fullstiff|tempered|0.1/0.5 | 0.0561 | 0.0004 | 0.0748 [0.00] | 0.0749 [0.00] | 0.0508 [0.55] | 0.0153 [0.99] | 0.0473 [0.98] | 0.0026 [1.00] | 0.0369 | 0.0180 | 0.0390 | 0.0002 |
| fullstiff|tempered|0.05/1.0 | 0.0607 | 0.0001 | 0.0776 [0.01] | 0.0792 [0.02] | 0.0423 [0.79] | 0.0128 [1.00] | 0.0464 [1.00] | 0.0026 [1.00] | 0.0415 | 0.0238 | 0.0425 | 0.0000 |
| graded|gauss|0.25/0.35 | 0.0043 | 0.0036 | 0.0041 [0.57] | 0.0046 [0.40] | 0.0087 [0.23] | 0.0029 [0.81] | 0.0037 [0.71] | 0.0036 [0.74] | 0.0062 | 0.0056 | 0.0069 | 0.0116 |
| graded|gauss|0.1/0.5 | 0.0039 | 0.0011 | 0.0021 [0.98] | 0.0024 [0.96] | 0.0052 [0.39] | 0.0015 [0.93] | 0.0023 [0.93] | 0.0015 [0.99] | 0.0080 | 0.0064 | 0.0078 | 0.0018 |
| graded|gauss|0.05/1.0 | 0.0104 | 0.0004 | 0.0032 [1.00] | 0.0043 [1.00] | 0.0034 [0.90] | 0.0010 [1.00] | 0.0028 [0.99] | 0.0014 [1.00] | 0.0322 | 0.0245 | 0.0201 | 0.0005 |
| graded|t3|0.25/0.35 | 0.0143 | 0.0119 | 0.0091 [0.82] | 0.0086 [0.78] | 0.0099 [0.64] | 0.0051 [0.95] | 0.0075 [0.90] | 0.0119 [0.90] | 0.0204 | 0.0196 | 0.0173 | 0.0253 |
| graded|t3|0.1/0.5 | 0.0138 | 0.0036 | 0.0051 [1.00] | 0.0050 [0.99] | 0.0062 [0.78] | 0.0022 [0.98] | 0.0041 [0.99] | 0.0029 [1.00] | 0.0339 | 0.0506 | 0.0173 | 0.0061 |
| graded|t3|0.05/1.0 | 0.0256 | 0.0010 | 0.0073 [1.00] | 0.0112 [0.99] | 0.0052 [0.93] | 0.0014 [1.00] | 0.0041 [0.99] | 0.0023 [1.00] | 0.0775 | 0.0644 | 0.0373 | 0.0012 |
| graded|tempered|0.25/0.35 | 0.0129 | 0.0022 | 0.0127 [0.47] | 0.0131 [0.37] | 0.0136 [0.43] | 0.0070 [0.93] | 0.0101 [0.99] | 0.0026 [1.00] | 0.0125 | 0.0064 | 0.0117 | 0.0010 |
| graded|tempered|0.1/0.5 | 0.0072 | 0.0003 | 0.0056 [0.95] | 0.0057 [0.91] | 0.0088 [0.45] | 0.0032 [0.94] | 0.0050 [0.98] | 0.0011 [1.00] | 0.0088 | 0.0041 | 0.0072 | 0.0002 |
| graded|tempered|0.05/1.0 | 0.0050 | 0.0001 | 0.0025 [1.00] | 0.0026 [1.00] | 0.0059 [0.55] | 0.0014 [0.98] | 0.0026 [0.97] | 0.0005 [1.00] | 0.0092 | 0.0055 | 0.0058 | 0.0000 |


## Corrections after independent adversarial review (2026-10-25; scripts /tmp/atom-adv/)
Hold: optimiser and objective correct (independent implementation to 2.6e-15; Newton polish changes KL <=1.1e-5; F convex); consistent q -> exact recovery (max KL 1.4e-8 over 400 fits); headline arithmetic reproduces (geometric mean over cells of ratio of cell means); seeds disjoint; learned-C features use no truth. The atom-level fix for full-conditionals-stiff and size-graded worlds holds in every variant (oracle 0.03-0.13x of uniform atoms; ordering-level GLS 0.56-1.27x).
Wrong or overstated:
- Learned C "never worse than 1.32x" was seed luck: on fresh seeds worst cell 25.4x, overall 0.649. Cause: the unclamped ridge predictor extrapolates log-variance far outside its training range (-18.7..35.9 vs -6.9..6.1), concentrating all weight on a few contexts. Clamping to the training range removes the catastrophes.
- "Rare catastrophic worlds for every method" is false: the oracle and 1000-sample data arms have none; catastrophes come from the estimated-weight arms (learned, per-pattern learned, IRLS).
- The per-pattern learned C (0.346x) is given the test world's generator label: not "no truth at test".
- "Even the oracle hurts in equal-tempered" comes from the oracle reading the true logit per context; a generator-level oracle ties uniform.
- The 1000-sample data arm only ties the plain smoothed empirical joint from the same samples (0.262 vs 0.260; empirical better at n=3). Combining data with the model was not tested. It also loses one cell at Dirichlet 3.0 (weight concentration again).
- Shared bias across a variable's atoms hurts the self-estimated and learned arms; the oracle barely moves.
Survives: atom level >> ordering level with correct precision, robust to seeds, concentration, correlated noise and n=4 with 1000 worlds/cell. Estimated weights need bounds (clamp, minimum weight, shrinkage) before any claim.
