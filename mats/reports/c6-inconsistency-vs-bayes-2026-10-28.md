# C6: is inconsistency minimisation enough for Bayes probabilities? (2026-10-28, revised after review)

Analytic note for crux C6 in world-critique/DESIGN-v1.md.
- Reviewed adversarially by GPT-6 (Codex): work/c6/review-codex.md. This version corrects every point the review raised
  (see the last section).
- Code: work/c6/c6_counterexamples.py (numpy + scipy, one process, a few seconds). Every number below is in
  work/c6/output.txt.

**Answer.** These examples establish three things:
1. Choosing one best configuration or hypothesis (a point estimate) can badly misrepresent predictive probabilities,
   even when its energy is the exact −log joint. See (a) and (c1).
2. Fitting residuals cannot recover information that only the prior carries. See (b).
3. A correctly specified variational free energy recovers Bayes exactly.

They do **not** refute the program's distribution-level objectives. The atom energy optimises over distributions μ and
receives every conditioning subset, including the unconditional marginals. Given correctly specified inputs, it
recovers the true joint; see (c2).

## (a) Point estimate vs volume (Gemini), W1

Setup:
- Hidden R is one of 120 pairs of the 16 coordinates, uniform prior. Cells are g ~ N(0,1), items v ~ N(g, 0.3^2),
  and each observation is y ~ N(v, 0.3^2).
- Six items, one observation each:
  - N = (0,0,1^14), y = −1.2
  - P = (1,1,0^14), y = +1.2
  - M1 = (1,1,s) and M2 = (1,1,1−s), with s = 1010..., y = +1.2 each
  - Q0 = (0,1,0^14) and Q1 = (1,0,0^14), y = +0.5 each
- Query: a new item x_q = 0^16. We report P(y_q > 0 | D).

Under R1 = {0,1}, P, M1 and M2 share one cell. That is the sharpest fit, and the query cell then holds N (−1.2).
Under every other R, the query cell holds positive items.

| quantity | value |
|---|---|
| argmin over R of E_marg(R) = −log p(D,R), with g and v integrated out | R1 |
| argmin of the pure point energy E_joint(R,g,v) = −log p(D,R,g,v) | R1 (same) |
| gap from R1 to the next-best hypothesis | 1.17 nats |
| hypotheses predicting y_q > 0 | the other 119 (median +1.49 nats above R1; 77 within 2 nats) |
| posterior mass: P(R1\|D) vs the opposite group | 0.048 vs 0.952 (ratio 19.8) |
| P(y_q>0), best single hypothesis R1 (exact within R1) / joint-MAP plug-in / Bayes | 0.039 / 0.008 / **0.816** |
| E[y_q], R1 vs Bayes | −1.02 vs +0.50 |
| KL(Bayes predictive \|\| R1 predictive) | 3.50 nats (numerical quadrature) |
| weights ∝ exp(−E_marg/T), P(y_q>0) at T = 0.25 / 0.5 / 1 / 2 | 0.26 / 0.71 / 0.82 / 0.84 |

Choosing one hypothesis fails even after integrating out the continuous latents inside it. That is E_marg, which
equals E_joint at its conditional minimum minus ½ log det(2πΣ_post). The distribution-level fix is exact: the minimum
over w of [E_w E_marg − H(w)] is 7.881728, equal to −log p(D), and is reached at w = posterior. This example refutes
choosing a single hypothesis. It does not refute minimising inconsistency over distributions.

## (b) Nullspace (Kimi), W2

Setup:
- 3 items v ~ N(0,1) and 2 sources with bias b ~ N(0, 0.25^2).
- Each source quotes each item once (SPOT quote = v + b + N(0, 0.3^2)), and all 6 quotes read 1.0.
- Query: v_1.

The residual is identical along a whole line of solutions:
- Rank 4 for 5 unknowns, so the nullity is 1. The null direction is (1,1,1,−1,−1), the shift gauge.
- The residual is 0 along the whole line, with v_1 unbounded.
- v_1 − v_2 is identical everywhere on the line, so residual fitting determines it.
- v_1 is not gauge-invariant, so this is not a mere reparameterisation of the query.

Two residual-optimal points:
- The min-norm point (pseudo-inverse, or gradient descent from 0) gives v_1 = 0.400.
- The prior-chosen point on the line gives v_1 = 0.914. This is the infinite-data limit of the posterior mean; the
  finite-data mean is 0.878 at m = 1.
- Their prior log density ratio is 4.63 nats, a density ratio of about 102. That is a ratio of point densities, not
  the odds of an event. Kimi's e^2 threshold corresponds to 2 nats.

Bayes:

| setting | v_1 (mean ± sd) | P(v_1 > 0.5) |
|---|---|---|
| m = 1 quote per item-source pair | 0.878 ± 0.263 | 0.925 |
| m = 10 | 0.911 ± 0.181 | 0.988 |
| m = 1000 | 0.914 ± 0.169 | 0.993 |
| m = 1000, prior τ_b = 1 | 0.400 ± 0.447 | |
| m = 1000, prior τ_b = 4 | 0.040 ± 0.566 | |

- The posterior sd shrinks, but only to a floor of 1/√35 = 0.169 that the prior sets, never to 0.
- Changing the prior moves v_1 from 0.91 to 0.04 with the same set of residual optima.
- Control: adding m quotes from a known-unbiased source removes the nullspace, and the sd falls to 0.009 at m = 1000.
- Two conclusions survive:
  - Along the null direction, only the prior supplies location and spread.
  - Probabilities need the curvature, not only the minimiser. Turning a point estimate into a probability, such as
    0 or 1 for v_1 > 0.5, is an extra plug-in convention.

## (c1) Codex: two global optima, unique Bayes (W3-style discrete relations; a star graph, not the C2 chain benchmark)

Setup:
- A root bit z with a report "z = 1" of likelihood ratio 81.
- 4 leaves c_j. Each has a report "c_j = 0" (error 0.20) and a relation "c_j = z" (error 0.25).
- Objective: E = −log p(z, c, reports), up to a state-independent constant.

Results:
- The global minimisers are exactly (z=0, c=0000) and (z=1, c=0000). Their energies are equal and their query
  predictions differ.
- Bayes gives a unique P(z=1|D) = 0.872.
- With the likelihood ratio lowered by 5% there is a single minimiser, z = 0, while Bayes gives 0.866. So the failure
  holds on an open set of parameters.
- The unweighted violation count also gives z = 0.
- As in (a), this refutes choosing a single configuration of a correctly weighted energy, not the distribution-level
  version.

## (c2, revised) The atom energy with correctly specified inputs

The original c2 was wrong and is withdrawn (see the corrections section).

Revised check: F(μ) = Σ w·μ(v_S)·KL(μ(X_i|v_S) || q_a) − γH(μ) on 2 bits, minimised exactly. The world is
z1 ~ Bern(0.8) and z2 = z1 xor Bern(0.001).

| inputs | P(z1=1) at γ = 0 / 1e-4 | KL(p \|\| μ) at γ = 0 / 1e-4 |
|---|---|---|
| true full conditionals | 0.8000 / 0.7929 | 2e-13 / 1.5e-4 |
| true conditionals + true marginals (the atom set the real estimator uses) | 0.8000 / 0.8000 | 9e-15 / 5e-10 |
| symmetric copy conditionals (the old c2 input) | 0.5000 / 0.5000 | 0.19 / 0.19 |
| symmetric copy conditionals + true marginals | 0.7995 / 0.7994 | 2e-4 / 2e-4 |

Reading:
- With correctly specified, positive inputs, the atom energy returns the true joint. The marginal atoms make it robust
  to near-deterministic conditionals.
- With γ = 1e-4 and only near-deterministic full conditionals, the entropy term shifts the answer slightly (0.793).
- The old 0.5 came from inputs that themselves encode 0.5, not from the energy.
- This is consistent with the 10-25 atom-energy note. Whether the atom energy is Bayes-consistent is a question about
  its inputs: whether the conditionals and marginals are calibrated, and how inconsistent inputs are weighted. Nothing
  here shows that the objective itself fails.

## What a Bayes-consistent inconsistency energy needs (sufficient conditions, not necessity)

Gibbs variational identity: for any distribution μ over latents h,

  E_μ[−log p(D|h) − log p0(h)] − H(μ) = KL(μ || p(h|D)) − log p(D).

Hence the following is **sufficient** for exact Bayes predictions. Optimise over an unrestricted family of
distributions μ an energy that is the calibrated negative log-likelihood plus the negative log prior, in fixed log-
probability units with unit entropy weight. Then answer queries by E_μ[p(query|h)].

The examples show what can go wrong when parts are removed:

| part removed | example | result |
|---|---|---|
| optimising over distributions (pick one configuration instead) | (a), (c1) | wrong, even with a correct energy |
| the prior | (b) | a nullspace, with location and spread undetermined |
| calibrated weights | (c1), unweighted count | wrong |

They do not show that each part is necessary in every world. Other exact routes exist; the atom energy given
consistent conditionals and marginals in (c2) is one.

Point vs Bayes in general: a point prediction differs from the Bayes predictive whenever the posterior is not a point
mass, even for an identifiable, unimodal Gaussian posterior (it misses the predictive variance). Non-identifiability
(b) and unequal-volume multimodality (a, c1) make the gap large, up to a sign flip of the predicted mean in (a).

How this bears on the earlier GLS and atom work:
- The ordering-level GLS energy normalises exp(−E) over configurations (gls-energy/atom/atom_energy.py, normalise). It
  is distribution-level, so (a) and (c1) do not apply to it, and its input conditionals can already encode prior
  information.
- The atom energy's KL terms contain conditional entropies. γ is not a Bayesian temperature, and the error-precision
  weights for noisy conditional logits are not, by themselves, calibrated likelihoods of evidence.

Caveats:
- The data sets are constructed, not sampled. They prove existence, not how often this happens in W1/W2.
- In (a), the observation sizes were picked by a scan (`python3 c6_counterexamples.py scan`, plus variants tried in the
  session).

## Corrections after review (GPT-6 Codex, work/c6/review-codex.md)

1. **Old c2 "continuum of optima": false as implemented.** With η = 1e-9 the conditionals are positive, and the unique
   optimum is ((1−η)/2, η/2, η/2, (1−η)/2), with marginal 0.5. A continuum exists only at η = 0, and the 0.01 grid
   missed the off-diagonal mass. Withdrawn.
2. **Old c2 stripped the unconditional marginals the real estimator receives**, and its symmetric copy conditionals
   encode 0.5 rather than the stated 0.8 world. It was an example of misspecified inputs, not a failure of the atom
   energy. It is replaced by the revised c2, where correctly specified inputs give 0.8000.
3. **The GLS link was wrong.** The GLS energy normalises over configurations, so "likelihood only, no marginalisation"
   was unsupported. The claims that γ is a temperature and that F "reconciles, does not infer" are also removed.
4. **W1:** the KL value is numerical quadrature, and the log-det identity holds for E_joint at its conditional minimum.
   Both now stated. The scope is limited to choosing a single hypothesis.
5. **W2:**
   - The sd does shrink, from 0.263 to a floor of 0.169; the earlier "does not shrink" was wrong.
   - 0.914 is the prior-selected optimum, the m → ∞ limit, not the m = 1 mean of 0.878.
   - 4.63 nats is a log density ratio (about 102×).
   - The 0/1 point probability is a plug-in convention.
6. **W3:** the example is a star graph, not the C2 chain benchmark; relabelled. Its energies are −log joint only up to
   a constant.
7. **The necessity claim is withdrawn.** The final section now states sufficient conditions (unrestricted μ, fixed
   log-probability units). It no longer claims point and Bayes agree except under non-identifiability or
   multimodality.
