# Tiny-PFN mechanistic interpretability: plan (opus-mats, 2026-10-26 ~23:35Z)

Kelvin: "We need a mechanistic understanding of the errors from bayes." First understand how the network gets to
approximate Bayes on its training distribution, then its deviations. Starts when tiny-pfn models land
(work/tiny-pfn/, report outbox/tiny-pfn-2026-10-27.md). Target model: smallest config within 0.02 nats of Bayes,
all seeds (5; no keys arm in this world).

## Bayes intermediates (world, Kelvin 23:36Z/23:40Z: 3 independent sources, no duplicates, no shared errors;
## unknown per-source noise variance from {0.1, 0.5, 2.0}; 4 labelled + 1 query question)
1. Per source s: labelled residuals r_s = o_s - y (4 numbers), and their sum of squares.
2. Posterior over each source's variance (factorises over sources).
3. Precision weights; mixture predictive mean/variance for the query.

## Steps
1. Locate: for each intermediate, per layer and token type, retrained ridge probes (episode-level split, lambda by
   inner CV); baselines: shuffled targets, untrained network with the same seed.
2. Intervene: patch the representation of one intermediate from an episode that differs only in it (e.g. same query
   readings, different partition posterior); check the output shift against the Bayes-predicted shift.
3. Replace: substitute the hypothesised computation (a head's pattern by the GLS-weight pattern; a residual component
   by its probe reconstruction). Score = fraction of performance retained,
   (NLL_mean-ablated - NLL_replaced) / (NLL_mean-ablated - NLL_model). Unreplaced remainder = not yet understood.
4. Deviations: per-episode delta = model - Bayes; locate which step's approximation carries it; build the mechanistic
   estimator and compare with sampling on MSE vs compute (ARC criterion), over seeds and held-out prior variants;
   it must beat the flat "network = Bayes" line past where sampling crosses it.

## Guards
>=3 seeds; untrained control; source-permutation symmetry checks; correct random baselines; adversarial review
before reporting.
