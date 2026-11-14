"""cmlib: replace a layer's activations by a simple distribution q and measure what the downstream loses.

Modules: stats (ESS, AUC, tail diagnostics, bootstrap), ratio (calibrated classifier density ratios),
estimators (direct gap, cell RMS gap, Cov_q(G, w), tempered paths, regions), qfam (conditional /
linear Gaussian q, subset knock-out / knock-in), consistency (cross-layer push-forward checks),
dl (description-length bookkeeping), synth (known-answer worlds used by tests and VALIDATION.md).
"""
from . import stats, ratio, estimators, qfam, consistency, dl  # noqa: F401
