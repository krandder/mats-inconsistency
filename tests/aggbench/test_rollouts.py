"""E4 (rollouts of an imperfect model as sources): exactness of both ceilings, by brute force and Monte Carlo."""
import numpy as np
import pytest

from aggbench import make_env
from aggbench.baselines import oracle_clearing
from aggbench.envs import (RolloutConfig, report_posterior_logit, rollout_report_posterior_logit,
                           rollout_trace_posterior_logit, rollout_weights)
from aggbench.metrics import paired_difference
from aggbench.verify import brute_force_rollouts, oracle_check

EXACT_MODEL = dict(calibration=1.0, recency=1.0, read_noise=0.0, shared_bias=0.0, step_noise=0.0)


def _logistic_fit(X, y, iters=25):
    X1 = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X1 @ w))
        H = X1.T @ (X1 * (p * (1 - p))[:, None])
        w += np.linalg.solve(H, X1.T @ (y - p))
    return w


@pytest.mark.parametrize("overrides", [{}, dict(n_rollouts=3, calibration=0.6, recency=1.3, prior=0.3),
                                       dict(n_rollouts=10, read_noise=1.0, shared_bias=0.0, step_noise=0.1),
                                       EXACT_MODEL, dict(EXACT_MODEL, n_rollouts=12)])
def test_ceilings_equal_brute_force(overrides):
    batch = make_env("E4", **overrides).sample("smoke", 40)
    assert brute_force_rollouts(batch) < 1e-12
    assert oracle_check(make_env("E4", **overrides))["verified"]


def test_brute_force_catches_a_wrong_ceiling():
    """Mutation: the ceiling computed with the wrong weights (no recency decay) must fail the check."""
    batch = make_env("E4").sample("smoke", 24)
    batch.oracle["W"] = (batch.oracle["W"] > 0) * batch.config["calibration"]
    assert brute_force_rollouts(batch) > 1e-3


def _fixed_structure_draws(env, n, seed=11):
    rng = np.random.default_rng(seed)
    mu, order = env.structure(rng)
    y, x, e, b, trace = env.simulate(mu, order, rng, n)
    return mu, order, y, trace


def test_reports_ceiling_is_the_posterior_monte_carlo():
    """For one fixed structure, logistic regression of Y on the simulated reports recovers the ceiling's weights."""
    env = make_env("E4")
    mu, order, y, trace = _fixed_structure_draws(env, 400_000)
    reports = np.nansum(trace, axis=2)
    c = RolloutConfig()
    W = rollout_weights(order, len(mu), c.calibration, c.recency)
    length = (order >= 0).sum(1)
    w_oracle = np.array([rollout_report_posterior_logit(r, W, length, mu, c, 0.0) for r in np.eye(len(W))])
    w_fit = _logistic_fit(reports, y.astype(float))
    assert np.abs(w_fit[:-1] - w_oracle).max() < 0.02 and abs(w_fit[-1]) < 0.02


def test_trace_ceiling_is_the_posterior_monte_carlo():
    """Same for the trace ceiling, on a small config (every trace entry is a regression feature)."""
    env = make_env("E4", n_rollouts=3, n_pieces=4, rollout_len=(1, 3))
    c = env.config
    mu, order, y, trace = _fixed_structure_draws(env, 400_000, seed=5)
    mask = np.isfinite(trace[0])
    w_oracle = np.zeros(trace.shape[1:])
    for idx in zip(*np.nonzero(mask)):   # the ceiling is linear in the trace: read off its weight on each entry
        t = np.where(mask, 0.0, np.nan)
        t[idx] = 1.0
        w_oracle[idx] = rollout_trace_posterior_logit(t, order, mu, c, 0.0)
    # every rollout starts from the same value logit0 + b: one regression feature, weight = sum over rollouts
    w_oracle[0, 0] = w_oracle[:, 0].sum()
    mask[1:, 0] = False
    X = trace[:, mask]
    w_oracle = w_oracle[mask]
    w_fit = _logistic_fit(X, y.astype(float))
    assert np.abs(w_fit[:-1] - np.array(w_oracle)).max() < 0.03 and abs(w_fit[-1]) < 0.02


def test_exact_model_reduces_to_overlapping_observers():
    """With an exact model the rollouts are E2-style observers: reports = local posteriors, the reports
    ceiling equals E2's formula, and the trace ceiling equals Bayes on the union of conditioning sets."""
    batch = make_env("E4", **EXACT_MODEL).sample("dev", 100)
    l0 = batch.public["prior_logit"]
    np.testing.assert_allclose(batch.reports["logit"], batch.local["logit"], atol=1e-12)
    f = oracle_clearing(batch)["logit"]
    N = batch.reports["logit"].shape[1]
    for e in range(len(batch)):
        cov, mu = batch.oracle["cov"][e], batch.oracle["mu"][e]
        e2 = report_posterior_logit(batch.reports["logit"][e], cov, mu, np.ones(N), np.zeros(N), l0)
        assert abs(f[e] - e2) < 1e-6
        lam = 2 * mu * batch.oracle["x"][e]
        union = l0 + lam[cov.any(0)].sum()
        tr = rollout_trace_posterior_logit(batch.obs[e], batch.oracle["order"][e], mu,
                                           RolloutConfig(**EXACT_MODEL), l0)
        assert abs(tr - union) < 1e-6


def test_order_matters_only_for_the_imperfect_model():
    """Two rollouts that read the same pieces in opposite orders agree under an exact model and disagree
    under recency decay: the imperfect model is not order-consistent."""
    order = np.array([[0, 1, 2], [2, 1, 0]])
    mu = np.array([0.3, 0.5, 0.2])
    for kw, same in [(dict(EXACT_MODEL, n_rollouts=2, n_pieces=3, rollout_len=(3, 3)), True),
                     (dict(n_rollouts=2, n_pieces=3, rollout_len=(3, 3), step_noise=0.0), False)]:
        env = make_env("E4", **kw)
        _, _, _, _, trace = env.simulate(mu, order, np.random.default_rng(0), 50)
        r = np.nansum(trace, axis=2)
        assert np.allclose(r[:, 0], r[:, 1]) == same


def test_ceiling_order():
    """Information order: Bayes on the whole context >= trace ceiling >= reports ceiling (mean regret)."""
    env = make_env("E4")
    dv = env.sample("dev", 2000)
    c = env.config
    rep = oracle_clearing(dv)
    tr = {"logit": np.array([rollout_trace_posterior_logit(dv.obs[e], dv.oracle["order"][e], dv.oracle["mu"][e], c,
                                                           dv.public["prior_logit"]) for e in range(len(dv))])}
    assert paired_difference(tr, rep, dv)["ci_high"] < 0
    assert paired_difference(dv.bayes, tr, dv)["ci_high"] < 0


def test_e4_deterministic_and_shapes():
    a, b = make_env("E4").sample("train", 10), make_env("E4").sample("train", 10)
    np.testing.assert_array_equal(a.reports["logit"], b.reports["logit"])
    np.testing.assert_array_equal(a.keys["noisy"], b.keys["noisy"])
    assert a.obs.shape == (10, 6, 7) and a.keys["full"].shape == (10, 6, 24)
    np.testing.assert_allclose(np.nansum(a.obs, axis=2), a.reports["logit"])
    assert (a.oracle["cov"].sum(2) == a.oracle["length"]).all()
    with pytest.raises(ValueError):
        make_env("E4", rollout_len=(3, 9))
