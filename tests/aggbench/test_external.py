"""Closed-form literature methods (aggbench.external.closed_form): numpy only."""
import numpy as np

from aggbench import make_env
from aggbench.baselines import oracle_clearing
from aggbench.envs import View
from aggbench.external import SOURCES
from aggbench.external import closed_form as cf


def test_pif_full_keys_equals_reports_ceiling_e2_e3():
    for name in ("E2", "E3-sybil", "E3-collude"):
        b = make_env(name).sample("dev", 200)
        got = cf.pif_oracle(b.view("full"))["logit"]
        ref = oracle_clearing(b)["logit"]
        np.testing.assert_allclose(got, ref, atol=1e-8)  # colluders' keys show the pooled coverage they report on
    b = make_env("E3-biased").sample("dev", 200)  # the keys do not reveal the distortion: PIF differs there
    assert np.abs(cf.pif_oracle(b.view("full"))["logit"] - oracle_clearing(b)["logit"]).max() > 0.1


def test_pif_full_keys_equals_bayes_e1_aligned_and_unaligned():
    b = make_env("E1").sample("dev", 100)
    for keys in (b.view("full").keys, b.agent_inputs("full", alignment="unaligned")[1]):
        f = cf.pif_oracle(cf.with_keys(b.view("full"), keys))
        np.testing.assert_allclose(f["mean"], b.bayes["mean"], atol=1e-8)
        np.testing.assert_allclose(f["prec"], b.bayes["prec"], rtol=1e-6, atol=1e-6)


def test_pif_compound_is_scaled_sum():
    rng = np.random.default_rng(0)
    n, N = 4000, 5
    X = rng.normal(size=(n, N))
    v = View("bernoulli", {"logit": X}, np.zeros((n, N, 0)), {"prior_logit": 0.0, "n_pieces": 1})
    fit = cf.fit_pif_compound(v)
    got = cf.pif_oracle(v, fit)["logit"]
    d, r = fit["delta"], fit["rho"]
    np.testing.assert_allclose(got, d / (d + (N - 1) * r) * X.sum(1), rtol=1e-8)


def test_ici_gains_sum_to_identity_and_self_fusion():
    rng = np.random.default_rng(1)
    n, d = 50, 3
    M = rng.normal(size=(n, d, d))
    Ca = M @ np.swapaxes(M, 1, 2) + np.eye(d)
    M = rng.normal(size=(n, d, d))
    Cb = M @ np.swapaxes(M, 1, 2) + np.eye(d)
    x = rng.normal(size=(n, d))
    # an estimate fused with itself is unchanged (ICI removes the full common information)
    xf, Cf = cf._ici_pair(x, Ca, x, Ca, np.linspace(0, 1, 11))
    np.testing.assert_allclose(xf, x, atol=1e-8)
    np.testing.assert_allclose(Cf, Ca, atol=1e-8)
    # unbiasedness K + L = I: fusing two equal means returns that mean for any covariances
    xf, _ = cf._ici_pair(x, Ca, x, Cb, np.linspace(0, 1, 11))
    np.testing.assert_allclose(xf, x, atol=1e-8)


def test_ci_weights_on_simplex_and_single_agent():
    b = make_env("E1", n_agents=1).sample("dev", 20)
    f = cf.covariance_intersection(b.view("none"))
    np.testing.assert_allclose(f["mean"], b.reports["mean"][:, 0], atol=1e-10)
    b = make_env("E1").sample("dev", 20)
    w = cf.covariance_intersection(b.view("none"))["omega"]
    assert (w >= 0).all() and np.allclose(w.sum(1), 1)


def test_winkler_equal_independent_errors():
    rng = np.random.default_rng(2)
    n, N, d = 20000, 4, 2
    theta = rng.normal(size=(n, d))
    m = theta[:, None, :] + rng.normal(size=(n, N, d))
    v = View("gaussian", {"mean": m, "prec": np.broadcast_to(np.eye(d), (n, N, d, d))}, np.zeros((n, N, 0)),
             {"d": d})
    S = cf.fit_winkler(v, theta)
    f = cf.winkler_gls(v, S)
    np.testing.assert_allclose(f["mean"], m.mean(1), atol=0.02)
    np.testing.assert_allclose(f["prec"][0], np.eye(d) * N, rtol=0.05)


def test_surprisingly_popular_prelec_example():
    # "Is Philadelphia the capital of Pennsylvania?": 65% say yes, but they predict 75% will say yes -> "no" wins.
    votes = np.array([[1] * 65 + [0] * 35])
    pred = np.array([[0.75] * 100])
    assert cf.surprisingly_popular(votes, pred)[0] == 0


def test_sources_record_provenance():
    for name, s in SOURCES.items():
        assert "kind" in s and ("paper" in s), name
        if "vendored" in s["kind"]:
            assert len(s["commit"]) == 40 and s["license"]
