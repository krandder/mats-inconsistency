"""Exactness of the Bayes ceilings, oracle clearing, splits and determinism."""
import numpy as np
import pytest

from aggbench import make_env, episode_seeds, verify_disjoint
from aggbench.baselines import oracle_clearing
from aggbench.envs import (DiscreteEnv, gaussian_global, gaussian_local, report_posterior_logit,
                           union_identified)
from aggbench.splits import split_of
from aggbench.verify import brute_force_bernoulli, brute_force_gaussian

DISCRETE = ["E2", "E3-sybil", "E3-biased", "E3-collude"]


# ---------------------------------------------------------------- Bayes ceilings vs brute force

@pytest.mark.parametrize("name", DISCRETE)
def test_discrete_bayes_equals_enumeration(name):
    batch = make_env(name).sample("smoke", 40)
    assert brute_force_bernoulli(batch) < 1e-12


def test_discrete_duplicates_counted_once():
    batch = make_env("E2").sample("smoke", 200)
    l0 = batch.public["prior_logit"]
    naive = l0 + (batch.local["logit"] - l0).sum(1)  # counts shared pieces several times
    lam = 2 * batch.oracle["mu"] * batch.oracle["x"]
    union = batch.oracle["cov"].any(1)
    assert np.allclose(batch.bayes["logit"], l0 + (lam * union).sum(1))
    overlap = batch.oracle["cov"].sum(1).max(1) > 1
    assert overlap.any() and not np.allclose(naive[overlap], batch.bayes["logit"][overlap])


def test_gaussian_bayes_equals_joint_conditioning():
    batch = make_env("E1").sample("smoke", 40)
    assert brute_force_gaussian(batch) < 1e-9


def _linear_map(fn, A, Bf, sig, shape):
    """Matrix of the linear map s -> posterior mean computed by fn, and the posterior precision."""
    cols = []
    for j in range(int(np.prod(shape))):
        e = np.zeros(int(np.prod(shape)))
        e[j] = 1
        mean, prec = fn(A, Bf, sig, e.reshape(shape))
        cols.append(mean)
    return np.stack(cols, -1), prec


def test_gaussian_posteriors_match_monte_carlo():
    """Sample (theta, s) from one fixed structure; regress theta on s; compare to the analytic posterior."""
    batch = make_env("E1").sample("smoke", 1)
    A, Bf, sig = batch.oracle["A"][0], batch.oracle["Bf"][0], batch.oracle["sig"][0]
    N, m, d = A.shape
    rng = np.random.default_rng(123)
    n = 200_000
    theta = rng.standard_normal((n, d))
    eta = rng.standard_normal((n, Bf.shape[-1]))
    s = (np.einsum("imd,nd->nim", A, theta) + np.einsum("imr,nr->nim", Bf, eta)
         + sig[None, :, None] * rng.standard_normal((n, N, m)))
    # global posterior
    G, P = _linear_map(gaussian_global, A, Bf, sig, (N, m))
    S = s.reshape(n, -1)
    coef = np.linalg.lstsq(S, theta, rcond=None)[0].T
    resid = theta - S @ coef.T
    assert np.abs(coef - G).max() < 0.02
    assert np.abs(np.cov(resid.T) - np.linalg.inv(P)).max() < 0.02
    # agent 0's local posterior
    G0, P0 = _linear_map(lambda *a: tuple(x[0] for x in gaussian_local(*a)), A, Bf, sig, (N, m))
    coef0 = np.linalg.lstsq(s[:, 0], theta, rcond=None)[0].T
    assert np.abs(coef0 - G0[:, :m]).max() < 0.02
    assert np.abs(np.cov((theta - s[:, 0] @ coef0.T).T) - np.linalg.inv(P0)).max() < 0.02


def _logistic_fit(X, y, iters=25):
    X1 = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X1 @ w))
        H = X1.T @ (X1 * (p * (1 - p))[:, None])
        w += np.linalg.solve(H, X1.T @ (y - p))
    return w


def test_reports_ceiling_matches_monte_carlo():
    """With the union NOT identified, oracle clearing is the exact posterior given the reports:
    the logistic regression of Y on the reports must recover its weights."""
    cov = np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 1]], dtype=bool)
    mu = np.array([0.3, 0.6, 0.5, 0.2, 0.4])
    assert not union_identified(cov)
    one, zero = np.ones(3), np.zeros(3)
    w_oracle = np.array([report_posterior_logit(e, cov, mu, one, zero, 0.0) for e in np.eye(3)])
    rng = np.random.default_rng(7)
    n = 400_000
    y = rng.random(n) < 0.5
    x = np.where(y[:, None], mu, -mu) + rng.standard_normal((n, 5))
    reports = (2 * mu * x) @ cov.T.astype(float)
    w_fit = _logistic_fit(reports, y.astype(float))
    assert np.abs(w_fit[:3] - w_oracle).max() < 0.02 and abs(w_fit[3]) < 0.02
    assert not np.allclose(w_oracle, 1.0)  # differs from the global posterior's weights


# ---------------------------------------------------------------- oracle clearing == Bayes

def test_oracle_clearing_equals_bayes_e1():
    batch = make_env("E1").sample("dev", 60)
    f = oracle_clearing(batch)
    assert np.abs(f["mean"] - batch.bayes["mean"]).max() < 1e-8
    assert np.abs(f["prec"] - batch.bayes["prec"]).max() < 1e-6


@pytest.mark.parametrize("name", DISCRETE)
def test_oracle_clearing_equals_bayes_when_identified(name):
    batch = make_env(name, n_agents=6, n_pieces=4, pieces_per_agent=(1, 3)).sample("dev", 300)
    ident = np.array([union_identified(c) for c in batch.oracle["cov"]])
    assert ident.sum() >= 20
    f = oracle_clearing(batch)
    assert np.abs(f["logit"][ident] - batch.bayes["logit"][ident]).max() < 1e-8


def test_oracle_is_clearing_limit():
    """Closed form = Lam -> 0 limit of (C^T C + eps M^-1) w = C^T r, summed."""
    batch = make_env("E2").sample("dev", 5)
    for e in range(5):
        cov, mu = batch.oracle["cov"][e].astype(float), batch.oracle["mu"][e]
        r = batch.reports["logit"][e]
        w = np.linalg.solve(cov.T @ cov + 1e-9 * np.diag(1 / (2 * mu ** 2)), cov.T @ r)
        closed = report_posterior_logit(r, cov > 0, mu, np.ones(len(r)), np.zeros(len(r)), 0.0)
        assert abs(w.sum() - closed) < 1e-5


def test_adversary_roles_present():
    for name, role, count in [("E3-sybil", 3, 3), ("E3-biased", 2, 1), ("E3-collude", 1, 3)]:
        batch = make_env(name).sample("dev", 10)
        assert ((batch.oracle["role"] == role).sum(1) == count).all()
    b = make_env("E3-collude").sample("dev", 10)
    for e in range(10):
        rows = b.oracle["cov"][e][b.oracle["role"][e] == 1]
        assert (rows == rows[0]).all()


# ---------------------------------------------------------------- splits and seeds

def test_splits_disjoint_and_guarded():
    seeds = {s: episode_seeds(s, 5000) for s in ("train", "dev", "test", "smoke")}
    assert verify_disjoint(*seeds.values())
    assert all(split_of(int(v[0])) == k and split_of(int(v[-1])) == k for k, v in seeds.items())
    assert not verify_disjoint(episode_seeds("train", 10), episode_seeds("train", 10, start=5))
    with pytest.raises(ValueError):
        episode_seeds("train", 10, start=10_000_000 - 5)
    with pytest.raises(PermissionError):
        make_env("E2").sample("test", 3)
    assert len(make_env("E2").sample("test", 3, allow_test=True)) == 3


@pytest.mark.parametrize("name", ["E1"] + DISCRETE)
def test_deterministic(name):
    a, b = make_env(name).sample("train", 20), make_env(name).sample("train", 20)
    for field in ("truth", "obs"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    for k in a.bayes:
        np.testing.assert_array_equal(a.bayes[k], b.bayes[k])
    for v in ("noisy", "full"):
        np.testing.assert_array_equal(a.keys[v], b.keys[v])
    c = make_env(name).sample("dev", 20)
    assert not np.allclose(a.obs[np.isfinite(a.obs)][:20], c.obs[np.isfinite(c.obs)][:20])


def test_views_hide_oracle():
    batch = make_env("E2").sample("dev", 4)
    for vis, dim in [("none", 0), ("noisy", 24), ("full", 24)]:
        view = batch.view(vis)
        assert view.keys.shape == (4, 5, dim)
        assert not hasattr(view, "oracle") and not hasattr(view, "truth")


def test_config_validation():
    with pytest.raises(ValueError):
        make_env("E1", obs_dim=4, coverage=3)
    with pytest.raises(ValueError):
        DiscreteEnv(n_agents=3, n_colluders=3, n_sybil=1)
