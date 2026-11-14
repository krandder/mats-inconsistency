"""Structure zoo: exactness of every ceiling (brute force, simulator Monte Carlo, closed forms), interface,
held-out guard."""
import itertools

import numpy as np
import pytest

from aggbench import make_env
from aggbench.baselines import all_baselines, oracle_clearing
from aggbench.metrics import per_episode, sigmoid
from aggbench.verify import brute_force_tabular, oracle_check
from aggbench.zoo import FAMILIES, HELDOUT, ZOO, list_envs, posterior_logit, tabular_ceilings

ZNAMES = [n for n in ZOO if hasattr(ZOO[n](), "tables")]   # tabular worlds; Z8 is linear-Gaussian


def env_of(name, **kw):
    return make_env(name, allow_heldout=True, **kw)


# ---------------------------------------------------------------- ceilings vs brute force

@pytest.mark.parametrize("name", ZNAMES)
def test_brute_force_matches(name):
    env = env_of(name)
    batch = env.sample("smoke", 30)
    assert brute_force_tabular(env, batch) < 1e-12
    assert oracle_check(env)["verified"]


@pytest.mark.parametrize("name", ZNAMES)
def test_tables_match_simulator(name):
    """The closed-form tables are the simulator's true law: P(w), P(w, o_i) and P(w, o_i, o_j) (conditional
    independence given w) from 200k simulated draws of one fixed structure."""
    env = env_of(name)
    rng = np.random.default_rng(5)
    st = env.structure(rng)
    prior_w, target, lik, _ = env.tables(st)
    n = 200_000
    w, sym = env.simulate(st, rng, n)
    W, O = lik.shape[1], lik.shape[2]
    assert np.abs(np.bincount(w, minlength=W) / n - prior_w).max() < 5e-3
    for i in range(lik.shape[0]):
        emp = np.bincount(w * O + sym[:, i], minlength=W * O).reshape(W, O) / n
        assert np.abs(emp - prior_w[:, None] * lik[i]).max() < 5e-3, i
    emp = np.bincount((w * O + sym[:, 0]) * O + sym[:, 1], minlength=W * O * O).reshape(W, O, O) / n
    model = prior_w[:, None, None] * lik[0][:, :, None] * lik[1][:, None, :]
    assert np.abs(emp - model).max() < 5e-3
    assert np.allclose(lik.sum(-1), 1.0) and np.isclose(prior_w.sum(), 1.0)


def test_simulator_check_has_power():
    """A wrong table (Z1 with every shared parent replaced by a fair coin) fails the simulator check."""
    env = env_of("Z1")
    rng = np.random.default_rng(5)
    st = env.structure(rng)
    bad = dict(st, rho=np.full_like(st["rho"], 0.5))
    prior_w, _, lik, _ = env.tables(bad)
    w, sym = env.simulate(st, rng, 200_000)
    W, O = lik.shape[1], lik.shape[2]
    emp = np.bincount(w, minlength=W) / len(w)
    assert np.abs(emp - prior_w).max() > 2e-2


def _bayes_mc(env, st, n, seed):
    prior_w, target, lik, _ = env.tables(st)
    w, sym = env.simulate(st, np.random.default_rng(seed), n)
    with np.errstate(divide="ignore"):
        s = np.log(prior_w) + np.log(lik[np.arange(lik.shape[0]), :, sym]).sum(1)   # (n, W)
    logit = np.logaddexp.reduce(s[:, target == 1], 1) - np.logaddexp.reduce(s[:, target == 0], 1)
    return target[w], logit


@pytest.mark.parametrize("name", ZNAMES)
def test_bayes_calibrated_on_simulated_draws(name):
    """E[Y | p] = p for the Bayes posterior: per decile of p, |mean(Y) - mean(p)| within 4 standard errors."""
    env = env_of(name)
    st = env.structure(np.random.default_rng(11))
    y, logit = _bayes_mc(env, st, 60_000, 12)
    p = sigmoid(logit)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    for k in np.unique(b):
        m = b == k
        se = np.sqrt(max(p[m].mean() * (1 - p[m].mean()), 1e-4) / m.sum())
        assert abs(y[m].mean() - p[m].mean()) < 4 * se + 1e-3, (k, m.sum())


# ---------------------------------------------------------------- closed forms and structure-specific facts

def test_z1_closed_form():
    env = env_of("Z1")
    batch = env.sample("dev", 50)
    for e in range(50):
        g, rho, s = batch.oracle["group"][e], batch.oracle["rho"][e], batch.oracle["s"][e]
        x = batch.obs[e, :, 0]
        edges = env.edges()
        sym = batch.oracle["sym"][e]
        _, _, lik, _ = env.tables_for(batch, e)
        # per-agent likelihood of its symbol given its parent bit u (u=1 rows: any w with that bit)
        lu = np.zeros((len(g), 2))
        from aggbench.zoo import binned_normal
        for i in range(len(g)):
            lu[i] = binned_normal(np.array([-s[i], s[i]]), np.ones(2), edges)[:, sym[i]]
        logp = []
        for y in (0, 1):
            v = np.log(0.5) + sum(np.log(lu[i, y]) for i in range(len(g)) if g[i] < 0)
            for k in range(len(rho)):
                members = np.flatnonzero(g == k)
                tot = sum((rho[k] if z == y else 1 - rho[k]) * np.prod(lu[members, z]) for z in (0, 1))
                v += np.log(tot)
            logp.append(v)
        assert abs((logp[1] - logp[0]) - batch.bayes["logit"][e]) < 1e-9
        assert np.isfinite(x).all()


def test_z1_shared_parent_caps_evidence():
    """Many agents of one group cannot push the posterior past what their parent says: |logit| <= log(rho/(1-rho))."""
    env = env_of("Z1", n_agents=12, n_groups=1, direct_prob=0.0, strength_range=(3.0, 3.0))
    batch = env.sample("dev", 200)
    cap = np.log(batch.oracle["rho"][:, 0] / (1 - batch.oracle["rho"][:, 0]))
    assert (np.abs(batch.bayes["logit"]) <= cap + 1e-9).all()
    l0 = batch.public["prior_logit"]
    summed = (batch.reports["logit"] - l0).sum(1)
    assert (np.abs(summed) > cap).mean() > 0.9           # the independent product overshoots


def test_z2_magnitude_reports_are_prior_and_reports_lose_information():
    env = env_of("Z2")
    batch = env.sample("dev", 400)
    mag = batch.oracle["type"] == 2
    assert np.abs(batch.reports["logit"][mag]).max() < 1e-9
    thr = batch.oracle["type"] == 1
    assert set(np.unique(batch.obs[..., 0][thr])) <= {0.0, 1.0}
    gap = per_episode(oracle_clearing(batch), batch)["kl"]
    assert gap.mean() > 1e-3 and (gap > -1e-12).all()
    no_mag = ~mag.any(1)
    assert np.abs(oracle_clearing(batch)["logit"][no_mag] - batch.bayes["logit"][no_mag]).max() < 1e-8


def test_z4_hypothesis_report_closed_form():
    """A hypothesis agent that drew h=1 reports log E[q | Y=1 side]: P(Y=1 | h=1) = E[q^2] / E[q] under the prior
    of D (prior 0.5), and its samples are exchangeable given D."""
    env = env_of("Z4")
    batch = env.sample("dev", 40)
    for e in range(40):
        m = batch.oracle["model"][e]
        pd, q = env._model_tables({k: v[e] for k, v in batch.oracle.items()})
        for i in np.flatnonzero(m >= 0):
            pD = 0.5 * pd[m[i], 0] + 0.5 * pd[m[i], 1]
            qq = q[m[i]]
            h = batch.obs[e, i, 0]
            p1 = (pD * qq * qq).sum() / (pD * qq).sum() if h == 1 else (pD * qq * (1 - qq)).sum() / (pD * (1 - qq)).sum()
            assert abs(sigmoid(batch.reports["logit"][e, i]) - p1) < 1e-12


def test_z4_many_samples_converge_to_model_posterior_not_certainty():
    """Mixture semantics: with 60 samples of one model and no evidence agents, Bayes approaches q(D), it does not
    add up the samples' log-odds."""
    env = env_of("Z4", n_agents=60, n_models=1, hypothesis_prob=1.0)
    batch = env.sample("dev", 100)
    l0 = 0.0
    summed = (batch.reports["logit"] - l0).sum(1)
    assert np.abs(batch.bayes["logit"]).max() < 8 and np.abs(summed).mean() > 3 * np.abs(batch.bayes["logit"]).mean()


def test_z5_full_joint_enumeration():
    """Enumerate the full joint table P(w, o_1..o_N) over all 8 * 2^N cells and condition on the observed o."""
    env = env_of("Z5", n_agents=5)
    batch = env.sample("dev", 20)
    for e in range(20):
        prior_w, target, lik, _ = env.tables_for(batch, e)
        sym = tuple(batch.oracle["sym"][e])
        num = den = 0.0
        for w in range(8):
            for o in itertools.product((0, 1), repeat=5):
                p = prior_w[w] * np.prod([lik[i, w, o[i]] for i in range(5)])
                if o == sym:
                    den += p
                    num += p * target[w]
        assert abs(num / den - sigmoid(batch.bayes["logit"][e])) < 1e-12


@pytest.mark.parametrize("name", ["Z6", "Z6-parity3"])
def test_parity_closed_form_and_uninformative_reports(name):
    """P(parity = 1 | readings) = (1 - prod_j (1 - 2 p_j)) / 2 with p_j = P(B_j = 1 | readings of bit j)."""
    env = env_of(name)
    batch = env.sample("dev", 100)
    assert np.abs(batch.reports["logit"]).max() < 1e-12
    assert np.abs(oracle_clearing(batch)["logit"]).max() < 1e-12
    nb = env.config.n_bits
    for e in range(100):
        var, acc, o = batch.oracle["var"][e], batch.oracle["acc"][e], batch.obs[e, :, 0]
        prod = 1.0
        for j in range(nb):
            lo = sum(np.log(acc[i] / (1 - acc[i])) * (1 if o[i] == 1 else -1) for i in np.flatnonzero(var == j))
            prod *= 1 - 2 * sigmoid(lo)
        assert abs((1 - prod) / 2 - sigmoid(batch.bayes["logit"][e])) < 1e-12
    assert np.abs(batch.bayes["logit"]).mean() > 0.3


def test_z7_regime_dependent_reliability_matters():
    """Bayes jointly infers the regime: it beats the reports' independent product, and agents with a negative
    strength in some regime exist."""
    env = env_of("Z7")
    batch = env.sample("dev", 1000)
    assert (batch.oracle["s"] < 0).any(axis=2).mean() > 0.3
    tr = env.sample("train", 1000)
    f, _ = all_baselines(tr, batch)
    assert per_episode(f["sum_logodds"], batch)["regret"].mean() > 0.01


# ---------------------------------------------------------------- interface, registry, guard

@pytest.mark.parametrize("name", ZNAMES)
def test_interface(name):
    env = env_of(name)
    batch = env.sample("dev", 8)
    N = batch.public["n_agents"]
    assert batch.kind == "bernoulli" and batch.truth.shape == (8,) and set(np.unique(batch.truth)) <= {0, 1}
    assert batch.reports["logit"].shape == (8, N) and np.isfinite(batch.reports["logit"]).all()
    assert np.isfinite(batch.bayes["logit"]).all() and np.isfinite(oracle_clearing(batch)["logit"]).all()
    for vis in ("none", "noisy", "full"):
        v = batch.view(vis)
        assert v.keys.shape[:2] == (8, N) and (v.keys.shape[2] == 0) == (vis == "none")
        assert not hasattr(v, "oracle") and not hasattr(v, "truth")
        obs, keys, pub = batch.agent_inputs(vis)
        assert obs.shape[:2] == (8, N) and np.isfinite(obs).all()
    assert not np.array_equal(batch.keys["noisy"], batch.keys["full"])
    a, b = env.sample("train", 6), env_of(name).sample("train", 6)
    np.testing.assert_array_equal(a.obs, b.obs)
    np.testing.assert_array_equal(a.bayes["logit"], b.bayes["logit"])


@pytest.mark.parametrize("name", ZNAMES)
def test_ceiling_ordering(name):
    """bayes <= reports ceiling <= every report-only baseline (mean regret, up to sampling noise)."""
    env = env_of(name)
    tr, dv = env.sample("train", 400), env.sample("dev", 400)
    f, _ = all_baselines(tr, dv)
    reg = {k: per_episode(v, dv)["regret"] for k, v in f.items()}
    kl = per_episode(f["oracle_clearing"], dv)["kl"]
    assert kl.min() > -1e-9
    for k in ("linear_pool", "log_pool", "sum_logodds", "extremized_log_pool"):
        d = reg[k] - reg["oracle_clearing"]
        assert d.mean() >= -3 * d.std() / np.sqrt(len(d)) - 1e-12, k


def test_registry_and_heldout_guard():
    infos = list_envs()
    names = [d["name"] for d in infos]
    assert names[:6] == ["E1", "E2", "E3-sybil", "E3-biased", "E3-collude", "E4"]
    assert set(ZOO) <= set(names) and all(d["family"] == FAMILIES[d["name"]] for d in infos)
    assert "hidden-provenance" in {d["family"] for d in infos if not d["heldout"]}
    fam = {d["family"] for d in infos}
    assert {"common-cause", "nonlinear", "heavy-tail", "hypothesis-samples", "multi-question",
            "complementary-xor", "context-reliability"} <= fam
    assert {d["name"] for d in infos if d["heldout"]} == set(HELDOUT)
    open_ = list_envs(include_heldout=False)
    assert not any(d["heldout"] for d in open_)
    assert {d["family"] for d in infos if d["heldout"]} - {d["family"] for d in open_} == {"heavy-tail",
                                                                                         "context-reliability"}
    for name in HELDOUT:
        with pytest.raises(PermissionError):
            make_env(name)
        assert make_env(name, allow_heldout=True).name == name
    with pytest.raises(ValueError):
        make_env("Z99")


def test_posterior_logit_handles_impossible_states():
    prior = np.array([0.5, 0.5, 0.0])
    target = np.array([1, 0, 1])
    lik = np.array([[[0.2, 0.8], [0.6, 0.4], [1.0, 0.0]]])
    assert np.isclose(posterior_logit(prior, target, lik[:, :, 0]), np.log(0.2 / 0.6))
    rep, bayes, ceil = tabular_ceilings(prior, target, lik, np.array([1]))
    assert np.isclose(bayes, np.log(0.8 / 0.4)) and np.isclose(ceil, bayes) and np.isclose(rep[0], bayes)


# ---------------------------------------------------------------- Z8 hidden provenance (linear-Gaussian)

def test_z8_brute_force_and_monte_carlo_reports_ceiling():
    env = env_of("Z8")
    assert oracle_check(env)["max_abs_diff"] < 1e-12
    batch = env.sample("dev", 1)
    cov, mu = batch.oracle["cov"][0], batch.oracle["mu"][0]
    N, K = cov.shape
    beta = np.array([env.report_ceiling(r, cov, mu) for r in np.eye(N)])      # ceiling is linear, l0 = 0
    rng = np.random.default_rng(3)
    n, tau = 300_000, env.config.read_noise
    y = rng.random(n) < 0.5
    x = np.where(y[:, None], mu, -mu) + rng.standard_normal((n, K))
    reads = x[:, None, :] + tau * rng.standard_normal((n, N, K))
    rep = (env.weights(cov, mu)[None] * reads).sum(-1)
    X1 = np.hstack([rep, np.ones((n, 1))])
    w = np.zeros(N + 1)
    for _ in range(25):
        p = 1 / (1 + np.exp(-X1 @ w))
        w += np.linalg.solve(X1.T @ (X1 * (p * (1 - p))[:, None]), X1.T @ (y - p))
    assert np.abs(w[:N] - beta).max() < 0.03 and abs(w[N]) < 0.02


def test_z8_content_and_own_models_do_not_fingerprint():
    batch = env_of("Z8").sample("dev", 300)
    obs, _, _, own = batch.agent_inputs("none", alignment="unaligned", own_model=True)
    K = obs.shape[-1]
    for e in range(300):
        vals = obs[e][np.isfinite(obs[e])]
        assert len(np.unique(vals)) == len(vals)            # no two readings are the same number
    assert len(np.unique(own[..., K:][own[..., :K] > 0])) <= 3
    tr = env_of("Z8").sample("train", 2000)
    f, _ = all_baselines(tr, batch)
    assert per_episode(f["oracle_clearing"], batch)["regret"].mean() > 0.005   # reports lose information


def test_z8_full_provenance_identifies_shared_pieces_and_none_hides_them():
    """Unaligned Z8: the `full` key's identity block moves with each coordinate, so readings of one piece can be
    grouped and Bayes rebuilt from the view; at `none` there is no identity at all."""
    env = env_of("Z8")
    batch = env.sample("dev", 200)
    obs, keys, _, own = batch.agent_inputs("full", alignment="unaligned", own_model=True)
    K, tau = env.config.n_pieces, env.config.read_noise
    for e in range(200):
        ids = np.rint(keys[e, :, 2 * K:] * K).astype(int) - 1          # piece id per coordinate, -1 = not read
        mu = own[e, :, K:2 * K]
        tot = env.logit0
        for k in range(K):
            m = ids == k
            if m.any():
                tot += 2 * mu[m][0] * obs[e][m].sum() / (m.sum() + tau ** 2)
        assert abs(tot - batch.bayes["logit"][e]) < 1e-9
    obs_n, keys_n, _, own_n = batch.agent_inputs("none", alignment="unaligned", own_model=True)
    assert keys_n.shape[-1] == 0 and own_n.shape[-1] == 2 * K          # own model: coverage and mu, no identity


@pytest.mark.parametrize("name,kw", [("Z1-blind", {"n_groups": 1}), ("Z4-blind", {"n_models": 1})])
def test_none_floor_equals_bayes_with_a_single_source(name, kw):
    from aggbench.zoo import none_floor_logit
    env = env_of(name, **kw)
    batch = env.sample("dev", 100)
    assert np.abs(none_floor_logit(env, batch) - batch.bayes["logit"]).max() < 1e-9


@pytest.mark.parametrize("name", ["Z1-blind", "Z4-blind"])
def test_none_floor_is_at_least_bayes_and_calibrated(name):
    """The floor at `none` is a posterior given less than Bayes: regret >= 0 on average (KL >= 0 per episode),
    strictly above Bayes here, and calibrated: E[Y | p] = p per decile."""
    from aggbench.zoo import none_floor_logit
    env = env_of(name)
    batch = env.sample("dev", 3000)
    f = {"logit": none_floor_logit(env, batch)}
    m = per_episode(f, batch)
    assert m["kl"].min() > -1e-9 and m["kl"].mean() > 1e-3
    p = sigmoid(f["logit"])
    edges = np.quantile(p, np.linspace(0, 1, 11))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
    for k in range(10):
        s = b == k
        assert abs(batch.truth[s].mean() - p[s].mean()) < 4 * np.sqrt(0.25 / s.sum()) + 1e-3


def test_none_floor_unknown_where_not_computed():
    from aggbench.report import floor_without_provenance
    for name in ("E1", "Z8"):
        env = make_env(name)
        assert floor_without_provenance(env, env.sample("dev", 3)) is None


# ---------------------------------------------------------------- Z9 k-bit parity

@pytest.mark.parametrize("k", [3, 4, 6])
def test_z9_parity_of_marginals_closed_form(k):
    """Bayes = (1 - prod_j (1 - 2 p_j)) / 2 over the parity bits, p_j the posterior of bit j from its readings;
    distractor readings drop out; every reading alone is uninformative (reports = prior = reports ceiling)."""
    env = env_of("Z9", n_bits=k)
    batch = env.sample("dev", 150)
    assert brute_force_tabular(env, batch) < 1e-12
    assert np.abs(batch.reports["logit"]).max() < 1e-12
    assert np.abs(oracle_clearing(batch)["logit"]).max() < 1e-12
    for e in range(150):
        var, acc, o = batch.oracle["var"][e], batch.oracle["acc"][e], batch.obs[e, :, 0]
        assert set(range(k)) <= set(var.tolist())                   # every parity bit is read
        prod = 1.0
        for j in range(k):
            lo = sum(np.log(acc[i] / (1 - acc[i])) * (1 if o[i] == 1 else -1) for i in np.flatnonzero(var == j))
            prod *= 1 - 2 * sigmoid(lo)
        assert abs((1 - prod) / 2 - sigmoid(batch.bayes["logit"][e])) < 1e-12
    assert (batch.oracle["var"] >= k).any()                          # distractor readers exist


@pytest.mark.parametrize("name,k", [("Z9", 3), ("Z9", 4), ("Z9-hard", 3)])
def test_z9_no_pair_of_readings_informs_y(name, k):
    """For k >= 3, P(Y = 1 | o_i, o_j) = 1/2 for every pair of agents and every pair of readings (exact, from the
    tables): only higher-order combinations carry information."""
    env = env_of(name, n_bits=k)
    st = env.structure(np.random.default_rng(1))
    prior_w, target, lik, _ = env.tables(st)
    N = lik.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            joint = np.einsum("w,wa,wb->wab", prior_w, lik[i], lik[j])          # P(w, o_i, o_j)
            p1 = joint[target == 1].sum(0) / joint.sum(0)
            assert np.abs(p1 - 0.5).max() < 1e-12


@pytest.mark.parametrize("name,k,lo,hi", [("Z9", 3, 0.45, 0.55), ("Z9", 4, 0.40, 0.50), ("Z9", 6, 0.32, 0.42),
                                          ("Z9-hard", 3, 0.17, 0.23), ("Z9-hard", 4, 0.06, 0.10),
                                          ("Z9-hard", 6, 0.0, 0.02)])
def test_z9_prior_to_bayes_gap(name, k, lo, hi):
    """Z9's defaults leave >= 0.3 nats between the prior and Bayes for k = 3, 4, 6 (so a model that solves parity
    is separable from one that outputs the prior); Z9-hard keeps the first, noisier setting."""
    batch = env_of(name, n_bits=k).sample("dev", 3000)
    gap = per_episode({"logit": np.zeros(3000)}, batch)["kl"].mean()
    assert lo < gap < hi
