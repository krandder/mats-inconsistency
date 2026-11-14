"""Z9 knobs added 2026-10-22 (learnability review): randomised readers per bit (no count leak) and biased bits."""
import numpy as np
import pytest

from aggbench import make_env
from aggbench.verify import brute_force_tabular
from aggbench.zoo import KParityEnv

KNOBS = [{"readers_random": True}, {"bit_prob": 0.8}, {"bit_prob": 0.65, "readers_random": True},
         {"readers_random": True, "readers_alpha": 1.0, "readers_per_bit": 6}]


def test_defaults_unchanged():
    a, b = make_env("Z9").sample("dev", 40), make_env("Z9", readers_random=False, bit_prob=0.5).sample("dev", 40)
    assert np.array_equal(a.obs, b.obs) and np.allclose(a.bayes["logit"], b.bayes["logit"])


@pytest.mark.parametrize("kw", KNOBS)
def test_brute_force_and_tables_match_simulator(kw):
    env = make_env("Z9", **kw)
    assert brute_force_tabular(env, env.sample("smoke", 20)) < 1e-12
    rng = np.random.default_rng(3)
    st = env.structure(rng)
    prior_w, target, lik, _ = env.tables(st)
    n = 200_000
    w, sym = env.simulate(st, rng, n)
    W, O = lik.shape[1], lik.shape[2]
    assert np.abs(np.bincount(w, minlength=W) / n - prior_w).max() < 5e-3
    for i in range(lik.shape[0]):
        emp = np.bincount(w * O + sym[:, i], minlength=W * O).reshape(W, O) / n
        assert np.abs(emp - prior_w[:, None] * lik[i]).max() < 5e-3


@pytest.mark.parametrize("kw", KNOBS)
def test_bayes_calibrated_and_prior_logit(kw):
    env = make_env("Z9", **kw)
    b = env.sample("dev", 4000)
    p = 1 / (1 + np.exp(-b.bayes["logit"]))
    assert abs(p.mean() - b.truth.mean()) < 0.03
    assert abs(1 / (1 + np.exp(-env.prior_logit())) - b.truth.mean()) < 0.03


def test_readers_random_varies_readers_per_bit():
    env = make_env("Z9", readers_random=True)
    counts = [np.bincount(env.structure(np.random.default_rng(s))["var"], minlength=env.n_vars())[:4]
              for s in range(50)]
    assert len({tuple(c) for c in counts}) > 10
    fixed = make_env("Z9")
    assert all((np.bincount(fixed.structure(np.random.default_rng(s))["var"], minlength=6)[:4] == 3).all()
               for s in range(10))


def test_biased_bits_make_single_readings_informative():
    env = make_env("Z9", bit_prob=0.8)
    b = env.sample("dev", 200)
    assert np.abs(b.local["logit"] - env.prior_logit()).max() > 0.05      # a single reading now informs Y
    flat = make_env("Z9").sample("dev", 200)
    assert np.allclose(flat.local["logit"], 0.0, atol=1e-9)


def test_knob_validation():
    with pytest.raises(ValueError):
        KParityEnv("Z9", bit_prob=1.0)
    with pytest.raises(ValueError):
        KParityEnv("Z9", readers_per_bit=None, readers_random=True)
