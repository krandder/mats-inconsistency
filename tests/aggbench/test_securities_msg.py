import math

import numpy as np
import pytest

from aggbench import securities_msg as sm
from aggbench.analysis import gsigma as gs
from aggbench.parity_learn import DEV_SEED, Z9Spec, _kl, sample


@pytest.mark.parametrize("spec", [Z9Spec(), Z9Spec(readers_random=True), sm.SPECS["lr"][0]])
def test_sample_world_reproduces_parity_learn_sample(spec):
    a = sample(spec, 400, np.random.default_rng(5))
    b = sm.sample_world(spec, 400, np.random.default_rng(5))
    assert np.array_equal(a["tokens"], b["tokens"]) and np.array_equal(a["y"], b["y"])
    assert np.allclose(a["bayes"], b["bayes"])
    assert np.array_equal(sm.target(spec.k)[b["w"]], b["y"].astype(int))
    agree = (((b["w"][:, None] >> b["var"]) & 1) == b["read"]).mean()
    assert abs(agree - np.mean(spec.acc)) < 0.01                 # readings are of omega's bits at the drawn accuracy


def test_market_readout_with_bit_securities_is_the_parity_formula():
    rng = np.random.default_rng(0)
    eta = rng.normal(0, 2, (50, 4))
    ind = sm.BITS[:, :4].astype(float)
    p = 1 / (1 + np.exp(-eta))
    py1 = (1 - np.prod(1 - 2 * p, 1)) / 2
    assert np.allclose(sm.market_logit_np(eta, ind), np.log(py1) - np.log1p(-py1))


def test_oracle_closed_form_matches_the_review_reference():
    spec = Z9Spec()                                           # fixed readers: the review's 0.0147 at 5.49 bits
    dev = sm.sample_world(spec, 4000, np.random.default_rng(DEV_SEED))
    f, sym, possible = sm.oracle_closed(dev, spec, 1)
    assert abs(_kl(f, dev["bayes"]).mean() - 0.0147) < 0.002
    assert possible == 41                                     # 45 unordered pairs, 5 of them the zero trade
    f2, _, possible2 = sm.oracle_closed(dev, spec, 2)
    assert _kl(f2, dev["bayes"]).mean() < _kl(f, dev["bayes"]).mean() and possible2 <= 256


def test_span_regret_agrees_with_gsigma_and_is_zero_on_the_bit_span():
    cell = gs.z9p2_cell(1, readers_random=True)
    ev = gs.Evaluator(cell)
    for sigma, phi in ((("Y",), sm.y_security()), (("maj",), cell.sec["maj"][:, None].astype(float)),
                       (("b1", "b2", "b3", "b4"), 2.0 * sm.BITS[:, :4] - 1)):
        assert abs(ev.full(sigma)["regret_sum"] * math.log(2) - sm.span_regret(cell, phi)) < 1e-7
    assert sm.span_regret(cell, sm.basket_securities(16)) < 1e-9          # baskets span the bits
    assert sm.span_regret(cell, sm.basket_securities(2)) > 0.05            # two bits are not enough


def test_angles():
    par = 2.0 * sm.BITS[:, :4] - 1
    a = sm.angles(sm.basket_securities(16), par)
    assert max(a["angles_deg"]) < 1e-5 and a["energy_outside"] < 1e-10
    a = sm.angles(sm.y_security(), par)
    assert min(a["angles_deg"]) > 89.9                                     # Y is orthogonal to every bit


@pytest.mark.parametrize("arm,b", [("settle_market", 2), ("yonly_mlp", 4), ("ysec_market", 2), ("oraclephi_market", 6)])
def test_train_arm_smoke(arm, b):
    pytest.importorskip("torch")
    r = sm.train_arm(arm, b, 0, "rr3", steps=30, eval_every=15, n_val=200, n_dev=200, n_boot=50, n_struct=1, threads=2)
    assert r["distinct_symbols"] <= 2 ** b and r["id_bits"] + r["amount_bits"] == b
    assert np.isfinite(r["regret"]["mean"]) and r["securities"]["R_mean_nats"] >= -1e-9
    assert np.array(r["phi"]).shape == (64, r["S"])


def test_leak_reduced_spec_oracle_and_mc_span_regret():
    spec, ov = sm.SPECS["lr"]
    dev = sm.sample_world(spec, 4000, np.random.default_rng(DEV_SEED))
    f, _, _ = sm.oracle_closed(dev, spec, 1)
    assert abs(_kl(f, dev["bayes"]).mean() - 0.006) < 0.002            # main README: 0.006 at 5.49 bits
    cells = (gs.z9p2_cell(s, **ov) for s in range(50))                  # 13 agents: Monte Carlo R
    cell = next(c for c in cells if {f"b{j}" for j in range(1, 5)} <= {v for r in c.reads for v in r})
    assert cell.N == 13                                                  # (a structure that reads every parity bit)
    assert sm.span_regret(cell, 2.0 * sm.BITS[:, :4] - 1, n_mc=2000) < 1e-9
    assert sm.span_regret(cell, sm.y_security(), n_mc=2000) > 0.1
