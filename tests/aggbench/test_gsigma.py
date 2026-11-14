"""Listed-securities gap G(Sigma) and the log-likelihood span view (aggbench/analysis/gsigma.py)."""
import itertools
import math

import numpy as np
import pytest

from aggbench.analysis import gsigma as gs


def bits_of(c):
    return sorted(n for n in c.sec if n.startswith("b"))


# ----------------------------------------------------------------------------- G(Sigma)

def test_xor_gap_is_one_bit_and_listing_both_bits_closes_it():
    c = gs.xor_cell(1.0)
    ev = gs.Evaluator(c)
    assert ev.G(["Y"])[0] == pytest.approx(1.0, abs=1e-12)
    assert ev.G(["b1", "b2"])[0] == pytest.approx(0.0, abs=1e-12)
    # one-shot, listing b1 alone does not help: the b2 reader's forecasts on Y and b1 are both 1/2
    assert ev.G(["Y", "b1"])[0] == pytest.approx(1.0, abs=1e-12)
    # sequentially (b1 reader first, the b2 reader sees the b1 price) {Y, b1} closes the gap; the other order does not
    assert gs.g_sequential(c, ["Y", "b1"], [0, 1]) == pytest.approx(0.0, abs=1e-12)
    assert gs.g_sequential(c, ["Y", "b1"], [1, 0]) == pytest.approx(1.0, abs=1e-12)
    # a single security correlated with both bits already makes every forecast injective
    assert ev.G(["and"])[0] == pytest.approx(0.0, abs=1e-12)


def test_and_forecasts_lose_nothing_but_summed_y_trades_do():
    for acc in (1.0, 0.9):
        ev = gs.Evaluator(gs.and_cell(acc))
        r = ev.full(["Y"])
        assert r["G"] == pytest.approx(0.0, abs=1e-12)          # forecasts are one-to-one in the evidence
        assert r["residual_kl_bits"] > 0.1 and r["regret_sum"] > 0.01
        assert ev.full(["b1", "b2"])["regret_sum"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("make,sigma", [
    (lambda: gs.z9_cell(3, n_bits=3), ["b1", "b2", "b3"]),
    (lambda: gs.z9_cell(4, n_bits=4), ["b1", "b2", "b3", "b4"]),
    (lambda: gs.z9p2_cell(1), ["b1", "b2", "b3", "b4"]),
    (lambda: gs.z6_cell(2), ["b1", "b2"]),
    (lambda: gs.z5_cell(5), ["A", "B", "C"]),
    (lambda: gs.copy_cell(0.85, 0.95), ["S"]),
    (lambda: gs.e3_sybil_cell(0, n_pieces=6, pieces_per_agent=(2, 4)), [f"x{k}" for k in range(1, 7)]),
])
def test_true_sufficient_securities_give_zero_gap_and_zero_regret(make, sigma):
    ev = gs.Evaluator(make())
    r = ev.full(sigma)
    assert abs(r["G"]) < 1e-9
    assert abs(r["regret_sum"]) < 1e-7


def test_y_only_gap_on_parity_is_all_information():
    for k in (3, 4):
        ev = gs.Evaluator(gs.z9_cell(0, n_bits=k))
        assert ev.G(["Y"])[0] == pytest.approx(ev.hy - ev.hyx, abs=1e-10)


def test_gap_never_increases_when_a_security_is_added():
    c = gs.z9_cell(7, n_bits=3)
    ev = gs.Evaluator(c)
    pool = ["Y", "b1", "b2", "b3", "s2", "d1", "maj"]
    for size in range(len(pool)):
        for sub in itertools.combinations(pool, size):
            g = ev.G(list(sub))[0]
            for z in pool:
                if z not in sub:
                    assert ev.G(list(sub) + [z])[0] <= g + 1e-10


def test_monte_carlo_path_agrees_with_enumeration(monkeypatch):
    c = gs.z9_cell(2, n_bits=3)
    exact = gs.Evaluator(c).full(["Y", "b1", "maj"])
    monkeypatch.setattr(gs, "EXACT_LIMIT", 4)
    ev = gs.Evaluator(c, n_mc=40000, seed=1)
    assert ev.mc
    mc = ev.full(["Y", "b1", "maj"])
    assert abs(mc["G"] - exact["G"]) < 5 * mc["G_se"] + 1e-3
    assert abs(mc["pool"] - exact["pool"]) < 5 * mc["pool_se"] + 1e-3


# ----------------------------------------------------------------------------- span view

def _brute_regret(c, sigma):
    """E_x KL(p(Y | x) || q(Y | x)) by plain enumeration of observation tuples."""
    cls = gs.forecast_classes(c, sigma)
    tr = gs.agent_trades(c, sigma, cls)
    phi = gs.basis(c, sigma)
    tot = 0.0
    for x in itertools.product(*[range(L.shape[1]) for L in c.lik]):
        pw = c.prior * np.prod([c.lik[i][:, o] for i, o in enumerate(x)], 0)
        px = pw.sum()
        if px == 0:
            continue
        s = np.log(c.prior) + sum(phi @ tr[i]["theta"][cls[i][0][o]] for i, o in enumerate(x))
        q = np.exp(s - s.max())
        py = np.bincount(c.target, pw / px)
        qy = np.bincount(c.target, q / q.sum())
        tot += px * sum(a * math.log2(a / b) for a, b in zip(py, qy) if a > 0)
    return tot


@pytest.mark.parametrize("sigma", [["Y"], ["Y", "b1"], ["maj"], ["b1", "b2", "b3"], ["Y", "s2"]])
def test_summed_regret_is_gap_plus_pooling_loss(sigma):
    c = gs.z9_cell(1, n_bits=3)
    r = gs.Evaluator(c).full(sigma)
    assert r["regret_sum"] == pytest.approx(_brute_regret(c, sigma), abs=1e-9)


def test_zero_residual_implies_zero_regret():
    cases = [(gs.xor_cell(0.9), [["Y"], ["b1", "b2"], ["and"], ["Y", "b1"]]),
             (gs.and_cell(0.9), [["Y"], ["Y", "b1"], ["b1", "b2"]]),
             (gs.copy_cell(0.85, 0.95), [["Y"], ["S"], ["Y", "S"]]),
             (gs.z6_cell(0), [["Y"], ["b1", "b2"], ["and"]]),
             (gs.z5_cell(0), [["Y"], ["Y", "A"], ["A", "B", "C"]]),
             (gs.z9_cell(0, n_bits=3), [["Y"], ["b1", "b2", "b3", "d1", "d2"], ["maj"]])]
    n_zero = 0
    for c, sets in cases:
        ev = gs.Evaluator(c)
        for s in sets:
            r = ev.full(s)
            if r["residual_kl_max_bits"] < 1e-9:
                n_zero += 1
                assert abs(r["regret_sum"]) < 1e-9
    assert n_zero >= 5


@pytest.mark.parametrize("make", [lambda: gs.xor_cell(0.9), lambda: gs.and_cell(0.9), lambda: gs.copy_cell(0.85, 0.95)])
def test_y_only_trades_are_out_of_span_and_lose(make):
    r = gs.Evaluator(make()).full(["Y"])
    assert r["residual_kl_bits"] > 0.1 and r["regret_sum"] > 0.01


def test_residual_and_regret_vanish_together_on_the_mixing_probe():
    prev = None
    for t in (0.0, 1e-3, 1e-2, 1e-1):
        r = gs.Evaluator(gs.mix_cell("and", t)).full(["Y"])
        if t == 0:
            assert r["residual_kl_max_bits"] < 1e-10 and abs(r["regret_sum"]) < 1e-10
        else:
            assert r["residual_kl_bits"] > 0 and r["regret_sum"] > 0
            if prev:
                assert r["regret_sum"] > prev
            prev = r["regret_sum"]


def test_trade_in_span_equals_log_likelihood_coefficients():
    c = gs.z6_cell(3)
    tr = gs.agent_trades(c, ["b1", "b2"])
    for i, t in enumerate(tr):
        j = int(c.reads[i][0][1]) - 1
        acc = c.lik[i][0, 0] if c.sec[c.reads[i][0]][0] == 0 else c.lik[i][0, 1]
        L = math.log(acc / (1 - acc))
        th = t["theta"][t["cls"]]                                   # (A, d), symbol order
        assert th[1, j] - th[0, j] == pytest.approx(2 * L, abs=1e-6)
        assert np.abs(th[:, 1 - j]).max() < 1e-8


# ----------------------------------------------------------------------------- tree width

def test_treewidth_known_graphs():
    assert gs.treewidth("abcd", [("a", "b"), ("b", "c"), ("c", "d")]) == 1
    assert gs.treewidth("abcd", [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]) == 2
    assert gs.treewidth("abcd", [("a", "b", "c", "d")]) == 3


def test_parity_market_treewidth_bits_versus_chain():
    c = gs.z9_cell(0, n_bits=4)
    assert gs.market_treewidth(c, bits_of(c)) == 4
    assert gs.market_treewidth(c, bits_of(c) + ["s2", "s3"]) == 2


# ----------------------------------------------------------------------------- best span and the osc bound

def test_span_dimension_and_principal_angles_on_parity():
    c = gs.z9_cell(0, n_bits=4)
    sv = gs.span_svd(c)
    read = set(r[0] for r in c.reads)
    assert sv["d_omega"] == len(read)                       # one direction per variable someone reads
    full = gs.principal_angles(c, sorted(read), svd=sv)
    assert full["angles_deg"].max() < 1e-4 and full["energy_outside"] < 1e-20
    y_only = gs.principal_angles(c, ["Y"], svd=sv)
    assert y_only["angles_deg"].min() > 89.9 and y_only["energy_outside"] == pytest.approx(1.0)


def test_kl_osc_bound_holds_and_is_tight_somewhere():
    worst = 0.0
    for c, sets in [(gs.and_cell(0.9), [["Y"], ["Y", "b1"]]), (gs.z6_cell(1), [["Y"], ["maj"]]),
                    (gs.z9_cell(2, n_bits=3), [["Y"], ["b1", "b2", "b3"]]), (gs.z5_cell(3), [["Y"]])]:
        for s in sets:
            r = gs.osc_bound_check(c, s)
            assert r["violations"] == 0 and r["y_above_w"] == 0
            worst = max(worst, r["max_ratio"])
    assert 0.5 < worst <= 1.0 + 1e-9
