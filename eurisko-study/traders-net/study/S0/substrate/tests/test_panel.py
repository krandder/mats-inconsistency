"""Known panel values on toys (S0 worker 4)."""
import math, os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import panel as P

p = np.array([0.9, 0.2, 0.6, 0.5]); y = np.array([1, 0, 0, 1])


def test_brier_bce():
    assert P.brier(p, y) == pytest.approx(0.165, abs=1e-12)
    assert P.bce(p, y) == pytest.approx(-(math.log(.9) + math.log(.8) + math.log(.4) + math.log(.5)) / 4, abs=1e-12)


def test_ece():
    assert P.ece(p, y, 10) == pytest.approx(0.35, abs=1e-12)
    assert P.ece(p, y, 2) == pytest.approx(0.05, abs=1e-12)
    assert P.ece([1.0, 0.0], [1, 0]) == 0.0          # p=1.0 lands in the last bin
    assert P.ece([0.7] * 10, [1] * 7 + [0] * 3) == pytest.approx(0.0, abs=1e-12)


def test_scores_by_family():
    s = P.scores(p, y, fam=np.array([0, 0, 1, 1]), names=["a", "b"])
    assert s["all"]["n"] == 4 and s["a"]["n"] == 2
    assert s["a"]["brier"] == pytest.approx((0.01 + 0.04) / 2)
    assert s["b"]["brier"] == pytest.approx((0.36 + 0.25) / 2)


def test_violations_values():
    #        0    1    2(and) 3(or) 4(taut) 5(contra) 6(not 0) 7,8,9 exhaustive
    pr = np.array([0.5, 0.5, 0.7, 0.3, 0.9, 0.2, 0.4, 0.2, 0.3, 0.4])
    cons = [("and", 2, 0, 1), ("or", 3, 0, 1), ("taut", 4), ("contra", 5), ("not", 6, 0),
            ("exhaustive", 7, 8, 9)]
    v = P.violations(pr, cons)
    assert v["and"]["mean"] == pytest.approx(0.2)
    assert v["or"]["mean"] == pytest.approx(0.2)
    assert v["taut"]["mean"] == pytest.approx(0.1)
    assert v["contra"]["mean"] == pytest.approx(0.2)
    assert v["not"]["mean"] == pytest.approx(0.1)
    assert v["exhaustive"]["mean"] == pytest.approx(0.1)
    assert v["all"]["n"] == 6


def test_frechet_lower_and_upper_or():
    pr = np.array([0.9, 0.9, 0.7, 0.0])
    assert P.violations(pr, [("and", 2, 0, 1)])["and"]["mean"] == pytest.approx(0.1)
    pr = np.array([0.2, 0.3, 0.0, 0.6])
    assert P.violations(pr, [("or", 3, 0, 1)])["or"]["mean"] == pytest.approx(0.1)
    # coherent prices give zero
    pr = np.array([0.6, 0.5, 0.3, 0.8])
    v = P.violations(pr, [("and", 2, 0, 1), ("or", 3, 0, 1)])
    assert v["all"]["max"] == pytest.approx(0.0)


def test_violations_only_before_resolution():
    Pt = np.array([[0.5, 0.5, 0.7], [0.5, 0.5, 0.9]])
    U = np.array([[True, True, True], [True, False, True]])   # q1 resolved at t=1
    v = P.violations(Pt, [("and", 2, 0, 1)], unresolved=U)
    assert v["and"]["n"] == 1 and v["and"]["mean"] == pytest.approx(0.2)


def test_gini():
    assert P.gini([0, 0, 0, 1]) == pytest.approx(0.75)
    assert P.gini([2, 2, 2]) == 0.0
    assert P.gini([-1, 1]) == pytest.approx(0.5)
    assert P.gini([0, 0]) == 0.0
    w = np.random.default_rng(0).exponential(size=200)
    brute = np.abs(w[:, None] - w[None, :]).sum() / (2 * 200 * w.sum())
    assert P.gini(w) == pytest.approx(brute, rel=1e-12)


def test_gini_null_positive_control():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    sig = np.zeros((50, 2000)); sig[:5] = 2 * y - 1        # 5 informed traders
    sig[5:] = rng.choice([-1, 1], size=(45, 2000))
    wf = lambda yy: (sig * (2 * yy - 1)).sum(1)            # P&L of +-1 bets
    r = P.gini_null(wf, y, n_null=30, seed=0)
    assert r["z"] > 3 and r["p"] < 0.05


def test_dead_bankrupt_cons_flops_ec():
    d = P.dead_bankrupt(act_count=[0, 1, 0, 3], wealth=[1, -1, 0, 2])
    assert d == {"dead_frac": 0.5, "bankrupt_frac": 0.5}
    c = P.conservation_residual(2.0, [1.0, 0.5], bias=0.4)
    assert c["abs"] == pytest.approx(0.1) and c["rel"] == pytest.approx(0.05)
    fc = P.FlopCounter().add(10).add(5, "bwd").add(1)
    assert fc.total == 16 and fc.by_kind == {"fwd": 11, "bwd": 5}
    assert P.net_forward_flops(1, 1, 1, d=1, H=1, L=1, mlp=1) == 2 * (1 + (3 + 2 + 1 + 2) + 1)
    t = np.arange(1, 101); s = P.ec_slopes(t, 3 * t ** 0.5)
    assert s["loglog"] == pytest.approx(0.5, abs=1e-9)
    assert P.ec_slopes(t, 2.0 * t)["lin_last_half"] == pytest.approx(2.0)


def test_panel_row_shapes():
    row = P.panel_row(p, y, fam=np.array([0, 0, 1, 1]), wealth=[1, 2, 3], act_count=[0, 1, 1],
                      flops=P.FlopCounter().add(7), cons=(1.0, [1.0], 0.0),
                      P=np.array([0.5, 0.5, 0.7]), constraints=[("and", 2, 0, 1)])
    assert row["brier"]["all"] == pytest.approx(0.165)
    assert row["viol_mean"]["and"] == pytest.approx(0.2)
    assert row["flops_cum"] == 7 and row["cons_resid_rel"] == 0.0
    assert row["dead_frac"] == pytest.approx(1 / 3)


def test_constraints_from_li():
    kind = ["base", "base", "and", "or", "taut_or", "impl", "contra", "ex3", "ex3", "ex3", "exp"]
    ops = np.full((11, 2), -1); ops[2] = (0, 1); ops[3] = (0, 1)
    group = np.full(11, -1); group[7:10] = 4
    c = P.constraints_from_li(kind, ops, group)
    assert ("and", 2, 0, 1) in c and ("or", 3, 0, 1) in c
    assert ("taut", 4) in c and ("taut", 5) in c and ("contra", 6) in c
    assert ("exhaustive", 7, 8, 9) in c and len(c) == 6
    c2 = P.constraints_from_li(["base", "twin"], np.array([[-1, -1], [0, -1]]), np.array([-1, -1]))
    assert c2 == [("equal", 1, 0)]
    assert P.violations(np.array([0.3, 0.5]), c2)["equal"]["mean"] == pytest.approx(0.2)
