import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pattern as P

def test_soup_rule_oracle():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, (3, 200)).astype(np.uint8)
    W, lo, hi, k = 32, 0, 150, 4
    pr = P.soup_pred(bits, W, lo, hi, k)
    for i in range(0, 150, 7):
        for q in range(3):
            h = [bits[q, i + W - 1 - a] for a in range(32)]
            exp = -1
            for j in range(1, 32 - k + 1):
                if h[j:j + k] == h[:k]:
                    exp = h[j - 1]; break
            assert pr[i, q] == exp

def test_periodic_exploits_ignorant_market():
    T = 3000; bits = np.array([[1, 0, 0] * (T // 3)] * 2, np.uint8)
    z = np.zeros((T - 32, 2)); fam = np.array([0, 0])
    out, mkt = P.run_class(bits, 32, z, 100, T - 32, fam, 1)
    assert out["kt-2"]["log"].sum() > 0.6 * (T - 132) * 2 * np.log(2)   # near ln2 per bit
    assert out["soup-2"]["share"].sum() > 0.45 * (T - 132) * 2           # 0.5 per bit
    assert out["corr-2"]["log"][-500:].mean() > 0.5 * 2

def test_perfect_market_not_exploitable():
    rng = np.random.default_rng(1); T = 3000
    bits = rng.integers(0, 2, (4, T)).astype(np.uint8)
    z = np.where(bits[:, 32:].T > 0, 9.0, -9.0)       # clairvoyant market
    out, _ = P.run_class(bits, 32, z, 100, T - 32, np.zeros(4, int), 1)
    for v in out.values():
        if v["log"] is not None: assert v["log"].sum() < 1e-3
        assert v["share"].sum() <= 2e-4 * v["share"].size  # clip at 1-1e-4 leaves 1e-4 per share

def test_slope():
    assert abs(P.slope(np.full(100, 2.0)) - 2.0) < 1e-12
