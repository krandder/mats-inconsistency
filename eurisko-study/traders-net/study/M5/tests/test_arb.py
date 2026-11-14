"""Tests for w2/arb.py that EXPLOIT-w2.md describes as inline but that were not in the code (added by the M5 lead).
Run: python3 -m pytest -q tests/test_arb.py   (or python3 tests/test_arb.py)"""
import itertools, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w2"))
import arb  # noqa: E402

VERT = {"and": [(a, b, a & b) for a in (0, 1) for b in (0, 1)],
        "or": [(a, b, a | b) for a in (0, 1) for b in (0, 1)]}


def d2(x, v):
    return sum((a - b) ** 2 for a, b in zip(x, v))


def test_projection_feasible_and_pythagoras():
    rng = random.Random(0)
    for kind in ("and", "or"):
        H = arb._halfspaces(kind)
        for _ in range(3000):
            x = [rng.random() for _ in range(3)]
            p = arb.proj_poly(x, kind)
            assert all(sum(c * v for c, v in zip(co, p)) <= r + 1e-6 for co, r in H), (kind, x, p)
            for v in VERT[kind]:  # projection onto a convex set is never farther from any member (Pythagoras)
                assert d2(p, v) <= d2(x, v) + 1e-9, (kind, x, p, v)


def test_simplex_projection():
    rng = random.Random(1)
    for _ in range(3000):
        x = [rng.random() for _ in range(3)]
        p = arb.proj_simplex(x)
        assert abs(sum(p) - 1) < 1e-9 and min(p) >= 0
        for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            assert d2(p, v) <= d2(x, v) + 1e-9


def test_bundle_riskless():
    """Every traded bundle earns >= violation - 0 >= TOL on every coherent outcome (5000 random price tuples)."""
    rng = random.Random(2)
    outs = {"and": VERT["and"], "or": VERT["or"], "ex3": [(1, 0, 0), (0, 1, 0), (0, 0, 1)], "twin": [(0, 0), (1, 1)]}
    n = 0
    for _ in range(5000):
        for kind, ys in outs.items():
            p = [rng.random() for _ in range(len(ys[0]))]
            for y in ys:
                traded, pnl = arb.bundle_pnl(kind, p, y)
                if traded:
                    n += 1
                    assert pnl >= arb.TOL - 1e-12, (kind, p, y, pnl)
        for kind, fixed in (("taut_or", 1), ("impl", 1), ("contra", 0)):
            pp = rng.random()
            traded, pnl = arb.bundle_pnl(kind, [pp], [fixed], fixed)
            if traded:
                n += 1
                assert pnl >= arb.TOL - 1e-12
    assert n > 10000


def test_belief_clipped():
    for kind, p in (("and", [0.9, 0.9, 0.1]), ("or", [0.1, 0.1, 0.9]), ("ex3", [0.9, 0.9, 0.9]), ("twin", [0.2, 0.8])):
        q = arb.belief(kind, p)
        assert all(arb.DELTA <= v <= 1 - arb.DELTA for v in q)
    assert arb.belief("contra", [0.4], 0) == [arb.DELTA]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("PASS", k)
