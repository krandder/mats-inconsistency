"""Tests for substrate/streams_ext.py (S0 worker 2). Run: python3 -m pytest -q tests/test_streams_ext.py
or python3 tests/test_streams_ext.py (no pytest needed)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import streams_ext as S  # noqa: E402

T = 24000


def _s(**kw):
    a = dict(Q=64, T=T, seed=3, regime_every=5000)
    a.update(kw)
    return S.make_streams(**a)


def test_seeded_determinism():
    a, b = _s(), _s()
    assert np.array_equal(a.bits, b.bits) and np.array_equal(a.fam_t, b.fam_t)
    assert np.array_equal(a.active, b.active) and a.events == b.events
    c = _s(seed=4)
    assert not np.array_equal(a.bits, c.bits)


def test_prng_no_shared_offsets():
    s = _s(regime_every=0)
    P = s.bits[s.fam0 == 5]
    assert len(P) == 8
    from numpy.lib.stride_tricks import sliding_window_view as swv
    for i in range(len(P)):
        v = swv(P[i, :2064], 64)                      # shifts 0..2000, the pi offset range
        for j in range(len(P)):
            if i == j:
                continue
            for sh in np.where((v == P[j, :64]).all(1))[0]:
                assert not np.array_equal(P[i, sh:], P[j, :T - sh]), (i, j, sh)
    c = np.corrcoef(P.astype(float))
    assert np.abs(c[~np.eye(8, dtype=bool)]).max() < 0.04
    assert abs(P.mean() - 0.5) < 0.01


def test_heldout_hidden_then_released():
    s = _s()
    ho = np.where(s.fam0 == S.HO)[0]
    assert len(ho) == 8 and ho.min() == s.n_main
    assert not s.active[ho, :s.release].any() and s.active[ho, s.release:].all()
    assert s.active[:s.n_main].all()
    per = [int(d.split("p=")[1].split()[0]) for d in np.array(s.desc)[ho]]
    assert all(17 <= p <= 31 for p in per)
    tr = [int(d.split("p=")[1].split()[0]) for d, f in zip(s.desc, s.fam0) if f == 0]
    assert all(2 <= p <= 16 for p in tr)
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  (torch missing: adapter checks skipped; use PYTHONPATH=wealth-net/.pylib)")
    else:
        a = S.to_array_stream(s)
        i0 = s.release - s.W                   # last position whose target is pre-release
        w = a.window([i0 - 1, i0 + s.W]).numpy()
        assert (w[0, ho] == 0.5).all()                       # nothing leaks before release
        assert set(np.unique(w[1, ho])) <= {0.0, 1.0}         # full window from release on
        assert (w[:, :s.n_main] != 0.5).all()
        m = a.label_mask([i0 - 1, i0]).numpy()
        assert not m[0, ho].any() and m[1, ho].all() and m[:, :s.n_main].all()
        assert np.array_equal(a.label([i0]).numpy()[0], s.bits[:, s.release])
    assert not (s.fam_t[ho] != S.HO).any()                # held-out never shifted


def test_regime_shift_quarter_and_paired():
    s = _s()
    sw = [e for e in s.events if e["kind"] == "switch"]
    bounds = sorted({e["t"] for e in sw})
    assert bounds == [5000, 10000, 15000, 20000]
    for b in bounds:
        qs = [e["q"] for e in sw if e["t"] == b]
        assert len(qs) == len(set(qs)) == round(0.25 * s.n_main)
        for e in sw:
            if e["t"] == b:
                assert e["from"] != e["to"] and s.fam_t[e["q"], b] == e["to"]
                assert s.fam_t[e["q"], b - 1] == e["from"]
    s0 = _s(regime_every=0)
    assert np.array_equal(s.bits[:, :5000], s0.bits[:, :5000])   # paired before first boundary
    assert np.array_equal(s.bits[s.n_main:], s0.bits[s.n_main:])  # held-out unaffected


def test_compounds_follow_current_operands():
    s = _s()
    import re
    for q in range(s.n_main):
        if s.fam0[q] not in (6, 7):
            continue
        a, b = map(int, re.findall(r"q(\d+)", s.desc[q]))
        end = min([e["t"] for e in s.events if e["kind"] == "switch" and e["q"] == q] + [T])
        x, y = s.bits[a, :end], s.bits[b, :end]
        want = (x & y) if s.fam0[q] == 6 else (x | y)
        assert np.array_equal(s.bits[q, :end], want)


def test_drop_in_make_questions():
    b, f, d = S.make_questions(64, 4000, 1)
    assert b.shape == (64, 4000) and f.shape == (64,) and len(d) == 64
    assert set(f.tolist()) == set(range(8))


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
