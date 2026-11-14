"""Cross-module checks against siblings' substrate code; each test skips until its file exists.
Seeded determinism of the streams, the prequential no-overlap contract, panel vs loop agreement.
torch comes from /home/agent/workspaces/eurisko-study/wealth-net/.pylib (see STATUS)."""
import importlib, os, sys
import numpy as np
import pytest
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
PYLIB = "/home/agent/workspaces/eurisko-study/wealth-net/.pylib"
if os.path.isdir(PYLIB) and PYLIB not in sys.path:
    sys.path.append(PYLIB)
import panel


def _mod(name):
    if not os.path.exists(os.path.join(HERE, name + ".py")):
        pytest.skip(f"{name}.py not present yet")
    try:
        return importlib.import_module(name)
    except ImportError as e:
        pytest.skip(f"{name} import failed: {e}")


def test_streams_seeded_determinism():
    s = _mod("streams")
    a = s.make_questions(16, 512, seed=7); b = s.make_questions(16, 512, seed=7)
    c = s.make_questions(16, 512, seed=8)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])


def test_streams_ext_seeded_determinism():
    s = _mod("streams_ext")
    kw = dict(Q=16, T=3000, heldout=2, regime_every=1000)
    a, b, c = s.make_streams(seed=5, **kw), s.make_streams(seed=5, **kw), s.make_streams(seed=6, **kw)
    assert np.array_equal(a.bits, b.bits) and np.array_equal(a.fam_t, b.fam_t)
    assert np.array_equal(a.active, b.active) and a.events == b.events
    assert not np.array_equal(a.bits, c.bits)


def _tiny_run(seed, delay=None):
    pq = _mod("prequential"); st = _mod("streams"); md = _mod("model")
    torch = pytest.importorskip("torch")
    torch.manual_seed(seed)
    bits, fam, _ = st.make_questions(16, 400, seed=seed)
    stream = pq.ArrayStream(bits, fam, W=16, delay=delay)
    net = md.Net(16, 16, d=16, H=2, L=1)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    return pq.run_prequential(net, stream, opt, B=8), fam, stream


def test_prequential_no_overlap_and_determinism():
    r1, fam, _ = _tiny_run(3)
    r2, _, _ = _tiny_run(3)
    np.testing.assert_array_equal(r1["z"], r2["z"])                 # seeded determinism
    m = r1["mask"]
    ps, ts = r1["pred_step"], r1["train_step"]
    trained = ts >= 0
    assert np.all(ps[trained] >= 0) and np.all(ts[trained] >= ps[trained])   # test, then train
    assert np.all(ps[m] >= 0)


def test_prequential_delay_no_overlap():
    delay = np.array([0, 3, 7, 0] * 4)
    r, fam, _ = _tiny_run(4, delay=delay)
    ts, ps = r["train_step"], r["pred_step"]
    B = r["B"]
    ok = ts >= 0
    # a label with delay D at position i is only trained in the block that contains i + D
    ii, qq = np.nonzero(ok)
    assert np.all(ts[ii, qq] == (ii + delay[qq]) // B)
    assert np.all(ts[ok] >= ps[ok])


def test_panel_matches_loop_scores():
    pq = _mod("prequential")
    r, fam, _ = _tiny_run(5)
    b_loop, r_loop = pq.per_family(r["z"], r["y"], r["mask"], fam)
    row = panel.panel_from_prequential(r, fam)
    for f in range(len(b_loop)):
        if not np.isnan(b_loop[f]):
            assert row["bce"][str(f)] == pytest.approx(b_loop[f], rel=1e-5)
            assert row["brier"][str(f)] == pytest.approx(r_loop[f], rel=1e-5)


def test_panel_violations_match_li_families():
    li = _mod("li_families")
    st = li.make_li_stream(seed=1, T=3000)
    rng = np.random.default_rng(0)
    p = rng.random((50, st.Q))
    ours = panel.violations(p, panel.constraints_from_li(st.kind, st.ops, st.group), tol=0.05)
    theirs = li.violations(p, st, tol=0.05)
    for k_ours, k_theirs in (("and", "and"), ("or", "or"), ("contra", "contra"),
                             ("exhaustive", "ex3"), ("equal", "twin")):
        if k_theirs in theirs:
            assert ours[k_ours]["mean"] == pytest.approx(theirs[k_theirs]["mean"], rel=1e-9)
            assert ours[k_ours]["frac_gt_tol"] == pytest.approx(theirs[k_theirs]["frac"], rel=1e-9)
    n_t = sum(theirs[k]["n"] for k in ("taut_or", "impl") if k in theirs)
    m_t = sum(theirs[k]["mean"] * theirs[k]["n"] for k in ("taut_or", "impl") if k in theirs) / n_t
    assert ours["taut"]["n"] == n_t and ours["taut"]["mean"] == pytest.approx(m_t, rel=1e-9)
