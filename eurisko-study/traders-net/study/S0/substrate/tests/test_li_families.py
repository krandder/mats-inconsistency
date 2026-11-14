"""Checks for li_families (S0 worker 3, PREREG-w3 arms 1-4). Run: python3 -m pytest -q tests/test_li_families.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from li_families import make_li_stream, DelayBook, violations, log_score_gain
from streams import ca, lfsr16

T = 5000
ST = make_li_stream(0, T)


def test_labels_exact():
    st, b = ST, ST.bits
    for q in np.where(st.kind == "taut_or")[0]:
        assert b[q].all()
    for q in np.where(st.kind == "impl")[0]:
        assert b[q].all()
    for q in np.where(st.kind == "contra")[0]:
        assert not b[q].any()
    for g in np.unique(st.group[st.group >= 0]):
        assert (b[st.group == g].sum(0) == 1).all()
    for q in np.where(st.kind == "twin")[0]:
        assert (b[q] == b[st.ops[q, 0]]).all()
    for q in np.where(st.kind == "and")[0]:
        assert (b[q] == (b[st.ops[q, 0]] & b[st.ops[q, 1]])).all()
    for q in np.where(st.kind == "or")[0]:
        assert (b[q] == (b[st.ops[q, 0]] | b[st.ops[q, 1]])).all()


def test_expensive_is_k_ahead():
    st = ST
    for q in np.where(st.kind == "exp")[0]:
        k = st.delay[q]; d = st.desc[q]
        if d.startswith("ca110"):
            row = int(d.split("row=")[1].split()[0], 16); g = ca(110, row, T + 2000)
        else:
            s = int(d.split("state=")[1].split()[0]); g = lfsr16(s, T + 2000)
        assert (st.bits[q] == g[k:k + T]).all()
        # window at t is the current stream g[t-W:t]
        t = 3000
        assert (st.window(t)[0, q] == g[t - st.W:t]).all()


def test_no_leak():
    st = ST
    ts = np.arange(st.t0, T, 7)
    w = st.window(ts)                                     # [n,Q,W]
    for i, t in enumerate(ts[::50]):
        for q in range(st.Q):
            u = np.arange(t - st.delay[q] - st.W, t - st.delay[q])
            # every non-unknown entry is a label with reveal time u+delay < t, posed at/after release
            known = (u >= 0) & (u >= st.release[q])
            assert (u[known] + st.delay[q] < t).all()
            wi = st.window(t)[0, q]
            assert (wi[~known] == 0.5).all()
            assert (wi[known] == st.bits[q, u[known]]).all()
    a, b = st.train_range; c, d = st.eval_range
    assert b <= c and a < b and c < d
    # label for eval positions is revealed inside the stream
    assert d + st.delay.max() <= T


def test_unresolved_history():
    st = ST
    q = np.where(st.kind == "taut_or")[0][0]
    r, D = st.release[q], st.delay[q]
    assert st.unresolved_history(r + D)[0, q]           # last revealed u would be r-1
    assert not st.unresolved_history(r + D + 1)[0, q]
    assert (st.window(r + D)[0, q] == 0.5).all()
    assert st.window(r + D + 1)[0, q, -1] == 1.0


def test_determinism():
    a, b, c = make_li_stream(1, 3000), make_li_stream(1, 3000), make_li_stream(2, 3000)
    assert (a.bits == b.bits).all() and (a.release == b.release).all() and a.desc == b.desc
    assert not (a.bits == c.bits).all()


def test_delaybook_settles_once_at_booked_price():
    st = ST
    book = DelayBook(st.delay)
    rng = np.random.default_rng(0)
    booked, settled = {}, 0
    for t in range(st.t0, 2600):
        p = rng.random(st.Q)                             # trade first, then settle (test, then train)
        book.book(t, p)
        for q in range(st.Q):
            booked[(q, t)] = p[q]
        ok, u, price, y = book.settle(t, st.bits)
        for q in np.where(ok)[0]:
            assert u[q] + st.delay[q] == t
            assert price[q] == booked.pop((q, u[q]))
            assert y[q] == st.bits[q, u[q]]
            settled += 1
    # everything with reveal time < 2600 was settled
    assert all(u + st.delay[q] >= 2600 for (q, u) in booked)
    assert book.open == len(booked) and settled > 0


def test_violations_and_log_score():
    st = ST
    p = st.bits[:, 100:200].T.astype(float)             # truth is fully coherent
    v = violations(p, st)
    for k, d in v.items():
        assert d["mean"] == 0.0, k
    v = violations(np.full((1, st.Q), 0.5), st, t=np.array([st.t0]))
    assert v["taut_or"]["n"] == 0                       # none released at t0
    v = violations(np.full((1, st.Q), 0.5), st, t=np.array([T - 1]))
    assert abs(v["taut_or"]["mean"] - 0.5) < 1e-12 and abs(v["ex3"]["mean"] - 0.5) < 1e-12
    assert v["pre:taut_or"]["n"] == 0                   # all have resolved history by T-1
    q = np.where(st.kind == "taut_or")[0][0]
    v = violations(np.full((1, st.Q), 0.5), st, t=np.array([st.release[q]]))
    assert v["pre:taut_or"]["n"] >= 1
    assert abs(log_score_gain(np.array(0.8), np.array(1)) - np.log(1.6)) < 1e-12


def test_positions_adapter_runs_prequential():
    import torch
    from prequential import run_prequential
    from model import Logistic
    st = make_li_stream(0, 1200)
    ps = st.as_positions()
    assert (ps.window([5]).numpy()[0] == st.window(5 + st.W)[0]).all()
    lm = Logistic(ps.Q, ps.W)
    out = run_prequential(lm, ps, torch.optim.Adam(lm.parameters(), 1e-2), B=8)
    # every live, in-stream-revealed (i, q) was predicted before trained, trained once
    i, q = np.nonzero(out["mask"])
    rev = i + st.W + st.delay[q] < st.T - 8
    assert (out["train_step"][i[rev], q[rev]] >= out["pred_step"][i[rev], q[rev]]).all()
