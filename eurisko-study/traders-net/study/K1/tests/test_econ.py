import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import econ


def test_determinism_and_conservation():
    for nic, Q in [(True, 64), (False, 64), (False, 1)]:
        a = econ.run(N=300, Q=Q, niches=nic, seed=3, T=400, verbose=False, runlog=False)
        b = econ.run(N=300, Q=Q, niches=nic, seed=3, T=400, verbose=False, runlog=False)
        ks = [k for k in a['final'] if k != 'wall_s']
        assert np.allclose([a['final'][k] for k in ks], [b['final'][k] for k in ks], equal_nan=True, rtol=0, atol=0)
        f = a['final']
        assert f['cons_resid_ledger'] < 1e-9 and f['cons_resid_settle'] < 1e-9


def test_null_breaks_skill_on_q1():
    s = econ.run(N=300, Q=1, niches=False, seed=1, T=1500, verbose=False, runlog=False)
    assert s['tail_brier_all'] < s['tail_brier_all_null']


def test_kelly_bayes_identity():
    # stake=1, no cost, no death/split, maker negligible: wealth ratio = likelihood ratio of beliefs vs price
    e = econ.Economy(2, 1, 1, 0, niches=False, c=0.0, subsidy=1e-12, split_at=1e9, die_at=0.0)
    st = econ.Stream(1, 0, batch=1)
    for t in range(50):
        bq, ctx, y, pt = st.step(t)
        w0 = e.w.copy()
        I, J = e.pairs(bq)
        b = np.clip(e.beliefs(I, ctx[J] & econ.MASK[e.k[I]]), e.eps, 1 - e.eps)
        p = (w0 * b).sum() / w0.sum()
        e.step(bq, ctx, y, pt, st.fam)
        exp = w0 * np.where(y[0], b / p, (1 - b) / (1 - p))
        assert np.allclose(e.w, exp, rtol=1e-6)


def test_dense_equals_pair_path():
    st1, st2 = econ.Stream(64, 5), econ.Stream(64, 5)
    a = econ.Economy(200, 64, 16, 5, niches=False, dense=True)
    b = econ.Economy(200, 64, 16, 5, niches=False, dense=False)
    for t in range(300):
        x1, x2 = st1.step(t), st2.step(t)
        a.step(*x1, st1.fam); b.step(*x2, st2.fam)
    assert np.allclose(a.w, b.w, rtol=1e-9, atol=1e-12) and (a.alive == b.alive).all()
    assert np.allclose(a.W, b.W) and np.allclose(a.counts, b.counts) and np.allclose(a.cum_wf, b.cum_wf)


if __name__ == '__main__':
    test_determinism_and_conservation(); test_null_breaks_skill_on_q1(); test_kelly_bayes_identity()
    test_dense_equals_pair_path(); print('4/4 PASS')
