"""Unit tests for R8-E1 (CPU). Run: cd li-net/r8e1; PYTHONPATH=/tmp/r4e3a/pylib:. python3 -m pytest -q test_r8e1.py"""
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e1v5 as E  # noqa: E402
import e1v3 as E3  # noqa: E402
import e1v4 as E4  # noqa: E402
import partA2 as A2  # noqa: E402
import rollouts as RO  # noqa: E402
import wm  # noqa: E402


def test_anchor_order_reproduces_standard_anchors_and_nests():
    for stream in (A2.STREAM_CAL2, A2.STREAM_TEST2):
        order = E.anchor_order(stream, 50)
        assert (order[:, :2] == A2.make_boards2(stream, 50)["anchors"]).all()
        assert all(sorted(r.tolist()) == list(range(36)) for r in order)
    a = E.anchor_order(A2.STREAM_TEST2, 50)
    b = E.anchor_order(A2.STREAM_TEST2, 50)
    assert (a == b).all()                                              # fixed per board, independent of any seed


def _bag(B=3, K=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    l0 = torch.randn(B, 36, generator=g)
    lprior = torch.randn(B, 36, generator=g)
    dl = torch.randn(B, K, 36, generator=g) * 2
    beta = torch.rand(B, K, generator=g) * 3 + 0.5
    valid = torch.ones(B, K, dtype=torch.bool)
    valid[0, -2:] = False
    return l0, lprior, dl, beta, valid


def test_anchor_token_identity_at_k0_and_zero_shift():
    torch.manual_seed(0)
    for pool in (E4.LoopedAttnT3(), wm.AttnL3()):
        keyn = E3.KeyNet("K3", 32)
        with torch.no_grad():
            pool.out.weight.normal_(0, 0.2)
        l0, lprior, dl, beta, valid = _bag()
        b0 = torch.full((3,), 5.0)
        e = torch.zeros(3, 0, 36)
        dla, ba, k, va = E.anchor_bag(keyn, l0, lprior, e, torch.zeros(3, 0), torch.zeros(3, 0, dtype=torch.bool), b0)
        out, w = pool(l0, dla, ba, k, va)
        assert torch.equal(out, l0)                                    # K = 0: exactly the root, bitwise
        dla, ba, k, va = E.anchor_bag(keyn, l0, lprior, dl, beta, valid, b0)
        assert dla.shape == (3, 8, 36) and float(dla[:, 0].abs().max()) == 0.0 and bool(va[:, 0].all())
        out, w = pool(l0, dla, ba, k, va)
        assert torch.allclose(out, l0 + (w[:, 1:] * dl.clamp(-8, 8)).sum(1), atol=1e-5)


def test_jacobi_prec_params_init_and_coef():
    J = E.JacobiPrecT4(T=4)
    assert E3.nparams(J) + E3.nparams(E3.KeyNet("K3", 128)) == 814
    lam = J.lam_rc(torch.tensor([[0.5, 2.0, 4.0]]))
    assert torch.allclose(lam, torch.ones_like(lam), atol=1e-6)          # lambda = 1 at init for every beta
    l0, lprior, dl, beta, valid = _bag(seed=1)
    k = F.normalize(torch.randn(3, 7, 16), dim=-1)
    with torch.no_grad():
        J.eta.uniform_(0.3, 1.2)
        J.P.uniform_(0.5, 1.5)
        J.s.uniform_(0.2, 1.5)
        J.a.fill_(1.0)
        J.b.fill_(0.7)
    out, x = J(l0, dl, beta, k, valid)
    w = J.coef(dl, beta, k, valid)
    assert torch.allclose(out, l0 + (w * dl.clamp(-8, 8)).sum(1), atol=1e-4)
    assert float(w[~valid].abs().max()) == 0.0


def test_jacobi_prec_identical_cluster_uses_read_precision():
    """N identical reads: cluster weight N beta / (lambda(beta) + N beta), lambda(beta) = softplus(l~) sigmoid(a - b beta)."""
    J = E.JacobiPrecT4(T=4)
    with torch.no_grad():
        J.a.fill_(2.0)
        J.b.fill_(1.5)
    for bval in (0.8, 3.0):
        lam = float(J.lam_rc(torch.tensor([[bval]]))[0, 0, 0])
        for N in (1, 4, 16):
            k = F.normalize(torch.randn(1, 1, 16), dim=-1).expand(1, N, 16).contiguous()
            dl = torch.randn(1, 1, 36).expand(1, N, 36).contiguous()
            beta = torch.full((1, N), bval)
            w = J.coef(dl, beta, k, torch.ones(1, N, dtype=torch.bool))
            assert torch.allclose(w[0].sum(0), torch.full((36,), N * bval / (lam + N * bval)), atol=1e-5)


def _tiny_setup(nb=6):
    torch.manual_seed(0)
    m = wm.WorldModel(d=32, layers=2, heads=2, ff=64).eval()
    B = A2.boards2_t(A2.STREAM_TEST2, nb, "cpu")
    ordT = torch.as_tensor(E.anchor_order(A2.STREAM_TEST2, nb))
    ev, l0 = E.root(m, B, ordT, 2)
    lprior = E.prior_logits(m, B)
    trust = wm.TrustNet(32 + 3).eval()
    anyp = {"looped_attn": (E3.KeyNet("K3", 32).eval(), E4.LoopedAttnT3().eval()),
            "attn_K3": (E3.KeyNet("K3", 32).eval(), wm.AttnL3().eval())}
    sc = {"alpha": 0.2, "eta": 0.08, "u_K1": 1.0, "u_K3": 0.5, "econ_R7_u": 1.3, "econ_R7_lam": 1.4, "econ_D3_u": 1.0,
          "econ_D3_lam": 1.0}
    P = E.Pools8(anyp, E3.KeyNet("K1", 32).eval(), wm.AttnL3().eval(), E3.KeyNet("K4", 32).eval(), E3.LoopedLinT4().eval(),
                 E3.KeyNet("K3", 32).eval(), E4.JacobiLinT4().eval(), E3.KeyNet("K3", 32).eval(), E.JacobiPrecT4().eval(), sc, 3.4)
    cfg = E.Cfg8(device="cpu", chunk=4096, d=32)
    return m, B, ordT, ev, l0, lprior, trust, P, cfg


def test_root_with_more_anchors_is_nested_and_m2_is_standard():
    m, B, ordT, ev, l0, lprior, trust, P, cfg = _tiny_setup()
    ev_std = A2.ev_from(B["s"], A2.cells_mask(B["anchors"], "cpu"))
    assert torch.equal(ev, ev_std)
    ev18, _ = E.root(m, B, ordT, 18)
    assert int((ev18 > 0).sum(1).min()) == 18 and bool(((ev > 0) <= (ev18 > 0)).all())


def test_pools_all_k0_returns_root_and_runs_with_reads():
    m, B, ordT, ev, l0, lprior, trust, P, cfg = _tiny_setup()
    e = torch.zeros(6, 0, 36)
    out, _ = P.all(l0, lprior, 2, e, e, torch.zeros(6, 0), torch.zeros(6, 0, 32), torch.zeros(6, 0, dtype=torch.bool))
    for a in E.POOL_NAMES:
        assert torch.equal(out[a], l0), a
    pt, _ = RO.sample_patches(l0, 5, torch.full((6,), 0.9), torch.Generator().manual_seed(1), 4.0)
    ro = E3.bag_reads3(m, B, ev, l0, pt, torch.full(pt.shape, 1.5), torch.Generator().manual_seed(2), 4096)
    ro["beta"] = E3.beta_of(trust, ro)
    out, W = P.all(l0, lprior, 2, ro["lr"], ro["dl"], ro["beta"], ro["hbar"], torch.ones(6, 5, dtype=torch.bool), need_w=True)
    for a in E.POOL_NAMES:
        assert out[a].shape == (6, 36) and bool(torch.isfinite(out[a]).all()), a
    for a in E.WPOOLS:
        assert W[a].shape == (6, 5, 36), a


@pytest.mark.parametrize("km,sm", E.CELLS)
def test_puct8_never_uses_unread_real_readings(km, sm):
    m, B, ordT, ev, l0, lprior, trust, P, cfg = _tiny_setup()
    anchors = A2.cells_mask(B["anchors"], "cpu").bool()
    _, picks = E.puct8(m, trust, P, B, ev, l0, lprior, True, torch.Generator().manual_seed(5), cfg, km, sm, 4)
    for p in picks:
        assert all(len(set(r.tolist())) == 6 for r in p)
    for w in range(1, 5):
        cov = E3.coverage(torch.cat(picks[:w - 1], 1), "cpu").bool() if w > 1 else torch.zeros(6, 36, dtype=torch.bool)
        B2 = dict(B)
        B2["s"] = B["s"] ^ (~cov & ~anchors)
        ev2 = A2.ev_from(B2["s"], A2.cells_mask(B2["anchors"], "cpu"))
        assert torch.equal(ev2, ev)
        _, p2 = E.puct8(m, trust, P, B2, ev2, l0, lprior, True, torch.Generator().manual_seed(5), cfg, km, sm, w)
        for i in range(w):
            assert torch.equal(picks[i], p2[i])


def test_load_checked_rejects_a_hash_mismatch():
    torch.manual_seed(0)
    m = wm.WorldModel(d=32, layers=2, heads=2, ff=64).eval()
    with tempfile.TemporaryDirectory() as d:
        E4.save_wm_npz(m, os.path.join(d, "s1", "theta_real_v2.npz"), {"seed": 1})
        m2, meta, ck = E.load_checked(d, 1, "cpu")
        assert ck["match"] and ck["n_params"] == m.n_params()
        with open(os.path.join(d, "s1", "theta_real_v2.npz"), "ab") as f:
            f.write(b"x")
        with pytest.raises(Exception):
            E.load_checked(d, 1, "cpu")


def test_precision_weighted_crowding_and_anchor_identity():
    torch.manual_seed(0)
    k = F.normalize(torch.randn(2, 5, 16), dim=-1)
    valid = torch.ones(2, 5, dtype=torch.bool)
    beta = torch.full((2, 5), 3.0)
    assert torch.allclose(E.crowding_prec(k, valid, beta, 3.0), wm.crowding(k, valid), atol=1e-6)
    beta2 = beta.clone()
    beta2[:, 0] = 9.0                                                   # a heavier anchor crowds every overlapping read more
    c1, c2 = E.crowding_prec(k, valid, beta, 3.0), E.crowding_prec(k, valid, beta2, 3.0)
    assert bool((c2 >= c1 - 1e-6).all())
    for pool in (E.LoopedAttnT3P(3.4), E.AttnL3P(3.4)):
        keyn = E3.KeyNet("K3", 32)
        with torch.no_grad():
            pool.out.weight.normal_(0, 0.2)
        l0, lprior, dl, beta, valid = _bag()
        e = torch.zeros(3, 0, 36)
        dla, ba, kk, va = E.anchor_bag(keyn, l0, lprior, e, torch.zeros(3, 0), torch.zeros(3, 0, dtype=torch.bool), torch.full((3,), 10.2))
        out, _ = pool(l0, dla, ba, kk, va)
        assert torch.equal(out, l0)
        dla, ba, kk, va = E.anchor_bag(keyn, l0, lprior, dl, beta, valid, torch.full((3,), 10.2))
        out, w = pool(l0, dla, ba, kk, va)
        assert bool(torch.isfinite(out).all()) and float(w[:, 1:][~valid].abs().max()) == 0.0


def test_fs_stream_interleaves_one_tile_per_block():
    n = 4
    tl6 = torch.tensor(E3.TILES)[torch.rand(n, 6).argsort(1)]
    tiles8 = torch.cat([tl6, tl6[:, :2]], 1)
    sk = torch.randint(0, 40, (n, 24))
    st = E.fs_stream(tiles8, sk)
    assert st.shape == (n, 32)
    assert torch.equal(st[:, 0::4], tiles8)
    assert torch.equal(st[:, [i for i in range(32) if i % 4]], sk)
    first24 = st[:, :24]
    for r in range(n):
        assert sorted(first24[r, 0::4].tolist()) == sorted(E3.TILES)
        assert sorted(first24[r, [i for i in range(24) if i % 4]].tolist()) == sorted(sk[r, :18].tolist())
