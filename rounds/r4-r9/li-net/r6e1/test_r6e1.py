"""Unit tests for R6-E1 (CPU). Run: cd li-net/r6e1; PYTHONPATH=/tmp/r4e3a/pylib:. python3 -m pytest -q test_r6e1.py"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e1v3 as E  # noqa: E402
import env6  # noqa: E402
import partA2 as A2  # noqa: E402
import wm  # noqa: E402


def test_tiles_partition_board():
    cells = sorted(c for t in E.TILES for c in env6.PATCHES[t])
    assert cells == list(range(36))


def test_key_variants_shapes_and_centring():
    torch.manual_seed(0)
    B, K = 3, 7
    hbar = torch.randn(B, K, 128)
    dl = torch.randn(B, K, 36)
    valid = torch.ones(B, K, dtype=torch.bool)
    valid[0, -2:] = False
    for v in E.VARIANTS:
        kn = E.KeyNet(v)
        k, s = kn(hbar, dl, valid)
        assert k.shape == (B, K, 16) and torch.allclose(k.norm(dim=-1), torch.ones(B, K), atol=1e-5)
        assert torch.allclose(s.norm(dim=-1), torch.ones(B, K), atol=1e-4) and bool((s >= 0).all())
    # K2/K4 are invariant to a bag-wide shift of hbar (bag centring); K1 is not
    shift = torch.randn(B, 1, 128) * 5
    for v, inv in (("K1", False), ("K2", True), ("K4", True)):
        kn = E.KeyNet(v)
        a, _ = kn(hbar, dl, valid)
        b, _ = kn(hbar + shift, dl, valid)
        assert torch.allclose(a[valid], b[valid], atol=1e-4) == inv
    # padding does not enter the bag mean
    kn = E.KeyNet("K2")
    a, _ = kn(hbar, dl, valid)
    h2 = hbar.clone()
    h2[0, -2:] = 1e3
    b, _ = kn(h2, dl, valid)
    assert torch.allclose(a[0, :5], b[0, :5], atol=1e-5)


def test_key_align_loss_zero_when_keys_match_salience():
    s = torch.nn.functional.normalize(torch.rand(2, 5, 16), dim=-1)
    valid = torch.ones(2, 5, dtype=torch.bool)
    assert float(E.key_align_loss(s, s, valid)) < 1e-10


def test_looped_lin_ignores_padding_and_is_small():
    torch.manual_seed(0)
    lp = E.LoopedLinT4(T=4, gamma=2.0, init_s=1.0)
    l0 = torch.zeros(1, 36)
    dl = torch.randn(1, 1, 36).expand(1, 6, 36).contiguous()
    beta = torch.ones(1, 6)
    k = torch.nn.functional.normalize(torch.randn(1, 1, 16), dim=-1).expand(1, 6, 16).contiguous()
    v1 = torch.zeros(1, 6, dtype=torch.bool)
    v1[0, 0] = True
    out1, x1 = lp(l0, dl, beta, k, v1)
    assert float(x1[0, 1:].abs().max()) == 0.0                      # padded reads contribute nothing
    assert E.nparams(lp) + E.nparams(E.KeyNet("K4")) < 8000


def test_prospective_reads_never_use_real_readings():
    torch.manual_seed(0)
    m = wm.WorldModel(d=32, layers=2, heads=2, ff=64).eval()
    B = A2.boards2_t(41, 3, "cpu")
    ev_root = A2.ev_from(B["s"], A2.cells_mask(B["anchors"], "cpu"))
    import rollouts as RO
    l0 = RO.forward_chunks(m, ev_root, B["note"])
    mu = torch.sigmoid(l0)
    hb1, dl1 = E.prospect(m, B, ev_root, l0, mu)
    B2 = dict(B)
    B2["s"] = ~B["s"]                                                # flip every real reading
    ev_root2 = A2.ev_from(B["s"], A2.cells_mask(B["anchors"], "cpu"))  # same root (anchors are root evidence)
    hb2, dl2 = E.prospect(m, B2, ev_root2, l0, mu)
    assert torch.allclose(hb1, hb2) and torch.allclose(dl1, dl2)
    assert hb1.shape == (3, 40, 32) and dl1.shape == (3, 40, 36)


def test_tau_tiers_flip_rates():
    torch.manual_seed(0)
    m = wm.WorldModel(d=32, layers=2, heads=2, ff=64).eval()
    B = A2.boards2_t(42, 1, "cpu")
    ev_root = A2.ev_from(B["s"], A2.cells_mask(B["anchors"], "cpu"))
    g = torch.Generator().manual_seed(1)
    for tv, d in ((0.3, 0.0), (1.5, 0.127), (2.5, 0.28)):
        tau = torch.full((20000,), tv)
        flip = torch.rand(20000, 36, generator=g) < E.delta_of(tau)[:, None]
        assert abs(float(flip.float().mean()) - d) < 0.005
