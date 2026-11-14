"""Unit tests for R7-E1 (CPU). Run: cd li-net/r7e1; PYTHONPATH=/tmp/r4e3a/pylib:. python3 -m pytest -q test_r7e1.py"""
import os
import sys
import tempfile
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e1v4 as E  # noqa: E402
import e1v3 as E3  # noqa: E402
import partA2 as A2  # noqa: E402
import rollouts as RO  # noqa: E402
import wm  # noqa: E402


def _bag(B=3, K=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = F.normalize(torch.randn(B, K, 16, generator=g), dim=-1)
    dl = torch.randn(B, K, 36, generator=g) * 2
    beta = torch.rand(B, K, generator=g) * 3.5 + 0.5
    valid = torch.ones(B, K, dtype=torch.bool)
    valid[0, -3:] = False
    valid[2, -1] = False
    l0 = torch.randn(B, 36, generator=g)
    return l0, dl, beta, k, valid


def test_jacobi_identical_cluster_is_precision_crowding():
    """N identical reads: one Jacobi step already solves the normal equations; cluster weight = N beta / (lambda + N beta)."""
    torch.manual_seed(0)
    J = E.JacobiLinT4(T=4)
    lam = float(J.lam()[0])
    assert abs(lam - 1.0) < 1e-6
    for N in (1, 2, 4, 8, 16):
        k = F.normalize(torch.randn(1, 1, 16), dim=-1).expand(1, N, 16).contiguous()
        dl = torch.randn(1, 1, 36).expand(1, N, 36).contiguous()
        beta = torch.full((1, N), 2.5)
        valid = torch.ones(1, N, dtype=torch.bool)
        out, _ = J(torch.zeros(1, 36), dl, beta, k, valid)
        Wn = N * 2.5 / (lam + N * 2.5)
        assert torch.allclose(out[0], Wn * dl[0, 0], atol=1e-5)
        w = J.coef(dl, beta, k, valid)
        assert torch.allclose(w[0].sum(0), torch.full((36,), Wn), atol=1e-5)


def test_jacobi_first_step_equals_econ_kc():
    l0, dl, beta, k, valid = _bag()
    dl = dl.clamp(-8, 8)
    J1 = E.JacobiLinT4(T=1)
    out, _ = J1(l0, dl, beta, k, valid)
    ref, _ = E.agg_econ(l0, dl, beta, k, valid, 1.0, 1.0)
    assert torch.allclose(out, ref, atol=1e-4)


def test_jacobi_coef_reproduces_forward_and_ignores_padding():
    l0, dl, beta, k, valid = _bag(seed=1)
    J = E.JacobiLinT4(T=4)
    with torch.no_grad():
        J.eta.uniform_(0.3, 1.2)
        J.P.uniform_(0.5, 1.5)
        J.lam_raw.uniform_(-1.0, 1.0)
        J.s.uniform_(0.2, 1.5)
    out, x = J(l0, dl, beta, k, valid)
    w = J.coef(dl, beta, k, valid)
    assert torch.allclose(out, l0 + (w * dl.clamp(-8, 8)).sum(1), atol=1e-4)
    assert float(w[~valid].abs().max()) == 0.0 and float(x[~valid].abs().max()) == 0.0
    dl2, beta2, k2 = dl.clone(), beta.clone(), k.clone()
    dl2[~valid] = 100.0
    beta2[~valid] = 50.0
    k2[~valid] = F.normalize(torch.randn(int((~valid).sum()), 16), dim=-1)
    out2, _ = J(l0, dl2, beta2, k2, valid)
    assert torch.allclose(out, out2, atol=1e-5)
    assert E3.nparams(J) + E3.nparams(E3.KeyNet("K3")) < 3000


def test_looped_attn_is_weight_tied_small_and_ignores_padding():
    torch.manual_seed(0)
    la = E.LoopedAttnT3(T=3)
    assert sum(isinstance(m, nn.MultiheadAttention) for m in la.modules()) == 1
    head = E3.nparams(E3.KeyNet("K3"))
    assert (E3.nparams(la) + head) / (E3.nparams(wm.AttnL3()) + head) <= 0.38
    with torch.no_grad():
        la.out.weight.normal_(0, 0.1)
    l0, dl, beta, k, valid = _bag(seed=2)
    out, w = la(l0, dl, beta, k, valid)
    assert float(w[~valid].abs().max()) == 0.0
    dl2 = dl.clone()
    dl2[~valid] = 100.0
    out2, w2 = la(l0, dl2, beta, k, valid)
    assert torch.allclose(out, out2, atol=1e-5) and torch.allclose(w[valid], w2[valid], atol=1e-5)


def test_econ_weights_formula():
    l0, dl, beta, k, valid = _bag(seed=3)
    w = E.econ_weights(beta, k, valid, 0.7, 1.3)
    v = valid.float()
    g = torch.relu(k @ k.transpose(1, 2)) ** 2
    for b_ in range(3):
        for r in range(9):
            if not valid[b_, r]:
                assert float(w[b_, r]) == 0.0
                continue
            c = sum(float(beta[b_, j] * g[b_, r, j] * v[b_, j]) for j in range(9))
            assert abs(float(w[b_, r]) - 0.7 * float(beta[b_, r]) / (1.3 + c)) < 1e-5


def test_theta_real_npz_roundtrip():
    torch.manual_seed(0)
    m = wm.WorldModel(d=32, layers=2, heads=2, ff=64).eval()
    B = A2.boards2_t(41, 5, "cpu")
    ev = A2.ev_from(B["s"], A2.cells_mask(B["anchors"], "cpu"))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s1", "theta_real_v2.npz")
        meta = E.save_wm_npz(m, p, {"seed": 1, "train_curve": [{"step": 0}]})
        assert os.path.exists(p) and not os.path.exists(p + ".tmp") and os.path.exists(p[:-4] + ".json")
        assert meta["sha256"] == E.sha256_file(p) and meta["n_params"] == m.n_params()
        m2, meta2 = E.load_wm_npz(p, "cpu")
        assert meta2["sha256"] == meta["sha256"] and meta2["arch"] == {"d": 32, "layers": 2, "heads": 2, "ff": 64}
        for (a, x), (b, y) in zip(m.state_dict().items(), m2.state_dict().items()):
            assert a == b and torch.equal(x, y)
        assert torch.equal(RO.forward_chunks(m, ev, B["note"]), RO.forward_chunks(m2, ev, B["note"]))


def _puct_setup(nb=6):
    torch.manual_seed(0)
    m = wm.WorldModel(d=32, layers=2, heads=2, ff=64).eval()
    B = A2.boards2_t(41, nb, "cpu")
    ev_root = A2.ev_from(B["s"], A2.cells_mask(B["anchors"], "cpu"))
    l0 = RO.forward_chunks(m, ev_root, B["note"])
    trust = wm.TrustNet(32 + 3).eval()
    P = SimpleNamespace(keys={"K3": E3.KeyNet("K3", 32).eval()}, attn={"K3": wm.AttnL3().eval()})
    with torch.no_grad():
        P.attn["K3"].out.weight.normal_(0, 0.1)
    cfg = E.Cfg7(device="cpu", chunk=4096)
    return m, B, ev_root, l0, trust, P, cfg


def test_puct7_never_uses_unread_real_readings():
    """For each wave w, flipping the real readings of every cell no earlier wave read (anchors excluded) leaves waves 1..w
    unchanged, for power sampling and greedy selection; picks are 6 distinct patches per wave."""
    m, B, ev_root, l0, trust, P, cfg = _puct_setup()
    anchors = A2.cells_mask(B["anchors"], "cpu").bool()
    for mode in ("power", "greedy"):
        _, picks = E.puct_search7(m, trust, P, B, ev_root, l0, True, torch.Generator().manual_seed(5), cfg, mode, "mixed", 4)
        for p in picks:
            assert p.shape == (6, 6) and all(len(set(r.tolist())) == 6 for r in p)
        for w in range(1, 5):
            cov = E3.coverage(torch.cat(picks[:w - 1], 1), "cpu").bool() if w > 1 else torch.zeros(6, 36, dtype=torch.bool)
            B2 = dict(B)
            B2["s"] = B["s"] ^ (~cov & ~anchors)
            ev2 = A2.ev_from(B2["s"], A2.cells_mask(B2["anchors"], "cpu"))
            assert torch.equal(ev2, ev_root)
            _, p2 = E.puct_search7(m, trust, P, B2, ev2, l0, True, torch.Generator().manual_seed(5), cfg, mode, "mixed", w)
            for i in range(w):
                assert torch.equal(picks[i], p2[i])
    # positive control: flipping cells that wave 1 read changes later waves somewhere
    _, picks = E.puct_search7(m, trust, P, B, ev_root, l0, True, torch.Generator().manual_seed(5), cfg, "power", "mixed", 4)
    cov1 = E3.coverage(picks[0], "cpu").bool()
    B3 = dict(B)
    B3["s"] = B["s"] ^ (cov1 & ~anchors)
    _, p3 = E.puct_search7(m, trust, P, B3, ev_root, l0, True, torch.Generator().manual_seed(5), cfg, "power", "mixed", 4)
    assert torch.equal(picks[0], p3[0]) and any(not torch.equal(picks[i], p3[i]) for i in (1, 2, 3))


def test_puct7_wave2_uses_amended_score():
    """Greedy wave 2 = top 6 of (1 + lam_U / 2 1[half]) (Var / (1 + crowd)^1.5)^2 recomputed from wave 1's reads
    (amended Reading 9: preference decayed and not squared, novelty-weighted variance squared)."""
    m, B, ev_root, l0, trust, P, cfg = _puct_setup()
    bag, picks = E.puct_search7(m, trust, P, B, ev_root, l0, True, torch.Generator().manual_seed(7), cfg, "greedy", "mixed", 2)
    g = RO.geo("cpu")
    keyn, attn = P.keys["K3"], P.attn["K3"]
    b1 = E.head_ro(bag, 6)
    valid = torch.ones(6, 6, dtype=torch.bool)
    with torch.no_grad():
        k, _ = keyn(b1["hbar"], b1["dl"], valid)
        mu = torch.sigmoid(attn(l0, b1["dl"], b1["beta"], k, valid)[0])
        _, dl_c = E3.prospect(m, B, ev_root, l0, mu)
        kc_, _ = keyn.embed(None, dl_c)
    crowd = (torch.relu(kc_ @ k.transpose(1, 2)) ** 2).sum(-1)
    var = (mu * (1 - mu)) @ g["pmask"].T
    score = (1 + 1.5 * g["in_nw"].float())[None] * (var / (1 + crowd) ** 1.5) ** 2
    assert torch.equal(torch.topk(score, 6, dim=1).indices, picks[1])
    assert torch.equal(b1["patch"], picks[0])


def test_head_ro_takes_first_reads():
    ro = {"lr": torch.randn(2, 30, 36), "patch": torch.arange(60).view(2, 30), "beta": torch.rand(2, 30)}
    h = E.head_ro(ro, 24)
    assert h["lr"].shape == (2, 24, 36) and torch.equal(h["patch"], ro["patch"][:, :24]) and h["beta"].shape == (2, 24)
