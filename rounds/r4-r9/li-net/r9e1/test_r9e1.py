"""Unit tests for R9-E1 (CPU). Run: cd li-net/r9e1; PYTHONPATH=/tmp/r4e3a/pylib:.:../r8e1 python3 -m pytest -q test_r9e1.py"""
import os
import sys
import tempfile

import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e1v6 as E  # noqa: E402
from e1v6 import A2, E3, E4, E5, RO, env6, wm  # noqa: E402


def test_select_strong_smallest_under_threshold_else_largest():
    assert E.select_strong({"30": 0.045, "32": 0.030, "34": 0.015}, (30, 32, 34), 0.035) == 32
    assert E.select_strong({"30": 0.030, "32": 0.020, "34": 0.010}, (30, 32, 34), 0.035) == 30
    assert E.select_strong({"30": 0.060, "32": 0.050, "34": 0.040}, (30, 32, 34), 0.035) == 34
    assert E.select_strong({"30": 0.060, "32": 0.035, "34": 0.010}, (30, 32, 34), 0.035) == 32      # <= is inclusive


def test_econ_prec_identical_cluster_closed_form():
    """N identical reads, no background: W(N) = N u beta / (lambda(beta) + N beta), lambda(beta) = softplus(a - b beta)."""
    u, a, b = 1.3, 2.0, 1.2
    for bval in (0.8, 1.5, 3.4):
        lam = float(E.econ_prec_lambda(torch.tensor(bval), a, b))
        for N in (1, 4, 16):
            k = F.normalize(torch.randn(1, 1, 16), dim=-1).expand(1, N, 16).contiguous()
            w = E.econ_prec_weights(torch.full((1, N), bval), k, torch.ones(1, N, dtype=torch.bool), u, a, b)
            assert abs(float(w.sum()) - N * u * bval / (lam + N * bval)) < 1e-5
    W = lambda N, bv: N * bv / (float(E.econ_prec_lambda(torch.tensor(bv), a, b)) + N * bv)
    assert W(16, 3.4) / W(1, 3.4) < W(16, 0.8) / W(1, 0.8)              # a smaller ridge at high precision dedups more
    assert float(E.econ_prec_lambda(torch.tensor(0.5), 4.0, 0.1)) > 2.5   # no ceiling at 2 (amended reading 5)


def test_econ_prec_reduces_to_econ_kc_at_a_constant_ridge():
    torch.manual_seed(0)
    k = F.normalize(torch.randn(3, 7, 16), dim=-1)
    beta = torch.rand(3, 7) * 3 + 0.5
    valid = torch.ones(3, 7, dtype=torch.bool)
    valid[0, -2:] = False
    lam = 1.37
    a = float(torch.log(torch.expm1(torch.tensor(lam))))               # softplus(a) = lam
    w1 = E.econ_prec_weights(beta, k, valid, 0.9, a, 0.0)
    w2 = E4.econ_weights(beta, k, valid, 0.9, lam)
    assert torch.allclose(w1, w2, atol=1e-6)
    dl = torch.randn(3, 7, 36)
    l0 = torch.randn(3, 36)
    o1, _ = E.agg_econ_prec(l0, dl, beta, k, valid, 0.9, a, 0.0)
    o2, _ = E4.agg_econ(l0, dl, beta, k, valid, 0.9, lam)
    assert torch.allclose(o1, o2, atol=1e-5)


def _tiny_setup(nb=6):
    torch.manual_seed(0)
    m = wm.WorldModel(d=32, layers=2, heads=2, ff=64).eval()
    B = A2.boards2_t(A2.STREAM_TEST2, nb, "cpu")
    ordT = torch.as_tensor(E5.anchor_order(A2.STREAM_TEST2, nb))
    ev, l0 = E5.root(m, B, ordT, 2)
    lprior = E5.prior_logits(m, B)
    trust = wm.TrustNet(32 + 3).eval()
    anyp = {"looped_attn": (E3.KeyNet("K3", 32).eval(), E5.LoopedAttnT3P(3.4).eval()),
            "attn_K3": (E3.KeyNet("K3", 32).eval(), E5.AttnL3P(3.4).eval())}
    sc = {"alpha": 0.2, "eta": 0.08, "u_K1": 1.0, "u_K3": 0.5, "econ_R7_u": 1.3, "econ_R7_lam": 1.4, "econ_D3_u": 1.0,
          "econ_D3_lam": 1.0, "econ_prec_u": 1.1, "econ_prec_a": 1.0, "econ_prec_b": 0.6}
    P = E.Pools9(anyp, E3.KeyNet("K1", 32).eval(), wm.AttnL3().eval(), E3.KeyNet("K4", 32).eval(), E3.LoopedLinT4().eval(),
                 E3.KeyNet("K3", 32).eval(), E4.JacobiLinT4().eval(), E3.KeyNet("K3", 32).eval(), E5.JacobiPrecT4().eval(), sc, 3.4)
    return m, B, ordT, ev, l0, lprior, trust, P


def test_pools9_k0_returns_root_and_runs_with_reads():
    m, B, ordT, ev, l0, lprior, trust, P = _tiny_setup()
    for M in (2, 32):
        evM, l0M = E5.root(m, B, ordT, M)
        e = torch.zeros(6, 0, 36)
        out, _ = P.all(l0M, lprior, M, e, e, torch.zeros(6, 0), torch.zeros(6, 0, 32), torch.zeros(6, 0, dtype=torch.bool))
        for a in E.POOL_NAMES:
            assert torch.equal(out[a], l0M), (a, M)                    # K = 0: every pool returns the root exactly
    pt, _ = RO.sample_patches(l0, 5, torch.full((6,), 0.9), torch.Generator().manual_seed(1), 4.0)
    ro = E3.bag_reads3(m, B, ev, l0, pt, torch.full(pt.shape, 1.5), torch.Generator().manual_seed(2), 4096)
    ro["beta"] = E3.beta_of(trust, ro)
    out, W = P.all(l0, lprior, 2, ro["lr"], ro["dl"], ro["beta"], ro["hbar"], torch.ones(6, 5, dtype=torch.bool), need_w=True)
    for a in E.POOL_NAMES:
        assert out[a].shape == (6, 36) and bool(torch.isfinite(out[a]).all()), a
    for a in E.WPOOLS:
        assert W[a].shape == (6, 5, 36), a
    k3, _ = P.any["attn_K3"][0].embed(None, ro["dl"])
    w = E.econ_prec_weights(ro["beta"], k3, torch.ones(6, 5, dtype=torch.bool), 1.1, 1.0, 0.6)
    assert torch.allclose(out["econ_prec"], l0 + (w[..., None] * ro["dl"]).sum(1), atol=1e-5)


def test_mode_b_adds_bounded_read_increments_to_the_strong_root():
    """Reading 3: Mode B reads are the standard reads (2 anchors + patch); every pool adds their Delta l over l0(2) to l0(M_0);
    mix averages the reads' beliefs (it ignores the root)."""
    m, B, ordT, ev, l0, lprior, trust, P = _tiny_setup()
    ev32, l032 = E5.root(m, B, ordT, 32)
    pt, _ = RO.sample_patches(l032, 4, torch.full((6,), 0.9), torch.Generator().manual_seed(3), 4.0)
    ro = E3.bag_reads3(m, B, ev, l0, pt, torch.full(pt.shape, 0.3), torch.Generator().manual_seed(4), 4096)
    ro["beta"] = E3.beta_of(trust, ro)
    assert torch.allclose(ro["dl"], ro["lr"] - l0[:, None], atol=1e-6)
    res = E5.run_prefixes(P, l032, lprior, 32, ro, (0, 2, 4))
    assert torch.equal(res[0]["mmd"], l032) and torch.equal(res[0]["looped_attn"], l032)
    v = torch.ones(6, 4, dtype=torch.bool)
    assert torch.allclose(res[4]["mmd"], wm.agg_mmd(l032, ro["dl"], ro["beta"], v, P.sc["eta"]), atol=1e-5)
    assert torch.allclose(res[4]["mix"], wm.agg_mix(ro["lr"], v), atol=1e-5)
    # Mode A at M_0 = 32: patch cells the root has seen keep the root's readings; a read inside the root changes nothing
    inside = torch.zeros(6, 1, dtype=torch.long)
    for r in range(6):
        seen = set(ordT[r, :32].tolist())
        ids = [p for p in range(env6.NPATCH) if set(RO.geo("cpu")["patches"][p].tolist()) <= seen]
        inside[r, 0] = ids[0] if ids else 0
    roA = E3.bag_reads3(m, B, ev32, l032, inside, torch.full(inside.shape, 2.5), torch.Generator().manual_seed(5), 4096)
    ok = [r for r in range(6) if set(RO.geo("cpu")["patches"][int(inside[r, 0])].tolist()) <= set(ordT[r, :32].tolist())]
    assert ok and float(roA["dl"][ok].abs().max()) < 1e-5


def _syn_any(nM=7, na=5, R=12):
    L0S = torch.arange(nM).float()[:, None, None].expand(nM, na, 36).clone() * 10 + torch.arange(na).float()[None, :, None]
    DLS = torch.arange(nM).float()[:, None, None, None].expand(nM, na, R, 36).clone() * 100 + torch.arange(R).float()[None, None, :, None]
    BES = torch.arange(nM).float()[:, None, None].expand(nM, na, R).clone() + 0.5
    REFS = torch.rand(nM, na, 36)
    return L0S, torch.zeros(na, 36), DLS, BES, REFS


def test_any_bag9_modes_take_the_right_reads_anchor_and_target():
    L0S, LPa, DLS, BES, REFS = _syn_any(R=64)
    M0T = torch.tensor(E.M0S, dtype=torch.float32)
    KST = torch.tensor(E.KS_TRAIN)
    g = torch.Generator().manual_seed(0)
    for force in (None, "A", "B"):
        l0, lp, dl, be, valid, b0, y, Mi, K, mB = E.any_bag9(L0S, LPa, DLS, BES, REFS, M0T, KST, 3.0, 400, 0, 5, g, 32, 32, 0.5,
                                                            "cpu", force)
        src_obs = (dl[:, 0, 0] // 100).long()
        assert torch.equal(src_obs, torch.where(mB, torch.zeros_like(Mi), Mi))   # Mode B: the standard (M_0 = 2) reads
        assert torch.equal((l0[:, 0] // 10).long(), Mi)                          # the anchor is always l0(M_0)
        assert torch.allclose(b0, M0T[Mi] * 3.0 / 6.0)
        assert not bool((mB & (Mi == 0)).any())                                  # at M_0 = 2 the modes coincide (always A)
        assert torch.equal(valid.sum(1), K)
        if force == "A":
            assert not bool(mB.any())
        if force == "B":
            assert torch.equal(mB, Mi > 0)
        if force is None:
            share = float(mB[Mi > 0].float().mean())
            assert 0.35 < share < 0.65


def test_pfs_bag_prefixes():
    g = torch.Generator().manual_seed(0)
    PDL = torch.randn(2, 6, 32, 36)
    PBE = torch.rand(2, 6, 32)
    l0, dl, be, valid, y = E.pfs_bag(torch.randn(6, 36), PDL, PBE, torch.rand(6, 36), 300, 0, 4, g, "cpu")
    ks = valid.sum(1)
    assert int(ks.min()) >= 1 and int(ks.max()) <= 32 and len(set(ks.tolist())) > 20
    assert bool((valid[:, :-1].int() >= valid[:, 1:].int()).all())                # prefixes


def test_load_checked_rejects_a_hash_mismatch():
    torch.manual_seed(0)
    m = wm.WorldModel(d=32, layers=2, heads=2, ff=64).eval()
    with tempfile.TemporaryDirectory() as d:
        E4.save_wm_npz(m, os.path.join(d, "s1", "theta_real_v2.npz"), {"seed": 1})
        m2, meta, ck = E5.load_checked(d, 1, "cpu")
        assert ck["match"] and ck["n_params"] == m.n_params()
        with open(os.path.join(d, "s1", "theta_real_v2.npz"), "ab") as f:
            f.write(b"x")
        with pytest.raises(Exception):
            E5.load_checked(d, 1, "cpu")


def _tiny_cfg(out):
    return E.Cfg9(out=out, device="cpu", threads=2, d=32, n_cal=40, n_any=20, n_d3=20, n_pfs=20, pool=32, pool_any=32, n_ref=16,
                  agg_steps=12, agg_batch=16, trust_steps=10, fit_steps=6, n_test=6)


@pytest.fixture(scope="module")
def tiny_fit():
    torch.manual_seed(0)
    m = wm.WorldModel(d=32, layers=2, heads=2, ff=64).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    with tempfile.TemporaryDirectory() as d:
        cfg = _tiny_cfg(d)
        orig = env6.bayes_marginals

        def boom(*a, **k):
            raise AssertionError("exact Bayes called during calibration")
        env6.bayes_marginals = boom
        try:
            gen = torch.Generator().manual_seed(1)
            trust, P, info = E.fit9(m, cfg, torch.device("cpu"), lambda *a: None, gen)
        finally:
            env6.bayes_marginals = orig
        yield m, cfg, trust, P, info, gen


def test_fit9_runs_without_exact_bayes_and_selects_a_candidate(tiny_fit):
    m, cfg, trust, P, info, gen = tiny_fit
    assert info["M_strong"] in (30, 32, 34)
    assert set(info["strong_selection"]["kl_ref36_root"]) == {"30", "32", "34"}
    assert info["params"]["jacobi_prec"] == 814 and info["params"]["econ_prec"] == 3
    for k in ("econ_prec_u", "econ_prec_a", "econ_prec_b", "econ_D3_u", "econ_D3_lam", "eta", "alpha"):
        assert k in info["scalars"]
    assert info["scalars"]["econ_prec_b"] >= 0
    assert any(k.endswith("_B") for k in info["heldout_any_bce_to_ref"])


def test_test9_reports_every_rule_input(tiny_fit):
    m, cfg, trust, P, info, gen = tiny_fit
    Pf = torch.as_tensor(env6.placements()).double()
    R, raw = E.test9(m, trust, P, cfg, torch.device("cpu"), lambda *a: None, gen, Pf, info["M_strong"])
    Ms = info["M_strong"]
    for M in E.M0S:
        for a in E.ANY:
            assert R["identity_K0"][f"{a}_M{M}_kl_diff"] < 1e-7
    for md in ("A", "B"):
        for M in (18, 30, 32, 34):
            for a in E.POOL_NAMES:
                for K in E.KS_SR:
                    assert f"sr{md}{M}_kl_{a}_K{K}" in R
        assert R[f"sr{md}{Ms}_kl_looped_attn_K0"] == pytest.approx(R["kl_root"][str(Ms)], abs=1e-6)
    for a in E.POOL_NAMES:
        for K in E.KS_FS:
            assert f"fs_kl_{a}_K{K}" in R
    for t in (0.3, 1.5, 2.5):
        for N in (1, 2, 4, 8, 16):
            assert "W_econ_prec" in R["controlled"][f"t{t}_N{N}"] and "mse_econ_prec" in R["controlled"][f"t{t}_N{N}"]
    assert "srSA_NW" in raw and "srSB_SE" in raw and "K24_jacobi_prec" in raw["fs_NW"]
