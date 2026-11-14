"""Unit tests for R5-E1 (CPU). Run: cd li-net/r5e1; PYTHONPATH=/tmp/r4e3a/pylib:. python3 -m pytest -q test_r5e1.py"""
import copy
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import env6  # noqa: E402
import rollouts as RO  # noqa: E402
import run  # noqa: E402
import wm  # noqa: E402

HARNESS = "/home/agent/workspaces/eurisko-study/lir-agent/harness"


def test_placements_match_harness_enumeration():
    """Our enumeration equals the harness's r2 enumeration (same algorithm, imported unchanged) for 6x6, ships (4,3,2)."""
    sys.path.insert(0, HARNESS)
    from r2.core import Env7Config, placements as h_placements
    Ph = h_placements(Env7Config(W=6, H=6, ships=(4, 3, 2)))
    P = env6.placements()
    assert P.shape == Ph.shape == (49504, 36)
    assert set(map(bytes, np.packbits(P, 1))) == set(map(bytes, np.packbits(Ph, 1)))
    assert (P.sum(1) == 9).all()


def test_exact_bayes_vectorised_equals_bruteforce():
    b = env6.make_boards(99, 20)
    Pf = torch.as_tensor(env6.placements()).double()
    m = env6.bayes_marginals(torch.as_tensor(b["ev"]), torch.as_tensor(b["note"]), Pf).numpy()
    for i in range(20):
        bf = env6.bayes_bruteforce(b["ev"][i], int(b["note"][i]))
        assert np.abs(m[i] - bf).max() < 1e-10
    assert np.allclose(m.sum(1), 9.0)


def test_board_generators_match_spec():
    b = env6.make_boards(5, 20000)
    assert (b["occ"].sum(1) == 9).all()
    assert ((b["ev"] > 0).sum(1) == 6).all()
    assert abs(b["note_true"].mean() - 0.30) < 0.015
    rep_hit = b["ev"] == 2
    probed = b["ev"] > 0
    flips = (rep_hit != b["occ"]) & probed
    assert abs(flips.sum() / probed.sum() - 0.15) < 0.01
    g = torch.Generator().manual_seed(3)
    tbd = env6.torch_boards(20000, g, torch.as_tensor(env6.placements()))
    assert abs(float(tbd["note_true"].float().mean()) - 0.30) < 0.015
    pr = tbd["ev"] > 0
    fl = ((tbd["ev"] == 2) != tbd["occ"]) & pr
    assert abs(float(fl.sum()) / float(pr.sum()) - 0.15) < 0.01


def test_patch_geometry():
    assert env6.PATCHES.shape == (40, 6)
    assert sorted(set(env6.OVERLAP.ravel().tolist())) == [0, 1, 2, 3, 4, 6]
    assert env6.IN_NW.sum() == 22 and env6.IN_SE.sum() == 22 and (env6.SECTOR == 2).sum() == 4


def test_rollout_invariants():
    torch.manual_seed(0)
    m = wm.WorldModel(d=32, layers=2, heads=2, ff=64).eval()
    b = env6.boards_to_torch(env6.make_boards(7, 6), "cpu")
    l0 = RO.forward_chunks(m, b["ev"], b["note"])
    g = torch.Generator().manual_seed(1)
    patch, half = RO.sample_patches(l0, 5, torch.full((6,), 0.8), g)
    assert bool((torch.as_tensor(env6.IN_NW)[patch[half]]).all()) and bool((torch.as_tensor(env6.IN_SE)[patch[~half]]).all())
    tau = RO.draw_tau(patch.shape, g, "cpu")
    ro = RO.bag_rollouts(m, b, l0, patch, tau, g)
    P6 = torch.as_tensor(env6.PATCHES)[patch]                                   # [6,5,6]
    assert bool((ro["cells"][..., :, None] == P6[..., None, :]).any(-1).all())  # every sampled cell inside the patch
    assert all(len(set(r.tolist())) == 4 for r in ro["cells"].reshape(-1, 4))   # 4 distinct cells
    # leave-one-out logit of each sampled cell equals a fresh pass with that cell re-masked
    i, j = 2, 3
    cells, vals = ro["cells"][i, j], ro["vals"][i, j]
    rev = torch.zeros(1, 36, dtype=torch.bool)
    val = torch.zeros(1, 36, dtype=torch.long)
    rev[0, cells] = True
    val[0, cells] = vals.long()
    for t in range(4):
        rv = rev.clone()
        rv[0, cells[t]] = False
        lt = m(b["ev"][i:i + 1], b["note"][i:i + 1], rv, val)[0]
        assert abs(float(lt[0, cells[t]]) - float(ro["lr"][i, j, cells[t]])) < 1e-5
    lf = m(b["ev"][i:i + 1], b["note"][i:i + 1], rev, val)[0]
    other = torch.ones(36, dtype=torch.bool)
    other[cells] = False
    assert float((lf[0, other] - ro["lr"][i, j, other]).abs().max()) < 1e-5
    assert torch.allclose(ro["dl"], ro["lr"] - l0[:, None])


def test_aggregator_identities():
    torch.manual_seed(0)
    B, K = 3, 5
    l0 = torch.randn(B, 36)
    dl = torch.randn(B, 1, 36).expand(B, K, 36).contiguous()
    lr = l0[:, None] + dl
    valid = torch.ones(B, K, dtype=torch.bool)
    # mix of K identical rollouts is that rollout
    assert torch.allclose(torch.sigmoid(wm.agg_mix(lr, valid)), torch.sigmoid(lr[:, 0]), atol=1e-6)
    # crowding with identical keys = K
    k = torch.nn.functional.normalize(torch.randn(B, 1, 16), dim=-1).expand(B, K, 16)
    assert torch.allclose(wm.crowding(k, valid), torch.full((B, K), float(K)))
    beta = torch.ones(B, K)
    assert torch.allclose(wm.agg_kc(l0, dl, beta, k, valid, 1.0), l0 + dl[:, 0], atol=1e-5)
    assert torch.allclose(wm.agg_mmd(l0, dl, beta, valid, 1.0), l0 + K * dl[:, 0], atol=1e-5)
    # padding is ignored by attn_L3
    att = wm.AttnL3()
    v2 = valid.clone()
    v2[:, -1] = False
    a, w = att(l0, dl, beta, k, v2)
    assert float(w[:, -1].abs().max()) == 0.0


def test_metric_functions():
    p = torch.tensor([0.0, 0.3, 1.0, 0.5])
    assert float(wm.bern_kl(p, p).abs().max()) < 1e-5
    assert float(wm.bern_jsd(p, p).abs().max()) < 1e-6
    assert float(wm.bern_jsd(torch.tensor([0.1]), torch.tensor([0.9]))) > 0.3


def test_lir_arm_zero_gradient_and_identical_belief():
    """Part B, LIR-Masked: theta_real gets no reward gradient (measured with requires_grad on) and is bit-identical."""
    cfg = run.Cfg(device="cpu", d=32, layers=2, heads=2, ff=64, rl_steps=3, groups=4, gsize=4, truth_batch=8,
                  rl_eval_every=3, n_test=20)
    torch.manual_seed(0)
    m = wm.WorldModel(cfg.d, cfg.layers, cfg.heads, cfg.ff)
    T = env6.boards_to_torch(env6.make_boards(2, 20), "cpu")
    Pf = torch.as_tensor(env6.placements()).double()
    Pstar = env6.bayes_marginals(T["ev"], T["note"], Pf).float()
    Pt = torch.as_tensor(env6.placements())
    log = lambda *a: None
    res, rows = run.part_b(m, "lir", cfg, torch.device("cpu"), log, T, Pstar, Pt)
    assert res["mask_check"]["max_reward_grad_real"] == 0.0
    assert res["mask_check"]["max_param_change_real"] == 0.0
    post = [e for e in res["evals"] if e["tag"] == "post_rl"][0]
    assert post["drift_linf"] == 0.0 and post["swap_belief_change_max"] == 0.0
    res2, _ = run.part_b(m, "tokmask", cfg, torch.device("cpu"), log, T, Pstar, Pt)
    assert res2["steps"][0]["grad_real"] > 0.0


# ---- added with the amendment (PREREG-AMENDMENTS.md): Part A v2, bounded-context reads
def test_v2_reads_see_only_root_and_patch():
    import partA2
    torch.manual_seed(0)
    m = wm.WorldModel(d=32, layers=2, heads=2, ff=64).eval()
    B = partA2.boards2_t(31, 4, "cpu")
    ev_root = partA2.ev_from(B["s"], partA2.cells_mask(B["anchors"], "cpu"))
    g = torch.Generator().manual_seed(2)
    idx = torch.tensor([0, 0, 1, 2])
    patch = torch.tensor([5, 5, 30, 12])
    clean = partA2.reads(m, B, idx, ev_root, patch, torch.full((4,), 0.3), g)
    # repeated clean reads of the same patch on the same board are identical
    assert torch.equal(clean["lr"][0], clean["lr"][1]) and torch.equal(clean["hbar"][0], clean["hbar"][1])
    # a clean read equals one call with exactly the root readings plus the patch readings
    pm = torch.zeros(36, dtype=torch.bool)
    pm[torch.as_tensor(env6.PATCHES[30])] = True
    ev = torch.where(pm & (ev_root[1] == 0), 1 + B["s"][1].long(), ev_root[1])
    z = torch.zeros(1, 36, dtype=torch.bool)
    l = m(ev[None], B["note"][1:2], z, z.long())[0]
    assert float((l[0] - clean["lr"][2]).abs().max()) < 1e-5
    # corrupted reads differ only through the patch cells' readings
    gg = torch.Generator().manual_seed(3)
    cor = partA2._read(m, B, torch.tensor([1] * 50), ev_root, torch.tensor([30] * 50), torch.full((50,), 1.5), gg)
    assert float((cor["lr"] - clean["lr"][2]).abs().max()) > 0.0


def test_v2_union_call_covers_union():
    import partA2
    m = wm.WorldModel(d=32, layers=2, heads=2, ff=64).eval()
    B = partA2.boards2_t(32, 3, "cpu")
    ev_root = partA2.ev_from(B["s"], partA2.cells_mask(B["anchors"], "cpu"))
    patch = torch.tensor([[0, 1], [10, 10], [39, 0]])
    _, cov = partA2.union_call(m, B, ev_root, patch)
    exp = [len(set(env6.PATCHES[a]) | set(env6.PATCHES[b])) / 36 for a, b in patch.tolist()]
    assert np.allclose(cov.numpy(), exp)
