"""ForecastWorld: exact ceilings (A) full-info Bayes and (B) forecasts + exact source sets (GLS)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from aggbench import forecast_world as fw  # noqa: E402


def lsig(x):
    return torch.nn.functional.logsigmoid(x)


def logloss(probit, y):
    lp, lm = torch.special.log_ndtr(probit), torch.special.log_ndtr(-probit)
    return float(-(y * lp + (1 - y) * lm).mean())


@pytest.fixture(scope="module")
def env():
    torch.set_num_threads(2)
    return fw.ForecastWorld(alpha=1.0, N=6, k=3)


def test_local_and_ceiling_a_match_dense_numpy(env):
    ep = env.sample("smoke", 8, chunk=1)
    A, sq, B, sd = env.A.numpy(), env.s_q.numpy(), env.B.numpy(), env.src_sd.numpy()
    s = ep["s"].numpy()
    for e in range(8):
        for i in range(env.config.N + 1):
            idx = np.flatnonzero((ep["union"][e] if i == env.config.N else ep["mask"][e, i]).numpy())
            Bu = B[idx].reshape(-1, env.config.D)
            r = np.repeat(sd[idx] ** 2, env.config.rows)
            P = np.eye(env.config.D) + Bu.T @ (Bu / r[:, None])
            mu = np.linalg.solve(P, Bu.T @ (s[e, idx].reshape(-1) / r))
            var = np.einsum("qd,qd->q", A, np.linalg.solve(P, A.T).T)
            u = A @ mu / np.sqrt(var + sq ** 2)
            got = ep["bayes_probit"][e] if i == env.config.N else ep["probit"][e, i]
            np.testing.assert_allclose(got.numpy(), u, atol=1e-9)


def test_sets_have_k_distinct_training_sources(env):
    ep = env.sample("smoke", 64, chunk=2)
    assert torch.all(ep["mask"].sum(-1) == env.config.k)
    assert torch.all(ep["mask"][..., env.config.M:] == 0)          # held-out sources never read in training
    held = env.sample("smoke", 512, chunk=2, heldout=True)
    assert held["mask"][..., env.config.M:].sum() > 0


def test_key_is_sum_of_source_embeddings(env):
    ep = env.sample("smoke", 16, chunk=3, key_sigma=0.0)
    torch.testing.assert_close(ep["key"], ep["mask"] @ env.E)


def test_probit_inversion_recovers_projection(env):
    ep = env.sample("smoke", 16, chunk=4)
    torch.testing.assert_close(ep["probit"] * ep["denom"], ep["mu"] @ env.A.T, atol=1e-10, rtol=0)


def test_ceiling_b_equals_a_for_one_forecaster():
    e1 = fw.ForecastWorld(alpha=1.0, N=1, k=4)
    ep = e1.sample("smoke", 64, chunk=5, ceiling_b=True)
    torch.testing.assert_close(ep["oracle_probit"], ep["bayes_probit"], atol=1e-6, rtol=0)
    torch.testing.assert_close(ep["probit"][:, 0], ep["bayes_probit"], atol=1e-10, rtol=0)


def test_ceiling_b_equals_a_when_forecasts_reveal_all_readings():
    # Q >= D and disjoint reading sets: each mu_i is fully revealed and the union statistic is the sum
    e = fw.ForecastWorld(D=8, Q=12, M=12, M_heldout=0, rows=2, src_factors=3, q_factors=3, N=2, k=2, alpha=0.0)
    ep = e.sample("smoke", 400, chunk=6, ceiling_b=True)
    disjoint = ep["mask"].sum(1).max(-1).values <= 1
    assert disjoint.sum() > 20
    torch.testing.assert_close(ep["oracle_probit"][disjoint], ep["bayes_probit"][disjoint], atol=1e-6, rtol=0)


def test_ceiling_b_strictly_below_a_with_little_overlap():
    # two forecasters, disjoint rich readings (k*src_factors = 32 > Q = 20): forecasts lose information
    e = fw.ForecastWorld(N=2, k=4, alpha=0.0)
    ep = e.sample("smoke", 1024, chunk=7, ceiling_b=True)
    kl = fw.logit_metrics(fw.probit_to_logit(ep["oracle_probit"]), ep["bayes_probit"], ep["y"])["kl"]
    assert kl.min() >= -1e-12 and kl.mean() > 3e-4   # small but real: ~0.001 nats at this config


def test_ceiling_b_rank_tolerance_stable(env):
    ep = env.sample("smoke", 32, chunk=8, ceiling_b=False)
    full = env.sample("smoke", 32, chunk=8)          # same episodes, with X kept
    b1 = env.ceiling_b(full, rank_tol=1e-7)
    b2 = env.ceiling_b(full, rank_tol=1e-11)
    torch.testing.assert_close(b1, b2, atol=1e-5, rtol=0)
    torch.testing.assert_close(full["probit"], ep["probit"])


def test_ceiling_b_equals_direct_gls_from_readings(env):
    """B computed from inverted forecasts equals the projection posterior computed from the raw readings s."""
    ep = env.sample("smoke", 8, chunk=9)
    b = env.ceiling_b(ep)
    c = env.config
    w = torch.linalg.solve_triangular(env.L, ep["s"].reshape(8, -1, 1), upper=False)[..., 0]
    colmask = ep["mask"].repeat_interleave(c.rows, dim=-1)
    for e in range(8):
        H = torch.einsum("idq,ds->iqs", ep["X"][e], env.G) * colmask[e][:, None, :]
        Hw = H.reshape(-1, H.shape[-1]) @ env.L
        _, S, Vh = torch.linalg.svd(Hw, full_matrices=False)
        V = Vh[S > 1e-9 * S[0]].T
        mean = env.F @ (V @ (V.T @ w[e]))
        var = 1 - ((env.A @ env.F @ V) ** 2).sum(-1)
        torch.testing.assert_close(b[e], env.A @ mean / torch.sqrt(var + env.s_q ** 2), atol=1e-7, rtol=0)


def test_monte_carlo_calibration_and_ordering():
    """A, B and local forecasts are calibrated; expected log loss A <= B <= best pools and single forecaster."""
    e = fw.ForecastWorld(alpha=1.0, N=5, k=3)
    ep = e.sample("smoke", 6000, chunk=10, ceiling_b=True)
    y = ep["y"]
    for name, u in [("A", ep["bayes_probit"]), ("B", ep["oracle_probit"]), ("local", ep["probit"][:, 0])]:
        p = torch.special.ndtr(u).flatten()
        yy = y.flatten()
        bins = torch.clamp((p * 10).long(), max=9)
        for b in range(10):
            sel = bins == b
            if sel.sum() > 2000:
                gap = float((yy[sel].mean() - p[sel].mean()).abs())
                se = float(np.sqrt(0.25 / sel.sum().item()))
                assert gap < 4 * se + 0.01, (name, b, gap)
    la, lb, l1 = logloss(ep["bayes_probit"], y), logloss(ep["oracle_probit"], y), logloss(ep["probit"][:, 0], y)
    assert la <= lb + 2e-3 and lb < l1
    lo = fw.probit_to_logit(ep["probit"])
    lmean = float(torch.nn.functional.binary_cross_entropy_with_logits(lo.mean(1), y))
    assert lb < lmean


def test_kl_nonnegative_and_zero_at_bayes(env):
    ep = env.sample("smoke", 64, chunk=11)
    m = fw.logit_metrics(fw.probit_to_logit(ep["bayes_probit"]), ep["bayes_probit"], ep["y"])
    assert np.abs(m["kl"]).max() < 1e-10
    m2 = fw.logit_metrics(fw.pool_mean_logodds(fw.probit_to_logit(ep["probit"])), ep["bayes_probit"], ep["y"])
    assert m2["kl"].min() >= -1e-12


def test_splits_disjoint_and_deterministic(env):
    a = env.sample("train", 4, chunk=0)
    b = env.sample("train", 4, chunk=0)
    c = env.sample("dev", 4, chunk=0)
    torch.testing.assert_close(a["theta"], b["theta"])
    assert not torch.allclose(a["theta"], c["theta"])


def test_popularity_zipf_and_top_share():
    shares = [fw.ForecastWorld(alpha=a, k=3).top_share(3, n=4000) for a in (0.0, 1.0, 2.0)]
    assert shares[0] < shares[1] < shares[2]
    w = fw.ForecastWorld(alpha=1.5).popularity()
    ws = torch.sort(w[w > 0], descending=True).values
    torch.testing.assert_close(ws / ws[0], torch.arange(1, len(ws) + 1, dtype=ws.dtype) ** -1.5)


def test_aggregator_permutation_invariant():
    e = fw.ForecastWorld(N=5, k=3)
    ep = e.sample("smoke", 4, chunk=12)
    tok, lo = fw.token_features(ep, "key", 0.5)
    perm = torch.tensor([3, 1, 4, 0, 2])
    for readout in ("diag", "full"):
        m = fw.make_aggregator(tok.shape[-1], e.config.Q, width=32, layers=2, seed=0, readout=readout).eval()
        with torch.no_grad():
            torch.testing.assert_close(m(tok, lo), m(tok[:, perm], lo[:, perm]), atol=1e-5, rtol=0)


# ---------------------------------------------------------------- learned forecasters (2026-10-22 redirect)
def test_raw_observations_are_lossless_and_leave_episodes_unchanged(env):
    a = env.sample("smoke", 16, chunk=20, obs=True)
    b = env.sample("smoke", 16, chunk=20)
    for key in ("theta", "y", "probit", "bayes_probit", "mask"):
        torch.testing.assert_close(a[key], b[key])
    s_back = (a["obs"] @ env.R_obs)[..., :env.config.rows]          # R_obs orthogonal: s is recoverable exactly
    torch.testing.assert_close(s_back, a["s"], atol=1e-10, rtol=0)
    nu = (a["obs"] @ env.R_obs)[..., env.config.rows:]
    assert abs(float(nu.std()) - env.config.nuis_sd) < 0.2
    assert torch.all(a["mask"].gather(-1, a["idx"]) == 1)


def test_instance_unchanged_by_learned_extras():
    e = fw.ForecastWorld(alpha=1.0)
    # checksums of the instance at commit 37b6efa (the exact-forecaster sweep), before the learned extras existed
    for name, val in [("A", 38.20543520179409), ("s_q", 10.782770185844452), ("B", 899.2372021094909),
                      ("src_sd", 63.39942859073391), ("E", 633.7620496901633), ("rank_all", 1275.0)]:
        assert abs(float(getattr(e, name).abs().sum()) - val) < 1e-9, name
    assert e.desc.shape == (e.config.M + e.config.M_heldout, e.config.desc_dim)


def test_descriptions_carry_loading_pattern():
    e = fw.ForecastWorld()
    # sources that load on more shared factors have more similar descriptions than unrelated random controls
    load = (e.B.norm(dim=1) > 0).double()
    overlap = load @ load.T
    sim = torch.corrcoef(e.desc)
    simr = torch.corrcoef(e.desc_random)
    off = ~torch.eye(len(load), dtype=torch.bool)
    c_meaning = np.corrcoef(overlap[off].numpy(), sim[off].numpy())[0, 1]
    c_random = np.corrcoef(overlap[off].numpy(), simr[off].numpy())[0, 1]
    assert c_meaning > 0.3 and abs(c_random) < 0.15


@pytest.mark.parametrize("arm", ["fc", "fckey", "wide", "bar"])
def test_learned_system_permutation_invariant(arm):
    e = fw.ForecastWorld(N=5, k=3)
    ep = e.sample("smoke", 4, chunk=21, obs=True)
    m = fw.make_learned_system(e, arm, width=32, layers=2, seed=0).eval()
    perm = torch.tensor([3, 1, 4, 0, 2])
    ep2 = dict(ep, idx=ep["idx"][:, perm], mask=ep["mask"][:, perm])
    with torch.no_grad():
        torch.testing.assert_close(m(ep)[0], m(ep2)[0], atol=1e-4, rtol=0)


def test_fc_forecasts_get_no_aggregator_gradient():
    e = fw.ForecastWorld(N=4, k=2)
    ep = e.sample("smoke", 4, chunk=22, obs=True)
    m = fw.make_learned_system(e, "fc", width=32, layers=1, seed=0)
    out, loc = m(ep)
    out.sum().backward()
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in m.fore.parameters())


# ---------------------------------------------------------------- gate 2026-10-23 additions
def test_disjoint_sets_are_disjoint():
    e = fw.ForecastWorld(alpha=1.0, N=10, k=3, disjoint=True)
    ep = e.sample("smoke", 64, chunk=30, ceiling_b=True)
    assert torch.all(ep["mask"].sum(-1) == 3) and torch.all(ep["mask"].sum(1) <= 1)
    assert torch.all(ep["union"].sum(-1) == 30)


def test_fairbar_equivariant_and_reads_duplicates():
    e = fw.ForecastWorld(N=5, k=3)
    ep = e.sample("smoke", 4, chunk=31, obs=True)
    m = fw.make_learned_system(e, "fairbar", width=32, layers=2, seed=0).eval()
    perm = torch.tensor([3, 1, 4, 0, 2])
    ep2 = dict(ep, idx=ep["idx"][:, perm], mask=ep["mask"][:, perm])
    torch.manual_seed(0)
    with torch.no_grad():
        a = m(ep)[0]
        torch.manual_seed(0)
        b = m(ep2)[0]
    assert a.shape == (4, e.config.Q) and torch.isfinite(a).all()
    # slot tags are random per call; with a zero slot table the output is exactly permutation invariant
    with torch.no_grad():
        m.slot.zero_()
        torch.testing.assert_close(m(ep)[0], m(ep2)[0], atol=1e-4, rtol=0)


def test_split_embedding_random_control_is_independent():
    from aggbench.controls import random_weights
    m = fw.make_aggregator(36, 20, width=32, layers=1, split=20)

    class W:
        def __init__(self, module):
            self.module = module
    r = random_weights(W(m), 1).module
    x = torch.randn(2, 5, 36)
    lo = torch.randn(2, 5, 20)
    with torch.no_grad():
        assert not torch.allclose(m(x, lo), r(x, lo))
        m.embed.ef.weight.zero_()
        assert not torch.allclose(r.embed.ef.weight, m.embed.ef.weight)
