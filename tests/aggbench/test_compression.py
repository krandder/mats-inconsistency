import numpy as np
import pytest

pytest.importorskip("scipy")

from aggbench import make_env
from aggbench.baselines import extremized, oracle_clearing
from aggbench.compression import (BudgetError, Channel, LLRQuantizer, PooledQuant, QueryProtocol, agent_statistic,
                                  cell_bounds, check_message_locality, curve, gaussian_cell_decoder,
                                  gaussian_report_model, k_of_n_messages, llr_encoder, llr_quantizer_forecast,
                                  make_cenv, queried_forecast, quantize, top_w_forecast, top_w_messages,
                                  _piece_inputs)
from aggbench.metrics import per_episode
from aggbench.report import prior_forecast


@pytest.fixture(scope="module")
def e2():
    env = make_env("E2")
    return env, env.sample("dev", 600)


def test_channel_enforces_budget():
    ch = Channel(2)
    assert ch.send(np.array([0, 3])).tolist() == [0, 3]
    with pytest.raises(BudgetError):
        ch.send(np.array([4]))
    with pytest.raises(BudgetError):
        ch.send(np.array([0.5]))           # a real-valued width is not a budget
    with pytest.raises(BudgetError):
        Channel(0).send(np.array([1]))
    assert Channel(3).total_bits(5) == 15


@pytest.mark.parametrize("b", [1, 2, 3])
def test_llr_quantizer_respects_b_exactly(e2, b):
    env, dv = e2
    for fam in ("own", "shared"):
        q = LLRQuantizer(b, 1.0, fam, pooled=2.0)
        _, sym = llr_quantizer_forecast(env, dv, q, M=64)
        per_agent = [len(np.unique(sym[:, i])) for i in range(sym.shape[1])]
        assert max(per_agent) <= 2 ** b
        assert len(np.unique(sym)) == 2 ** b          # the code is used in full on 600 episodes


@pytest.mark.parametrize("b", [2, 4, 8])
def test_top_w_and_k_of_n_respect_budget(e2, b):
    _, dv = e2
    x, mu, _ = _piece_inputs(dv)
    for q in range(1, b + 1):
        sent, used = top_w_messages(x, mu, b, q)
        assert used.max() <= b
        sent, used = k_of_n_messages(x, mu, b, q)
        assert (used.sum(1) <= x.shape[1] * b).all()
        assert (sent.sum(-1) * q == used).all()        # a chosen agent sends all its readings at q bits


def test_pooled_quant_distinct_values(e2):
    _, dv = e2
    pq = PooledQuant(2, "mean", 1.0, 2.0, 1.0, centroids=np.array([-3.0, -1.0, 1.0, 3.0]))
    v = pq.values(dv.local["logit"], 0.0)
    assert len(np.unique(v)) == 4


def test_b_infinite_recovers_uncompressed(e2):
    env, dv = e2
    fc, _ = llr_quantizer_forecast(env, dv, LLRQuantizer(None))
    assert np.allclose(fc["logit"], oracle_clearing(dv)["logit"], atol=1e-8)
    pq = PooledQuant(None, "mean", alpha=1.7)
    assert np.allclose(pq.forecast(dv.local["logit"], 0.0)["logit"], extremized(dv.view("none"), 1.7)["logit"])
    fc, _ = top_w_forecast(dv, None, None, 1.0)
    assert np.allclose(fc["logit"], dv.bayes["logit"], atol=1e-9)


def test_b_zero_is_the_prior(e2):
    env, dv = e2
    fc, sym = llr_quantizer_forecast(env, dv, LLRQuantizer(0))
    assert (sym == 0).all()
    assert np.allclose(per_episode(fc, dv)["kl"], per_episode(prior_forecast(dv), dv)["kl"])
    assert np.allclose(PooledQuant(0).forecast(dv.local["logit"], 0.0)["logit"], 0.0)


def test_high_b_approaches_oracle_clearing(e2):
    env, dv = e2
    fc, _ = llr_quantizer_forecast(env, dv, LLRQuantizer(8, 1.0, "own"), M=512)
    k8 = per_episode(fc, dv)["kl"].mean()
    ko = per_episode(oracle_clearing(dv), dv)["kl"].mean()
    assert abs(k8 - ko) < 0.003


def test_locality_of_llr_encoder(e2):
    env, dv = e2
    obs, _, _, own = dv.agent_inputs("none", alignment="unaligned", own_model=True)
    enc = llr_encoder(dv, LLRQuantizer(3, 1.0, "own"))
    assert check_message_locality(enc, obs, own)
    received = np.random.default_rng(0).normal(size=obs.shape[:2])          # queries (offsets) received
    assert check_message_locality(enc, obs, own, received)

    def mixing(obs, own, received=None):                                     # uses everyone's evidence
        return quantize(np.nansum(obs, (1, 2))[:, None] + 0 * obs[..., 0], np.array([0.0]))
    with pytest.raises(AssertionError):
        check_message_locality(mixing, obs, own)


def _mc_posterior(env, dv, e, quant, n_mc=400_000, seed=1):
    """Monte Carlo P(Y=1 | cells) for episode e's fixed structure (E2), by counting simulated cell patterns."""
    rng = np.random.default_rng(seed)
    cov, mu = dv.oracle["cov"][e], dv.oracle["mu"][e]
    y = rng.random(n_mc) < 0.5
    x = (2 * y[:, None] - 1) * mu + rng.standard_normal((n_mc, len(mu)))
    r = (2 * mu * x) @ cov.T
    h = (2 * mu ** 2) @ cov.T
    scale = np.sqrt(h ** 2 + 2 * h)
    sym = quant.encode(r, np.broadcast_to(scale, r.shape), 0.0)
    target = quant.encode(dv.local["logit"][e][None], scale[None], 0.0)[0]
    hit = (sym == target).all(1)
    return y[hit].mean(), hit.sum()


@pytest.mark.parametrize("b", [1, 2])
def test_exact_decoder_matches_monte_carlo(e2, b):
    env, dv = e2
    q = LLRQuantizer(b, 1.0, "own")
    fc, _ = llr_quantizer_forecast(env, dv, q, M=4096)
    checked = 0
    for e in range(40):
        p_mc, n_hit = _mc_posterior(env, dv, e, q)
        if n_hit < 3000:
            continue
        p = 1 / (1 + np.exp(-fc["logit"][e]))
        assert abs(p - p_mc) < 4 * np.sqrt(p_mc * (1 - p_mc) / n_hit) + 0.005, (e, p, p_mc, n_hit)
        checked += 1
        if checked >= 4:
            break
    assert checked >= 2


def test_decoder_handles_exact_duplicates():
    env = make_env("E3-sybil")
    dv = env.sample("dev", 60)
    q = LLRQuantizer(2, 1.0, "own")
    r, scale = agent_statistic(dv)
    sym = q.encode(r, scale, 0.0)
    lo, hi = cell_bounds(sym, q.thresholds(scale, 0.0))
    model = gaussian_report_model(dv)
    full = gaussian_cell_decoder(model, lo, hi, 0.0, M=2048)
    # drop the copies by hand: keep one agent per distinct coverage row
    out = []
    for e in range(len(dv)):
        _, keep = np.unique(dv.oracle["cov"][e], axis=0, return_index=True)
        keep = np.sort(keep)
        m = tuple(a[e:e + 1][:, keep] if a.ndim == 2 else a[e:e + 1][:, keep][:, :, keep] for a in model)
        out.append(gaussian_cell_decoder(m, lo[e:e + 1, keep], hi[e:e + 1, keep], 0.0, M=2048)[0])
    assert np.allclose(full, out, atol=0.03)


def test_queried_q0_is_one_shot(e2):
    env, dv = e2
    quant = LLRQuantizer(2, 1.0, "own")
    one, sym1 = llr_quantizer_forecast(env, dv, quant, M=256)
    proto = QueryProtocol(quant, q=0, kappa=0.7, centroids=np.array([-1.0, -0.3, 0.3, 1.0]))
    r, scale = agent_statistic(dv)
    sym, offs, qsym = proto.run(r, scale, 0.0)
    assert (offs == 0).all() and (qsym == 0).all() and np.array_equal(sym, sym1)
    assert np.allclose(queried_forecast(env, dv, proto, M=256)["logit"], one["logit"])


def test_queried_message_depends_on_own_inputs_and_query(e2):
    env, dv = e2
    obs, _, _, own = dv.agent_inputs("none", alignment="unaligned", own_model=True)
    quant = LLRQuantizer(2, 1.0, "own")
    proto = QueryProtocol(quant, q=2, kappa=0.5, centroids=np.array([-1.0, -0.3, 0.3, 1.0]))
    r, scale = agent_statistic(dv)
    _, offs, qsym = proto.run(r, scale, 0.0)
    assert len(np.unique(qsym)) <= 4                     # the downlink respects q bits
    assert check_message_locality(llr_encoder(dv, quant), obs, own, offs)


def test_grouped_zoo_keeps_bayes_and_reads_several_symbols():
    base = make_env("Z9")
    g = make_cenv("Z9-p2")
    bb, gb = base.sample("dev", 50), g.sample("dev", 50)
    assert gb.obs.shape[1] == bb.obs.shape[1] // 2 and gb.obs.shape[2] == 2
    assert np.allclose(gb.bayes["logit"], bb.bayes["logit"])
    assert np.allclose(gb.local["logit"], 0.0, atol=1e-9)      # two readings of a 4-bit parity: no own signal
    fc, _ = llr_quantizer_forecast(g, gb, LLRQuantizer(None))
    assert np.allclose(fc["logit"], gb.oracle["report_logit"], atol=1e-9)
    fc, _ = llr_quantizer_forecast(g, gb, LLRQuantizer(2, 1.0, "own"))
    assert np.allclose(fc["logit"], 0.0, atol=1e-9)             # self-summary is useless here


def test_curve_rho_and_b90():
    c = curve({0: 0.4, 1: 0.1, 2: 0.05, 4: 0.03, None: 0.0}, floor_regret=0.0, ref_regret=0.02)
    assert c["rho"][0] == 0 and abs(c["rho"][2] - 0.875) < 1e-12 and c["b90"] == 4
    assert c["b90_full_context_transformer"] == 2


def test_query_is_a_pure_read_of_the_consensus_state(e2):
    env, dv = e2
    quant = LLRQuantizer(2, 1.0, "own")
    proto = QueryProtocol(quant, q=None, kappa=0.5, centroids=np.array([-2.0, -0.5, 0.5, 2.0]))
    r, scale = agent_statistic(dv)
    state, offs, qsym = proto.run(r, scale, 0.0)
    assert state.shape == r.shape and state.dtype.kind == "i" and state.max() < 4   # consensus = b-bit messages only
    frozen = state.copy()
    frozen.setflags(write=False)
    for t in range(state.shape[1]):                     # each query is recomputed from the state prefix alone
        assert np.allclose(qsym[:, t], proto.consensus_estimate(frozen[:, :t]))
    assert np.array_equal(state, frozen)


def test_fixed_state_respects_S_and_unbounded_is_running_sum(e2):
    from aggbench.compression import FixedStateChain
    _, dv = e2
    r = dv.local["logit"]
    for S in (1, 3, 6):
        v, writes = FixedStateChain(S, R=6.0, w=0.8).run(r, 0.0)
        assert writes.max() < 2 ** S and len(np.unique(v)) <= 2 ** S
    v, writes = FixedStateChain(None, w=0.7).run(r, 0.0)
    assert writes is None and np.allclose(v, 0.7 * r.sum(1))
    assert np.allclose(FixedStateChain(0).forecast(r, 0.0)["logit"], 0.0)


def test_fixed_state_large_S_does_not_overflow(e2):
    from aggbench.compression import FixedStateChain
    _, dv = e2
    r = dv.local["logit"]
    v64, _ = FixedStateChain(64, R=12.0, w=1.0).run(r, 0.0)
    assert np.allclose(v64, np.clip(r.sum(1), -12, 12), atol=1e-9) or np.abs(r.sum(1)).max() > 12


def test_raw_code_is_lossless_when_it_fits_and_prior_at_b0():
    from aggbench.compression import raw_code_forecast
    g = make_cenv("Z9-p2")
    gb = g.sample("dev", 40)
    assert np.allclose(raw_code_forecast(g, gb, 2)["logit"], gb.bayes["logit"], atol=1e-9)    # 2 binary readings
    assert np.allclose(raw_code_forecast(g, gb, 0)["logit"], 0.0, atol=1e-9)


def test_gaussian_channel_capacity_and_quantised_eval_respects_b():
    torch = pytest.importorskip("torch")
    from aggbench.compression import _net_v2, gauss_bits
    assert abs(gauss_bits() - 1.0) < 1e-12                  # P = 1, sigma^2 = 1/3: one bit per dimension
    for b in (1, 2, 4):
        net = _net_v2(5, b, "gauss")
        x = torch.randn(64, 3, 5)
        q = net.message(x, "quant")
        assert set(torch.unique(q).tolist()) <= {-1.0, 1.0} and q.shape[-1] == b
        sym = net.symbols(x).numpy().astype(np.int64)
        Channel(b).send(sym)
        noisy = net.message(x, "train")
        assert noisy.shape == q.shape and not torch.equal(noisy, q)


def test_blind_batch_denies_oracle_access(e2):
    from aggbench.compression import blind
    _, dv = e2
    bb = blind(dv)
    with pytest.raises(AssertionError):
        bb.oracle["cov"]
    with pytest.raises(AssertionError):
        "cov" in bb.oracle


def test_fair_z9_code_is_blind_and_equals_bayes_once_structure_is_sent():
    from aggbench.compression import FairZ9Code, PiecesPerAgent, blind
    env = PiecesPerAgent("Z9", 2, accuracy_range=(0.9, 0.9))     # fixed accuracy: the vars are the whole structure
    dv = env.sample("dev", 60)
    code = FairZ9Code(env, "pair", 0, n_prior=5000)
    assert code.lossless_bits() == 6
    for b in (None, 6, 7):                                      # the message carries which bit + the reading
        fc, sym = code.forecast(blind(dv), b)
        assert np.allclose(fc["logit"], dv.bayes["logit"], atol=1e-9)
    fc, sym = code.forecast(blind(dv), 3)
    assert sym.max() < 8
    assert per_episode(fc, dv)["kl"].mean() > 0.05               # 3 bits cannot say which bit and the reading


def test_fair_llr_decoder_is_blind_and_matches_oracle_with_point_structure():
    from aggbench.compression import FairLLR, blind, gaussian_cell_decoder, quantize, cell_bounds
    env = make_env("E2")
    dv = env.sample("dev", 6)
    model = gaussian_report_model(dv)
    q = FairLLR(2, 1.0, 2.0)
    th = q.thresholds(0.0)
    lo, hi = cell_bounds(quantize(dv.local["logit"], th), th)
    oracle = gaussian_cell_decoder(model, lo, hi, 0.0, M=2048)
    for e in range(len(dv)):                                    # prior = the true structure: fair == oracle
        point = tuple(m[e:e + 1] for m in model)
        sub = dv.__class__(dv.env, dv.config, dv.kind, dv.seeds[e:e + 1], dv.truth[e:e + 1], dv.obs[e:e + 1],
                           {k: v[e:e + 1] for k, v in dv.local.items()}, {k: v[e:e + 1] for k, v in dv.reports.items()},
                           {k: v[e:e + 1] for k, v in dv.keys.items()}, {k: v[e:e + 1] for k, v in dv.bayes.items()},
                           {}, dv.public)
        fc, _ = q.forecast(blind(sub), point, M=2048)
        assert abs(fc["logit"][0] - oracle[e]) < 0.05
