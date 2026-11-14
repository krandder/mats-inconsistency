"""Sequential aggregation protocol (aggbench.sequential): exact references, cascade metric, locality, determinism."""
import numpy as np
import pytest

from aggbench import make_env
from aggbench.sequential import (
    ActionModel, AppendKV, AttentionView, MeanView, agent_inputs_dict, check_sequential_locality,
    coarse_action_cascade, cascade_curve, exact_inputs, exact_sequential_bayes, full_context_bar, kv_init,
    linear_protocol, order_sensitivity, resample_private, run_protocol, sample_orders, scalar_report_sequential)

EXACT_ENVS = ["E1", "E2", "E3-sybil", "E3-biased", "E3-collude", "Z1", "Z2", "Z4", "Z5", "Z6", "Z8"]


def _batch(name, n=64, split="smoke"):
    return make_env(name).sample(split, n)


def _final(f, kind):
    return f["logit"] if kind == "bernoulli" else f["mean"]


@pytest.mark.parametrize("name", EXACT_ENVS)
def test_exact_sequential_bayes_recovers_bayes_in_every_order(name):
    b = _batch(name)
    p, inp = exact_sequential_bayes(b), exact_inputs(b)
    for s in range(3):
        f = run_protocol(p, inp, sample_orders(b.seeds, b.obs.shape[1], s)).forecast
        assert np.abs(_final(f, b.kind) - _final(b.bayes, b.kind)).max() < 1e-9
        if b.kind == "gaussian":
            assert np.abs(f["prec"] - b.bayes["prec"]).max() < 1e-8
    assert order_sensitivity(p, inp, b, n_orders=4)["variance"]["mean"] < 1e-20


def test_exact_reference_is_not_trivially_bayes():
    """Mutation check: dropping the dedup of shared pieces (E2) or one agent (zoo) breaks exactness."""
    b = _batch("E2")
    p, inp = exact_sequential_bayes(b), exact_inputs(b)
    K = inp["seen"].shape[-1]
    p.readout = lambda s: {"logit": b.public["prior_logit"] + s.v[..., K:].sum((1, 2))}   # counts shared pieces twice
    f = run_protocol(p, inp, sample_orders(b.seeds, b.obs.shape[1])).forecast
    assert np.abs(f["logit"] - b.bayes["logit"]).max() > 1e-3
    z = _batch("Z1")
    pz, iz = exact_sequential_bayes(z), exact_inputs(z)
    iz["loglik"][:, 0] = 0.0
    f = run_protocol(pz, iz, sample_orders(z.seeds, z.obs.shape[1])).forecast
    assert np.abs(f["logit"] - z.bayes["logit"]).max() > 1e-3


@pytest.mark.parametrize("name", ["E1", "E2", "Z1"])
def test_resample_private_is_consistent(name):
    """The resampled batch is a valid episode: its recomputed Bayes equals the exact reference on it, and only the
    chosen agent's observation changes."""
    b = _batch(name)
    agent = sample_orders(b.seeds, b.obs.shape[1])[:, 2]
    rb = resample_private(b, agent, seed=1)
    f = run_protocol(exact_sequential_bayes(rb), exact_inputs(rb), sample_orders(b.seeds, b.obs.shape[1])).forecast
    assert np.abs(_final(f, b.kind) - _final(rb.bayes, b.kind)).max() < 1e-9
    changed = ~np.isclose(np.nan_to_num(rb.obs), np.nan_to_num(b.obs)).reshape(len(b), b.obs.shape[1], -1).all(-1)
    others = np.ones_like(changed)
    others[np.arange(len(b)), agent] = False
    assert not changed[others].any() and changed.any()
    assert np.array_equal(rb.truth, b.truth)


def _cascade(name):
    b = make_env(name).sample("dev", 600)
    am = ActionModel.fit(make_env(name).sample("train", 1000))
    protos = {"exact": exact_sequential_bayes(b), "coarse": coarse_action_cascade(b, am),
              "scalar": scalar_report_sequential(b)}
    return cascade_curve(protos, exact_inputs, b, n_boot=200)


@pytest.mark.parametrize("name", ["E2", "Z1"])
def test_coarse_action_baseline_shows_a_cascade(name):
    c = _cascade(name)
    ret, msg = c["coarse"]["retention"], c["coarse"]["message_retention"]
    assert c["exact"]["retention"] == pytest.approx([1.0] * len(ret))
    # late agents' private information stops reaching the final answer and their messages
    assert ret[-1] < 0.5 * ret[0] and ret[-1] < 0.2
    assert msg[-1] < 0.5 * msg[0] and msg[-1] < 0.1
    # the scalar-report baseline passes every agent's evidence on (no cascade), whatever its regret
    assert min(c["scalar"]["message_retention"]) > 0.99


def test_order_sensitivity_of_coarse_actions():
    b = make_env("E2").sample("dev", 400)
    p = coarse_action_cascade(b, ActionModel.fit(make_env("E2").sample("train", 1000)))
    assert order_sensitivity(p, exact_inputs(b), b, n_orders=6)["variance"]["mean"] > 0.1


@pytest.mark.parametrize("name", ["E1", "E2", "Z1"])
def test_reference_protocols_are_local(name):
    b = _batch(name)
    inp, order = exact_inputs(b), sample_orders(b.seeds, b.obs.shape[1])
    assert check_sequential_locality(exact_sequential_bayes(b), inp, order)["passed"]
    if b.kind == "bernoulli":
        am = ActionModel.fit(_batch(name, 200, "train"))
        assert check_sequential_locality(coarse_action_cascade(b, am), inp, order)["passed"]


@pytest.mark.parametrize("mode", ["queried", "broadcast"])
def test_linear_protocol_modes_are_local(mode):
    b = _batch("E2")
    inp = agent_inputs_dict(b, "noisy", "unaligned")
    p = linear_protocol(inp, w_msg=8, w_view=3, mode=mode)
    assert p.mode == mode
    tr = run_protocol(p, inp, sample_orders(b.seeds, b.obs.shape[1]))
    assert tr.views[1].shape == (len(b), 3) and tr.messages[0].shape == (len(b), 8)
    assert check_sequential_locality(p, inp, tr.order)["passed"]


def test_locality_check_catches_a_leaking_encoder():
    b = _batch("E2")
    inp = agent_inputs_dict(b, "noisy", "unaligned")
    p = linear_protocol(inp, w_msg=8, w_view=3)
    enc = p.encode
    p.encode = lambda x, i, v, t: enc(x, i, v, t) + np.nan_to_num(x["obs"]).mean((1, 2))[:, None]  # reads everyone
    with pytest.raises(AssertionError, match="message"):
        check_sequential_locality(p, inp, sample_orders(b.seeds, b.obs.shape[1]))


def test_locality_check_catches_a_leaking_query_and_a_peeking_view():
    b = _batch("E2")
    inp = agent_inputs_dict(b, "noisy", "unaligned")
    order = sample_orders(b.seeds, b.obs.shape[1])
    p = linear_protocol(inp, w_msg=8, w_view=3)
    q = p.query
    p.query = lambda x, i, t: q(x, i, t) + np.nan_to_num(x["obs"]).sum((1, 2))[:, None]   # query reads everyone
    with pytest.raises(AssertionError, match="query"):
        check_sequential_locality(p, inp, order)
    # a consensus view that peeks at the inputs of the run (here through the agent's query call) sees agents
    # that have not reported yet: the causality check catches it
    p = linear_protocol(inp, w_msg=8, w_view=3)
    q, v, seen = p.query, p.view, {}

    def query(x, i, t):
        seen["x"] = x
        return q(x, i, t)
    p.query = query
    p.view = lambda s, qq, t: v(s, qq, t) + np.nan_to_num(seen["x"]["obs"]).sum((1, 2))[:, None]
    with pytest.raises(AssertionError, match="after position"):
        check_sequential_locality(p, inp, order)


def test_declared_widths_are_enforced():
    b = _batch("E2")
    inp = agent_inputs_dict(b)
    p = linear_protocol(inp, w_msg=8, w_view=3)
    p.w_msg = 6
    with pytest.raises(ValueError, match="message"):
        run_protocol(p, inp, sample_orders(b.seeds, b.obs.shape[1]))


def test_orders_and_runs_are_deterministic():
    b = _batch("Z1")
    N = b.obs.shape[1]
    o1, o2 = sample_orders(b.seeds, N), sample_orders(b.seeds, N)
    assert np.array_equal(o1, o2) and not np.array_equal(o1, sample_orders(b.seeds, N, order_seed=1))
    assert all(sorted(r) == list(range(N)) for r in o1)
    am = ActionModel.fit(_batch("Z1", 200, "train"))
    f1 = run_protocol(coarse_action_cascade(b, am), exact_inputs(b), o1).forecast
    f2 = run_protocol(coarse_action_cascade(b, am), exact_inputs(b), o2).forecast
    assert np.array_equal(f1["logit"], f2["logit"])
    r1, r2 = resample_private(b, o1[:, 0], seed=3), resample_private(b, o1[:, 0], seed=3)
    assert np.array_equal(r1.obs, r2.obs)
    c1 = cascade_curve({"x": exact_sequential_bayes(b)}, exact_inputs, b, positions=[0], n_boot=50)
    c2 = cascade_curve({"x": exact_sequential_bayes(b)}, exact_inputs, b, positions=[0], n_boot=50)
    assert c1["x"]["influence"] == c2["x"]["influence"]


def test_queried_view_is_the_default_and_broadcast_is_the_comparison():
    inp = agent_inputs_dict(_batch("E2"))
    assert linear_protocol(inp, 8, 3).mode == "queried"
    assert linear_protocol(inp, 8, 3, mode="broadcast").mode == "broadcast"


def test_full_context_hook_finds_the_transformer(monkeypatch):
    import aggbench.transformer as T
    calls = {}
    monkeypatch.setattr(T, "evaluate_full_context_transformer", lambda env, **kw: calls.setdefault("kw", kw))
    out = full_context_bar(make_env("E2"), visibility="noisy", n_train=10)
    assert out["available"] and calls["kw"] == {"visibility": "noisy", "n_train": 10}


def test_attention_view_reads_the_matching_entry():
    """Queried view = cross-attention read of the KV memory: a sharp query retrieves the value with the matching key."""
    n = 4
    upd, view = AppendKV(2), AttentionView(3, temperature=0.01)
    s = kv_init(n)
    s = upd(s, np.tile([1.0, 0.0, 5.0, 5.0, 5.0], (n, 1)), 0)
    s = upd(s, np.tile([0.0, 1.0, -7.0, 0.0, 1.0], (n, 1)), 1)
    out = view(s, np.tile([0.0, 1.0], (n, 1)), 2)
    assert np.allclose(out, [[-7.0, 0.0, 1.0]] * n, atol=1e-6)
    assert np.allclose(view(kv_init(n), np.zeros((n, 2)), 0), 0)


def test_views_accept_torch_tensors():
    torch = pytest.importorskip("torch")
    upd = AppendKV(2)
    s = upd(kv_init(4), torch.randn(4, 5), 0)
    s = upd(s, torch.randn(4, 5), 1)
    assert AttentionView(3)(s, torch.randn(4, 2), 2).shape == (4, 3)
    assert MeanView(3)(s, None, 2).shape == (4, 3)
    assert MeanView(3)(kv_init(4), None, 0).shape == (4, 3)
