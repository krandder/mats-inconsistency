"""Trader transformer (aggbench.trader_transformer): causal locality, knobs, order, losses, a CPU smoke run."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aggbench import make_env  # noqa: E402
from aggbench.metrics import summarize  # noqa: E402
from aggbench.trader_transformer import (  # noqa: E402
    TraderTransformer, prefix_bayes_logits, resolve_attn_layers, trader_context)

torch.set_num_threads(2)


def _ctx(n=32, **kw):
    return trader_context(make_env("E2").sample("smoke", n), **kw)


def _model(ctx, **kw):
    hp = dict(width=16, layers=4, heads=2, steps=5, batch=16)
    hp.update(kw)
    return TraderTransformer(0, ctx.tokens.shape[-1], **hp)


def test_resolve_attn_layers():
    assert resolve_attn_layers("all", 4) == (0, 1, 2, 3)
    assert resolve_attn_layers("last", 4) == (3,)
    assert resolve_attn_layers("first", 4) == (0,)
    assert resolve_attn_layers("every_other", 4) == (1, 3)
    assert resolve_attn_layers("every_other", 3) == (0, 2)
    assert resolve_attn_layers("none", 3) == ()
    assert resolve_attn_layers([2, 0], 3) == (0, 2)
    with pytest.raises(ValueError):
        resolve_attn_layers([5], 3)


@pytest.mark.parametrize("token_state", ["kv", "recurrent", "feedback"])
@pytest.mark.parametrize("attn_layers", ["all", "last", "every_other"])
@pytest.mark.parametrize("pos", ["none", "learned"])
def test_causal_locality(token_state, attn_layers, pos):
    """Perturbing trader j > t leaves every output at positions <= t unchanged; earlier traders do matter."""
    ctx = _ctx()
    m = _model(ctx, token_state=token_state, attn_layers=attn_layers, pos=pos)
    m.module.eval()
    x = torch.as_tensor(ctx.tokens)
    rng = np.random.default_rng(0)
    with torch.no_grad():
        base = m.module(x)
        N = x.shape[1]
        for t in range(N - 1):
            x2 = x.clone()
            x2[:, t + 1:] += torch.as_tensor(rng.normal(size=x2[:, t + 1:].shape), dtype=x.dtype)
            assert torch.allclose(m.module(x2)[:, :t + 1], base[:, :t + 1], atol=1e-5)
        x3 = x.clone()
        x3[:, 0] += 1.0
        assert not torch.allclose(m.module(x3)[:, -1], base[:, -1], atol=1e-5)   # the last trader reads trader 0


def test_no_attention_layers_means_no_cross_trader_information():
    ctx = _ctx()
    m = _model(ctx, attn_layers="none")
    x = torch.as_tensor(ctx.tokens)
    with torch.no_grad():
        x2 = x.clone()
        x2[:, :-1] += 1.0
        assert torch.allclose(m.module(x2)[:, -1], m.module(x)[:, -1], atol=1e-6)


def test_order_is_random_per_episode_seeded_and_recorded():
    b = make_env("E2").sample("smoke", 64)
    c1, c2, c3 = trader_context(b), trader_context(b), trader_context(b, order_seed=1)
    assert np.array_equal(c1.order, c2.order) and np.array_equal(c1.tokens, c2.tokens)
    assert not np.array_equal(c1.order, c3.order)
    assert len({tuple(o) for o in c1.order}) > 1
    from aggbench.transformer import full_context
    raw = full_context(b, "none", "unaligned").tokens
    assert np.array_equal(c1.tokens, raw[np.arange(len(b))[:, None], c1.order])


@pytest.mark.parametrize("name", ["E2", "Z1", "Z8"])
def test_prefix_bayes_ends_at_bayes(name):
    b = make_env(name).sample("smoke", 64)
    ctx = trader_context(b, with_prefix_bayes=True)
    assert np.abs(ctx.prefix_bayes[:, -1] - b.bayes["logit"]).max() < 1e-9
    assert np.array_equal(prefix_bayes_logits(b, ctx.order), ctx.prefix_bayes)


def test_z8_prefix_bayes_matches_closed_form():
    """Z8 prefix Bayes after the first t+1 traders = l0 + sum_k 2 mu_k (sum of their reads of k) / (n_k + tau^2)."""
    env = make_env("Z8")
    b = env.sample("smoke", 64)
    ctx = trader_context(b, with_prefix_bayes=True)
    cov, mu, reads, tau2 = b.oracle["cov"], b.oracle["mu"], np.nan_to_num(b.obs), env.config.read_noise ** 2
    for t in range(cov.shape[1]):
        idx = ctx.order[:, :t + 1]
        c = np.take_along_axis(cov, idx[..., None], 1)
        r = np.take_along_axis(reads, idx[..., None], 1) * c
        n_k, s_k = c.sum(1), r.sum(1)
        ref = env.logit0 + np.where(n_k > 0, 2 * mu * s_k / (n_k + tau2), 0.0).sum(1)
        assert np.abs(ctx.prefix_bayes[:, t] - ref).max() < 1e-9


def test_loss_knob():
    ctx = _ctx()
    logits = torch.randn(32, ctx.tokens.shape[1])
    y = torch.randint(0, 2, (32,)).float()
    final, every = _model(ctx, loss="final"), _model(ctx, loss="every_prefix")
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    assert torch.isclose(final.loss_fn(logits, y), bce(logits[:, -1], y))
    assert torch.isclose(every.loss_fn(logits, y), bce(logits, y[:, None].expand_as(logits)))
    # only every_prefix sends gradient to early positions
    lg = logits.clone().requires_grad_(True)
    final.loss_fn(lg, y).backward()
    assert lg.grad[:, :-1].abs().sum() == 0
    lg.grad = None
    every.loss_fn(lg, y).backward()
    assert lg.grad[:, 0].abs().sum() > 0
    distill = _model(ctx, loss="distill")
    assert distill.uses_oracle and not every.uses_oracle
    with pytest.raises(ValueError, match="prefix_bayes"):
        distill.fit(ctx, np.zeros(32))
    cd = _ctx(with_prefix_bayes=True)
    distill.fit(cd, np.zeros(32))


@pytest.mark.parametrize("token_state,layout", [("kv", "trader"), ("recurrent", "trader"), ("feedback", "trader"),
                                                ("kv", "grid")])
def test_cpu_smoke_e2_learns_above_the_prior(token_state, layout):
    """SMOKE only (no verdict): a small model trained briefly on E2 beats the prior."""
    env = make_env("E2")
    tr, dv = env.sample("train", 3000), env.sample("dev", 500)
    ctr, cdv = trader_context(tr, "noisy", layout=layout), trader_context(dv, "noisy", layout=layout)
    m = TraderTransformer(0, ctr.tokens.shape[-1], width=32, layers=2, heads=2, steps=250, batch=256,
                          token_state=token_state, loss="every_prefix", layout=layout).fit(ctr, tr.truth)
    reg = summarize(m.predict(cdv), dv, n_boot=200)["regret"]["mean"]
    prior = summarize({"logit": np.zeros(len(dv))}, dv, n_boot=200)["regret"]["mean"]
    assert reg < 0.7 * prior


# ----------------------------------------------------------------------------- grid layouts (Kelvin's 3-D version)

def _grid(layout, layers=3, n=16, **kw):
    ctx = trader_context(make_env("E2").sample("smoke", n), "noisy", layout=layout)
    m = TraderTransformer(0, ctx.tokens.shape[-1], width=16, layers=layers, heads=4, steps=5, batch=8, layout=layout,
                          **kw)
    m.module.eval()
    return ctx, m


def test_piece_tokens():
    b = make_env("E2").sample("smoke", 16)
    ctx = trader_context(b, "noisy", layout="grid")
    n, N, T, F = ctx.tokens.shape
    assert T == b.config["n_pieces"]
    seen = np.isfinite(b.obs).sum(-1)[np.arange(n)[:, None], ctx.order]
    assert np.array_equal(ctx.tokens[..., 0].sum(-1), seen)
    # present cells come first in each trader, and carry the observed values
    assert all(np.all(np.diff(row) <= 0) for row in ctx.tokens[..., 0].reshape(-1, T))
    vals = np.sort(ctx.tokens[0, 0, :, 1][ctx.tokens[0, 0, :, 0] == 1])
    agent = ctx.order[0, 0]
    assert np.allclose(vals, np.sort(b.obs[0, agent][np.isfinite(b.obs[0, agent])]))
    with pytest.raises(ValueError, match="piece"):
        trader_context(make_env("Z1").sample("smoke", 4), layout="grid")


@pytest.mark.parametrize("layout", ["grid", "axial", "grid_full"])
def test_grid_cone_locality(layout):
    """Perturbing any cell outside the cone {t' <= t, r' <= r} leaves cell (t, r) unchanged."""
    ctx, m = _grid(layout)
    x = torch.as_tensor(ctx.tokens)
    n, N, T, _ = x.shape
    rng = np.random.default_rng(0)
    with torch.no_grad():
        base = m.module.cells(x)
        for t, r in [(0, 0), (2, 1), (T - 1, N - 2), (3, N - 1)]:
            outside = torch.ones(N, T, dtype=torch.bool)
            outside[:r + 1, :t + 1] = False
            x2 = x + outside[None, :, :, None] * torch.as_tensor(rng.normal(size=x.shape), dtype=x.dtype)
            assert torch.allclose(m.module.cells(x2)[:, r, t], base[:, r, t], atol=1e-5)
        # the last cell's cone is everything: the first cell matters
        x3 = x.clone()
        x3[:, 0, 0] += 1.0
        assert not torch.allclose(m.module.cells(x3)[:, -1, -1], base[:, -1, -1], atol=1e-5)


@pytest.mark.parametrize("layout,one_layer_diagonal", [("grid", False), ("axial", False), ("grid_full", True)])
def test_diagonal_paths(layout, one_layer_diagonal):
    """Axial-by-head grid: cell (t-1, r-1) reaches (t, r) only through two layers (token axis, then trader axis or the
    reverse); the full cone reaches it in one."""
    for layers, expect in [(1, one_layer_diagonal), (2, True)]:
        ctx, m = _grid(layout, layers=layers)
        x = torch.as_tensor(ctx.tokens)
        with torch.no_grad():
            x2 = x.clone()
            x2[:, 0, 0] += 3.0
            moved = not torch.allclose(m.module.cells(x2)[:, 1, 1], m.module.cells(x)[:, 1, 1], atol=1e-6)
        assert moved == expect, (layout, layers)


def test_token_heads_knob():
    from aggbench.trader_transformer import grid_masks
    ms = grid_masks(3, 4, "grid", heads=4, token_heads=3, layers=2)
    assert ms[0].shape == (4, 12, 12)
    assert (ms[0][:3] == ms[0][0]).all() and not (ms[0][3] == ms[0][0]).all()
    _, m = _grid("grid", token_heads=4)            # all heads on the token axis: no information across traders
    ctx, _ = _grid("grid")
    x = torch.as_tensor(ctx.tokens)
    with torch.no_grad():
        x2 = x.clone()
        x2[:, :-1] += 1.0
        assert torch.allclose(m.module(x2)[:, -1], m.module(x)[:, -1], atol=1e-6)


def test_grid_fit_predict_and_layout_check():
    ctx, m = _grid("grid")
    m.fit(ctx, np.zeros(len(ctx.order)))
    assert m.predict(ctx)["logit"].shape == (len(ctx.order),)
    with pytest.raises(ValueError, match="layout"):
        m.predict(_ctx())
    with pytest.raises(ValueError, match="kv"):
        _grid("grid", token_state="recurrent")


# ----------------------------------------------------------------------------- review fixes (gemini vision review V1, V3, V4)

def test_device_cpu_explicit_and_cuda_when_present():
    ctx = _ctx()
    m = _model(ctx, device="cpu")
    assert m.device == "cpu"
    m.fit(ctx, np.zeros(32))
    assert m.predict(ctx)["logit"].shape == (32,)
    assert next(m.module.parameters()).device.type == "cpu"           # back on cpu for the controls
    auto = _model(ctx)
    assert auto.device == ("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        auto.fit(ctx, np.zeros(32))
        assert np.allclose(auto.predict(ctx)["logit"], auto.predict(ctx)["logit"])


@pytest.mark.parametrize("layout", ["trader", "grid"])
def test_input_changes_with_visibility(layout):
    b = make_env("E2").sample("smoke", 16)
    tok = {v: trader_context(b, v, layout=layout).tokens for v in ("none", "noisy", "full")}
    assert tok["none"].shape[-1] < tok["noisy"].shape[-1] == tok["full"].shape[-1]
    assert not np.array_equal(tok["noisy"], tok["full"])


def test_deepsets_trader_encoder():
    b = make_env("E2").sample("smoke", 32)
    ctx = trader_context(b, "noisy", trader_encoder="deepsets")
    assert ctx.tokens.ndim == 4 and ctx.layout == "trader"
    m = TraderTransformer(0, ctx.tokens.shape[-1], width=16, layers=2, heads=2, steps=5, batch=16,
                          trader_encoder="deepsets")
    m.module.eval()
    x = torch.as_tensor(ctx.tokens)
    with torch.no_grad():
        base = m.module(x)
        # empty piece cells do not matter; the piece order inside a trader does not matter
        x2 = x.clone()
        x2[..., 1:] += (1 - x2[..., :1]) * 5.0
        assert torch.allclose(m.module(x2), base, atol=1e-5)
        perm = torch.randperm(x.shape[2])
        assert torch.allclose(m.module(x[:, :, perm]), base, atol=1e-5)
        x3 = x.clone()
        x3[:, 1:] += 1.0                                                  # causal: position 0 sees only itself
        assert torch.allclose(m.module(x3)[:, 0], base[:, 0], atol=1e-5)
    m.fit(ctx, b.truth)
    with pytest.raises(ValueError, match="deepsets"):
        m.predict(trader_context(b, "noisy"))
