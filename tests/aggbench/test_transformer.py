"""full_context_transformer: interface, controls, learns structure no report baseline can (needs torch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aggbench import make_env  # noqa: E402
from aggbench.metrics import per_episode  # noqa: E402
from aggbench.baselines import oracle_clearing  # noqa: E402
from aggbench.transformer import decompose_regret, evaluate_full_context_transformer, full_context  # noqa: E402


@pytest.mark.parametrize("name,alignment", [("E1", "unaligned"), ("E2", "aligned"), ("E2", "unaligned"),
                                            ("E4", "unaligned"), ("Z1", "unaligned"), ("Z7", "aligned")])
def test_tokens_one_per_agent(name, alignment):
    batch = make_env(name, allow_heldout=True).sample("dev", 5)
    for vis in ("none", "noisy", "full"):
        ctx = full_context(batch, vis, alignment)
        assert ctx.tokens.shape[:2] == (5, batch.obs.shape[1]) and np.isfinite(ctx.tokens).all()


def test_learns_xor_beyond_every_report_baseline():
    """Z6 with full keys: every report is the prior, so every report-only method has regret = prior's regret;
    the transformer reading the raw readings and keys must do much better, and its controls must not."""
    env = make_env("Z6")
    rec = evaluate_full_context_transformer(env, n_train=6000, n_eval=600, visibility="full", alignment="aligned",
                                            steps=600, n_boot=100, threads=2, regime="pool")
    m = rec["methods"]
    dv = env.sample("dev", 600)
    prior_regret = per_episode({"logit": np.zeros(600)}, dv)["regret"].mean()
    trained = m["full_context_transformer"]["regret"]["mean"]
    assert trained < 0.5 * prior_regret
    assert m["full_context_transformer:random_network"]["regret"]["mean"] > trained + 0.05
    assert m["full_context_transformer:untrained_init"]["regret"]["mean"] > trained + 0.05
    assert set(rec["controls"]["full_context_transformer"]) == {"random_network", "untrained_init"}


def test_gaussian_head():
    env = make_env("E1")
    rec = evaluate_full_context_transformer(env, n_train=2000, n_eval=200, steps=200, n_boot=50, threads=2,
                                            regime="pool")
    f = rec["forecasts"]["trained"]
    assert f["mean"].shape == (200, 4) and f["prec"].shape == (200, 4, 4)
    assert (np.linalg.eigvalsh(f["prec"]) > 0).all()
    assert np.isfinite(rec["methods"]["full_context_transformer"]["regret"]["mean"])


def test_decompose_regret_attributes_loss_to_messages():
    """Z6 with scalar reports as messages: every report is the prior, so the best-from-messages forecast is the
    prior; the whole regret of a report aggregator is information loss, aggregation error is about 0. With raw
    readings plus keys as messages, the information loss shrinks."""
    env = make_env("Z6")
    tr, dv = env.sample("train", 4000), env.sample("dev", 500)
    agg = oracle_clearing(dv)                                  # the exact reports ceiling (= prior here)
    rep = lambda b: b.reports["logit"][..., None]
    d = decompose_regret(agg, rep(tr), rep(dv), tr, dv, steps=300, width=32, n_boot=100, threads=2)
    assert abs(d["aggregation_error"]["mean"]) < 0.02
    assert abs(d["information_loss"]["mean"] - d["total"]["mean"]) < 0.02
    rich = lambda b: np.concatenate([b.obs, b.keys["full"]], -1)
    d2 = decompose_regret(agg, rich(tr), rich(dv), tr, dv, steps=600, width=32, n_boot=100, threads=2)
    assert d2["information_loss"]["mean"] < 0.5 * d["information_loss"]["mean"]
    assert d2["predictor"]["labels"].startswith("bayes")


def test_report_job_validates_and_saves(tmp_path):
    """A transformer job's result passes results.validate and is saved for resume."""
    from aggbench.report import _fct_job_saved, _job_file
    job = ("Z6", "unaligned", "none", 600, 2, True, 60)
    r = _fct_job_saved((job, tmp_path))
    assert "bayes" in r["methods"] and r["learned_arms"] == ["full_context_transformer"]
    assert _job_file(tmp_path, job).exists()


def test_fresh_stream_never_reuses_and_is_deterministic():
    from aggbench.transformer import FreshStream
    env = make_env("E2")
    a = list(FreshStream(env, "none", "unaligned", batch=32, steps=4, samplers=0))
    b = list(FreshStream(env, "none", "unaligned", batch=32, steps=4, samplers=2))
    assert len(a) == 4 and all(np.array_equal(x[0], y[0]) for x, y in zip(a, b))
    flat = np.concatenate([t.reshape(len(t), -1) for t, _ in a])
    assert len(np.unique(flat, axis=0)) == len(flat)                 # 128 distinct episodes
    stream = FreshStream(env, "none", "unaligned", batch=32, steps=4, samplers=0)
    assert stream.n_episodes == 128 and not np.array_equal(stream.stats_tokens[:32], a[0][0])


def test_fresh_regime_job_in_a_process_pool(tmp_path):
    """The report runs fresh jobs inside ProcessPoolExecutor workers, which start their own sampler pools."""
    from concurrent.futures import ProcessPoolExecutor
    from aggbench.report import _fct_job_saved, _job_file
    job = ("Z6", "unaligned", "none", 0, 2, True, 40, "fresh")
    with ProcessPoolExecutor(1) as ex:
        r = ex.submit(_fct_job_saved, (job, tmp_path)).result()
    assert r["extra"]["regime"] == "fresh" and r["splits"]["train"]["n"] == 40 * 512
    assert _job_file(tmp_path, job).name.endswith("-fresh.json")
