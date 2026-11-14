"""Baselines, metrics and the results schema."""
import numpy as np
import pytest

from aggbench import make_env
from aggbench.baselines import (all_baselines, extremized, fit_extremize, linear_pool, log_pool,
                                sum_logodds)
from aggbench.envs import View
from aggbench.metrics import gaussian_kl, per_episode, summarize
from aggbench.results import make_result, split_record, validate
from aggbench.verify import oracle_check


def _view(logits, prior_logit=0.0):
    logits = np.asarray(logits, float)
    return View("bernoulli", {"logit": logits}, np.zeros(logits.shape + (0,)), {"prior_logit": prior_logit})


def test_bernoulli_pools_by_hand():
    v = _view([[1.0, 3.0], [-2.0, 0.0]], prior_logit=0.5)
    assert np.allclose(log_pool(v)["logit"], [2.0, -1.0])
    assert np.allclose(sum_logodds(v)["logit"], [3.5, -2.5])
    assert np.allclose(extremized(v, 2.0)["logit"], [3.5, -2.5])
    p = (1 / (1 + np.exp(-v.reports["logit"]))).mean(1)
    assert np.allclose(linear_pool(v)["logit"], np.log(p / (1 - p)))


def test_log_pool_is_bayes_when_all_agents_see_everything():
    env = make_env("E2", n_agents=3, n_pieces=6, pieces_per_agent=(6, 6))
    batch = env.sample("dev", 50)
    assert np.allclose(log_pool(batch.view())["logit"], batch.bayes["logit"])


def test_gaussian_sum_is_bayes_without_overlap_or_shared_noise():
    env = make_env("E1", n_agents=3, d=3, obs_dim=1, coverage=3, n_factors=0)
    batch = env.sample("dev", 30)
    f = sum_logodds(batch.view())
    assert np.allclose(f["mean"], batch.bayes["mean"]) and np.allclose(f["prec"], batch.bayes["prec"])


def test_extremize_fit_recovers_known_alpha():
    # one agent whose report is the Bayes logit divided by 3: the best alpha is 3
    batch = make_env("E2", n_agents=1, n_pieces=6, pieces_per_agent=(6, 6)).sample("train", 4000)
    v = _view(batch.bayes["logit"][:, None] / 3)
    assert abs(fit_extremize(v, batch.truth) - 3.0) < 0.05


@pytest.mark.parametrize("name", ["E1", "E2", "E3-collude"])
def test_metric_identities(name):
    env = make_env(name)
    tr, dv = env.sample("train", 300), env.sample("dev", 1500)
    forecasts, _ = all_baselines(tr, dv)
    zero = per_episode(forecasts["bayes"], dv)
    assert np.allclose(zero["regret"], 0) and np.allclose(zero["kl"], 0, atol=1e-12)
    for f in forecasts.values():
        m = per_episode(f, dv)
        assert (m["kl"] > -1e-9).all()
        s = summarize(f, dv, n_boot=300)
        # E[regret] = E[KL]; the two estimates agree within the regret CI half-width (generous)
        half = s["regret"]["ci_high"] - s["regret"]["ci_low"]
        assert abs(s["regret"]["mean"] - s["kl"]["mean"]) < half + 1e-9
        assert s["regret"]["ci_low"] <= s["regret"]["mean"] <= s["regret"]["ci_high"]


def test_gaussian_kl_known_value():
    m0, p0 = np.zeros((1, 2)), np.eye(2)[None]
    m1, p1 = np.ones((1, 2)), 2 * np.eye(2)[None]
    # 0.5*(tr(2I) + 1^T 2I 1 - 2 + log det I - log det 2I) = 0.5*(4 + 4 - 2 - 2 log 2)
    assert np.isclose(gaussian_kl(m0, p0, m1, p1)[0], 0.5 * (6 - 2 * np.log(2)))


def test_result_schema():
    env = make_env("E2")
    tr, dv = env.sample("train", 100), env.sample("dev", 100)
    forecasts, _ = all_baselines(tr, dv)
    methods = {k: summarize(f, dv, n_boot=100) for k, f in forecasts.items()}
    common = dict(run_id="t", track="EXPLORATORY", env=env, visibility="full",
                  splits={"train": split_record(tr, "train"), "eval": split_record(dv, "dev")},
                  oracle_check=oracle_check(env), allow_dirty=True)  # tests also run on uncommitted trees
    r = make_result(methods=methods, **common)
    assert r["harness"]["git_commit"] and len(r["harness"]["sha256"]) == 64
    with pytest.raises(ValueError):  # a learned arm without its random-network control
        make_result(methods=methods, learned_arms=["log_pool"], **common)
    ok = make_result(methods=methods, learned_arms=["log_pool"],
                     controls={"log_pool": {"random_network": "linear_pool", "untrained_init": "sum_logodds"}},
                     **common)
    validate(ok)
    with pytest.raises(ValueError):
        make_result(methods=methods, smoke=True, **common)  # smoke output must be track SMOKE


def test_result_rejects_dirty_harness(monkeypatch):
    import aggbench.results as R
    env = make_env("E2")
    dv = env.sample("dev", 30)
    methods = {"bayes": summarize(dv.bayes, dv, n_boot=50)}
    common = dict(run_id="t", track="EXPLORATORY", env=env, visibility="full",
                  splits={"eval": split_record(dv, "dev")}, oracle_check=oracle_check(env), methods=methods)
    clean = dict(R.harness_info(), git_dirty=False)
    monkeypatch.setattr(R, "harness_info", lambda: dict(clean, git_dirty=True))
    with pytest.raises(ValueError, match="git_dirty"):
        make_result(**common)
    r = make_result(allow_dirty=True, **common)
    assert r["harness"]["dirty_allowed"] is True
    validate(r)                                   # the explicit, recorded allowance survives re-validation
    r["harness"].pop("dirty_allowed")
    with pytest.raises(ValueError, match="git_dirty"):
        validate(r)
    validate(r, allow_dirty=True)
    monkeypatch.setattr(R, "harness_info", lambda: clean)
    assert "dirty_allowed" not in make_result(**common)["harness"]


def test_harness_info_without_git_uses_env_and_sha(monkeypatch):
    """A checkout without git metadata (remote GPU VM): commit from AGGBENCH_GIT_COMMIT, clean iff the sources
    hash to AGGBENCH_EXPECT_SHA256."""
    import subprocess
    import aggbench.results as R
    real = R.harness_info()
    monkeypatch.setattr(R.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="", stderr=""))
    monkeypatch.setenv("AGGBENCH_GIT_COMMIT", "abc123")
    monkeypatch.setenv("AGGBENCH_EXPECT_SHA256", real["sha256"])
    h = R.harness_info()
    assert h["git_commit"] == "abc123" and h["git_dirty"] is False and "git_source" in h
    monkeypatch.setenv("AGGBENCH_EXPECT_SHA256", "0" * 64)
    assert R.harness_info()["git_dirty"] is True


def test_report_jobs_resume(tmp_path):
    import json
    from aggbench.report import _job_file, run_jobs
    job = ("Z6", "unaligned", "none", 20000, 2, False, 3000)
    _job_file(tmp_path, job).write_text(json.dumps({"saved": True}))
    assert run_jobs([job], workers=1, job_dir=tmp_path) == [{"saved": True}]
