"""Smoke tests of the learned literature methods (aggbench.external.learned, tabpfn_adapter, report). Need torch;
the TabPFN test also needs the tabpfn package and cached v2 weights (~/.cache/tabpfn)."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aggbench import make_env  # noqa: E402
from aggbench.controls import run_with_controls  # noqa: E402
from aggbench.envs import View  # noqa: E402
from aggbench.external.learned import ScalarFreshStream, locality, maker, rich_inputs  # noqa: E402
from aggbench.transformer import FreshStream  # noqa: E402

ARCHS = ("deepsets", "set_transformer", "tarmac_r1", "tarmac_r3")
SMALL = {"steps": 6, "batch": 32, "eval_every": 2, "patience": 4}


def _finite(arms):
    for f in arms.values():
        for v in f.values():
            assert np.isfinite(v).all()


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("env_name", ["E2", "E1"])
def test_rich_pool_regime_permutation_invariance_and_locality(arch, env_name):
    b = make_env(env_name).sample("train", 300)
    x = rich_inputs(b, "noisy", "unaligned")
    _finite(run_with_controls(maker(arch, x, **SMALL), x, b.truth, x))
    agg = maker(arch, x, **SMALL)(0).fit(x, b.truth)
    assert agg.best_step is not None and agg.best_step <= SMALL["steps"]
    perm = np.random.default_rng(0).permutation(x.tokens.shape[1])
    f0, f1 = agg.predict(x), agg.predict(replace(x, tokens=x.tokens[:, perm]))
    for k in f0:
        np.testing.assert_allclose(f0[k], f1[k], rtol=1e-4, atol=1e-4)
    if arch != "set_transformer":
        assert locality(agg, x) is True


@pytest.mark.parametrize("env_name", ["E2", "E1"])
def test_fresh_regime_trains_on_new_episodes(env_name):
    env = make_env(env_name)
    stream = FreshStream(env, "noisy", "unaligned", batch=16, steps=3, samplers=0)
    dv = rich_inputs(env.sample("dev", 50), "noisy", "unaligned")
    _finite(run_with_controls(maker("tarmac_r1", stream), stream, None, dv))
    seen = [xb.shape for xb, _ in stream]
    assert len(seen) == 3 and seen[0][0] == 16
    scalar = ScalarFreshStream(env, batch=16, steps=3)
    _finite(run_with_controls(maker("set_transformer", scalar), scalar, None, env.sample("dev", 50).view("none")))


def _permuted(view, perm):
    return View(view.kind, {k: v[:, perm] for k, v in view.reports.items()}, view.keys[:, perm], dict(view.public))


@pytest.mark.parametrize("env_name", ["E2", "E1"])
def test_scalar_smoke_and_permutation_invariance(env_name):
    b = make_env(env_name).sample("train", 300)
    v = b.view("noisy")
    _finite(run_with_controls(maker("set_transformer", v, **SMALL), v, b.truth, v))
    agg = maker("set_transformer", v, **SMALL)(0).fit(v, b.truth)
    perm = np.random.default_rng(0).permutation(v.keys.shape[1])
    f0, f1 = agg.predict(v), agg.predict(_permuted(v, perm))
    for k in f0:
        np.testing.assert_allclose(f0[k], f1[k], rtol=1e-4, atol=1e-4)


def test_tabpfn_smoke():
    pytest.importorskip("tabpfn")
    if not list((Path.home() / ".cache" / "tabpfn").glob("tabpfn-v2-classifier*.ckpt")):
        pytest.skip("TabPFN v2 weights not cached")
    from aggbench.external.tabpfn_adapter import run_tabpfn_with_controls
    b = make_env("E2").sample("train", 120)
    v = rich_inputs(b, "noisy", "unaligned")
    arms = run_tabpfn_with_controls(v, b.truth, v, n_context=100)
    assert set(arms) == {"trained", "control:shuffled_labels", "control:random_network"}
    for f in arms.values():
        assert f["logit"].shape == (120,) and np.isfinite(f["logit"]).all()


def test_report_pif_reads_provenance_keys_not_own_model():
    """In rich cells the PIF row must use the noisy provenance keys, never the agents' own sensor models."""
    from aggbench.external import closed_form as cf
    from aggbench.external.report import run_cell
    from aggbench.metrics import summarize
    r = run_cell("E2", "noisy-aligned", n_train=200, n_eval=300, n_boot=50, part="closed", allow_dirty=True)
    dv = make_env("E2").sample("dev", 300)
    want = summarize(cf.pif_oracle(dv.view("noisy")), dv, n_boot=50)["regret"]["mean"]
    assert abs(r["methods"]["pif_oracle"]["regret"]["mean"] - want) < 1e-12


def test_report_learned_part_smoke():
    from aggbench.external.report import merge, run_cell, table
    kw = dict(n_train=200, n_eval=100, n_boot=20, allow_dirty=True, steps=3,
              skip=("tabpfn", "set_transformer", "tarmac_r3", "deepsets"))
    a = run_cell("E2", "noisy-unaligned", part="closed", **kw)
    b = run_cell("E2", "noisy-unaligned", part="learned", **kw)
    m = merge([a, b])
    assert {"tarmac_r1", "tarmac_r1_pool"} <= set(m["learned_arms"])
    assert m["extra"]["locality_check"]["tarmac_r1"] is True
    assert "tarmac_r1_pool" in table([m])
