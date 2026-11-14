"""Controls for learned aggregators (needs torch: run with the trader-cat venv, see aggbench/README.md)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aggbench import make_env  # noqa: E402
from aggbench.controls import random_weights, run_with_controls  # noqa: E402


class TinyAgg:
    """Weighted mean of report log-odds, one learned weight per position."""

    def __init__(self, seed, n_agents):
        torch.manual_seed(seed)
        self.module = torch.nn.Linear(n_agents, 1)

    def fit(self, view, truth, steps=200):
        x = torch.tensor(view.reports["logit"], dtype=torch.float32)
        y = torch.tensor(truth, dtype=torch.float32)
        opt = torch.optim.Adam(self.module.parameters(), lr=0.05)
        for _ in range(steps):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(self.module(x)[:, 0], y).backward()
            opt.step()

    def predict(self, view):
        with torch.no_grad():
            x = torch.tensor(view.reports["logit"], dtype=torch.float32)
            return {"logit": self.module(x)[:, 0].double().numpy()}


def test_random_weights_changes_weights_keeps_scale():
    agg = TinyAgg(0, 50)
    rnd = random_weights(agg, seed=1)
    w, wr = agg.module.weight, rnd.module.weight
    assert not torch.allclose(w, wr)
    assert abs(wr.std().item() / w.std().item() - 1) < 0.5
    assert torch.equal(agg.module.weight, w)  # original untouched
    assert torch.equal(random_weights(agg, 1).module.weight, wr)  # deterministic


def test_run_with_controls():
    env = make_env("E2")
    tr, dv = env.sample("train", 500), env.sample("dev", 200)
    make = lambda seed: TinyAgg(seed, 5)
    arms = run_with_controls(make, tr.view("none"), tr.truth, dv.view("none"), seed=0)
    assert set(arms) == {"trained", "control:untrained_init", "control:random_network"}
    for f in arms.values():
        assert f["logit"].shape == (200,) and np.isfinite(f["logit"]).all()
    again = run_with_controls(make, tr.view("none"), tr.truth, dv.view("none"), seed=0)
    for k in arms:
        np.testing.assert_allclose(arms[k]["logit"], again[k]["logit"])
