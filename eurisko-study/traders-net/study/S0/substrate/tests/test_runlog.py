"""RunLog round trip and the train/eval non-overlap guard (S0 worker 4)."""
import json, os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import panel, runlog


def test_roundtrip(tmp_path):
    with runlog.RunLog(tmp_path, "r1", meta={"seed": 3}) as rl:
        rl.log(0, {"loss": 0.5, "brier": {"all": 0.2, "ca30": 0.25}})
        rl.log(100, panel.panel_row([0.9, 0.2], [1, 0], wealth=[1, 0]))
    d = runlog.read_run(tmp_path / "r1.csv")
    assert d[("loss", "all")] == [(0, 0.5)]
    assert d[("brier", "ca30")] == [(0, 0.25)]
    assert d[("brier", "all")][1] == (100, pytest.approx(0.025))
    assert d[("gini", "all")][0][1] == pytest.approx(0.5)
    assert json.load(open(tmp_path / "r1.meta.json"))["seed"] == 3
    # appending to an existing run keeps one header
    with runlog.RunLog(tmp_path, "r1") as rl:
        rl.log(200, {"loss": 0.4})
    lines = open(tmp_path / "r1.csv").read().splitlines()
    assert lines[0] == ",".join(runlog.FIELDS) and sum(l.startswith("run_id") for l in lines) == 1


def test_guard_single_pass():
    g = runlog.PrequentialGuard()
    keys = [(q, t) for t in range(5) for q in range(3)]
    for t in range(5):
        batch = [(q, t) for q in range(3)]
        g.eval_many(batch); g.train_many(batch)
    assert g.trained == set(keys)
    with pytest.raises(AssertionError):
        g.eval((0, 2))               # eval on train data
    with pytest.raises(AssertionError):
        g.train((0, 99))             # train before test
    with pytest.raises(AssertionError):
        g.train((0, 1))              # second epoch in single-pass mode


def test_guard_stress_counts_in_sample():
    g = runlog.PrequentialGuard(stress=True)
    for ep in range(2):
        for t in range(4):
            g.eval((0, t)); g.train((0, t))
    assert g.in_sample_evals == 4
