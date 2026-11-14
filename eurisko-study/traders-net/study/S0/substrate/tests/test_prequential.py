"""Tests for prequential.py. Run: PYTHONPATH=<pylib>:.. python3 tests/test_prequential.py (or pytest)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from streams import make_questions
from model import Net, Logistic
from prequential import ArrayStream, run_prequential, run_multiepoch, assert_disjoint, per_family

torch.set_num_threads(1)
Q, W, T = 16, 8, 8 + 256


def _run(seed, delay=None, B=4):
    torch.manual_seed(seed)
    b, f, _ = make_questions(Q, T, seed)
    s = ArrayStream(b, f, W, delay=delay)
    m = Net(Q, W, d=16, H=2, L=1)
    return s, run_prequential(m, s, torch.optim.Adam(m.parameters(), 1e-3), B=B)


def test_seeded_determinism():
    _, r1 = _run(3)
    _, r2 = _run(3)
    _, r3 = _run(4)
    assert np.array_equal(r1["z"], r2["z"])
    assert not np.array_equal(r1["z"], r3["z"])


def test_predict_before_train_once():
    _, r = _run(1)
    ps, ts = r["pred_step"], r["train_step"]
    assert (ps >= 0).all() and (ts >= 0).all()
    assert (ts >= ps).all()          # label used only at or after the step that scored it
    assert (ts == ps).all()          # delay 0: trained in the same block, after the test


def test_delay_windows_and_training():
    D = np.where(np.arange(Q) % 2 == 0, 0, 64)
    s, r = _run(1, delay=D, B=8)
    ps, ts = r["pred_step"], r["train_step"]
    ok = ts >= 0
    assert (ts[ok] - ps[ok] >= 0).all()
    # delayed questions: trained exactly ceil-block later, untrained tail = last 64 positions
    assert (ts[:-64, 1::2] - ps[:-64, 1::2] == 64 // 8).all()
    assert (ts[-64:, 1::2] < 0).all()
    # window of a delayed question ends at bit index i+W-1-D (only revealed bits)
    i = 100
    w = s.window(np.array([i]))[0]
    assert torch.equal(w[1], torch.tensor(s.bits[1, i - 64:i - 64 + W], dtype=torch.float32))
    assert torch.equal(w[0], torch.tensor(s.bits[0, i:i + W], dtype=torch.float32))


def test_multiepoch_disjoint_and_raises():
    b, f, _ = make_questions(Q, T, 2)
    s = ArrayStream(b, f, W)
    try:
        assert_disjoint(s, np.arange(0, 100), np.arange(99, 150))
        raise RuntimeError("overlap not detected")
    except AssertionError:
        pass
    torch.manual_seed(0)
    m = Net(Q, W, d=16, H=2, L=1)
    rows, info = run_multiepoch(m, s, torch.optim.Adam(m.parameters(), 1e-3), epochs=2, B=32)
    assert info["n_train"] + info["n_eval"] == s.n_pos and len(rows) == 3


def test_per_family_known_values():
    z = np.zeros((4, 2)); y = np.array([[1, 0]] * 4); m = np.ones((4, 2), bool)
    b, r = per_family(z, y, m, np.array([0, 1]), n_fam=2)
    assert np.allclose(b, np.log(2)) and np.allclose(r, 0.25)


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
