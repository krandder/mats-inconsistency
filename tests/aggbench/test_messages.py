import numpy as np
import pytest

from aggbench import envs
from aggbench.messages import check_locality


def _batch():
    env = envs.make_env("E2")
    return env.sample("smoke", 16)


def test_agent_inputs_shapes_and_copy():
    b = _batch()
    obs, keys, public = b.agent_inputs("noisy")
    assert obs.shape[:2] == keys.shape[:2] == (len(b), b.obs.shape[1])
    obs[...] = 0.0
    assert not np.all(b.obs[np.isfinite(b.obs)] == 0.0)


def test_local_encoder_passes():
    b = _batch()
    obs, keys, _ = b.agent_inputs("full")
    local = lambda o, k: np.concatenate([np.nan_to_num(o), k], axis=-1)
    assert check_locality(local, obs, keys)


def test_mixing_encoder_fails():
    b = _batch()
    obs, keys, _ = b.agent_inputs("full")
    def mixing(o, k):
        m = np.nan_to_num(o)
        return m + m.mean(axis=1, keepdims=True)
    with pytest.raises(AssertionError):
        check_locality(mixing, obs, keys)
