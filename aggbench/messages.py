"""High-dimensional messages (2026-10-20): each agent sends a vector value (and key) computed from its own
observation only; the aggregator sees messages, never raw observations.

check_locality(encode, obs, keys) verifies the rule: perturbing other agents' inputs must not change
agent i's message. `encode(obs, keys) -> messages` takes arrays shaped like Batch.agent_inputs()
((n, n_agents, ...)) and returns (n, n_agents, m).
"""
import numpy as np


def check_locality(encode, obs, keys, trials=3, seed=0, atol=1e-9):
    """Raise AssertionError if any agent's message depends on another agent's observation or key."""
    rng = np.random.default_rng(seed)
    base = np.asarray(encode(obs, keys))
    n_agents = obs.shape[1]
    for _ in range(trials):
        i = int(rng.integers(n_agents))
        o2, k2 = obs.copy(), keys.copy()
        others = [j for j in range(n_agents) if j != i]
        finite = np.isfinite(o2[:, others])
        noise = rng.normal(size=o2[:, others].shape)
        o2[:, others] = np.where(finite, o2[:, others] + noise, o2[:, others])
        k2[:, others] = k2[:, others] + rng.normal(size=k2[:, others].shape)
        out = np.asarray(encode(o2, k2))
        if not np.allclose(out[:, i], base[:, i], atol=atol, equal_nan=True):
            raise AssertionError(f"agent {i}'s message changed when other agents' inputs changed")
    return True
