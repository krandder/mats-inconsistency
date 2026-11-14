"""agent_inputs(alignment="unaligned"): same information, no shared slot positions."""
import dataclasses

import numpy as np
import pytest

from aggbench import make_env
from aggbench.envs import alignment_transforms, gaussian_global, slot_layout
from aggbench.metrics import per_episode
from aggbench.verify import brute_force_bernoulli, brute_force_gaussian


def _undo(batch, obs_u, keys_u, T):
    """Invert the hidden transforms (T orthogonal: inverse = transpose)."""
    S, obs_slotted, blocks = slot_layout(batch)
    Ti = np.swapaxes(T, -1, -2)
    perm = np.isin(T, (0.0, 1.0)).all((-1, -2))

    def back(x):
        idx = np.argmax(Ti, -1)
        p = np.take_along_axis(x, idx[..., None], axis=2)
        r = np.einsum("enab,enbw->enaw", Ti, np.nan_to_num(x))
        return np.where(perm[..., None, None], p, r)

    obs = back(obs_u[..., None])[..., 0] if obs_slotted else obs_u
    keys = keys_u.copy()
    if keys.shape[-1]:
        for start, s, w in blocks:
            blk = keys[..., start:start + s * w].reshape(*keys.shape[:2], s, w)
            keys[..., start:start + s * w] = back(blk).reshape(*keys.shape[:2], s * w)
    return obs, keys


CASES = [("E1", False), ("E1", True), ("E2", False), ("E3-sybil", False), ("E3-collude", False),
         ("E4", False), ("Z1", False), ("Z4", False)]


@pytest.mark.parametrize("name,mixing", CASES)
def test_unaligned_is_invertible_and_hidden(name, mixing):
    batch = make_env(name).sample("dev", 40)
    for vis in ("none", "noisy", "full"):
        obs_a, keys_a, _ = batch.agent_inputs(vis)
        obs_u, keys_u, _ = batch.agent_inputs(vis, alignment="unaligned", mixing=mixing)
        assert obs_u.shape == obs_a.shape and keys_u.shape == keys_a.shape
        T = alignment_transforms(batch, mixing)
        obs_b, keys_b = _undo(batch, obs_u, keys_u, T)
        np.testing.assert_allclose(obs_b, obs_a, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(keys_b, keys_a, atol=1e-12)
        if vis == "full":
            changed = (not np.allclose(obs_u, obs_a, equal_nan=True)) or not np.allclose(keys_u, keys_a)
            assert changed
    # deterministic in the episode seed, different across agents and episodes
    T1, T2 = alignment_transforms(batch, mixing), alignment_transforms(make_env(name).sample("dev", 40), mixing)
    np.testing.assert_array_equal(T1, T2)
    flat = T1.reshape(-1, *T1.shape[-2:])
    S = flat.shape[-1]
    distinct = len({t.round(12).tobytes() for t in flat})
    assert distinct >= min(len(flat), np.prod(np.arange(1, S + 1))) // 2   # varies across agents and episodes
    # default is unchanged
    np.testing.assert_array_equal(batch.agent_inputs("full")[0], batch.obs)


@pytest.mark.parametrize("name", ["E2", "E3-sybil", "E3-collude"])
def test_unaligned_plus_hidden_permutation_gives_the_same_posterior(name):
    batch = make_env(name).sample("dev", 60)
    obs_u, keys_u, _ = batch.agent_inputs("full", alignment="unaligned")
    obs_b, _ = _undo(batch, obs_u, keys_u, alignment_transforms(batch))
    assert brute_force_bernoulli(dataclasses.replace(batch, obs=obs_b)) < 1e-12
    assert brute_force_bernoulli(dataclasses.replace(batch, obs=obs_u)) > 1e-3  # position alone is now wrong


def test_e1_rotation_keeps_information_in_the_view():
    """E1 with orthogonal mixing: s' = Q s. With the provenance key at `full` (Q A, Q B, log sigma) the exact
    posterior from the unaligned view alone (no Q) equals Bayes. The own model (Q A, Q R Q^T) gives each agent its
    exact local posterior but not the shared-factor structure, so provenance `none` cannot reach Bayes."""
    batch = make_env("E1").sample("dev", 30)
    obs_u, keys_u, _, own = batch.agent_inputs("full", alignment="unaligned", mixing=True, own_model=True)
    c = batch.config
    N, m, d, r = obs_u.shape[1], obs_u.shape[2], c["d"], c["n_factors"]
    for e in range(30):
        A = keys_u[e, :, :m * d].reshape(N, m, d)
        Bf = keys_u[e, :, m * d:m * d + m * r].reshape(N, m, r)
        sig = np.exp(keys_u[e, :, -1])
        mean, prec = gaussian_global(A, Bf, sig, obs_u[e])
        assert np.abs(mean - batch.bayes["mean"][e]).max() < 1e-9
        assert np.abs(prec - batch.bayes["prec"][e]).max() < 1e-8
        # own model: A_i and R_i reproduce the agent's local posterior exactly
        Ao = own[e, :, :m * d].reshape(N, m, d)
        R = own[e, :, m * d:].reshape(N, m, m)
        for i in range(N):
            P = Ao[i].T @ np.linalg.solve(R[i], Ao[i]) + np.eye(d)
            assert np.abs(P - batch.local["prec"][e, i]).max() < 1e-8
    assert brute_force_gaussian(batch) < 1e-9
    with pytest.raises(ValueError):
        make_env("E2").sample("dev", 2).agent_inputs("full", alignment="unaligned", mixing=True)


def _dedup_by_position(obs, keys, l0):
    """Slot-equivariant dedup: piece at position k counted once, from the first agent that has it, with the
    strength that agent's key gives for position k."""
    n, N, K = obs.shape
    mu = keys[..., K:2 * K]
    out = np.full(n, l0)
    for e in range(n):
        for k in range(K):
            have = np.flatnonzero(np.isfinite(obs[e, :, k]))
            if len(have):
                i = have[0]
                out[e] += 2 * mu[e, i, k] * obs[e, i, k]
    return {"logit": out}


def _dedup_by_value(obs, keys, l0):
    """Dedup by exact value: the same piece seen by two agents is the same float in both observations."""
    n, N, K = obs.shape
    mu = keys[..., K:2 * K]
    out = np.full(n, l0)
    for e in range(n):
        seen = {}
        for i in range(N):
            for k in np.flatnonzero(np.isfinite(obs[e, i])):
                seen[float(obs[e, i, k])] = 2 * mu[e, i, k] * obs[e, i, k]
        out[e] += sum(seen.values())
    return {"logit": out}


def test_position_dedup_fails_when_unaligned():
    batch = make_env("E2").sample("dev", 400)
    l0 = batch.public["prior_logit"]
    obs_a, keys_a, _ = batch.agent_inputs("full")
    obs_u, keys_u, _ = batch.agent_inputs("full", alignment="unaligned")
    aligned = per_episode(_dedup_by_position(obs_a, keys_a, l0), batch)["regret"].mean()
    unaligned = per_episode(_dedup_by_position(obs_u, keys_u, l0), batch)["regret"].mean()
    assert abs(aligned) < 1e-9          # aligned slots hand the overlap to a slot-indexed model: exactly Bayes
    assert unaligned > 0.05             # unaligned: positions collide at random
    # floor (README): from unaligned obs and own models alone (provenance none), identical values reveal a
    # shared piece, so Bayes is reachable
    obs_n, keys_n, _, own = batch.agent_inputs("none", alignment="unaligned", own_model=True)
    assert keys_n.shape[-1] == 0
    by_value = per_episode(_dedup_by_value(obs_n, own, l0), batch)["regret"].mean()
    assert abs(by_value) < 1e-9


def test_unseen_pieces_not_identifiable_by_position():
    """Aligned: across episodes, piece k is unseen by agent i with the same rate for every k. Unaligned: for one
    episode, the set of NaN positions is a uniformly random subset per agent, so two agents' NaN patterns overlap
    as random subsets do (no excess agreement)."""
    batch = make_env("E2").sample("dev", 2000)
    obs_u, _, _ = batch.agent_inputs("none", alignment="unaligned")
    seen_u = np.isfinite(obs_u)
    seen_a = np.isfinite(batch.obs)
    K = seen_a.shape[-1]
    both_a = (seen_a[:, 0] & seen_a[:, 1]).sum(1).mean()
    both_u = (seen_u[:, 0] & seen_u[:, 1]).sum(1).mean()
    expected = (seen_a[:, 0].sum(1) * seen_a[:, 1].sum(1) / K).mean()
    assert abs(both_u - expected) < 0.1 and abs(both_a - expected) < 0.1   # E2 coverage itself is random
    # sybil copies share coverage exactly when aligned, not when unaligned
    b = make_env("E3-sybil").sample("dev", 300)
    s = b.oracle["role"] == 3
    oa = np.isfinite(b.obs)
    ou = np.isfinite(b.agent_inputs("none", alignment="unaligned")[0])
    same = lambda o: np.mean([(o[e][s[e]] == o[e][s[e]][0]).all() for e in range(300)])
    assert same(oa) == 1.0 and same(ou) < 0.05


def _likelihood_description(batch, e):
    """Each agent's own-likelihood parameters, built from the true structure (oracle), independently of the keys."""
    o, env = batch.oracle, batch.env
    if batch.kind == "gaussian":
        A, Bf, sig = o["A"][e], o["Bf"][e], o["sig"][e]
        R = np.einsum("iar,ibr->iab", Bf, Bf) + sig[:, None, None] ** 2 * np.eye(A.shape[1])
        return np.concatenate([A.reshape(len(A), -1), R.reshape(len(A), -1)], 1)
    if env in ("E2", "E3-sybil", "E3-collude", "Z8"):
        cov = o["cov"][e]
        return np.concatenate([cov, cov * o["mu"][e]], 1)
    if env.startswith("Z1"):
        g, rho = o["group"][e], o["rho"][e]
        return np.stack([(g < 0) * 1.0, np.where(g < 0, 1.0, rho[np.maximum(g, 0)]), o["s"][e]], 1)
    if env.startswith("Z4"):
        m = o["model"][e]
        return np.stack([(m >= 0) * 1.0, np.where(m >= 0, o["mu"][e][np.maximum(m, 0)], 0.0), o["s"][e]], 1)
    if env == "Z6":
        return np.concatenate([np.eye(2)[o["var"][e]], o["acc"][e][:, None]], 1)
    raise KeyError(env)


@pytest.mark.parametrize("name", ["E1", "E2", "E3-collude", "Z1", "Z1-blind", "Z4", "Z4-blind", "Z6", "Z8"])
def test_own_model_is_the_likelihood_and_nothing_more(name):
    """At every visibility the own model equals the agent's own-likelihood parameters built from the oracle: no
    parent-group or model identity (Z1/Z4), no shared-factor loadings (E1)."""
    batch = make_env(name).sample("dev", 20)
    for vis in ("none", "noisy", "full"):
        obs, keys, _, own = batch.agent_inputs(vis, own_model=True)
        assert keys.shape == batch.keys[vis].shape
        for e in range(20):
            np.testing.assert_allclose(own[e], _likelihood_description(batch, e), atol=1e-12)
        obs_u, keys_u, _, own_u = batch.agent_inputs(vis, alignment="unaligned", own_model=True)
        np.testing.assert_array_equal(obs_u, batch.agent_inputs("none", alignment="unaligned")[0])
        assert own_u.shape == own.shape


def test_z6_bayes_from_unaligned_inputs_and_own_models():
    """Z6 at provenance none: each agent's own model says which bit it reads and how well; that is enough."""
    batch = make_env("Z6").sample("dev", 200)
    obs, keys, _, own = batch.agent_inputs("none", alignment="unaligned", own_model=True)
    nb = make_env("Z6").config.n_bits
    for e in range(200):
        var, acc, o = own[e, :, :nb].argmax(1), own[e, :, nb], obs[e, :, 0]
        prod = 1.0
        for j in range(nb):
            lo = sum(np.log(acc[i] / (1 - acc[i])) * (1 if o[i] == 1 else -1) for i in np.flatnonzero(var == j))
            prod *= 1 - 2 / (1 + np.exp(-lo))
        assert abs((1 - prod) / 2 - 1 / (1 + np.exp(-batch.bayes["logit"][e]))) < 1e-12


@pytest.mark.parametrize("name", ["Z1", "Z1-blind"])
def test_z1_bayes_from_unaligned_inputs_and_own_models(name):
    """Z1 at provenance none: the own model has no parent identity, but agents with one parent share its rho
    exactly, which groups them (a fingerprint), and the closed form then gives Bayes. Z1-blind draws rho from two
    levels, so grouping by rho merges different parents and misses Bayes."""
    from aggbench.zoo import binned_normal, bin_of
    env = make_env(name)
    batch = env.sample("dev", 100)
    obs, _, _, own = batch.agent_inputs("none", alignment="unaligned", own_model=True)
    edges = env.edges()
    errs = []
    for e in range(100):
        direct, rho, s = own[e, :, 0], own[e, :, 1], own[e, :, 2]
        sym = bin_of(obs[e, :, 0], edges)
        lu = np.stack([binned_normal(np.array([-s[i], s[i]]), np.ones(2), edges)[:, sym[i]] for i in range(len(s))])
        logp = []
        for y in (0, 1):
            v = np.log(0.5) + np.log(lu[direct == 1, y]).sum()
            for r in np.unique(rho[direct == 0]):
                mem = np.flatnonzero((direct == 0) & (rho == r))
                v += np.log(sum((r if z == y else 1 - r) * np.prod(lu[mem, z]) for z in (0, 1)))
            logp.append(v)
        errs.append(abs(logp[1] - logp[0] - batch.bayes["logit"][e]))
    assert (max(errs) < 1e-9) == (name == "Z1")
