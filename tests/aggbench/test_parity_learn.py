import numpy as np
import pytest

from aggbench.compression import PiecesPerAgent, agent_tokens
from aggbench.parity_learn import (NV, Z9Spec, bayes_logit, biased_bit_curriculum, fixed_identity_curriculum,
                                   prior_logit, sample, tokens_from)


@pytest.mark.parametrize("kw", [{}, {"readers_random": True}, {"bit_prob": 0.8},
                                {"readers_random": True, "readers_alpha": 1.0, "readers_per_bit": 6}])
def test_tokens_and_bayes_match_the_env(kw):
    env = PiecesPerAgent("Z9", 2, **kw)
    b = env.sample("dev", 300)
    kw = {k: v for k, v in kw.items()}
    var, acc = b.oracle["var"], b.oracle["acc"]
    read = np.asarray(b.oracle["base_sym"])                     # Z9 symbols are the readings 0/1
    tok, keys = tokens_from(var, acc, read, 2)
    assert np.allclose(tok, agent_tokens(b)[..., :tok.shape[-1]], atol=1e-6)
    q = kw.get("bit_prob", 0.5)
    assert np.allclose(bayes_logit(var, acc, read, q, 4), b.bayes["logit"], atol=1e-8)
    assert abs(prior_logit(Z9Spec(**kw)) - env.public()["prior_logit"]) < 1e-12


@pytest.mark.parametrize("spec", [Z9Spec(), Z9Spec(readers_random=True), Z9Spec(bit_prob=0.8)])
def test_sampler_is_calibrated_and_shaped(spec):
    d = sample(spec, 20000, np.random.default_rng(0))
    assert d["tokens"].shape == (20000, 7, 2 + 2 * (NV + 1))
    p = 1 / (1 + np.exp(-d["bayes"]))
    assert abs(p.mean() - d["y"].mean()) < 0.015
    if not spec.readers_random:                                  # exactly 3 readers per parity bit
        onehot = d["keys"].reshape(20000, 7, 2, NV + 1)[..., :NV].sum((1, 2))
        assert (onehot[:, :4] == 3).all()


def test_curricula_keep_identities_and_split_into_agents():
    for spec, _ in fixed_identity_curriculum():
        d = sample(spec, 500, np.random.default_rng(1))
        used = d["keys"].reshape(500, spec.n_agents, 2, NV + 1)[..., :NV].sum((0, 1, 2))
        assert (used[spec.k:4] == 0).all() and used[4:].sum() > 0     # bits >= k unread, distractors stay 4, 5
    assert [s.bit_prob for s, _ in biased_bit_curriculum()] == [0.8, 0.65, 0.5]


def test_train_arm_smoke_runs_and_respects_channel():
    pytest.importorskip("torch")
    from aggbench.parity_learn import train_arm
    stages = fixed_identity_curriculum(steps=(4, 4, 6))
    r = train_arm("smoke", stages, b=2, channel="gauss", dec_keys=True, aux=1.0, batch=32, eval_every=5, n_val=64,
                  n_dev=64, finetune=3, n_boot=50, device="cpu")
    assert r["distinct_symbols"] <= 4 and r["supervision"] == "Y labels only"   # aux off at b < 4 (B12)
    assert "regret_after_sign_finetune" in r
    r = train_arm("smoke", [(Z9Spec(), 6)], channel="none", decoder="isab", batch=16, eval_every=3, n_val=32,
                  n_dev=32, n_boot=20, device="cpu")
    assert np.isfinite(r["regret"]["mean"])


def test_blind_count_code_finds_the_count_leak_and_readers_random_shrinks_it():
    from aggbench.parity_learn import _local_configs, named_codes, tabulated_forecast, _kl, DEV_SEED
    code, L = named_codes()["count of parity 1-readings (log2 3 bits)"]
    out = {}
    for tag, spec in (("fixed", Z9Spec()), ("rr", Z9Spec(readers_random=True))):
        dv = sample(spec, 1500, np.random.default_rng(DEV_SEED))
        f, cov = tabulated_forecast(code, L, spec, _local_configs(dv["tokens"]), n_tab=300_000)
        out[tag] = float(_kl(f, dv["bayes"]).mean())
    assert out["fixed"] < 0.2 < out["rr"]           # review B2: ~0.16 with exactly 3 readers per bit


def test_leak_reduced_spec_count_code_near_prior():
    from aggbench.parity_learn import _local_configs, named_codes, tabulated_forecast, _kl, DEV_SEED
    spec = Z9Spec(readers_random=True, readers_alpha=1.0, readers_per_bit=6)
    code, L = named_codes()["count of parity 1-readings (log2 3 bits)"]
    dv = sample(spec, 2000, np.random.default_rng(DEV_SEED))
    f, _ = tabulated_forecast(code, L, spec, _local_configs(dv["tokens"]), n_tab=300_000)
    prior = float(_kl(np.full(2000, 0.0), dv["bayes"]).mean())
    count = float(_kl(f, dv["bayes"]).mean())
    assert prior > 0.2 and prior - count < 0.03          # the count code keeps < ~10% of the information


def test_queried_encoder_reads_only_earlier_messages():
    torch = pytest.importorskip("torch")
    from aggbench.parity_learn import build
    spec = Z9Spec(readers_random=True, readers_alpha=1.0, readers_per_bit=6)
    net = build(16, 14, 4, "gauss", "transformer", False, queried=True, power_norm=True, p=2)
    net.eval()
    x = torch.tensor(sample(spec, 8, np.random.default_rng(0))["tokens"])
    m1 = net.message(x, "quant")
    x2 = x.clone()
    x2[:, 5:] += torch.randn_like(x2[:, 5:])
    m2 = net.message(x2, "quant")
    assert torch.equal(m1[:, :5], m2[:, :5])          # agent t never sees later agents
    assert set(torch.unique(m1).tolist()) <= {-1.0, 1.0}


def test_queried_fsq_usage_term_averages_all_agents():
    torch = pytest.importorskip("torch")
    from aggbench.parity_learn import build
    spec = Z9Spec(readers_random=True, readers_alpha=1.0, readers_per_bit=6)
    net = build(16, 14, 6, "gauss", "transformer", False, queried=True, fsq=True, p=2)
    x = torch.tensor(sample(spec, 64, np.random.default_rng(0))["tokens"])
    seen, orig = [], net._fsq

    def spy(u):
        out = orig(u)
        seen.append(float(net.last_usage))
        return out
    net._fsq = spy
    net.message(x, "train")
    assert len(seen) == x.shape[1]                    # one usage term per agent
    assert abs(float(net.last_usage) - float(np.mean(seen))) < 1e-6
