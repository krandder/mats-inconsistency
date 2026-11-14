"""Regression test asked for by hawks/E4.md item 7.6 / reshape (6), against the current harness:
in the protected configuration the forecast (and belief-claim) logits are EXACTLY independent of every
reward-trained group; in M1 (gating all) they are not. Run: pytest -q experiments/E4/test_e4_leak.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leak_check as LC


def test_protected_forecast_independent_of_reward_groups():
    ad, cls, groups = LC.CONFIGS["P (M2: role passes, goal deleted)"]
    for g in groups:
        r = LC.run_one(ad, cls, g)
        assert r["forecast"]["logit_maxabs_change"] == 0.0 and r["forecast"]["loss_grad_norm"] == 0.0
        assert r["belief"]["logit_maxabs_change"] == 0.0 and r["belief"]["loss_grad_norm"] == 0.0


def test_gating_all_leaks():
    ad, cls, groups = LC.CONFIGS["M1 (three adapters, gating all)"]
    r = LC.run_one(ad, cls, "action")
    assert r["forecast"]["logit_maxabs_change"] > 0 and r["forecast"]["loss_grad_norm"] > 0
