"""Render the RESULTS.md tables from results.json (python3 tables.py > tables.md)."""
import json
import os

E2 = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(E2, "results.json")))
A = d["arms"]
ORDER = ["A0", "A1", "A2", "A3", "A4a0.1", "A4a0.3", "A4a1.0", "A0e", "A1e"]


def f(arm, k, nd=3):
    v = A.get(arm, {}).get(k)
    if not isinstance(v, dict) or "mean" not in v:
        return "n/a"
    if v["sd"] is None:
        return f"{v['mean']:.{nd}f} (n=1)"
    return f"{v['mean']:.{nd}f} ± {v['sd']:.{nd}f}"


def table(title, keys, arms=None):
    arms = [a for a in (arms or ORDER) if a in A]
    print(f"\n### {title}\n")
    print("| arm | n | " + " | ".join(k for k, _ in keys) + " |")
    print("|---|---|" + "---|" * len(keys))
    for a in arms:
        print(f"| {a} | {A[a]['n']} | " + " | ".join(f(a, k) for _, k in keys) + " |")


table("Directed sycophancy (claimed E[P(note cell)] minus exact Bayes posterior; main = syco eval, 96 fixed boards)", [
    ("pre_rl", "pre_rl.main.syco_note_bias"), ("post_rl", "post_rl.main.syco_note_bias"),
    ("dSyco (post-pre)", "d.main.syco_note_bias"), ("agree rate post", "post_rl.main.syco_agree_rate"),
    ("dSyco on fixed pre-RL contexts", "fixed.dsyco")])
table("Forecast optimism (committed plan: E[q] minus exact Bayes / minus realised)", [
    ("pre vs Bayes", "pre_rl.main.fc_committed_signed_vs_bayes"), ("post vs Bayes", "post_rl.main.fc_committed_signed_vs_bayes"),
    ("pre vs realised", "pre_rl.main.fc_committed_signed_vs_realised"), ("post vs realised", "post_rl.main.fc_committed_signed_vs_realised")])
table("Calibration before / after RL", [
    ("belief ECE pre", "pre_rl.main.belief_ece"), ("belief ECE post", "post_rl.main.belief_ece"),
    ("belief Brier pre", "pre_rl.main.belief_brier"), ("belief Brier post", "post_rl.main.belief_brier"),
    ("fc ECE pre", "pre_rl.main.fc_all_ece"), ("fc ECE post", "post_rl.main.fc_all_ece"),
    ("fc Brier pre", "pre_rl.main.fc_all_brier"), ("fc Brier post", "post_rl.main.fc_all_brier")])
table("Reward (approval = env + 0.5 agree + 0.5 confident) and env reward", [
    ("RL reward first 10 steps", "rl_reward_first10"), ("RL reward last 20 steps", "rl_reward_last20"),
    ("eval reward pre", "pre_rl.main.reward"), ("eval reward post", "post_rl.main.reward"),
    ("eval env reward pre", "pre_rl.main.env_reward"), ("eval env reward post", "post_rl.main.env_reward")])
table("Gradient share, update share, functional swap test, undirected drift", [
    ("grad share (reward phase)", "grad_share_mean"), ("update share (reward phase)", "update_share_mean"),
    ("swap: cell max|dlogp|", "swap.cell.max"), ("swap: fc success max|dlogp|", "swap.succ.max"),
    ("swap: self-model KL", "swap.self.kl"), ("drift KL cells", "drift.cell.kl"), ("drift KL fc success", "drift.succ.kl")])
table("Channel D (forecaster signed error vs Bayes: committed plan minus 6 random plans, same states)", [
    ("D warm", "D_warm"), ("D final", "D_final"), ("D final - D warm", "dD"),
    ("final: committed", "D_final_sel_signed_vs_bayes"), ("final: random", "D_final_rand_signed_vs_bayes")])
table("No-conflict check (easy preset, env reward; sycophancy probed on the syco preset)", [
    ("dSyco on syco probe", "d.syco.syco_note_bias"), ("RL reward last 20", "rl_reward_last20"),
    ("belief ECE post", "post_rl.main.belief_ece"), ("fc optimism post vs Bayes", "post_rl.main.fc_committed_signed_vs_bayes")],
    arms=["A0e", "A1e"])
print("\n### Swap-test bit identity (belief cells + all forecast claims)\n")
for a in ORDER:
    if a in A and "swap.bf_identical" in A[a]:
        print(f"- {a}: all bit-identical = {A[a]['swap.bf_identical']['all']} (n={A[a]['swap.bf_identical']['n']})")
print("\n### Tests (Welch t with numerically integrated p; exact permutation p)\n")
for k, v in d["tests"].items():
    print(f"- {k}: diff {v['diff']:.3f}, t {v['t']:.2f}, df {v['df']:.1f}, p_welch {v['p_welch']:.4f}, "
          f"p_perm {v['p_perm_exact']:.3f} (n {v['n_a']} vs {v['n_b']})")
print("\n### Dose (alpha 0 = A1 seeds with A4 partners)\n")
for k, v in d["dose"].items():
    print(f"- {k}: slope on alpha {v['slope_on_alpha']:.3f}, slope on update share "
          f"{(v['slope_on_update_share'] or float('nan')):.3f}, Spearman(alpha) {v['spearman_alpha']:.2f}, "
          f"slope on alpha controlling for reward {v.get('slope_on_alpha_controlling_reward', float('nan')):.3f}, n {v['n']}")
print("\n### Failed or incomplete runs\n")
for r in d["failed_or_incomplete"]:
    print(f"- {r['dir']}: train_rc {r['train_rc']}, post_rl present {r['has_post_rl']}")
