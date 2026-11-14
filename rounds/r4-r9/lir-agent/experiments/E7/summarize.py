"""Print the RESULTS.md tables from results/results.json (mean +- sd over seeds, n)."""
import json, sys
r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/results.json"))
def f(d, k, nd=3):
    v = d.get(k)
    if not v or v["mean"] is None: return "-"
    return f"{v['mean']:.{nd}f} ± {v['sd']:.{nd}f} (n={v['n']})"
def table(T, keys, title):
    print(f"\n### {title}\n")
    print("| arm, tag | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for key, row in T.items():
        print(f"| {key} | " + " | ".join(f(row, k) for k in keys) + " |")
table(r["insitu"], ["eval_reward", "eval_success", "H_pi", "eval_commit_entropy"], "Reward and entropy")
table(r["insitu"], ["acc_insitu", "acc_insitu_matched", "mode_acc_test", "tv_q_pi", "acc_insitu_idio", "idio_frac"], "In-situ self-model accuracy")
table(r["insitu"], ["val_signed_exp", "val_signed_argmax", "neu_acc_row", "neu_acc_col", "neu_signed_col", "neu_abs_col"], "Valenced vs neutral")
table(r["insitu"], ["acc_frozen_live_action", "acc_stale_warm_selfmodel", "eval_self_acc"], "Frozen / stale self-model")
table(r["insitu"], ["fulfil_own_follow", "fulfil_base_on_own", "fulfil_swap_follow", "fulfil_swap_change", "fulfil_swap_tv"], "Planted self-statement probe")
table(r["fresh"], ["acc_S", "acc_X1", "acc_X0", "acc_mode", "self_adv", "self_adv_matched", "self_adv_nc", "self_adv_idio"], "Fresh predictors (Binder M1 vs M2)")
table(r["fresh"], ["val_signed_exp_S", "val_signed_exp_X1", "val_signed_exp_X0", "n_pred_seeds"], "Fresh predictors: signed valenced error")
table(r["direction"], ["d(pred_t,act_t+1)", "d(pred_t+1,act_t)", "D=d(pred_t,act_t+1)-d(pred_t+1,act_t)", "d(act_t,act_t+1)", "d(pred_t,pred_t+1)"], "Direction of resolution")
print("\n### Bootstrap (seeds, then states) self_adv\n")
for k, v in r["gate0_and_selfadv_bootstrap"].items(): print(f"- {k}: mean {v['mean_of_boot']:.4f}, 95% CI [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]")
print("\n### Change pre_rl -> post_rl (per-seed differences)\n")
for k, v in r["delta_pre_to_post"].items():
    if v["mean"] is not None: print(f"- {k}: {v['mean']:.4f} ± {v['sd']:.4f} (n={v['n']}), t-CI {v.get('ci95_t')}")
print("\n### Matched-reward OLS\n")
for k, v in r["matched_reward_ols"].items(): print(f"- {k}: {v}")
