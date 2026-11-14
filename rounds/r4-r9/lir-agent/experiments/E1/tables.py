"""Markdown tables for RESULTS.md, every cell read from results.json (mean ± sd over seeds, n)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
A = R["arms"]
ORDER = ["P-on", "D-on", "F-on", "P-off", "D-off", "F-off"]
def f(x, nd=3):
    if not x or x.get("mean") is None: return "n/a"
    sd = x.get("sd")
    return f"{x['mean']:.{nd}f} ± {sd:.{nd}f}" if sd is not None else f"{x['mean']:.{nd}f} (n=1)"
def table(title, tag, keys):
    out = [f"**{title}** (tag `{tag}`)", "", "| arm | n | " + " | ".join(k for k, _ in keys) + " |", "|---|---|" + "---|" * len(keys)]
    for a in ORDER:
        if a not in A: continue
        src = A[a][tag] if tag != "steps" else A[a]["steps"]
        n = A[a]["post_rl"]["env_reward"]["n"]
        out.append(f"| {a} | {n} | " + " | ".join(f(src.get(k), nd) for _, (k, nd) in [(None, (kk, d)) for kk, d in [v for _, v in keys]]) + " |")
    return "\n".join(out) + "\n"
K1 = [("reward", ("reward", 3)), ("env reward", ("env_reward", 3)), ("success", ("success", 3)),
      ("ECE committed (tok)", ("fc_committed_ece_tok", 3)), ("ECE committed (stated)", ("fc_committed_ece_stated", 3)),
      ("Brier committed (tok)", ("fc_committed_brier_tok", 3)), ("optimism committed (stated)", ("fc_committed_opt_vs_real_stated", 3)),
      ("agree false note (stated)", ("agree_false_note_stated", 3))]
K2 = [("ECE non-chosen (tok)", ("fc_nonchosen_ece_tok", 3)), ("optimism random cand. (tok)", ("fc_random_opt_vs_real_tok", 3)),
      ("optimism committed (tok)", ("fc_committed_opt_vs_real_tok", 3)), ("selection gap (tok)", ("selection_gap_tok", 3)),
      ("conf>=0.7 committed (stated)", ("fc_committed_conf07_stated", 3)), ("agree false note (tok)", ("agree_false_note_tok", 3)),
      ("note bias vs Bayes", ("note_bias_vs_bayes", 3)), ("sit. log score vs state", ("sit_logscore_vs_state", 3)),
      ("sit. KL vs Bayes", ("sit_kl_vs_bayes", 3)), ("sit. ECE (tok)", ("sit_ece_tok", 3))]
K3 = [("R_grad_belief max", ("R_grad_belief_maxabs", 4)), ("R_grad_forecast max", ("R_grad_forecast_maxabs", 4)),
      ("cos(reward, belief grad)", ("cos_reward_vs_B", 3)), ("cos(reward, forecast grad)", ("cos_reward_vs_F", 3)),
      ("reward step: mean abs dE[q] belief", ("e1_R_dbel_abs", 4)), ("reward step: dE[q] note cell, false notes", ("e1_R_dbel_note_false", 4)),
      ("reward step: max abs dE[q] note cell, false", ("e1_R_dbel_note_false_maxabs", 3)),
      ("reward step: mean abs dP(success)", ("e1_R_dfc_abs", 4)), ("calibration lag (Brier pre-F minus post-F)", ("lag", 4)),
      ("prequential note bias (mean over steps)", ("syco_note_bias", 3)), ("prequential note bias (max step)", ("syco_note_bias_max", 3))]
s = "\n".join([table("Table 1. Primary and outcome metrics, before RL", "pre_rl", K1),
               table("Table 2. Primary and outcome metrics, mid-RL (step 30)", "rl30", K1),
               table("Table 3. Primary and outcome metrics, after RL (step 60, pre-registered)", "post_rl", K1),
               table("Table 4. Secondary metrics after RL", "post_rl", K2),
               table("Table 5. Gradient localisation, leak and lag (per-step logs, mean over RL steps then over seeds)", "steps", K3)])
tr = ["**Table 6. Prequential note-cell bias (E[q] minus exact Bayes) in 10-step windows (RL batch scored before its update), mean ± sd over seeds**", "",
      "| arm | 1-10 | 11-20 | 21-30 | 31-40 | 41-50 | 51-60 |", "|---|---|---|---|---|---|---|"]
for a in ORDER:
    if a in A:
        tr.append(f"| {a} | " + " | ".join(f(x) for x in A[a]["traj10"]["syco_note_bias"]) + " |")
s += "\n" + "\n".join(tr) + "\n"
open(os.path.join(HERE, "tables.md"), "w").write(s)
print(s)
