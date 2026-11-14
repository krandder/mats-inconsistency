"""Render markdown tables (mean +- sd over seeds) from results.json for RESULTS.md."""
import json, os, sys
D = os.environ.get("E4_OUT", os.path.dirname(os.path.abspath(__file__)))
res = json.load(open(os.path.join(D, "results.json")))
ARMS = ["B0", "M0", "M1", "P", "twin"]


def f(v, nd=3):
    if not v:
        return "-"
    sd = v["sd"]
    return f"{v['mean']:+.{nd}f} ± {sd:.{nd}f}" if sd == sd else f"{v['mean']:+.{nd}f} (n=1)"


def table(preset, cols, title):
    print(f"\n**{title}** ({preset}; source `results.json: agg[\"{preset}|<arm>\"]`, per-seed files `runs/{preset}/<arm>-s<seed>/e4_fixed.jsonl`)\n")
    print("| arm | seeds | " + " | ".join(c[1] for c in cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for a in ARMS:
        d = res["agg"].get(f"{preset}|{a}")
        if not d:
            continue
        print(f"| {a} | {d['seeds']} | " + " | ".join(f(d.get(c[0])) for c in cols) + " |")


for p in sorted({k.split("|")[0] for k in res["agg"]}):
    table(p, [("op_env_reward_pre", "env reward pre"), ("op_env_reward_post", "env reward post"), ("op_reward_post", "approval reward post"),
              ("kl_init_total_post", "KL to init post"), ("last_step", "last RL step")], "Reward and KL")
    table(p, [("fx_fc_brier_post", "fc Brier post"), ("fx_fc_brier_delta", "Δ fc Brier"), ("fx_fc_ece_delta", "Δ fc ECE"),
              ("fx_fc_signed_vs_realised_delta", "Δ fc optimism vs realised"), ("fx_fc_signed_vs_bayes_delta", "Δ fc signed vs Bayes"),
              ("fx_sit_brier_delta", "Δ report Brier"), ("fx_sit_ece_delta", "Δ report ECE"), ("fx_sit_signed_vs_bayes_delta", "Δ report signed vs Bayes")],
          "Fixed teacher-forced set: forecasts and situation reports (Δ = post − pre)")
    table(p, [("fx_fc_note_bias_index_pre", "fc note idx pre"), ("fx_fc_note_bias_index_post", "fc note idx post"),
              ("fx_sit_note_bias_index_pre", "report note idx pre"), ("fx_sit_note_bias_index_post", "report note idx post"),
              ("fx_pol_note_bias_index_post", "policy first-shot note idx post"), ("commit_note_bias_first_post", "commit first=note excess post"),
              ("commit_note_bias_inplan_post", "commit note-in-plan excess post")], "Note-bias index")
    table(p, [("rel_swap_flip_pre", "swap flip pre"), ("rel_swap_flip_post", "swap flip post"), ("rel_top_down03_flip_pre", "top−0.3 flip pre"),
              ("rel_top_down03_flip_post", "top−0.3 flip post"), ("rel_second_up03_flip_post", "2nd+0.3 flip post"),
              ("op_reliance_swap_argmax_change_post", "on-policy swap flip post"), ("dfc_inconsistent_argmax_pre", "DFC inconsistency pre"),
              ("dfc_inconsistent_argmax_post", "DFC inconsistency post")], "Forecast reliance and decision-forecast consistency")
    table(p, [("gain_env", "Δ env reward"), ("gain_agree", "Δ w_agree·agree"), ("gain_conf", "Δ w_conf·conf"), ("gain_total", "Δ approval total"),
              ("agree_rate_post", "agree rate post"), ("conf_rate_post", "conf rate post")], "Approval-gain split")
    table(p, [("sel_op_fc_committed_signed_vs_realised_minus_fx_fc_signed_vs_realised", "selection: Δoptimism on-policy − fixed"),
              ("sel_op_fc_committed_brier_minus_fx_fc_brier", "selection: ΔBrier on-policy − fixed"),
              ("sel_op_fc_all_signed_vs_bayes_minus_fx_fc_signed_vs_bayes", "selection: Δsigned-vs-Bayes on-policy(all cands) − fixed")],
          "Selection channel (on-policy drift − fixed-set drift)")
    dc = {k: v for k, v in res["data_channel"].items() if k.startswith(p + "|")}
    if dc:
        print(f"\n**Data channel: fixed-set Δ(arm) − Δ(no-RL twin), paired by seed** ({p}; `results.json: data_channel`)\n")
        keys = sorted({k.split("|")[2] for k in dc if not k.endswith("curve_mean")})
        print("| arm | " + " | ".join(keys) + " |"); print("|---|" + "---|" * len(keys))
        for a in ARMS[:-1]:
            print(f"| {a} | " + " | ".join(f(dc.get(f"{p}|{a}|{k}")) for k in keys) + " |")
        print(f"\n**Same, curve-mean version (POST HOC: mean over common RL checkpoints)** ({p})\n")
        print("| arm | " + " | ".join(keys) + " |"); print("|---|" + "---|" * len(keys))
        for a in ARMS[:-1]:
            print(f"| {a} | " + " | ".join(f(dc.get(f"{p}|{a}|{k}|curve_mean")) for k in keys) + " |")
    mt = {k: v for k, v in res["matched"].items() if k.startswith(p + "|")}
    if mt:
        print(f"\n**Matched env-reward bins (all checkpoints pooled; `results.json: matched`)** ({p})\n")
        print("| bin | arm | n ckpts | fc signed vs Bayes | fc Brier | report note idx | agree | conf | KL | swap flip |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for k, v in sorted(mt.items()):
            _, b, a = k.split("|")
            print(f"| {b} | {a} | {v['n_ckpts']} | {f(v['fx_fc_signed_vs_bayes'])} | {f(v['fx_fc_brier'])} | {f(v['fx_sit_note_bias_index'])} | "
                  f"{f(v['agree_rate'])} | {f(v['conf_rate'])} | {f(v['kl_init_total'])} | {f(v['rel_swap_flip'])} |")
