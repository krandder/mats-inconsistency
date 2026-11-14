"""Print the regret-vs-history table (markdown) from result-*.json in this directory.

Usage: python3 table.py [dir]
"""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
ENVS = ["E2", "E3-sybil", "E3-biased", "E3-collude"]
HS = [0, 4, 16, 64]
ARMS = ["icl_attn", "bayes_logreg", "conj_bayes", "hedge_logpool", "wealth_kelly", "icl_attn:control:random_network",
        "icl_attn:control:untrained_init"]


def cell(m):
    r = m["regret"]
    return f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"


for env in ENVS:
    res = json.loads((D / f"result-{env}.json").read_text())
    M = dict(res["methods"])
    ph = D / f"posthoc-logreg-{env}.json"
    post = json.loads(ph.read_text()) if ph.exists() else None
    if post:
        M.update({k: v for k, v in post["methods"].items() if k.startswith("bayes_logreg")})
    print(f"\n**{env}** (n = {M['bayes']['regret']['n']} dev tasks; oracle_clearing {cell(M['oracle_clearing'])}; "
          f"extremized_log_pool {cell(M['extremized_log_pool'])})\n")
    print("| arm | " + " | ".join(f"H={h}" for h in HS) + " |")
    print("|---|" + "---|" * len(HS))
    for arm in ARMS:
        base, _, ctl = arm.partition(":")
        names = [f"{base}@H{h}" + (":" + ctl if ctl else "") for h in HS]
        if all(n in M for n in names):
            print(f"| {arm} | " + " | ".join(cell(M[n]) for n in names) + " |")
    pd = res["extra"]["paired_regret_differences"]
    print("| icl_attn - oracle_clearing (paired) | " + " | ".join(
        "{mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]".format(**pd[f"icl_attn@H{h}"]["vs_oracle_clearing"]) for h in HS) + " |")
    print("| icl_attn - conj_bayes (paired) | " + " | ".join(
        "{mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]".format(**res["extra"]["icl_minus_conj"][str(h)])
        if "icl_minus_conj" in res["extra"] else "n/a" for h in HS) + " |")
    if post and post["extra"].get("icl_attn_minus_bayes_logreg"):
        print("| icl_attn - bayes_logreg (paired) | " + " | ".join(
            "{mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]".format(**post["extra"]["icl_attn_minus_bayes_logreg"][str(h)])
            for h in HS) + " |")
