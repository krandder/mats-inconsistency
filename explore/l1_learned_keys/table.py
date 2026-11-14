"""Markdown tables from the L1 result JSONs (numbers copied from JSON, never hand-typed).

    python3 -m explore.l1_learned_keys.table results/explore/l1-learned-keys-2026-10-20
"""
import json
import sys
from pathlib import Path

ENVS = ["E1", "E2", "E3-sybil", "E3-biased", "E3-collude"]
VIS = ["none", "noisy", "full"]
ARMS = ["lk_clear", "lk_clear_cal", "attn_pool"]


def ci(s):
    return f"{s['mean']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]"


def load(d):
    out = {}
    for p in sorted(Path(d).glob("result-*.json")):
        r = json.loads(p.read_text())
        out[(r["env"]["name"], r["visibility"])] = r
    return out


def main(d):
    res = load(d)
    lines = ["Regret to bayes (nats/episode, 95% bootstrap CI over dev episodes):", "",
             "| env | vis | oracle_clearing | extremized_log_pool | " + " | ".join(ARMS) + " |",
             "|---" * (4 + len(ARMS)) + "|"]
    for e in ENVS:
        for v in VIS:
            r = res.get((e, v))
            if not r:
                continue
            m = r["methods"]
            cells = [ci(m[k]["regret"]) for k in ("oracle_clearing", "extremized_log_pool")]
            cells += [ci(m[f"{a}:trained"]["regret"]) if f"{a}:trained" in m else "n/a" for a in ARMS]
            lines.append(f"| {e} | {v} | " + " | ".join(cells) + " |")
    lines += ["", "Paired regret difference to oracle_clearing (arm minus oracle_clearing; > 0 means worse than the "
              "reports-level ceiling):", "",
              "| env | vis | extremized_log_pool | " + " | ".join(ARMS) + " |", "|---" * (3 + len(ARMS)) + "|"]
    for e in ENVS:
        for v in VIS:
            r = res.get((e, v))
            if not r:
                continue
            p = r["extra"]["paired_regret_difference"]
            cells = [ci(p["extremized_log_pool"]["oracle_clearing"])]
            cells += [ci(p[f"{a}:trained"]["oracle_clearing"]) if f"{a}:trained" in p else "n/a" for a in ARMS]
            lines.append(f"| {e} | {v} | " + " | ".join(cells) + " |")
    lines += ["", "Controls (regret to bayes, mean only): trained / untrained_init / random_network:", "",
              "| env | vis | " + " | ".join(ARMS) + " |", "|---" * (2 + len(ARMS)) + "|"]
    for e in ENVS:
        for v in VIS:
            r = res.get((e, v))
            if not r:
                continue
            m = r["methods"]
            cells = []
            for a in ARMS:
                if f"{a}:trained" not in m:
                    cells.append("n/a")
                    continue
                vals = [m[f"{a}:{k}"]["regret"]["mean"] for k in
                        ("trained", "control:untrained_init", "control:random_network")]
                cells.append(" / ".join(f"{x:.3f}" if abs(x) < 1e3 else f"{x:.1e}" for x in vals))
            lines.append(f"| {e} | {v} | " + " | ".join(cells) + " |")
    any_r = next(iter(res.values()))
    lines += ["", f"n dev = {any_r['splits']['eval']['n']} per cell, n train = {any_r['splits']['train']['n']}; "
              f"harness commit `{any_r['harness']['git_commit'][:10]}`, git_dirty={any_r['harness']['git_dirty']}, "
              f"code sha256 `{any_r['extra']['code_sha256'][:12]}`."]
    print("\n".join(lines))




def scaling(d):
    """lk_clear at visibility full across every run directory under d (main, long-training, more-data)."""
    rows = []
    for p in sorted(Path(d).rglob("result-*-full.json")):
        r = json.loads(p.read_text())
        if "lk_clear:trained" not in r["methods"]:
            continue
        pd = r["extra"]["paired_regret_difference"]["lk_clear:trained"]
        rows.append((r["env"]["name"], r["splits"]["train"]["n"], r["extra"]["steps"],
                     ci(r["methods"]["lk_clear:trained"]["regret"]), ci(pd["oracle_clearing"]),
                     ci(pd["extremized_log_pool"]), r["methods"]["lk_clear:control:random_network"]["regret"]["mean"],
                     str(p.parent.relative_to(d)) or "."))
    lines = ["| env | n train | steps | lk_clear regret | minus oracle_clearing | minus extremized_log_pool | "
             "random-network control | dir |", "|---" * 8 + "|"]
    for row in sorted(rows):
        lines.append("| " + " | ".join(str(x) for x in row[:6]) + f" | {row[6]:.3f} | {row[7]} |")
    print("\n".join(lines))


if __name__ == "__main__":
    (scaling if "--scaling" in sys.argv else main)(sys.argv[1])
