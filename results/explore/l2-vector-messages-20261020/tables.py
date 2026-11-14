"""Print the README tables from the result_*.json files in this directory (numbers are never hand-typed)."""
import json
from pathlib import Path

from aggbench.results import validate

HERE = Path(__file__).resolve().parent
ENVS = ["E1", "E2", "E3-sybil", "E3-biased", "E3-collude"]


def ci(x, p=3):
    return f"{x['mean']:.{p}f} [{x['ci_low']:.{p}f}, {x['ci_high']:.{p}f}]"


def main():
    print("| env | arm | m (key+value floats) | regret to bayes | regret minus oracle_clearing | minus extremized_log_pool | random-network control | untrained-init control | locality |")
    print("|---|---|---|---|---|---|---|---|---|")
    base_rows = []
    for env in ENVS:
        p = HERE / f"result_{env}.json"
        if not p.exists():
            print(f"| {env} | (missing) | | | | | | | |")
            continue
        r = json.loads(p.read_text())
        validate(r)
        M, X = r["methods"], r["extra"]
        for b in ("oracle_clearing", "extremized_log_pool", "log_pool", "sum_logodds"):
            print(f"| {env} | {b} (scalar reports) | 1 | {ci(M[b]['regret'])} | | | | | |")
        for m in X["message_dims"]:
            n = f"vecmsg_m{m}"
            pd = X["paired_differences"][n]
            print(f"| {env} | vector messages | {m}+{m} | {ci(M[n + ':trained']['regret'])} | "
                  f"{ci(pd['trained_minus_oracle_clearing'])} | {ci(pd['trained_minus_extremized_log_pool'])} | "
                  f"{ci(M[n + ':control:random_network']['regret'], 2)} | {ci(M[n + ':control:untrained_init']['regret'], 2)} | "
                  f"{'pass' if X['locality_check'][n]['passed'] else 'FAIL'} |")
        base_rows.append(f"{env}: harness {r['harness']['git_commit'][:10]} dirty={r['harness']['git_dirty']}, "
                         f"oracle_check max diff {r['oracle_check']['max_abs_diff']:.1e}, train n={r['splits']['train']['n']}, "
                         f"dev n={r['splits']['eval']['n']}, runner sha256 {X['runner_sha256'][:12]}")
    print()
    print("Declared distillation diagnostic (E2, soft targets sigmoid(bayes) on train):")
    print()
    print("| m (key+value floats) | regret to bayes | regret minus oracle_clearing | minus extremized_log_pool | random-network control | untrained-init control | locality |")
    print("|---|---|---|---|---|---|---|")
    rows = []
    for p in sorted(HERE.glob("result_E2-distill*.json")):
        r = json.loads(p.read_text())
        validate(r)
        M, X = r["methods"], r["extra"]
        for m in X["message_dims"]:
            n = f"vecmsg_distill_m{m}"
            pd = X["paired_differences"][n]
            rows.append((m, f"| {m}+{m} | {ci(M[n + ':trained']['regret'])} | {ci(pd['trained_minus_oracle_clearing'])} | "
                            f"{ci(pd['trained_minus_extremized_log_pool'])} | {ci(M[n + ':control:random_network']['regret'], 2)} | "
                            f"{ci(M[n + ':control:untrained_init']['regret'], 2)} | {'pass' if X['locality_check'][n]['passed'] else 'FAIL'} |"))
        base_rows.append(f"{p.name}: harness {r['harness']['git_commit'][:10]} dirty={r['harness']['git_dirty']}, "
                         f"runner sha256 {X['runner_sha256'][:12]}")
    for _, row in sorted(rows):
        print(row)
    print()
    for row in base_rows:
        print("- " + row)


if __name__ == "__main__":
    main()
