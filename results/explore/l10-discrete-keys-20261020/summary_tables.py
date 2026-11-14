"""Compact README tables, read from result_<env>.json (no hand-typed numbers).

    python3 results/explore/l10-discrete-keys-20261020/summary_tables.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENVS = ("E2", "E3-collude")


def cell(m, name, d=3):
    r = m.get(f"{name}:s0:trained", {}).get("regret")
    return "-" if r is None else f"{r['mean']:.{d}f} [{r['ci_low']:.{d}f}, {r['ci_high']:.{d}f}]"


def length_table(res):
    m = res["methods"]
    rows = ["| key size L | bits, Gumbel-sigmoid ST | bits, deterministic ST | bits, warm start | continuous R^L |",
            "|---|---|---|---|---|"]
    for L in (0, 1, 2, 3, 4, 6, 8, 12, 16, 24):
        if L == 0:
            rows.append(f"| 0 (no key) | {cell(m, 'none')} | - | - | - |")
            continue
        rows.append(f"| {L} | {cell(m, f'bits{L}')} | {cell(m, f'stbits{L}')} | {cell(m, f'wbits{L}')} | "
                    f"{cell(m, f'cont{L}')} |")
    return "\n".join(rows)


def reference_table(res):
    m, p = res["methods"], res["extra"]["paired_differences"]

    def r(name):
        x = m[name]["regret"]
        return f"{x['mean']:.3f} [{x['ci_low']:.3f}, {x['ci_high']:.3f}]"

    def d(arm, key):
        x = p[arm].get(key)
        return "-" if x is None else f"{x['mean']:+.4f} [{x['ci_low']:+.4f}, {x['ci_high']:+.4f}]"
    rows = ["| arm | regret to Bayes | minus no-key arm | minus oracle_clearing |", "|---|---|---|---|"]
    for b in ("oracle_clearing", "extremized_log_pool"):
        rows.append(f"| {b} (baseline, no learning) | {r(b)} | - | - |")
    for arm in ("none", "bits16", "stbits16", "wbits8@12000", "cont4", "cont16", "cont4@12000",
                "prov_bits", "prov_full", "prov_full@12000"):
        n = f"{arm}:s0"
        if f"{n}:trained" in m:
            rows.append(f"| {arm} | {r(f'{n}:trained')} | {d(n, 'trained_minus_none') if arm != 'none' else '-'} | "
                        f"{d(n, 'trained_minus_oracle_clearing')} |")
    return "\n".join(rows)


def content_table(res):
    a = res["extra"]["arms"]
    rows = ["| arm | key entropy | I(key; which pieces seen), probe, of H | I(key; # pieces) | I(key; sign of own log-odds) "
            "| I(key; size of own log-odds, 8 bins) | I(key; Y) |", "|---|---|---|---|---|---|---|"]
    for arm in ("bits16", "bag4x16", "stbits4", "stbits16", "wbits8", "wbits8@12000", "cont16", "prov_bits"):
        x = a.get(f"{arm}:s0")
        if not x:
            continue
        it, kc = x["interpretability"], x.get("key_content_bits")
        ent = it.get("key_entropy_bits")
        f = (lambda k: "-" if kc is None else f"{kc[k]:.2f} / {kc['H_' + k]:.2f}")
        rows.append(f"| {arm} | {'-' if ent is None else f'{ent:.2f}'} | "
                    f"{it['probe_seen_bits_total']:.2f} / {it['H_seen_total']:.2f} | {f('n_seen')} | {f('sign_local')} | "
                    f"{f('size_local')} | {f('Y')} |")
    return "\n".join(rows)


def main():
    for env in ENVS:
        res = json.loads((HERE / f"result_{env}.json").read_text())
        print(f"#### {env}: regret to Bayes (nats/episode, seed 0, 95% CI) by key size\n")
        print(length_table(res) + "\n")
        print(f"#### {env}: key arms against the no-key arm and oracle_clearing (paired, same dev episodes)\n")
        print(reference_table(res) + "\n")
        print(f"#### {env}: what the key says (bits; 'x / y' = information / entropy of the target)\n")
        print(content_table(res) + "\n")


if __name__ == "__main__":
    main()
