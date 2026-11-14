"""Markdown tables for RESULTS.md from vmout/item3_summary.json and vmout/GATES.json (reporting only, no new statistics).
  python tables3.py VMOUT > tables.md"""
import json, sys

V = sys.argv[1]
S = json.load(open(f"{V}/item3_summary.json"))
G = json.load(open(f"{V}/GATES.json"))
f3 = lambda x: "n/a" if x is None else f"{x:+.3f}"


def ci(c):
    return f"{c['mean']:+.3f} [{f3(c.get('lo'))}, {f3(c.get('hi'))}]"


print(f"Seeds analysed: {S['seeds']} (n = {S['n']}). Status by the pre-registered bars: **{S['status']}**.\n")
print("Primary (tau_b(R) - tau_b(R1) per seed, mean with 97.5% t-CI; P8 paired permutation test, one-sided):\n")
print("| contrast | per seed | mean [97.5% CI] | 95% CI | p(R > R1) | p(R1 > R) |")
print("|---|---|---|---|---|---|")
for r, v in S["primary"].items():
    ps = ", ".join(f"{x:+.3f}" for x in v["per_seed"])
    print(f"| {r} - R1 | {ps} | {ci(v['ci975'])} | [{f3(v['ci95']['lo'])}, {f3(v['ci95']['hi'])}] | "
          f"{v['perm_vs_R1']['p_one_sided']:.4f} | {v['perm_R1_vs']['p_one_sided']:.4f} |")
print(f"\nR4 hurts (97.5% upper bound < 0): {S['R4_hurts']}.\n")
print("Kendall tau-b against the truth T(s, i), 32 items, mean over seeds with 95% t-CI:\n")
print("| rule | tau_b mean [95% CI] | per seed | seeds |")
print("|---|---|---|---|")
for r, v in S["tau"].items():
    ps = ", ".join(f"{S['tau_per_seed'][str(s)][r]:.3f}" for s in v["seeds"])
    print(f"| {r} | {ci(v)} | {ps} | {v['seeds']} |")
nc = S["noisy_copy"]
print(f"\nNoisy copy (S = empty, bits(nz05 pipeline) - bits(g2 pipeline) per held maze; > 0 means g2 is better): "
      f"{ci(nc['across'])} bits/maze (95% t-CI over seeds), verdict **{nc['verdict']}**.\n")
print("| seed | gain g2 | gain nz05 | nz05 - g2 bits (95% CI over mazes) | truth g2 > nz05 |")
print("|---|---|---|---|---|")
for s, v in nc["per_seed"].items():
    print(f"| {s} | {v['gain_g2']:.2f} | {v['gain_nz05']:.2f} | {v['g2_minus_nz05']:+.2f} ± {v['ci95']:.2f} | {v['truth_g2_above']} |")
print("\nShare of seeds where a rule orders g2 vs nz05 as the truth does: " +
      ", ".join(f"{r} {x:.2f}" for r, x in nc["rule_correct_share"].items()) + ".\n")
print("Gates (every seed):\n")
print("| seed | G1 B2 held acc | G2 rho (r0 vs r1) | G3 noise ratio | U2 random | U3 null | U1 copyY R4 (reported) |")
print("|---|---|---|---|---|---|---|")
for s in range(5):
    g1, g2, g3 = G.get(f"G1:s{s}", {}), G.get(f"G2:s{s}", {}), G.get(f"G3:s{s}")
    g3s = (f"{g3['noise_ratio']:.3f} | {g3['sanity']['U2_random']} | {g3['sanity']['U3_null_R1_zero']} | "
           f"{g3['info']['U1_copyY_R4']:.0f}") if g3 else "not run (no prices) | | |"
    print(f"| {s} | {g1.get('acc', float('nan')):.4f} | {g2.get('rho', float('nan')):.3f} | {g3s} |")
sec = S["secondary"]
print("\nSecondary (no verdicts). Per-context tau_b, mean [95% CI]:\n")
print("| rule | " + " | ".join(sec["per_context"]) + " |")
print("|---|" + "---|" * len(sec["per_context"]))
for r in S["tau"]:
    print(f"| {r} | " + " | ".join(ci(sec["per_context"][c][r]) for c in sec["per_context"]) + " |")
lc = sec["lambda_curve_in_sample"]
i0 = lc["grid"].index(0.0) if 0.0 in lc["grid"] else 0
print(f"\nIn-sample lambda curve (exploratory): argmax lambda = {lc['argmax']:.4g}, tau {lc['tau_max']:.3f}; "
      f"lambda 0 (R1) tau {lc['tau'][i0]:.3f}; lambda 1 (R4) tau "
      f"{min(zip(lc['grid'], lc['tau']), key=lambda p: abs(p[0] - 1))[1]:.3f}.\n")
print("Taus against P8's control truth (free-latent GStack - pipeline) at N = 4096:\n")
print("| rule | tau_b mean [95% CI] |")
print("|---|---|")
for r, v in sec["control_truth_tau"].items():
    print(f"| {r} | {ci(v)} |")
if "p8_n1024_prices_vs_n4096_truth" in sec:
    p = sec["p8_n1024_prices_vs_n4096_truth"]
    print(f"\nP8's N = 1024 prices (seeds {p['seeds']}) scored against the new N = 4096 truth: " +
          ", ".join(f"{r} {ci(v)}" for r, v in p["tau"].items()) +
          f". R4 - R1 {ci(p['R4_minus_R1'])}, R8 - R1 {ci(p['R8_minus_R1'])} (95% CIs).\n")
ti = sorted(S["truth_items"].items(), key=lambda kv: -kv[1]["mean"])
print("Truth T (gain over base, bits/maze), mean over seeds [95% CI], top 8 and bottom 4:\n")
print("| item | T |")
print("|---|---|")
for k, v in ti[:8] + ti[-4:]:
    print(f"| {k} | {v['mean']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}] |")
