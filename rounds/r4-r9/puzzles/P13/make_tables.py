import json
R = json.load(open("/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P13/results/results.json"))
M = R["models"]
def f(x, d=2):
    return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
print("| model | arm | J | v9 | v33 | e19 | e33 | e59 | e99 @12n | e99 log loss raw / after post-hoc control @t_eq | trained |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for n, m in M.items():
    print(f"| {n} | {m.get('arm')} | {m.get('jopt')} | {f(m.get('v9'))} | {f(m.get('v33'))} | {f(m.get('e19_exact@12n'))} | {f(m.get('e33_exact@12n'))} | {f(m.get('e59_exact@12n'))} | {f(m.get('e99_exact@12n'))} | {f(m.get("e99_ll_raw@t_eq"),2)} / {f(m.get("e99_ll_ctrl@t_eq"),2)} | {m.get('trained')} |")
print()
print("| arm | n | e19 | e33 | e59 | e99 (mean, 95% Welch CI) |")
print("|---|---|---|---|---|---|")
for a, v in R["arms"].items():
    def c(k):
        d = v.get(k)
        if not d: return "-"
        ci = d.get("ci")
        return f"{d['mean']:.3f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
    print(f"| {a} | {len(v['seeds'])} | {c('e19_exact@12n')} | {c('e33_exact@12n')} | {c('e59_exact@12n')} | {c('e99_exact@t_eq')} |")
print()
print("| comparison | mean A | mean B | diff | Welch 95% CI | hier. bootstrap 95% CI | perm p |")
print("|---|---|---|---|---|---|---|")
for k, c in R["comparisons"].items():
    if "12n" in k and "e99" not in k: continue
    w, h = c["welch"], c["hboot"]
    print(f"| {k.replace(" | ", ", ")} | {c['mean_A']:.3f} | {c['mean_B']:.3f} | {w['diff']:+.3f} | [{w['lo']:.2f}, {w['hi']:.2f}] | [{h['lo']:.2f}, {h['hi']:.2f}] | {f(c.get('perm_p'))} |")
