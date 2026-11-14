"""M3 lead check: assert the numbers quoted in CURRENCY.md against their files. python3 check.py"""
import csv, json, os, re
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
def P(*a): return os.path.join(H, *a)
bad = []
def eq(name, got, want, tol):
    ok = abs(got - want) <= tol
    if not ok: bad.append(f"{name}: file {got} vs doc {want}")
cs = {r["currency"]: r for r in csv.DictReader(open(P("results/currency_summary.csv")))}
for c, m, s in [("cum", .043, .058), ("hl100", -.017, .030), ("hl1000", .001, .048), ("hl10000", .012, .056),
                ("win100", -.012, .044), ("win500", -.019, .029), ("win2000", .010, .045), ("gradnorm", .168, .070),
                ("gradnorm_win500", .127, .064), ("settle_loo_train", .417, .034), ("settle_loo_heldout", .576, .032),
                ("income_disjoint1k_stability_first5", .564, .066), ("income_disjoint1k_stability_last5", .355, .110),
                ("null_shuffled_oracle_sd", .044, 1)]:
    eq(c + " mean", float(cs[c]["mean"]), m, .0006); eq(c + " sd", float(cs[c]["sd"]), s, .0006) if s < 1 else None
sr = json.load(open(P("results/stop_rule.json")))
assert not sr["gradnorm"]["any_beats_by_0.05_in_3of3"] and not sr["gradnorm_win500"]["any_beats_by_0.05_in_3of3"]
for g, w in zip(sr["settle_loo_training_vs_gradnorm"], (.21, .25, .29)): eq("settle-gradnorm", g, w, .006)
pr = {(r["criterion"], r["frac"]): r for r in csv.DictReader(open(P("results/prune_final.csv")))}
for (c, f), m, s in [(("cum", "0.01"), .245, .382), (("cum", "0.2"), .821, .393), (("hl100", "0.01"), .156, .169),
                     (("hl100", "0.2"), 1.105, .233), (("random", "0.01"), .023, .014), (("random", "0.2"), .760, .120),
                     (("gradnorm", "0.01"), .009, .004), (("gradnorm", "0.2"), .392, .135)]:
    eq(f"prune {c} {f}", float(pr[(c, f)]["dbce_mean"]), m, .0006); eq(f"prune sd {c} {f}", float(pr[(c, f)]["dbce_sd"]), s, .0006)
wealth = [c for c, _ in pr if re.match(r"(cum|hl\d+|win\d+)$", c)]
for f in ("0.01", "0.2"):
    assert all(float(pr[(c, f)]["dbce_mean"]) > float(pr[("random", f)]["dbce_mean"]) for c in wealth), f
su = {r["arm"]: r for r in csv.DictReader(open(P("w2/results/summary.csv")))}
for a, m, s, lag, st in [("kelly", .690, .050, 9007, .0363), ("ema100", .074, .003, 188, .0375), ("ema1k", .334, .012, 881, .0363),
                         ("ema10k", .662, .035, 5793, .0363), ("fs1e-3", .047, .002, 120, .0373), ("fs1e-2", .054, .001, 118, .0461),
                         ("win500", .225, .014, 592, .0363), ("grad0.1", .397, .035, 1019, .0371), ("grad1", .336, .014, 597, .0443),
                         ("uniform", .556, .013, 10000, .5814), ("shuffled", .713, .073, 10000, .5975)]:
    eq(a + " post", float(su[a]["post2k_learnable_mean"]), m, .0006); eq(a + " post sd", float(su[a]["post2k_learnable_sd"]), s, .0006)
    eq(a + " lag", float(su[a]["lag_median_mean"]), lag, .6); eq(a + " stat", float(su[a]["stationary_mean"]), st, .00006)
eq("grad1 overall", float(su["grad1"]["overall_mean"]), .364, .0006); eq("fs overall", float(su["fs1e-3"]["overall_mean"]), .374, .0006)
de = json.load(open(P("w2/results/decision.json")))
assert de["best"] == "fs1e-3" and sorted(de["qualifying"]) == sorted(["ema100", "ema1k", "fs1e-3", "fs1e-2", "win500", "grad0.1"])
L = {}
for s in (1, 2, 3):
    for r in csv.DictReader(open(P(f"w2/results/switches_s{s}.csv"))):
        if r["to"] in ("periodic", "lfsr16"): L.setdefault((r["arm"], r["t"]), []).append(int(r["lag"]))
for a, want in [("kelly", (5486, 10000, 10000)), ("fs1e-3", (119, 121.5, 119))]:
    for t, w in zip(("10000", "20000", "30000"), want): eq(f"lag {a} {t}", float(np.median(L[(a, t)])), w, .01)
seeds = lambda a: np.array([float(x) for x in su[a]["post2k_learnable_seeds"].split("/")])
gap = np.minimum(seeds("grad1"), seeds("grad0.1")) - seeds("fs1e-3")
if gap.min() < .27: bad.append(f"fs vs grad per-seed gap {gap}")
print("FAIL\n" + "\n".join(bad) if bad else "OK: all CURRENCY.md numbers match their files"); raise SystemExit(1 if bad else 0)
