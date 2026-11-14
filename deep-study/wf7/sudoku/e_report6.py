"""e_report6.py -- tables for E3 (search candidates), E4 (a, b, c), E6, and the three figures.
Reads only what the other scripts saved. Usage: python -u e_report6.py > out/e_report6.out"""
import os, json, math
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import core

torch.set_num_threads(1); K, N = 6, 36
ARMN = {"S1": "S1  M_A, own confidence", "S2": "S2  M_A, least confident first + 5% floor (bad on purpose)",
        "S3": "S3  D6 (distilled from the search), own confidence", "S4": "S4  M_B, own confidence",
        "S5": "S5  M_A in the order of pi_trust"}
SETS = ("ALL", "HARD", "T12")
S = {s: {a: torch.load(f"out/search_{a}_{s}.pt") for a in ARMN if os.path.exists(f"out/search_{a}_{s}.pt")} for s in SETS}
S2 = {s: {a: torch.load(f"out/search_{a}_{s}_T2.pt") for a in ARMN if os.path.exists(f"out/search_{a}_{s}_T2.pt")} for s in SETS}   # post hoc check T2
Bz = {s: torch.load(f"out/base_{s}.pt") for s in SETS if os.path.exists(f"out/base_{s}.pt")}
REP = {}


FLM = 2 * 201030 * 36                       # one call of M_A (the pair check uses M_A only, no trust net)


def flops(r):
    """All networks counted. For the post hoc check T2 the check steps are pair-check calls (M_A only) plus, for an
    accepted candidate, the 36 route steps (M_A and the trust net)."""
    if r.get("chk") == "T2":
        route = 36 * (r["acc_idx"] >= 0).long()
        return (r["prop_steps"] * r["fl_prop"] + (r["chk_steps"] - route) * FLM + route * r["fl_chk"]).double()
    return (r["prop_steps"] * r["fl_prop"] + r["chk_steps"] * r["fl_chk"]).double()


def flops_pair_only(r):
    """What T2 would cost with the pair check alone (derived from the same run)."""
    route = 36 * (r["acc_idx"] >= 0).long()
    return (r["prop_steps"] * r["fl_prop"] + (r["chk_steps"] - route) * FLM).double()


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


# ----------------------------------------------------------------- E3: every candidate any search arm proposed
print("E3 (search part). Candidates proposed by the search arms, checked against T_A at the pre-set tau")
tot = dict(c=0, w=0, wa=0, rr=0, right=0)
for s in SETS:
    for a, r in S[s].items():
        c, w, wa, rr = int(r["n_cand"].sum()), int(r["n_wrong"].sum()), int(r["n_wrong_acc"].sum()), int(r["n_right_rej"].sum())
        tot["c"] += c; tot["w"] += w; tot["wa"] += wa; tot["rr"] += rr; tot["right"] += c - w
        print(f"   {s:5s} {a}: candidates {c:7d}; not the solution (so invalid) {w:7d}, of which accepted {wa}; "
              f"equal to the solution {c-w:6d}, of which rejected {rr}")
p, lo, hi = wilson(tot["wa"], tot["w"]); p2, lo2, hi2 = wilson(tot["rr"], tot["right"])
print(f"   TOTAL: invalid candidates {tot['w']}, accepted {tot['wa']} (rate {p:.2e}, 95% upper bound {hi:.2e}); "
      f"solution candidates {tot['right']}, rejected {tot['rr']} (rate {p2:.2e}, upper bound {hi2:.2e})")
REP["E3_search"] = dict(tot, fa_rate=p, fa_hi=hi, fr_rate=p2, fr_hi=hi2)
print("E3 (search part), POST HOC check T2 = pair check (one-cell contexts), then the same route threshold")
tot2 = dict(c=0, w=0, wa=0, rr=0, right=0)
for s in SETS:
    for a, r in S2[s].items():
        c, w, wa, rr = int(r["n_cand"].sum()), int(r["n_wrong"].sum()), int(r["n_wrong_acc"].sum()), int(r["n_right_rej"].sum())
        tot2["c"] += c; tot2["w"] += w; tot2["wa"] += wa; tot2["rr"] += rr; tot2["right"] += c - w
        print(f"   {s:5s} {a}: candidates {c:7d}; not the solution (so invalid) {w:7d}, of which accepted {wa}; "
              f"equal to the solution {c-w:6d}, of which rejected {rr}")
if tot2["c"]:
    p, lo, hi = wilson(tot2["wa"], tot2["w"]); p2, lo2, hi2 = wilson(tot2["rr"], tot2["right"])
    print(f"   TOTAL (T2): invalid candidates {tot2['w']}, accepted {tot2['wa']} (rate {p:.2e}, 95% upper bound {hi:.2e}); "
          f"solution candidates {tot2['right']}, rejected {tot2['rr']} (rate {p2:.2e}, upper bound {hi2:.2e})")
    REP["E3_search_T2"] = dict(tot2, fa_rate=p, fa_hi=hi, fr_rate=p2, fr_hi=hi2)

# ----------------------------------------------------------------- E4 (a), (b)
for s, SS, tag in [(s_, S, "") for s_ in SETS] + [(s_, S2, "_T2") for s_ in SETS]:
    S_cur = SS[s]
    if not S_cur:
        continue
    print(f"\nE4 on {s} ({'minimal puzzles' if s != 'T12' else 'puzzles with 12-16 clues'})" + ("   [POST HOC check T2]" if tag else "   [pre-registered check: route threshold]"))
    print(f"   {'search policy':60s} {'n':>4s} {'accepted':>9s} {'acc. correct':>13s} {'cand. 0 correct':>16s} {'cands to accept':>16s} {'FLOPs mean':>11s} {'median':>10s}")
    REP[f"E4_{s}{tag}"] = {}
    for a, r in S_cur.items():
        sol = r["sol"]; solved = r["acc_idx"] >= 0; correct = (r["answer"] == sol).all(1); f = flops(r)
        first_ok = (r["first"] == sol).all(1)
        REP[f"E4_{s}{tag}"][a] = dict(n=len(sol), accepted=float(solved.float().mean()), accepted_correct=float(correct[solved].float().mean()) if solved.any() else float("nan"),
                                n_wrong_answers=int((solved & ~correct).sum()), first_correct=float(first_ok.float().mean()),
                                cands_mean=float((r["acc_idx"][solved] + 1).float().mean()) if solved.any() else float("nan"),
                                flops_mean=float(f[solved].mean()) if solved.any() else float("nan"),
                                flops_median=float(f[solved].median()) if solved.any() else float("nan"), flops_all_mean=float(f.mean()))
        e = REP[f"E4_{s}{tag}"][a]
        if tag and solved.any():
            e["flops_pair_only_mean"] = float(flops_pair_only(r)[solved].mean()); e["flops_pair_only_median"] = float(flops_pair_only(r)[solved].median())
        print(f"   {ARMN[a]:60s} {e['n']:4d} {e['accepted']:9.4f} {e['accepted_correct']:13.4f} {e['first_correct']:16.4f} {e['cands_mean']:16.2f} {e['flops_mean']:11.3e} {e['flops_median']:10.3e}")
    if tag:
        print("      with the pair check alone (derived): FLOPs to accept, mean / median: " + "; ".join(
            f"{a} {REP[f'E4_{s}{tag}'][a].get('flops_pair_only_mean', float('nan')):.3e} / {REP[f'E4_{s}{tag}'][a].get('flops_pair_only_median', float('nan')):.3e}" for a in S_cur))
    arms = list(S_cur); agree_n = 0; agree_k = 0
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            ra, rb = S_cur[arms[i]], S_cur[arms[j]]; n = min(len(ra["sol"]), len(rb["sol"]))
            both = (ra["acc_idx"][:n] >= 0) & (rb["acc_idx"][:n] >= 0)
            same = (ra["answer"][:n][both] == rb["answer"][:n][both]).all(1)
            agree_n += int(both.sum()); agree_k += int(same.sum())
            fa_, fb_ = flops(ra)[:n][both], flops(rb)[:n][both]
            print(f"   same answer, {arms[i]} and {arms[j]}: {int(same.sum())} of {int(both.sum())} puzzles where both accepted one; "
                  f"FLOPs on those puzzles: {arms[i]} mean {float(fa_.mean()):.3e} median {float(fa_.median()):.3e}, {arms[j]} mean {float(fb_.mean()):.3e} median {float(fb_.median()):.3e} "
                  f"(ratio of means {float(fb_.mean() / fa_.mean()):.2f})")
    REP[f"E4_{s}{tag}"]["agree"] = (agree_k, agree_n)
    allacc = sum(int((r['acc_idx'] >= 0).sum()) for r in S_cur.values()); allwrong = sum(int(((r['acc_idx'] >= 0) & ~(r['answer'] == r['sol']).all(1)).sum()) for r in S_cur.values())
    print(f"   E4(a) on {s}{tag}: identical answers in {agree_k} of {agree_n} pairs; accepted answers that are not the solution: {allwrong} of {allacc}"
          f" -> {'PASS' if agree_k == agree_n and allwrong <= 0.001 * allacc else 'FAIL'}")

# ----------------------------------------------------------------- E4 (c): accuracy against FLOPs on HARD and ALL
def anytime(r, budgets):
    sol = r["sol"]; f0 = (r["nfree"] * r["fl_prop"]).double(); fa = flops(r)
    solved = r["acc_idx"] >= 0; correct = (r["answer"] == sol).all(1); first_ok = (r["first"] == sol).all(1)
    fa = torch.where(solved, fa, torch.full_like(fa, float("inf")))
    return np.array([float((((b >= fa) & correct) | ((b >= f0) & (b < fa) & first_ok)).float().mean()) for b in budgets])


def base_points(b):
    nf = b["nfree"].double(); FL = b["FL"]; pts = []
    pts.append(("K0 one pass of M_A, no check", float((nf * FL["mA6"]).mean()), float(b["K0"].float().mean())))
    for nm in ("mB6t", "mB6", "mB6L"):
        if f"K1_{nm}_margin" in b:
            pts.append((f"K1 one pass of {nm}, no check", float((nf * FL[nm]).mean()), float(b[f"K1_{nm}_margin"].float().mean())))
    if "KT" in b:
        pts.append(("KT one pass of mB6 in trust order", float((nf * (FL["mB6"] + FL["trustB"])).mean()), float(b["KT"].float().mean())))
    k2 = [(float((k * nf * FL["mB6"]).mean()), float(a.float().mean())) for k, a in zip(b["K2_k"], b["K2_acc"])]
    return pts, k2


budgets = np.logspace(8, 12.6, 80)
for s in ("HARD", "ALL"):
    if s not in Bz or not S[s]:
        continue
    pts, k2 = base_points(Bz[s]); b = Bz[s]
    print(f"\nE4(c) on {s}: accuracy against FLOPs (anytime rule for the search arms)")
    for nm, x, y in pts:
        print(f"   {nm:40s} FLOPs {x:.3e}  accuracy {y:.4f}")
    print("   K2 best of k passes of mB6 by its own log-probability: " + "  ".join(f"k={k}: {y:.4f} @ {x:.2e}" for k, (x, y) in zip(b["K2_k"], k2)))
    for rule in ("maxprob", "entropy"):
        if f"K1_mB6_{rule}" in b:
            print(f"   K1 one pass of mB6 with rule {rule}: accuracy {float(b[f'K1_mB6_{rule}'].float().mean()):.4f}")
    curves = {a: anytime(r, budgets) for a, r in S[s].items()}
    marks = [3e8, 1e9, 3e9, 1e10, 3e10, 1e11, 1e12]
    print("   budget (FLOPs per puzzle):            " + " ".join(f"{m:9.0e}" for m in marks))
    for a, r in S[s].items():
        print(f"   {a} (n={len(r['sol'])}) accuracy within budget: " + " ".join(f"{v:9.4f}" for v in anytime(r, marks)))
    for a, r in S2[s].items():
        print(f"   {a} with T2 (post hoc; n={len(r['sol'])}):      " + " ".join(f"{v:9.4f}" for v in anytime(r, marks)))
    # headline bins: S1 and S4 against the baseline points
    basel = [(x, y) for nm, x, y in pts if nm.startswith("K1") or nm.startswith("K0")] + k2
    best_base = max(y for x, y in basel)
    for a, SS, tag in (("S1", S, ""), ("S4", S, ""), ("S1", S2, "_T2"), ("S4", S2, "_T2")):
        if a not in SS[s]:
            continue
        r = SS[s][a]; final = float(anytime(r, [1e15])[0]); verdict = None; detail = ""
        for xb, yb in basel:
            ya = float(anytime(r, [xb])[0]); below = max(y for x, y in basel if x <= xb)
            if ya >= below + 0.10:
                verdict = "major"; detail = f"at {xb:.2e} FLOPs: {a} {ya:.4f} against best baseline of equal or lower FLOPs {below:.4f}"; break
        if verdict is None and final >= 0.99 and best_base < 0.90:
            verdict = "major"; detail = f"{a} reaches {final:.4f}; every baseline point is below 0.90"
        if verdict is None:
            if best_base >= 0.99:
                verdict = "null"; detail = f"best baseline point {best_base:.4f} >= 0.99: no headroom"
            elif final > best_base:
                xbest = min(x for x, y in basel if y == best_base)
                xs = [bb for bb in budgets if float(anytime(r, [bb])[0]) > best_base]
                ratio = (xs[0] / xbest) if xs else float("inf")
                verdict = "minor" if ratio > 10 else "better at <= 10x the FLOPs (between the pre-registered minor and major bins)"
                detail = f"final {final:.4f} against best baseline {best_base:.4f}; passes it at {ratio:.1f}x its FLOPs"
            else:
                verdict = "negative"; detail = f"final {final:.4f} against best baseline {best_base:.4f}"
        print(f"   HEADLINE E4(c) on {s}, {a}{' with the post hoc check T2' if tag else ''} against K0/K1/K2: {verdict.upper()} ({detail})")
        REP[f"E4c_{s}_{a}{tag}"] = dict(verdict=verdict, detail=detail, final=final, best_base=best_base)
    if s == "HARD":
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        cols = {"S1": "C0", "S2": "C1", "S3": "C2", "S4": "C3", "S5": "C4"}
        for a, v in curves.items():
            ax.plot(budgets, v, color=cols[a], alpha=.45, lw=1.2, label=ARMN[a].split("  ")[0] + ": " + ARMN[a].split("  ")[1].split(" (")[0] + ", route check (pre-registered)")
        for a, r in S2[s].items():
            ax.plot(budgets, anytime(r, budgets), color=cols[a], lw=2.0, label=ARMN[a].split("  ")[0] + ", pair check + route check (post hoc)")
        ax.plot([x for x, y in k2], [y for x, y in k2], "k--o", ms=3, lw=1, label="K2: best of k passes of M_B, own score, no check")
        for nm, x, y in pts:
            ax.plot([x], [y], "s" if nm.startswith("K1") else "^", ms=7, label=nm)
        ax.set_xscale("log"); ax.set_xlabel("FLOPs per puzzle (all networks counted)"); ax.set_ylabel("fraction of hard held-out puzzles answered correctly")
        ax.set_title("6x6 hard held-out puzzles: search corrected against T, and one-pass decoding"); ax.grid(alpha=.3); ax.legend(fontsize=5.8, loc="lower right")
        fig.tight_layout(); fig.savefig("fig/fig3_accuracy_vs_flops.png", dpi=150); plt.close(fig)

# ----------------------------------------------------------------- E6: fewer clues than any training puzzle of D6
if "ALL" in Bz and "T12" in Bz:
    print("\nE6. Fewer clues than any training puzzle of D6 (D6 saw 12-16 clues; FEW = puzzles of ALL with at most 11 clues)")
    few = Bz["ALL"]["nclue"] <= 11
    print(f"   one pass of M_A (K0), no check: accuracy on T12 {Bz['T12']['K0'].float().mean():.4f}; on FEW {Bz['ALL']['K0'][few].float().mean():.4f}; on all of ALL {Bz['ALL']['K0'].float().mean():.4f}")
    for nm in ("D6", "mB6"):
        if f"K1_{nm}_margin" in Bz["ALL"]:
            a_in = Bz["T12"][f"K1_{nm}_margin"].float().mean(); a_few = Bz["ALL"][f"K1_{nm}_margin"][few].float().mean()
            print(f"   one pass of {nm:4s}, no check: accuracy on T12 (12-16 clues) {a_in:.4f}; on FEW ({int(few.sum())} puzzles, 8-11 clues) {a_few:.4f}")
            REP[f"E6_onepass_{nm}"] = (float(a_in), float(a_few))
    for a in ("S1", "S4"):
        if a in S["ALL"] and a not in S["T12"]:
            r = S["ALL"][a]; sel = r["nclue"] <= 11; solved = (r["acc_idx"] >= 0) & sel; correct = (r["answer"] == r["sol"]).all(1); f = flops(r)
            print(f"   {ARMN[a]}: FEW only ({int(sel.sum())} puzzles): accepted {float(solved.sum() / sel.sum()):.4f}, wrong answers {int((solved & ~correct).sum())}, "
                  f"FLOPs {float(f[solved].mean()):.3e}, candidates {float((r['acc_idx'][solved] + 1).float().mean()):.2f}")
            REP[f"E6_{a}_FEW"] = dict(n=int(sel.sum()), accepted=float(solved.sum() / sel.sum()), wrong=int((solved & ~correct).sum()), flops=float(f[solved].mean()))
    for a in ("S3", "S1", "S4"):
        if a in S["ALL"] and a in S["T12"]:
            out = {}
            for nm, r, sel in (("T12", S["T12"][a], None), ("FEW", S["ALL"][a], S["ALL"][a]["nclue"] <= 11)):
                sel = torch.ones(len(r["sol"]), dtype=torch.bool) if sel is None else sel
                solved = (r["acc_idx"] >= 0) & sel; correct = (r["answer"] == r["sol"]).all(1); f = flops(r)
                out[nm] = dict(n=int(sel.sum()), accepted=float(solved.sum() / sel.sum()), correct=float(correct[solved].float().mean()),
                               wrong=int((solved & ~correct).sum()), flops=float(f[solved].mean()), cands=float((r["acc_idx"][solved] + 1).float().mean()))
            print(f"   {ARMN[a]}: T12: accepted {out['T12']['accepted']:.4f}, wrong answers {out['T12']['wrong']}, FLOPs {out['T12']['flops']:.3e}, candidates {out['T12']['cands']:.2f} | "
                  f"FEW: accepted {out['FEW']['accepted']:.4f}, wrong answers {out['FEW']['wrong']}, FLOPs {out['FEW']['flops']:.3e}, candidates {out['FEW']['cands']:.2f} | "
                  f"compute ratio FEW/T12 {out['FEW']['flops']/out['T12']['flops']:.2f}")
            REP[f"E6_{a}"] = out

# ----------------------------------------------------------------- figure 2: log T of valid grids and of invalid boards
meta = json.load(open("out/trust_meta_mA6.json")); tau = meta["tau"]
nll = np.load("out/e_trust6_mA6_nll.npz"); valid_lt = -nll["pi_trust_learned_on_real_grids"]
MA = core.Net(core.load_model("ckpt/mA6.pt")[0], K, "mA6")
ck = torch.load("ckpt/trust_mA6.pt", map_location="cpu"); m = core.RelFormer(**ck["cfg"]); m.load_state_dict(ck["state"]); tA = core.TrustPolicy(m, K)
wb = [r["wrong_boards"] for s in SETS for r in S[s].values() if len(r["wrong_boards"])]
if os.environ.get("NOFIG2"):
    wb = []
if wb:
    wb = torch.unique(torch.cat(wb), dim=0); wb = wb[torch.randperm(len(wb), generator=torch.Generator().manual_seed(1))[:1500]]
    wl = core.route_logT(MA, tA, wb)["logp"].numpy()
    print(f"\nfull log T (no early stop) of {len(wl)} distinct wrong search candidates: max {wl.max():.2f}, median {np.median(wl):.2f}, "
          f"share within 5 nats of tau {np.mean(wl > tau - 5):.4f}; held-out valid grids: min {valid_lt.min():.2f}, median {np.median(valid_lt):.2f}; tau {tau:.2f}")
    REP["wrong_cand_logT"] = dict(max=float(wl.max()), median=float(np.median(wl)), n=len(wl))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bins = np.linspace(min(wl.min(), -80), -15, 90)
    ax.hist(np.clip(wl, bins[0], None), bins=bins, alpha=.7, label=f"distinct wrong candidates proposed by the search arms (n={len(wl)})")
    ax.hist(valid_lt, bins=bins, alpha=.8, label=f"held-out valid grids (n={len(valid_lt)})")
    ax.axvline(tau, color="k", ls="--", lw=1, label=f"acceptance threshold tau = {tau:.2f} (from real validation grids)")
    ax.axvline(-math.log(28200960), color="g", ls=":", lw=1, label="uniform over the 28,200,960 valid grids")
    ax.set_xlabel("log T_A(board)"); ax.set_ylabel("boards"); ax.set_yscale("log"); ax.legend(fontsize=7)
    ax.set_title("6x6: T_A on valid grids and on the invalid boards the search proposes")
    fig.tight_layout(); fig.savefig("fig/fig2_T_separation.png", dpi=150); plt.close(fig)
json.dump(REP, open("out/e_report6.json", "w"), indent=1, default=float)
