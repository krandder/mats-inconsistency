"""Item 2 analysis, as pre-registered in PREREG.md. Reads vmout/results/ (and vmout/GATES.json), writes
results/summary.json and results/summary.txt.   python3 analyze.py
"""
import itertools
import json
import math
import os
import statistics as st

from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = f"{HERE}/vmout/results"
P10 = f"{HERE}/../../P10/vm_out/results"
OUT = f"{HERE}/results"
SEEDS = list(range(10))
FIXES = ("init", "warm", "both")
FEE = 0.005
THR = 0.01      # non-constant if min(p1, 1 - p1) > THR (primary)
STEPS = 20000
AGED = 7000     # auction secondary: last (re)spawn at least this many steps before the end
LEARNED = ("learned0", "learned1", "learned2")
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    lines.append(s)
    print(s)


def load(arm, s, d=RES):
    p = f"{d}/m2_{arm}_T1_seed{s}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def nonconst(p):
    return min(p, 1 - p) > THR


def live(o):
    return {r["slot"]: r for r in o["nmi"]["live"] if r.get("slot") in LEARNED}


def frac(o):
    lv = live(o)
    return sum(nonconst(lv[n]["p1"]) for n in LEARNED) / 3


def frac_aged(o):
    last = {n: 0 for n in LEARNED}
    for r in o["log"]["respawns"]:
        if r["slot"] in last:
            last[r["slot"]] = max(last[r["slot"]], r["step"])
    lv = live(o)
    aged = [n for n in LEARNED if last[n] <= STEPS - AGED]
    return (sum(nonconst(lv[n]["p1"]) for n in aged) / len(aged)) if aged else None, len(aged)


def collapse_step(o):
    for row in o.get("p1_trace_val", []):
        if all(not nonconst(p) for p in row[1:4]):
            return row[0]
    return None


def tci(x):
    x = [v for v in x if v is not None]
    n = len(x)
    if n == 0:
        return {"n": 0}
    m = st.mean(x)
    if n < 2:
        return {"mean": m, "n": n}
    sd = st.stdev(x)
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return {"mean": m, "lo": m - h, "hi": m + h, "sd": sd, "n": n}


def perm_p(d):
    """Exact two-sided sign-flip permutation p for the mean of paired differences d."""
    obs = abs(st.mean(d))
    hits = sum(abs(st.mean([s * v for s, v in zip(signs, d)])) >= obs - 1e-12 for signs in itertools.product((1, -1), repeat=len(d)))
    return hits / 2 ** len(d)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def fmt(c, k=3):
    if "lo" in c:
        return f"{c['mean']:.{k}f} [{c['lo']:.{k}f}, {c['hi']:.{k}f}] (n={c['n']})"
    return f"{c.get('mean', float('nan')):.{k}f} (n={c['n']})"


def main():
    os.makedirs(OUT, exist_ok=True)
    S = {"prereg_sha256": open(f"{HERE}/PREREG.sha256").read().split()[0]}
    gates = json.load(open(f"{HERE}/vmout/GATES.json")) if os.path.exists(f"{HERE}/vmout/GATES.json") else {}
    S["gates"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("p1_")} if isinstance(v, dict) else v for k, v in gates.items()}
    say("gates:", json.dumps(S["gates"]))

    arms = ["cvae"] + [f"cvae_{f}" for f in FIXES] + ["auction", "auction_both"]
    R = {a: {s: load(a, s) for s in SEEDS} for a in arms}
    R["reg2"] = {s: load("reg2", s) for s in SEEDS}
    S["n_files"] = {a: sum(o is not None for o in v.values()) for a, v in R.items()}
    say("files per arm:", S["n_files"])

    # ---- primary: cvae non-constant fraction, fix vs orig, paired by seed
    say("\nPRIMARY: fraction of the 3 learned cvae slots non-constant at the end (test, min(p1,1-p1) > 0.01)")
    fr = {a: {s: frac(o) for s, o in R[a].items() if o} for a in arms}
    prim, ps = {}, []
    base = fr["cvae"]
    for f in FIXES:
        a = f"cvae_{f}"
        ss = [s for s in SEEDS if s in fr[a] and s in base]
        d = [fr[a][s] - base[s] for s in ss]
        p = perm_p(d) if d else 1.0
        ps.append(p)
        prim[a] = {"seeds": ss, "frac": [fr[a][s] for s in ss], "frac_orig": [base[s] for s in ss],
                   "mean_frac": st.mean(fr[a][s] for s in ss) if ss else None, "diff": tci(d), "perm_p": p,
                   "slots_nonconst": f"{round(sum(fr[a][s] for s in ss) * 3)}/{3 * len(ss)}"}
    for f, pa in zip(FIXES, holm(ps)):
        prim[f"cvae_{f}"]["holm_p"] = pa
    prim["cvae_orig"] = {"frac": [base[s] for s in sorted(base)], "slots_nonconst": f"{round(sum(base.values()) * 3)}/{3 * len(base)}"}
    for a, v in prim.items():
        say(f"  {a:14s} non-constant slots {v['slots_nonconst']}" + (f"  diff vs orig {fmt(v['diff'])}  perm p {v['perm_p']:.4f}  Holm p {v['holm_p']:.4f}" if "diff" in v else ""))
    tot = sum(round(sum(fr[a].values()) * 3) for a in (f"cvae_{f}" for f in FIXES))
    n_tot = sum(3 * len(fr[f"cvae_{f}"]) for f in FIXES)
    S["primary"] = prim
    S["primary"]["pooled_fix_slots_nonconst"] = f"{tot}/{n_tot}"
    if tot == 0:
        S["primary"]["upper95_per_slot_rate_per_arm"] = 1 - 0.05 ** (1 / 30)
        S["primary"]["upper95_per_slot_rate_pooled"] = 1 - 0.05 ** (1 / n_tot)

    # ---- secondaries
    say("\nSECONDARY per arm (seed-level t 95% CI)")
    sec = {}
    for a in arms:
        os_ = {s: o for s, o in R[a].items() if o}
        if not os_:
            continue
        rows = {}
        for s, o in os_.items():
            tr = o["test_royalty"]
            Gs = [tr[n]["G"] for n in LEARNED]
            lv = live(o)
            nc = [n for n in LEARNED if nonconst(lv[n]["p1"])]
            rows[s] = {"frac": frac(o), "n_G_above_fee": sum(g > FEE for g in Gs), "best_G": max(Gs),
                       "best_gain_resample": max(tr[n]["gain_resample"] for n in LEARNED),
                       "best_gain_vs_const": max(tr[n]["gain_vs_const"] for n in LEARNED),
                       "max_nmi_noncell_nonconst": max([lv[n]["max_nmi_noncell"] for n in nc], default=None),
                       "max_nmi_cell_nonconst": max([lv[n]["max_nmi_cell"] for n in nc], default=None),
                       "collapse_step_val": collapse_step(o), "nll": o["test_nll_ehat_per_cell"],
                       "respawns": len(o["log"]["respawns"]), "learned_listed_end": sum(o["listed_final"][n] for n in LEARNED),
                       "surviving_G_total": o["surviving_G_total"], "planted_G": tr["planted"]["G"],
                       "p1_end": [lv[n]["p1"] for n in LEARNED]}
            if a.startswith("auction"):
                rows[s]["frac_aged"], rows[s]["n_aged"] = frac_aged(o)
        agg = {"seeds_with_learned_G_above_fee": sum(r["n_G_above_fee"] > 0 for r in rows.values()), "n": len(rows),
               "best_G": tci([r["best_G"] for r in rows.values()]), "nll": tci([r["nll"] for r in rows.values()]),
               "collapse_steps_val": [r["collapse_step_val"] for r in rows.values()],
               "never_collapsed_val": sum(r["collapse_step_val"] is None for r in rows.values()),
               "respawns": tci([r["respawns"] for r in rows.values()]), "frac": tci([r["frac"] for r in rows.values()])}
        cs = [r["collapse_step_val"] for r in rows.values() if r["collapse_step_val"] is not None]
        agg["median_collapse_step_val"] = st.median(cs) if cs else None
        nr = [r["max_nmi_noncell_nonconst"] for r in rows.values() if r["max_nmi_noncell_nonconst"] is not None]
        agg["max_nmi_noncell_nonconst"] = max(nr) if nr else None
        if a.startswith("auction"):
            agg["frac_aged"] = tci([r["frac_aged"] for r in rows.values()])
        reg = {s: R["reg2"][s]["test_nll_per_cell"] for s in rows if R["reg2"].get(s)}
        agg["nll_minus_reg2"] = tci([rows[s]["nll"] - reg[s] for s in reg])
        orig = {s: frac_o["test_nll_ehat_per_cell"] for s, frac_o in R["cvae"].items() if frac_o and s in rows}
        agg["nll_minus_cvae_orig"] = tci([rows[s]["nll"] - orig[s] for s in orig])
        if "_warm" in a or "_both" in a:
            agg["collapsed_before_step4000_val"] = sum(r["collapse_step_val"] is not None and r["collapse_step_val"] < 4000 for r in rows.values())
        sec[a] = {"agg": agg, "per_seed": rows}
        say(f"  {a:18s} frac {fmt(agg['frac'], 2)} | seeds w/ learned G>fee {agg['seeds_with_learned_G_above_fee']}/{len(rows)}"
            f" | best G {fmt(agg['best_G'], 4)} | NLL-reg2 {fmt(agg['nll_minus_reg2'], 3)} | val collapse median {agg['median_collapse_step_val']}"
            f" never {agg['never_collapsed_val']} | respawns {agg['respawns'].get('mean', 0):.1f}"
            + (f" | aged frac {fmt(agg['frac_aged'], 2)}" if "frac_aged" in agg else "")
            + (f" | collapsed before 4000 (no rate grad) {agg['collapsed_before_step4000_val']}/{len(rows)}" if "collapsed_before_step4000_val" in agg else ""))
    reg = [R["reg2"][s]["test_nll_per_cell"] for s in SEEDS if R["reg2"].get(s)]
    sec["reg2"] = {"nll": tci(reg), "per_seed": reg}
    say(f"  reg2 NLL {fmt(tci(reg))}")
    S["secondary"] = sec

    # ---- A/A vs P10, seeds 0-4
    say("\nA/A vs P10 (seeds 0-4)")
    aa = {}
    for a in ("cvae", "auction", "reg2"):
        mine, theirs = [], []
        for s in range(5):
            o, q = R[a].get(s), load(a, s, P10)
            if not (o and q):
                continue
            k = "test_nll_per_cell" if a == "reg2" else "test_nll_ehat_per_cell"
            mine.append(o[k]); theirs.append(q[k])
            if a != "reg2":
                aa.setdefault(a + "_frac", []).append([frac(o), frac(q)])
        aa[a + "_nll"] = {"here": mine, "p10": theirs, "diff": tci([x - y for x, y in zip(mine, theirs)])}
        say(f"  {a} NLL here {[round(x, 3) for x in mine]} P10 {[round(x, 3) for x in theirs]} diff {fmt(aa[a + '_nll']['diff'])}"
            + (f" | frac here/P10 {aa.get(a + '_frac')}" if a != "reg2" else ""))
    S["aa"] = aa

    # ---- verdict
    gate_fail = any(isinstance(v, dict) and v.get("pass") is False for v in gates.values()) or "failed_gate" in gates
    s2_failed = len(gates.get("stage2_failed_jobs", []))
    if gate_fail or s2_failed > 0.1 * gates.get("stage2_jobs", 40):
        verdict = "Problem"
    else:
        sig = [f for f in FIXES if prim[f"cvae_{f}"]["holm_p"] < 0.05]
        gain = [f for f in sig if prim[f"cvae_{f}"]["mean_frac"] >= 0.5 and sec[f"cvae_{f}"]["agg"]["seeds_with_learned_G_above_fee"] >= 5]
        verdict = "Gain" if gain else ("Informative" if sig else "No gain")
    S["verdict"] = verdict
    say("\nVERDICT (PREREG rules):", verdict)
    json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
