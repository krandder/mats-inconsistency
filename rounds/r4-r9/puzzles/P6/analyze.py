"""P6 analysis: results/train/*.json + results/eval/*.jsonl -> results/summary.json and results/chart.json.

O1   log2 L_reached (tie-break: iterations to the 256 cap).
O2   L*_4: largest test L with exact >= 0.9 and Wilson 95% lower bound >= 0.8 at D4 = min(4L, 32768)
     (log2; 4 = no length passes).
CO   continuous co-primary: mean nats per bit over L 32..8192 at D4 (lower is better).
Paired contrasts E-clone minus F / PBT / EMA by seed, paired bootstrap 95% CI over seeds (10000 resamples).
"""
import glob, json, math, os, random, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ARMS = ["R0", "DTL", "R1", "F", "EC", "PBT", "EMA"]
LS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
VM_RATE = 0.71


def load_eval(path):
    cells = {}
    for line in open(path):
        r = json.loads(line)
        cells[r["cell"]] = r
    return cells


def lstar(cells, key="exact4", lo="wilson_lo4"):
    ok = [c["L"] for c in cells.values() if c["cell"].startswith("L") and c[key] >= 0.9 and c[lo] >= 0.8]
    return max(ok) if ok else 16


def wilson_lo(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = p + z * z / (2 * n)
    return (c - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den


def lstar_diag(cells):
    ok = []
    for c in cells.values():
        if not c["cell"].startswith("L"):
            continue
        row = [r for r in c["grid"] if r[0] == c["L"]]
        if row and row[0][1] / c["n"] >= 0.9 and wilson_lo(row[0][1], c["n"]) >= 0.8:
            ok.append(c["L"])
    return max(ok) if ok else 16


def speed(cells):
    """learned speed v = L / smallest D with exact >= 0.9, at the largest L where exact >= 0.9 is reached."""
    best = None
    for c in sorted((c for c in cells.values() if c["cell"].startswith("L")), key=lambda c: c["L"]):
        ds = [r[0] for r in c["grid"] if r[1] / c["n"] >= 0.9]
        if ds:
            best = (c["L"], c["L"] / min(ds))
    return best


def model_row(arm, seed):
    name = f"{arm}_s{seed}"
    tr = json.load(open(os.path.join(RES, "train", name + ".json")))
    row = dict(arm=arm, seed=seed, L_reached=tr["L_reached"], log2_L_reached=math.log2(tr["L_reached"]),
               cap_it=tr.get("cap_it"), doublings=tr.get("doublings"), sec_per_iter=tr.get("sec_per_iter"),
               train_sec=tr.get("train_sec"))
    var = "plain" if arm in ("R0", "DTL") else "auction"
    ep = os.path.join(RES, "eval", f"{name}_{var}.jsonl")
    if os.path.exists(ep):
        cells = load_eval(ep)
        Lc = {c["L"]: c for c in cells.values() if c["cell"].startswith("L")}
        row.update(Lstar4=lstar(cells), log2_Lstar4=math.log2(lstar(cells)), Lstar_diag=lstar_diag(cells),
                   speed=speed(cells), eval_sec=sum(c.get("sec", 0) for c in cells.values()),
                   exact4={L: Lc[L]["exact4"] for L in sorted(Lc)}, nats4={L: Lc[L]["nats4"] for L in sorted(Lc)},
                   n_L=len(Lc))
        if all(L in Lc for L in LS):
            row["co_nats"] = st.mean(Lc[L]["nats4"] for L in LS)
        # overthinking retention: exact at the last grid depth minus its peak over D, mean over L
        ret = {L: Lc[L]["grid"][-1][1] / Lc[L]["n"] - max(r[1] for r in Lc[L]["grid"]) / Lc[L]["n"] for L in Lc}
        row["retention"] = ret
        row["retention_mean"] = st.mean(ret.values()) if ret else None
        row["resid_final"] = {L: Lc[L]["resid_final"] for L in Lc}
        row["gate_L32_D32"] = next((r[1] / Lc[32]["n"] for r in Lc[32]["grid"] if r[0] == 32), None) if 32 in Lc else None
        for k in ("pi256", "pi1024", "pi2048"):
            if k in cells:
                c = cells[k]
                row[k] = dict(all4_agree=c["all4_agree"], agree_with_std_init=c["agree_with_std_init"],
                              rel_state_dist=st.mean(c["rel_state_dist"]), exact_mean=c["exact_mean"],
                              exact_std_init=c["exact_std_init"])
        if var == "auction" and arm != "R1":
            row["halting"] = {L: dict(frac_halted=Lc[L].get("frac_halted"), halt_over_L=Lc[L].get("halt_over_L_mean"),
                                      steps=Lc[L].get("steps_applied_mean"), exact_given_halted=Lc[L].get("exact_given_halted"),
                                      s_small=Lc[L].get("s_small_frac"))
                              for L in Lc}
    sp = os.path.join(RES, "eval", f"{name}_single.jsonl")
    if os.path.exists(sp):
        sc = load_eval(sp)
        row["single_Lstar4"] = lstar(sc)
        row["single_log2_Lstar4"] = math.log2(lstar(sc))
        row["single_slot"] = next(iter(sc.values())).get("slot")
    if arm in ("F", "EC", "PBT", "EMA", "R1"):
        row.update(econ_dynamics(tr))
    return row


def econ_dynamics(tr):
    it_end = tr["args"]["iters"]
    ev = tr.get("events", [])
    rep = [e for e in ev if e["kind"] == "replace"]
    bail = [e for e in ev if e["kind"] == "bailout"]
    causes = {}
    for e in rep:
        c = e.get("comp", {})
        losses = {"gross": -min(c.get("gross", 0), 0), "price": c.get("price", 0), "tax": c.get("tax", 0), "cost": c.get("cost", 0)}
        k = max(losses, key=losses.get)
        causes[k] = causes.get(k, 0) + 1
    win = [r["win"] for r in tr["log"] if "win" in r]
    passes = sum(w["passes"] for w in win); rc = sum(w["rowchunks"] for w in win)
    last = tr["log"][-1]
    # entrant survival: of the entrants (non-founder lives) born at least h iterations before the end,
    # the fraction still alive h iterations after entry (lives: (id, born, died or None if alive at the end, founder))
    ents = [(l[1], l[2]) for l in tr.get("lives", []) if not l[3]]
    surv = {}
    for h in (500, 2000):
        el = [(b, d) for (b, d) in ents if b + h <= it_end]
        if el:
            surv[h] = sum(1 for (b, d) in el if d is None or d - b >= h) / len(el)
    return dict(exits=len(rep), bailouts=len(bail), exits_per_1000=1000 * len(rep) / it_end,
                bailouts_per_1000=1000 * len(bail) / it_end, pass_rate=passes / max(1, rc), exit_causes=causes,
                eff_n=last.get("eff_n"), gini=last.get("gini"), injected=tr.get("injected"),
                spear_w_ema=tr.get("spear_w_ema_final"), max_ledger_err=tr.get("max_ledger_err"), entrant_survival=surv,
                exit_its=[e["it"] for e in rep])


def boot_ci(d, n=10000, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choice(d) for _ in d) for _ in range(n))
    return [ms[int(0.025 * n)], ms[int(0.975 * n) - 1]]


def tci(v):
    if len(v) < 2:
        return None
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(len(v), 2.0)
    return t * st.stdev(v) / math.sqrt(len(v))


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "train", "*.json"))):
        b = os.path.basename(f)[:-5]
        arm, s = b.rsplit("_s", 1)
        rows.append(model_row(arm, int(s)))
    by = {a: [r for r in rows if r["arm"] == a] for a in ARMS}
    arms = {}
    for a, rs in by.items():
        if not rs:
            continue
        agg = dict(n=len(rs))
        for k in ("log2_L_reached", "cap_it", "log2_Lstar4", "co_nats", "retention_mean", "single_log2_Lstar4",
                  "exits", "exits_per_1000", "bailouts_per_1000", "pass_rate", "eff_n", "gini", "spear_w_ema",
                  "sec_per_iter", "train_sec", "eval_sec"):
            v = [r[k] for r in rs if r.get(k) is not None]
            if v:
                agg[k] = dict(mean=st.mean(v), ci95=tci(v), vals=v)
        v = [r["Lstar4"] for r in rs if "Lstar4" in r]
        if v:
            agg["Lstar4_median"] = st.median(v)
        agg["exact4_by_L"] = {L: st.mean(r["exact4"][L] for r in rs if L in r.get("exact4", {}))
                              for L in LS if any(L in r.get("exact4", {}) for r in rs)}
        agg["nats4_by_L"] = {L: st.mean(r["nats4"][L] for r in rs if L in r.get("nats4", {}))
                             for L in LS if any(L in r.get("nats4", {}) for r in rs)}
        arms[a] = agg
    contrasts = {}
    ec = {r["seed"]: r for r in by.get("EC", [])}
    for other in ("F", "PBT", "EMA"):
        ot = {r["seed"]: r for r in by.get(other, [])}
        seeds = sorted(set(ec) & set(ot))
        c = dict(seeds=seeds)
        for k in ("log2_L_reached", "log2_Lstar4", "co_nats", "cap_it"):
            d = [ec[s][k] - ot[s][k] for s in seeds if ec[s].get(k) is not None and ot[s].get(k) is not None]
            if len(d) >= 2:
                c[k] = dict(mean=st.mean(d), boot95=boot_ci(d), diffs=d)
        contrasts[f"EC-{other}"] = c
    # population matters: auction (a) minus richest single trader (b), per econ model with both
    pop = {}
    for a in ("F", "EC", "PBT", "EMA"):
        d = [r["log2_Lstar4"] - r["single_log2_Lstar4"] for r in by.get(a, []) if "single_log2_Lstar4" in r and "log2_Lstar4" in r]
        if d:
            pop[a] = dict(mean=st.mean(d), diffs=d)
    summ = dict(arms=arms, contrasts=contrasts, population=pop, models=rows)
    num = os.path.join(RES, "numerics.json")
    if os.path.exists(num):
        summ["numerics"] = json.load(open(num))
    json.dump(summ, open(os.path.join(RES, "summary.json"), "w"), indent=1, default=str)
    chart = dict(type="line", title="P6: exact match at D = min(4L, 32768) by test length (mean over seeds)",
                 x=dict(label="test length L", scale="log2", values=LS),
                 y=dict(label="exact match at D4", range=[0, 1]),
                 series=[dict(name=a, values=[arms[a]["exact4_by_L"].get(L) for L in LS]) for a in ARMS if a in arms])
    json.dump(chart, open(os.path.join(RES, "chart.json"), "w"), indent=1)
    for a in ARMS:
        if a in arms:
            g = arms[a]
            f = lambda k: (f"{g[k]['mean']:.3g}±{g[k]['ci95']:.2g}" if g.get(k) and g[k]["ci95"] is not None
                           else (f"{g[k]['mean']:.3g}" if g.get(k) else "-"))
            print(f"{a:4s} n={g['n']} log2L={f('log2_L_reached')} cap_it={f('cap_it')} log2L*4={f('log2_Lstar4')} "
                  f"co_nats={f('co_nats')} ret={f('retention_mean')} exits/1k={f('exits_per_1000')} "
                  f"single={f('single_log2_Lstar4')}")
    for k, c in contrasts.items():
        print(k, {kk: (round(v["mean"], 4), [round(x, 4) for x in v["boot95"]]) for kk, v in c.items() if kk != "seeds"})
    print("population (a)-(b) log2 L*4:", {a: round(v["mean"], 3) for a, v in pop.items()})


if __name__ == "__main__":
    main()
