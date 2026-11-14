"""P13 analysis (PREREG spec success/falsify + A6/A7): reads gpu/main_plan.json, the synced evals in results/vm/*/out and the
training logs; writes results/results.json and results/chart.json.

Eval names (phase 4): ev59_<run>_<jopt> (v9, v33, m59 at 12n; J option none also gives the gates), evT_<run> (e19, e33, e59
at 12n, chosen J), ev99_<run> (e99; mult 20 for the 20n-extension models), ev201_<run> (e201 stretch). M-ref: run "mref".
02:35Z addendum (spend): every model is scored on the first LIMIT (100) mazes of e19, e33, e59 and e99 (the Trader2D
test evals cover only those; DT-recall-eq and M-ref cover all mazes, and their all-maze values are reported as *_all).
02:40Z addendum: with the ECON-F0 main run cut, ECON-F0 is the 1.0 L4-h F0 pilot (pil_econ_F0, flagged in results).
Usage: analyze.py [--partial]"""
import glob, itertools, json, math, os, sys
import numpy as np
from scipy import stats

D = os.path.dirname(os.path.abspath(__file__))
CAL = ["raw", "clip4", "clip8", "clip16"] + [f"tau{2 ** (k / 2):.3g}" for k in range(-2, 17)]
JMACS = {"cap": 0, "none": 1, "scale": 2}            # tie order: fewer MACs first (none and scale cost the same)
TEST = ["e19", "e33", "e59", "e99"]
NS = {"e19": 19, "e33": 33, "e59": 59, "e99": 99, "e201": 201, "m59": 59, "v33": 33, "v9": 9}
RNG = np.random.default_rng(20260929)
LIMIT = 100                  # first 100 mazes of each test set (02:35Z addendum)


def find(name):
    g = sorted(glob.glob(os.path.join(D, "results", "vm", "*", "out", name)))
    return g[-1] if g else None


class Ev:
    def __init__(self, name, lim=LIMIT):
        j, z = find(name + ".json"), find(name + ".npz")
        self.ok = bool(j and z)
        self.lim, self.name = lim, name
        if self.ok:
            self.S = json.load(open(j))["sets"]
            self.z = np.load(z)

    def a(self, s, key):
        """per-maze array (records x mazes [x ...]); test sets are cut to the first LIMIT mazes"""
        x = self.z[f"{s}/{key}"]
        return x[:, :self.lim] if (self.lim and s in TEST + ["e201"]) else x

    def nmaz(self, s):
        return self.a(s, "exact").shape[1]

    def has(self, s):
        return self.ok and s in self.S

    def idx(self, s, t):
        return list(self.z[f"{s}/t"]).index(t)

    def exact(self, s, which="t_eq"):          # per-maze exact (0/1)
        t = {"t_eq": self.S[s]["t_eq"], "12n": min(self.S[s]["T12"], self.S[s]["Tmax"])}[which]
        return self.a(s, "exact")[self.idx(s, t)].astype(np.float64)

    def ll(self, s, which="t_eq", c=0):
        t = {"t_eq": self.S[s]["t_eq"], "12n": min(self.S[s]["T12"], self.S[s]["Tmax"])}[which]
        return self.a(s, "ll")[self.idx(s, t), :, c].astype(np.float64)

    def curve(self, s, key="exact"):
        return self.z[f"{s}/t"].tolist(), self.a(s, key).astype(np.float64).mean(1).tolist()

    def best_over_t(self, s, Tmax=None):
        t = np.array(self.z[f"{s}/t"]); e = self.a(s, "exact").astype(np.float64).mean(1)
        k = t <= (Tmax or t.max())
        return float(e[k].max()), int(t[k][np.argmax(e[k])])

    def slope(self, s):
        """slope of log mean|b| against log t over [n, 12n] (mean over mazes of the per-maze mean |b|)"""
        n = self.S[s]["n"]; t = np.array(self.z[f"{s}/t"]); mb = self.a(s, "meanb").astype(np.float64).mean(1)
        k = (t >= n) & (t <= 12 * n) & (mb > 0)
        return float(np.polyfit(np.log(t[k]), np.log(mb[k]), 1)[0]) if k.sum() > 2 else None


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    if len(b) == 1:                       # single model (M-ref): one-sample t on a minus its mean
        if len(a) < 2:
            return {"diff": d, "lo": None, "hi": None, "df": 0}
        se = a.std(ddof=1) / math.sqrt(len(a)); df = len(a) - 1
    else:
        va, vb = a.var(ddof=1) / len(a) if len(a) > 1 else 0, b.var(ddof=1) / len(b) if len(b) > 1 else 0
        se = math.sqrt(va + vb)
        if se == 0:
            return {"diff": d, "lo": d, "hi": d, "df": None}
        df = (va + vb) ** 2 / ((va ** 2 / (len(a) - 1) if len(a) > 1 else 0) + (vb ** 2 / (len(b) - 1) if len(b) > 1 else 0) or 1e-300)
    if se == 0:
        return {"diff": d, "lo": d, "hi": d, "df": df}
    q = stats.t.ppf(0.975, df)
    return {"diff": d, "lo": d - q * se, "hi": d + q * se, "df": df}


def hboot(A, B, n=10000):
    """A, B: lists of per-seed per-maze arrays. Seeds with replacement within arm, then mazes within seed."""
    def draw(X):
        X = [np.asarray(x) for x in X]
        k = len(X); m = len(X[0])
        si = RNG.integers(0, k, (n, k))
        M = np.stack(X)                                   # (k, m)
        mi = RNG.integers(0, m, (n, k, m))
        return M[si[:, :, None], mi].mean(2).mean(1)      # (n,)
    d = draw(A) - draw(B)
    return {"lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}


def perm_p(a, b):
    a, b = list(a), list(b)
    if len(b) < 2 or len(a) < 2:
        return None
    pool = np.array(a + b); obs = abs(np.mean(a) - np.mean(b)); cnt = tot = 0
    for c in itertools.combinations(range(len(pool)), len(a)):
        m = np.zeros(len(pool), bool); m[list(c)] = True
        tot += 1; cnt += abs(pool[m].mean() - pool[~m].mean()) >= obs - 1e-12
    return cnt / tot


def activity(train_json, first=0.10):
    T = json.load(open(train_json))
    it = max(l["it"] for l in T["log"])
    dg = [l["diag"] for l in T["log"] if "diag" in l and l["it"] >= first * it and l["diag"].get("iters")]
    fc = float(np.mean([d["frac_capped_pix"] for d in dg])) if dg else None
    resp = (T.get("stats") or {}).get("respawns", 0)
    return {"frac_capped_after10": fc, "respawns": resp, "active": bool((fc or 0) >= 0.01 or resp >= 1),
            "iters": it + 1, "train_sec": T.get("train_sec"), "min_wealth": min(d["min_wealth"] for d in dg) if dg else None}


def main():
    plan = json.load(open(os.path.join(D, "gpu", "main_plan.json")))
    sel = json.load(open(os.path.join(D, "results", plan["selection_file"])))
    runs = plan["runs"] + [{"name": "mref", "arm": "M-ref", "seed": 0, "kind": "mref"}]
    if not any(r["arm"] == "ECON-F0" for r in runs) and sel.get("selected") != "F0":
        runs.append({"name": "pil_econ_F0", "arm": "ECON-F0", "seed": 100, "kind": "trader", "a": "econ", "fix": "F0",
                     "flag": "1.0 L4-h pilot (main run cut, 02:40Z addendum)"})
    M = {}
    for r in runs:
        n, kind = r["name"], r.get("kind", "trader")
        jo = {j: Ev(f"ev59_{n}_{j}") for j in (["none", "scale", "cap"] if kind == "trader" else ["none"])}
        if n == "pil_econ_F0":
            jo = {"none": Ev("ev_pil_econ_F0_x")}
        M[n] = dict(r, ev59=jo, evT=Ev(f"evT_{n}"), ev99=Ev(f"ev99_{n}"), ev201=Ev(f"ev201_{n}"))
        tj = find(f"{n}.json") if kind != "mref" else None
        if kind == "dtr":
            tj = find(f"{n}.json")
        if tj and kind == "trader" and r["arm"].startswith("ECON"):
            M[n]["activity"] = activity(tj)
    arms = {}
    for r in runs:
        arms.setdefault(r["arm"], []).append(r["name"])
    # J option per Trader2D arm on m59 at 12n (highest seed-mean exact; ties to fewer MACs)
    jsel = {}
    for arm, ns in arms.items():
        if M[ns[0]].get("kind", "trader") != "trader":
            jsel[arm] = "none"; continue
        sc = {}
        for j in ["none", "scale", "cap"]:
            v = [M[n]["ev59"][j].exact("m59", "12n").mean() for n in ns if j in M[n]["ev59"] and M[n]["ev59"][j].has("m59")]
            if len(v) == len(ns):
                sc[j] = float(np.mean(v))
        jsel[arm] = min(sc, key=lambda j: (-sc[j], JMACS[j])) if sc else None
        jsel[arm + "_scores"] = sc
    out = {"jopt": jsel, "selection": {k: sel[k] for k in ["selected", "econ_fix_active", "F2_recorded", "gate"] if k in sel},
           "models": {}, "arms": {}, "comparisons": {}, "tiers": {}}
    per_maze = {}          # (run, set, which) -> array
    for arm, ns in arms.items():
        for n in ns:
            m = M[n]; j = jsel.get(arm) or "none"
            e59 = m["ev59"].get(j) or m["ev59"].get("none")
            rec = {"arm": arm, "seed": m.get("seed"), "jopt": j}
            if m.get("flag"):
                rec["flag"] = m["flag"]
            ctrl = 0
            if e59 and e59.has("m59"):
                ll59 = e59.z["m59/ll"][e59.idx("m59", min(e59.S["m59"]["T12"], e59.S["m59"]["Tmax"]))].astype(np.float64).mean(0)
                ctrl = int(np.argmin(ll59)); rec["ctrl"] = CAL[ctrl]
                rec["m59_exact_12n"] = float(e59.exact("m59", "12n").mean())
            g = m["ev59"].get("none")
            if g and g.has("v9") and g.has("v33"):
                rec["v9"] = float(g.exact("v9", "12n").mean()); rec["v33"] = float(g.exact("v33", "12n").mean())
                rec["trained"] = rec["v9"] >= 0.95 and rec["v33"] >= 0.5
            if arm == "M-ref":
                gm = Ev("ev_mref_gate")
                if gm.ok:
                    rec["gate_m13"] = gm.S["m13"]["exact@12n"]; rec["gate_m59"] = gm.S["m59"]["exact@12n"]
                    rec["gate_pass"] = rec["gate_m13"] >= 1.0 and rec["gate_m59"] >= 0.99
            for src, sets in [(m["evT"], ["e19", "e33", "e59"]), (m["ev99"], ["e99"]), (m["ev201"], ["e201"])]:
                for s in sets:
                    if not src.has(s):
                        continue
                    S = src.S[s]
                    for w in ["t_eq", "12n"]:
                        per_maze[(n, s, w)] = src.exact(s, w)
                        rec[f"{s}_exact@{w}"] = float(per_maze[(n, s, w)].mean())
                        rec[f"{s}_ll_raw@{w}"] = float(src.ll(s, w, 0).mean())
                        rec[f"{s}_ll_ctrl@{w}"] = float(src.ll(s, w, ctrl).mean())
                        rec[f"{s}_ll_raw_median@{w}"] = float(np.median(src.ll(s, w, 0)))
                    rec[f"{s}_t_eq"] = S["t_eq"]
                    rec[f"{s}_best_over_t"] = src.best_over_t(s)
                    rec[f"{s}_best_over_t_12n"] = src.best_over_t(s, S["T12"])
                    rec[f"{s}_slope_logb"] = src.slope(s)
                    rec[f"{s}_cycle@t_eq"] = S["cycle@t_eq"]; rec[f"{s}_cycle@12n"] = S["cycle@12n"]
                    rec[f"{s}_ece@t_eq"] = S.get(f"ece@{S['t_eq']}"); rec[f"{s}_auroc@t_eq"] = S.get(f"auroc_err@{S['t_eq']}")
                    rec[f"{s}_maxb@12n"] = S["maxb@12n"]
                    rec[f"{s}_n_mazes"] = src.nmaz(s)
                    full = Ev(src.name, lim=0) if src.nmaz(s) < S["N"] else None
                    if full:                      # DT-recall-eq and M-ref: all-maze values alongside the paired 100
                        for w in ["t_eq", "12n"]:
                            rec[f"{s}_exact_all@{w}"] = float(full.exact(s, w).mean())
                            rec[f"{s}_ll_raw_all@{w}"] = float(full.ll(s, w, 0).mean())
                        rec[f"{s}_n_mazes_all"] = S["N"]
                        rec[f"{s}_best_over_t_all"] = full.best_over_t(s)
            if "activity" in m:
                rec["activity"] = m["activity"]
            out["models"][n] = rec
    # arm summaries
    for arm, ns in arms.items():
        a = {"seeds": ns}
        for s in TEST + ["e201"]:
            for w in ["t_eq", "12n"]:
                v = [out["models"][n].get(f"{s}_exact@{w}") for n in ns]
                v = [x for x in v if x is not None]
                if v:
                    a[f"{s}_exact@{w}"] = {"mean": float(np.mean(v)), "per_seed": v,
                                           "ci": [welch(v, [0])["lo"], welch(v, [0])["hi"]] if len(v) > 1 else None}
        a["trained"] = sum(bool(out["models"][n].get("trained")) for n in ns)
        out["arms"][arm] = a

    def seedmeans(arm, s="e99", w="t_eq", only_trained=False):
        ns = [n for n in arms.get(arm, []) if (n, s, w) in per_maze and (not only_trained or out["models"][n].get("trained"))]
        return ns, [per_maze[(n, s, w)].mean() for n in ns], [per_maze[(n, s, w)] for n in ns]

    def compare(A, B, s="e99", w="t_eq", only_trained_B=False):
        na, a, pa = seedmeans(A, s, w); nb, b, pb = seedmeans(B, s, w, only_trained_B)
        if not a or not b:
            return None
        r = {"A": A, "B": B, "set": s, "at": w, "n_A": len(a), "n_B": len(b), "mean_A": float(np.mean(a)), "mean_B": float(np.mean(b))}
        r["welch"] = welch(a, b); r["hboot"] = hboot(pa, pb); r["perm_p"] = perm_p(a, b)
        r["ci_above_0"] = bool(r["welch"]["lo"] is not None and r["welch"]["lo"] > 0)
        if len(a) == 1 or len(b) == 1:
            r["flag"] = "one arm has a single model: interval is that value against the other arm's seed interval (not the pre-registered 2-arm Welch test)"
        return r

    act_arm = "ECON-fix" if sel.get("econ_fix_active") else ("ECON-active" if "ECON-active" in arms else None)
    pairs = [("ECON-fix", "DTR-eq"), ("ECON-fix", "ECON-nocap"), ("ECON-fix", "BPTT-dscov"), ("ECON-fix", "BPTT-progcov"),
             ("ECON-fix", "M-ref"), ("ECON-active", "ECON-nocap"), ("ECON-nocap", "BPTT-dscov")]
    C = out["comparisons"]
    for A, B in pairs:
        for s in TEST:
            for w in ["t_eq", "12n"]:
                c = compare(A, B, s, w)
                if c:
                    C[f"{A} - {B} | {s} @ {w}"] = c
    c = compare("ECON-fix", "DTR-eq", "e99", "t_eq", only_trained_B=True)
    if c:
        C["ECON-fix - DTR-eq(trained seeds) | e99 @ t_eq"] = c
    # tiers
    T = out["tiers"]; mods = out["models"]
    fix = sel.get("selected")
    ef = [mods[n] for n in arms.get("ECON-fix", [])]
    f0 = [mods[n] for n in arms.get("ECON-F0", [])] or ([mods[n] for n in arms.get("ECON-fix", [])] if fix == "F0" else [])
    mean = lambda L, k: float(np.mean([x[k] for x in L if x.get(k) is not None])) if any(x.get(k) is not None for x in L) else None
    if ef and f0:
        ll_fix = mean(ef, "e99_ll_ctrl@t_eq" if fix == "F0" else "e99_ll_raw@t_eq")
        s1 = {"fix": fix, "ECONfix_ll99": ll_fix, "F0_raw_ll99": mean(f0, "e99_ll_raw@t_eq"), "F0_ctrl_ll99": mean(f0, "e99_ll_ctrl@t_eq"),
              "ECONfix_exact99": mean(ef, "e99_exact@t_eq"), "F0_exact99": mean(f0, "e99_exact@t_eq")}
        if None not in s1.values():
            a1 = s1["ECONfix_ll99"] <= 0.1 * s1["F0_raw_ll99"]
            a2 = s1["ECONfix_ll99"] < s1["F0_ctrl_ll99"]
            a3 = s1["ECONfix_exact99"] >= s1["F0_exact99"] - 0.02
            s1["legs"] = {"<=0.1x F0 raw": a1, "< F0 post-hoc": a2, "exact within 2pp": a3}
            s1["holds"] = (a2 and a3) if fix in ("LB", "FL") else (a1 and a2 and a3)
            sl = [x["e99_slope_logb"] for x in ef if x.get("e99_slope_logb") is not None]
            if len(sl) > 1:
                ci = welch(sl, [0]); s1["slope_mean"] = float(np.mean(sl)); s1["slope_hi"] = ci["hi"]
                s1["label"] = "fixed" if ci["hi"] <= 0.1 else "slowed"
            s1["F2_holds"] = bool(sel.get("F2_recorded")) or not (a2 and a3)
            s1["F0_source"] = [x.get("flag", "main run") for x in f0]
        T["S1"] = s1
    mref = mods.get("mref", {})
    c_dtr = C.get("ECON-fix - DTR-eq | e99 @ t_eq"); c_dtr_t = C.get("ECON-fix - DTR-eq(trained seeds) | e99 @ t_eq")
    n_dtr_trained = out["arms"].get("DTR-eq", {}).get("trained", 0)
    if c_dtr and mref.get("e99_exact@t_eq") is not None:
        ef99 = c_dtr["mean_A"]
        s2 = {"ci_all": [c_dtr["welch"]["lo"], c_dtr["welch"]["hi"]], "dtr_trained": n_dtr_trained,
              "ci_trained": [c_dtr_t["welch"]["lo"], c_dtr_t["welch"]["hi"]] if c_dtr_t and n_dtr_trained >= 2 else None,
              "ECONfix_minus_Mref": ef99 - mref["e99_exact@t_eq"],
              "ll_raw_ECONfix": mean(ef, "e99_ll_raw@t_eq"), "ll_raw_DTR": mean([mods[n] for n in arms.get("DTR-eq", [])], "e99_ll_raw@t_eq")}
        leg_mref = s2["ECONfix_minus_Mref"] >= -0.02
        leg_ll = s2["ll_raw_ECONfix"] is not None and s2["ll_raw_DTR"] is not None and s2["ll_raw_ECONfix"] <= s2["ll_raw_DTR"]
        if n_dtr_trained >= 2:
            leg_ci = c_dtr["ci_above_0"] and (c_dtr_t is None or c_dtr_t["ci_above_0"])
            s2["holds"] = bool(leg_ci and leg_mref and leg_ll)
        else:
            s2["note"] = f"Bansal's recipe does not train reliably at this compute ({n_dtr_trained} of {len(arms.get('DTR-eq', []))}); S2 judged against M-ref only"
            s2["holds"] = bool(leg_mref and leg_ll)
        s2["legs"] = {"ci>0": c_dtr["ci_above_0"], ">= M-ref - 2pp": leg_mref, "raw ll <= DTR": leg_ll}
        T["S2"] = s2
        T["F1_holds"] = (not c_dtr["ci_above_0"]) or s2["ECONfix_minus_Mref"] < -0.02
    if act_arm:
        acts = [mods[n].get("activity", {}).get("active") for n in arms.get(act_arm, [])]
        c3 = C.get(f"{act_arm} - ECON-nocap | e99 @ t_eq")
        s3 = {"active_arm": act_arm, "active_per_seed": acts}
        if acts and not any(acts):
            s3["result"] = "economics inactive, untestable"; s3["holds"] = False
        elif c3:
            s3["ci"] = [c3["welch"]["lo"], c3["welch"]["hi"]]
            s3["holds"] = bool(all(acts) and c3["ci_above_0"])
            T["F3_holds"] = bool(any(acts) and not c3["ci_above_0"])
        T["S3"] = s3
    else:
        T["S3"] = {"result": "economics could not be made to bind (ladder found no active candidate)", "holds": False}
        T["F3_note"] = "the economy reduces to local losses on mazes"
    # 06:06Z addendum: the main ECON-fix runs' own activity (S3's per-seed rule), reported whatever the pilot said
    acts_fix = {n: mods[n].get("activity", {}).get("active") for n in arms.get("ECON-fix", [])}
    T["S3"]["ECON-fix_main_active_per_seed"] = acts_fix
    if acts_fix and not any(acts_fix.values()):
        T["S3"]["ECON-fix_main"] = "economics inactive, untestable (cap never binds, no respawn, in every seed)"
    c3b = C.get("ECON-nocap - BPTT-dscov | e99 @ t_eq")
    if c3b:
        T["S3b"] = {"ci": [c3b["welch"]["lo"], c3b["welch"]["hi"]], "holds": c3b["ci_above_0"]}
        T["F4_holds"] = bool(c3b["mean_B"] >= c3b["mean_A"])
    c4 = C.get("ECON-fix - M-ref | e99 @ t_eq")
    if c4 and mref.get("e99_best_over_t"):
        ef12 = mean(ef, "e99_exact@12n")
        T["S4"] = {"ci": [c4["welch"]["lo"], c4["welch"]["hi"]], "ECONfix_12n": ef12, "Mref_best_over_t_20n": mref["e99_best_over_t"],
                   "holds": bool(c4["ci_above_0"] and ef12 > mref["e99_best_over_t"][0])}
    T["Mref_gate"] = {k: mref.get(k) for k in ["gate_m13", "gate_m59", "gate_pass"]}
    json.dump(out, open(os.path.join(D, "results", "results.json"), "w"), indent=1, default=float)
    # chart: accuracy against n at t_eq (seed means, Welch CI) and the e99 curves
    chart = {"title": "P13: exact accuracy at equal test FLOPs against maze size", "x": [19, 33, 59, 99], "series": [], "curves99": []}
    for arm, a in out["arms"].items():
        chart["series"].append({"arm": arm, "y": [a.get(f"{s}_exact@t_eq", {}).get("mean") for s in TEST],
                                "ci": [a.get(f"{s}_exact@t_eq", {}).get("ci") for s in TEST], "n_seeds": len(arms[arm])})
        cs = [M[n]["ev99"].curve("e99") for n in arms[arm] if M[n]["ev99"].has("e99")]
        if cs:
            L = min(len(c[0]) for c in cs)
            chart["curves99"].append({"arm": arm, "t": cs[0][0][:L], "exact": np.mean([c[1][:L] for c in cs], 0).round(4).tolist()})
    json.dump(chart, open(os.path.join(D, "results", "chart.json"), "w"), indent=1)
    print(json.dumps({"jopt": jsel, "tiers": T}, indent=1, default=float))


if __name__ == "__main__":
    main()
