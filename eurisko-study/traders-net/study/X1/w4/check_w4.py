"""X1 worker 4: file checks + seed-4 reproduction of M8, M9, K1.
Usage: PYTHONPATH=/home/agent/workspaces/eurisko-study/wealth-net/.pylib python3 check_w4.py
Writes repro.csv (one row per seed-4 comparison) and check_w4.json (file checks + repro) next to this file.
Criterion (PREREG.md): x4 within 2 sd iff |x4 - m| <= 2 s (m, s = mean, sample sd of seeds 1-3); when s < 1e-4
the test is also reported with s floored at 1e-3."""
import csv, glob, hashlib, importlib.util, json, os, sys
from datetime import datetime, timezone
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.abspath(os.path.join(HERE, "..", ".."))          # study/
COPY = "/tmp/traders-net/X1w4/study"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
rows, filechecks = [], []


def jl(p): return json.load(open(p))


def fc(name, ok, got, want, src, kind="number"):
    filechecks.append(dict(check=name, kind=kind, ok=bool(ok), got=got, want=want, source=os.path.relpath(src, ORIG) if src.startswith(ORIG) else src))


def rep(ws, name, v123, x4, src123, src4, claim=""):
    v = [float(x) for x in v123]
    m, s = float(np.mean(v)), float(np.std(v, ddof=1))
    r = dict(ws=ws, quantity=name, s1=v[0], s2=v[1], s3=v[2], mean=m, sd=s, seed4=None, z=None,
             within_2sd=None, within_2sd_floor=None, claim=claim, src_seeds123=src123, src_seed4=src4)
    if x4 is not None and np.isfinite(x4):
        x4 = float(x4); r["seed4"] = x4
        r["z"] = (x4 - m) / s if s > 0 else float("inf")
        r["within_2sd"] = abs(x4 - m) <= 2 * s
        if s < 1e-4:
            r["within_2sd_floor"] = abs(x4 - m) <= 2 * 1e-3
    rows.append(r)
    return r


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def maybe(p):
    return p if os.path.exists(p) else None


# ------------------------------------------------------------------ M8
S, C = f"{ORIG}/M8", f"{COPY}/M8"
# file check the claim check.py skips (commented out after '#'): random bce_all_nolfsr 0.3089 +- 0.0195
for r in csv.DictReader(open(f"{S}/w2/results/summary.csv")):
    if r["arm"] in ("random", "absag"):
        want = {"random": (0.3089, 0.0195), "absag": (0.3106, 0.0183)}[r["arm"]]
        got = (float(r["bce_all_nolfsr_mean"]), float(r["bce_all_nolfsr_sd"]))   # file keeps 5 dp; quote is 4 dp half-up
        fc(f"M8 w2 {r['arm']} bce_all_nolfsr (M8.md table)", all(abs(g - q) <= 5e-5 + 1e-9 for g, q in zip(got, want)), got, want, f"{S}/w2/results/summary.csv")
cc = open(f"{S}/check.py").read()
live = [l for l in cc.splitlines() if 'eq("random_nl"' in l and ("#" not in l or l.index("#") > l.index('eq("random_nl"'))]
fc("M8 check.py random_nl check is live code (skeptic: it sits after a # on the absag_nl line)", bool(live),
   [l.strip() for l in cc.splitlines() if "random_nl" in l], "a live eq(...) call", f"{S}/check.py", kind="skeptic")

w1 = lambda a, s: f"{S}/w1/results/run-{a}-s{s}.json"
w1c = lambda a: f"{C}/w1/results/run-{a}-s4.json"
have1 = all(os.path.exists(w1c(a)) for a in ("none", "pos1", "neg1", "shuf1"))
Pn = lambda p: jl(p)["prequential_bce_no_lfsr16"]
for a, claim in (("shuf1", "sign of income is not a gate signal (shuf1 ~ pos1)"), ("neg1", "flipping sign costs little (>=0.005 in 2/3)"),
                 ("none", "income gating hurts vs no gate (none < pos1 3/3)")):
    v = [Pn(w1(a, s)) - Pn(w1("pos1", s)) for s in (1, 2, 3)]
    x4 = Pn(w1c(a)) - Pn(w1c("pos1")) if have1 else None
    rep("M8", f"w1 nolfsr {a} - pos1", v, x4, "M8/w1/results/run-*-s{1,2,3}.json", w1c(a), claim)
for a in ("none", "pos1", "neg1", "shuf1"):
    rep("M8", f"w1 prequential_bce {a}", [jl(w1(a, s))["prequential_bce"] for s in (1, 2, 3)],
        jl(w1c(a))["prequential_bce"] if os.path.exists(w1c(a)) else None, "M8/w1/results", w1c(a))
for k in ("neg_share_top10", "neg_share_all", "spearman_income", "spearman_absag"):
    rep("M8", f"w1 census none {k}", [jl(w1("none", s))["census"][k] for s in (1, 2, 3)],
        jl(w1c("none"))["census"][k] if os.path.exists(w1c("none")) else None, "M8/w1/results", w1c("none"))
if os.path.exists(w1c("none")):
    c4 = jl(w1c("none"))["census"]
    fc("M8 seed4 census rule3 (top10 neg share < all)", c4["neg_share_top10"] < c4["neg_share_all"],
       [c4["neg_share_top10"], c4["neg_share_all"]], "top10 < all", w1c("none"))

w2 = lambda a, s: jl(f"{S}/w2/results/run_{a}_s{s}.json")
w2c = lambda a: maybe(f"{C}/w2/results/run_{a}_s4.json")
fam = {}
for s in (1, 2, 3):
    for r in csv.DictReader(open(f"{S}/w2/results/family_s{s}.csv")): fam[(r["arm"], s)] = r
f4 = maybe(f"{C}/w2/results/family_s4.csv")
if f4:
    for r in csv.DictReader(open(f4)): fam[(r["arm"], 4)] = r
for k in ("bce_postshift_switched", "bce_all"):
    v = [w2("absag", s)[k] - w2("random", s)[k] for s in (1, 2, 3)]
    x4 = jl(w2c("absag"))[k] - jl(w2c("random"))[k] if w2c("absag") and w2c("random") else None
    rep("M8", f"w2 absag - random {k}", v, x4, "M8/w2/results/run_*_s{1,2,3}.json", f"{C}/w2/results/run_*_s4.json",
        "no gain over random reinit in 3/3 seeds (stop)" if k == "bce_postshift_switched" else "")
    for a in ("absag", "random"):
        rep("M8", f"w2 {a} {k}", [w2(a, s)[k] for s in (1, 2, 3)], jl(w2c(a))[k] if w2c(a) else None,
            "M8/w2/results", f"{C}/w2/results/run_{a}_s4.json")
for a in ("absag", "random"):
    rep("M8", f"w2 {a} bce_all_nolfsr", [float(fam[(a, s)]["bce_all_nolfsr"]) for s in (1, 2, 3)],
        float(fam[(a, 4)]["bce_all_nolfsr"]) if (a, 4) in fam else None, "M8/w2/results/family_s{1,2,3}.csv", f4 or "")
for k in ("bce_all_nolfsr", "postshift_switched_nolfsr"):
    rep("M8", f"w2 absag - random {k}", [float(fam[("absag", s)][k]) - float(fam[("random", s)][k]) for s in (1, 2, 3)],
        float(fam[("absag", 4)][k]) - float(fam[("random", 4)][k]) if ("absag", 4) in fam and ("random", 4) in fam else None,
        "M8/w2/results/family_s{1,2,3}.csv", f4 or "", "absag's whole-run lead is one family (lfsr16)" if k == "bce_all_nolfsr" else "")
if ("absag", 4) in fam and ("random", 4) in fam:
    from scipy.stats import fisher_exact
    L = jl(f"{S}/w2/results/decision.json")["lfsr16_learnt_by_arm"]
    la = sum(L["absag"]) + int(fam[("absag", 4)]["lfsr16_learnt"]); lr = sum(L["random"]) + int(fam[("random", 4)]["lfsr16_learnt"])
    fc("M8 exploratory: absag vs random lfsr16 learnt, seeds 1-4, one-sided Fisher p", True,
       dict(absag=f"{la}/4", random=f"{lr}/4", p=round(fisher_exact([[la, 4 - la], [lr, 4 - lr]], alternative="greater")[1], 3)),
       "seeds 1-3: absag 2/3, random 0/3, p=0.20", f4, kind="exploratory")
    fc("M8 seed4 lfsr16_learnt absag/random (exploratory)", True, [fam[("absag", 4)]["lfsr16_learnt"], fam[("random", 4)]["lfsr16_learnt"]], "-", f4, kind="exploratory")

# ------------------------------------------------------------------ M9
S, C = f"{ORIG}/M9", f"{COPY}/M9"
ver = jl(f"{S}/w3/results/VERDICT.json")
fc("M9 VERDICT trend=false x4, law_present false", all(not v["trend"] for v in ver["series"].values()) and not ver["law_present_B_FT"] and not ver["law_present_A_FT"],
   {k: v["trend"] for k, v in ver["series"].items()}, "all false", f"{S}/w3/results/VERDICT.json")
# byte-identical reruns of the M9 analyses done in /tmp/traders-net/X1w4/m9check (fit_lottery SOUP path made absolute)
for rel in ("w1/lottery_fit.csv", "w1/lottery_fit.json", "w1/predictions.csv", "w1/soup_runs.csv", "b/final_brier_summary.csv",
            "b/final_brier_raw.csv", "b/compute_to_brier_raw.csv", "b/compute_to_brier_summary.csv"):
    p = f"/tmp/traders-net/X1w4/m9check/study/M9/{rel}"
    if os.path.exists(p): fc(f"M9 rerun byte-identical {rel}", sha(p) == sha(f"{S}/{rel}"), sha(p)[:12], sha(f"{S}/{rel}")[:12], f"{S}/{rel}")
p = "/tmp/traders-net/X1w4/m9null/lottery_fit.json"
if os.path.exists(p):
    a, b = jl(p), jl(f"{S}/w1/lottery_fit.json")
    na = {r["family"]: r for r in csv.DictReader(open("/tmp/traders-net/X1w4/m9null/lottery_fit.csv"))}
    nb = {r["family"]: r for r in csv.DictReader(open(f"{S}/w1/lottery_fit.csv"))}
    fam4 = ["periodic", "prime", "and", "or"]
    fc("M9(a) shuffled-null p<=0.001 with fresh RNG seed 4 (fit_lottery default_rng(4))", all(float(na[k]["shuffle_p"]) <= 0.001 for k in fam4),
       {k: [na[k]["shuffle_p"], na[k]["shuffle_r2_p95"]] for k in fam4}, {k: [nb[k]["shuffle_p"], nb[k]["shuffle_r2_p95"]] for k in fam4},
       "/tmp/traders-net/X1w4/m9null/lottery_fit.csv (copy: X1/w4/m9a_null_seed4.csv)")

EXF = ["periodic", "ca110", "prime", "and", "or"]
exl = lambda d: float(np.mean([d["score"]["fam"][f]["bce"] for f in EXF]))
m9 = lambda arm, K, hw, s: jl(f"{S}/w3/results/vm/{arm}-K{K}-hw{hw}-s{s}.json")
def m9c(arm, K, hw):
    for d in (f"{C}/w3/results", f"{C}/w3/results/vm"):
        p = f"{d}/{arm}-K{K}-hw{hw}-s4.json"
        if os.path.exists(p): return jl(p)
    return None
for ser, arm, (k1, h1), (k64, h64), claim in (("B-FT", "B", (1, 128), (64, 2), "flat excl. LFSR16"),
                                              ("A-FT", "A", (1, 128), (64, 2), "thin traders cost +0.0068 (3/3)"),
                                              ("B-FS", "B", (1, 8), (64, 8), "no trend")):
    d1, d64 = m9c(arm, k1, h1), m9c(arm, k64, h64)
    for met, f in (("learn_bce_ex_lfsr", exl), ("learn_bce", lambda d: d["score"]["learn_bce"]), ("humility", lambda d: d["score"]["humility"])):
        v = [f(m9(arm, k64, h64, s)) - f(m9(arm, k1, h1, s)) for s in (1, 2, 3)]
        rep("M9", f"{ser} K64-K1 {met}", v, f(d64) - f(d1) if d1 and d64 else None, "M9/w3/results/vm", f"{C}/w3/results", claim if met == "learn_bce_ex_lfsr" else "")
    rep("M9", f"{ser} K64 eff_traders_per_q", [m9(arm, k64, h64, s)["eff_traders_per_q"] for s in (1, 2, 3)],
        d64["eff_traders_per_q"] if d64 else None, "M9/w3/results/vm", f"{C}/w3/results")

# ------------------------------------------------------------------ K1
S, C = f"{ORIG}/K1", f"{COPY}/K1"
for nic in ("on", "off"):
    o = lambda s: jl(f"{S}/runs/w1-n10k-q64-{nic}-s{s}/summary.json")
    c4 = maybe(f"{C}/runs/w1-n10k-q64-{nic}-s4/summary.json"); c4 = jl(c4) if c4 else None
    for k in ("alive", "gini_traders", "gini_null_traders", "top_species_share", "top_clade_share", "eff_clades"):
        rep("K1", f"w1 10k Q64-{nic} {k}", [o(s)["final"][k] for s in (1, 2, 3)], c4["final"][k] if c4 else None,
            f"K1/runs/w1-n10k-q64-{nic}-s*/summary.json", f"{C}/runs/w1-n10k-q64-{nic}-s4")
    g = lambda d: d["final"]["gini_traders"] - d["final"]["gini_null_traders"]
    rep("K1", f"w1 10k Q64-{nic} gini - null", [g(o(s)) for s in (1, 2, 3)], g(c4) if c4 else None, "K1/runs", f"{C}/runs",
        "Gini below beliefs-shuffled null" if nic == "on" else "")
    rg = lambda d: d["tail_brier_all"] - d["tail_brier_oracle_all"]
    rep("K1", f"w1 10k Q64-{nic} tail Brier - oracle", [rg(o(s)) for s in (1, 2, 3)], rg(c4) if c4 else None, "K1/runs", f"{C}/runs",
        "market Brier ~ oracle" if nic == "on" else "")

A2 = load_mod("k1w2an", f"{C}/w2/analyze.py"); A2.RUNS = f"{C}/runs"
sr = {(r["arm"], int(r["seed"])): r for r in csv.DictReader(open(f"{S}/w2/results/shift_runs.csv"))}
cr = {(r["run"].rsplit("-s", 1)[0], int(r["seed"])): r for r in csv.DictReader(open(f"{S}/w2/results/csweep_runs.csv"))}
for arm in ("S-on10k", "S-off10k"):
    x = None
    if os.path.exists(f"{C}/runs/w2-{arm}-s4/w2extra.json"):
        c, _ = A2.load(f"w2-{arm}-s4"); x = dict(**A2.block(c, 20000), **A2.shift_metrics(c))
    for k in ("recovery_steps", "excess_peak", "regret", "gini_gap", "newborn_share_trk_end"):
        if k in sr[(arm, 1)] and sr[(arm, 1)][k] not in ("", "nan"):
            rep("K1", f"w2 {arm} {k}", [float(sr[(arm, s)][k]) for s in (1, 2, 3)], x.get(k) if x else None,
                "K1/w2/results/shift_runs.csv", f"{C}/runs/w2-{arm}-s4")
for arm in ("C0-10k", "C0.1-10k", "S-off10k"):
    x = None
    if os.path.exists(f"{C}/runs/w2-{arm}-s4/w2extra.json"):
        c, _ = A2.load(f"w2-{arm}-s4"); x = A2.block(c, 10000)
    for k in ("regret", "gini_gap", "top_clade_share", "alive"):
        key = f"w2-{arm}"
        if (key, 1) in cr:
            rep("K1", f"w2 csweep {arm} {k}@10k", [float(cr[(key, s)][k]) for s in (1, 2, 3)], x.get(k) if x else None,
                "K1/w2/results/csweep_runs.csv", f"{C}/runs/w2-{arm}-s4")

A3 = load_mod("k1w3an", f"{C}/w3/analyze.py")
for arm in ("crp-within", "bh-within"):
    rule, topo = arm.split("-")
    o = [A3.run_stats(f"{S}/runs/w3-{rule}-{topo}-N10000-s{s}/metrics.csv") for s in (1, 2, 3)]
    p4 = maybe(f"{C}/runs/w3-{rule}-{topo}-N10000-s4/metrics.csv")
    p4 = p4 if os.path.exists(f"/tmp/traders-net/X1w4/logs/k1w3{rule}.done") else None   # metrics.csv grows during the run
    x = A3.run_stats(p4) if p4 else None
    for k in ("premium_agg", "brier2", "gini_T", "fund_alive_T"):
        rep("K1", f"w3 N10000 {arm} {k}", [d[k] for d in o], x[k] if x else None, f"K1/runs/w3-{rule}-{topo}-N10000-s*",
            p4 or "", "funds of funds lose share (premium << 0)" if (k == "premium_agg" and rule == "crp") else "")
summ = {r["arm"]: r for r in csv.DictReader(open(f"{S}/w3/out/summary_N10000.csv"))}
for arm in ("crp-within", "bh-within"):
    mine = [r for r in rows if r["quantity"] == f"w3 N10000 {arm} premium_agg"][0]
    fc(f"K1 w3 {arm} premium mean re-derived from run metrics", abs(mine["mean"] - float(summ[arm]["premium_agg"])) < 1e-4,
       round(mine["mean"], 5), round(float(summ[arm]["premium_agg"]), 5), f"{S}/w3/out/summary_N10000.csv")

# seed-4 K1 runs: money conservation and monoculture flags (K1.md: max residual 4.1e-12 over 141 runs; 4/49 arms mono)
res, mono = [], {}
for d in sorted(glob.glob(f"{C}/runs/*-s4")):
    A = A3.load(f"{d}/metrics.csv")[1]
    res.append(float(np.nanmax(np.abs(np.concatenate([A("cons_resid_ledger"), A("cons_resid_settle")])))))
    sj = maybe(f"{d}/summary.json")
    if sj: mono[os.path.basename(d)] = bool(jl(sj).get("monoculture"))
fc("K1 seed-4 runs conserve money (max |ledger/settle residual| < 1e-9)", max(res) < 1e-9, max(res), "< 1e-9 (K1.md: 4.1e-12)", f"{C}/runs")
fc("K1 seed-4 runs: no clade monoculture in the rerun arms", not any(mono.values()), mono, "all False", f"{C}/runs")
for a in ("absag", "random"):
    rep("M8", f"w2 {a} births_survived", [jl(f"{ORIG}/M8/w2/results/run_{a}_s{s}.json")["births_survived"] for s in (1, 2, 3)],
        jl(f"{COPY}/M8/w2/results/run_{a}_s4.json")["births_survived"] if os.path.exists(f"{COPY}/M8/w2/results/run_{a}_s4.json") else None,
        "M8/w2/results", f"{COPY}/M8/w2/results/run_{a}_s4.json")

# ------------------------------------------------------------------ write
keys = list(rows[0].keys())
with open(os.path.join(HERE, "repro.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
done = [r for r in rows if r["seed4"] is not None]
out = dict(checked_at=NOW, criterion="|x4-m|<=2s (s floored at 1e-3 when s<1e-4, reported separately)",
           n_quantities=len(rows), n_with_seed4=len(done), n_within_2sd=sum(bool(r["within_2sd"]) for r in done),
           outside_2sd=[{k: r[k] for k in ("ws", "quantity", "mean", "sd", "seed4", "z", "within_2sd_floor")} for r in done if not r["within_2sd"]],
           filechecks=filechecks, filechecks_ok=all(c["ok"] for c in filechecks if c["kind"] == "number"),
           skeptic_flags=[c["check"] for c in filechecks if c["kind"] == "skeptic" and not c["ok"]])
json.dump(out, open(os.path.join(HERE, "check_w4.json"), "w"), indent=1, default=float)
print(json.dumps({k: out[k] for k in ("checked_at", "n_quantities", "n_with_seed4", "n_within_2sd", "filechecks_ok")}))
for c in filechecks:
    if not c["ok"]: print("FILECHECK FAIL" if c["kind"] == "number" else "SKEPTIC FLAG", c["check"])
for r in out["outside_2sd"]: print("OUTSIDE", r)
