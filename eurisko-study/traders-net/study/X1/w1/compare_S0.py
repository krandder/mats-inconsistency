"""X1 worker 1: compare S0 headline claims (committed seeds) with a fresh seed.

Usage: python3 compare_S0.py COMMITTED_S0_DIR FRESH_S0_DIR --out OUT.json
Committed per-seed values are re-read from the per-seed files (and the quoted summary is
re-checked against them); fresh values come from w1 seed 4, w2 audit seed 4, w3 baseline seed 3,
w4 null_calib seed 3 in the clean copy. Rule (X1/w1/PREREG.md): |fresh-mean| <= 2 sd (sample sd);
sd == 0 -> 10% relative band (exact equality when mean == 0); 'count' rows are reported only.
"""
import csv, json, os, sys, statistics as st

C, F = sys.argv[1], sys.argv[2]
OUT = sys.argv[sys.argv.index("--out") + 1]
SELFTEST = "--selftest" in sys.argv  # fresh := first committed seed, to check the extractors
OVR = int(sys.argv[sys.argv.index("--fresh") + 1]) if "--fresh" in sys.argv else None  # tie-break: use this fresh seed for every row
J = lambda r, p: json.load(open(os.path.join(r, p)))


def w1(arm, key, fam, seeds):
    def g(R):
        out = []
        for s in seeds:
            d = J(R, f"w1/results/{arm}-s{s}.json"); i = d["families"].index(fam)
            if key == "gap": out.append(d["bce"][i] - d["in_bce"][i])
            elif key == "overlap": out.append(d["overlap"][fam])
            else: out.append(d[key][i])
        return out
    return g


def aud(key, seeds, sub=None):
    def g(R):
        out = []
        for s in seeds:
            v = J(R, f"results/w2/audit-s{s}.json")[key]
            out.append(v[sub] if sub else v)
        return out
    return g


def base(model, key, seeds):
    return lambda R: [J(R, f"w3/baseline-s{s}.json")[model][key] for s in seeds]


def null(arm, seeds):
    def g(R):
        rows = J(R, "w4/null_calib.json")["per_seed"]
        for s in seeds:  # tie-break seeds are written to null_calib_s<seed>.json by null_calib_s<seed>.py
            if os.path.exists(os.path.join(R, f"w4/null_calib_s{s}.json")): rows = rows + J(R, f"w4/null_calib_s{s}.json")["per_seed"]
        return [r["z_mean"] for r in rows if r["arm"] == arm and r["seed"] in seeds]
    return g


S123, S012 = (1, 2, 3), (0, 1, 2)
# id, claim as quoted in SUBSTRATE.md, kind, getter(root, seeds), committed seeds, fresh seeds
CL = [
 ("S0-1a", "pi prequential P4 BCE 0.6933 ± 0.0004", "stat", lambda s: w1("P4", "bce", "pi", s), S123, (4,)),
 ("S0-1b", "pi prequential P16 BCE 0.6957 ± 0.0004", "stat", lambda s: w1("P16", "bce", "pi", s), S123, (4,)),
 ("S0-1c", "pi multi-epoch held-out 0.830 ± 0.021", "stat", lambda s: w1("ME", "bce", "pi", s), S123, (4,)),
 ("S0-1d", "pi multi-epoch in-sample 0.562 ± 0.022", "stat", lambda s: w1("ME", "in_bce", "pi", s), S123, (4,)),
 ("S0-1e", "pi memorisation gap 0.267 ± 0.043", "stat", lambda s: w1("ME", "gap", "pi", s), S123, (4,)),
 ("S0-1f", "CA30 gap 0.004 ± 0.004 (prereg >0.05 not met)", "stat", lambda s: w1("ME", "gap", "ca30", s), S123, (4,)),
 ("S0-1g", "CA30 P4 BCE 0.6945 ± 0.0003", "stat", lambda s: w1("P4", "bce", "ca30", s), S123, (4,)),
 ("S0-1h", "LFSR16 P4 learnt in 1/3 seeds (BCE<0.3)", "count", lambda s: w1("P4", "bce", "lfsr16", s), S123, (4,)),
 ("S0-1i", "LFSR16 P16 learnt in 2/3 seeds (BCE<0.3)", "count", lambda s: w1("P16", "bce", "lfsr16", s), S123, (4,)),
 ("S0-1j", "leak: held-out LFSR16 overlap49 1.00", "stat", lambda s: w1("ME", "overlap", "lfsr16", s), S123, (4,)),
 ("S0-1k", "leak: held-out prime overlap49 1.00", "stat", lambda s: w1("ME", "overlap", "prime", s), S123, (4,)),
 ("S0-1l", "logistic baseline pi P4 0.714 ± 0.0001", "stat", lambda s: w1("P4", "bce_log", "pi", s), S123, (4,)),
 ("S0-2a", "pi shifted-copy share 1.0", "stat", lambda s: aud("A0_shift_copy_pi", s), S123, (4,)),
 ("S0-2b", "prng shifted-copy share 0.0", "stat", lambda s: aud("A1_shift_copy_prng", s), S123, (4,)),
 ("S0-2c", "prng max lag-0 |corr| 0.0096 ± 0.0030", "stat", lambda s: aud("A1_maxcorr_prng", s), S123, (4,)),
 ("S0-2d", "held-out periodic BCE first 1000 after release 0.0960 ± 0.0003", "stat", lambda s: aud("A1_ho_bce_first1000_after_release", s), S123, (4,)),
 ("S0-2e", "switched share 0.25 at every boundary (min over boundaries)", "stat", lambda s: (lambda R: [min(v) for v in aud("A2_switch_frac", s)(R)]), S123, (4,)),
 ("S0-2f", "switched-question BCE rise +0.193 ± 0.057", "stat", lambda s: aud("A2_shift_delta_mean", s), S123, (4,)),
 ("S0-2g", "prng BCE online logistic 0.8299 ± 0.0004", "stat", lambda s: aud("A1_bce", s, "prng"), S123, (4,)),
 ("S0-3a", "net delay0 BCE 0.3765 ± 0.006", "stat", lambda s: base("net", "fam:delay0:bce", s), S012, (3,)),
 ("S0-3b", "net delay64 BCE 0.327 ± 0.003", "stat", lambda s: base("net", "fam:delay64:bce", s), S012, (3,)),
 ("S0-3c", "net delay512 BCE 0.442 ± 0.013", "stat", lambda s: base("net", "fam:delay512:bce", s), S012, (3,)),
 ("S0-3d", "pre-resolution A∨¬A violation 0.480 ± 0.184", "stat", lambda s: base("net", "viol:pre:taut_or:mean", s), S012, (3,)),
 ("S0-3e", "pre-resolution A∧¬A violation 0.535 ± 0.067", "stat", lambda s: base("net", "viol:pre:contra:mean", s), S012, (3,)),
 ("S0-3f", "pre-resolution (A∧B)→A violation 0.554 ± 0.083", "stat", lambda s: base("net", "viol:pre:impl:mean", s), S012, (3,)),
 ("S0-3g", "pre-resolution triples violation 0.388 ± 0.148", "stat", lambda s: base("net", "viol:pre:ex3:mean", s), S012, (3,)),
 ("S0-3h", "logistic pre-resolution violation 0.500 (triples)", "stat", lambda s: base("logistic", "viol:pre:ex3:mean", s), S012, (3,)),
 ("S0-3i", "net triples over all live steps 0.132 ± 0.023", "stat", lambda s: base("net", "viol:ex3:mean", s), S012, (3,)),
 ("S0-4a", "Gini shuffled-label null z -0.19 ± 0.19", "stat", lambda s: null("null", s), S012, (3,)),
 ("S0-4b", "Gini positive control z 2.03 ± 0.18 (below prereg z>3)", "stat", lambda s: null("positive", s), S012, (3,)),
]

res = []
for cid, text, kind, mk, cs, fs in CL:
    r = {"id": cid, "claim": text, "kind": kind, "committed_seeds": cs, "fresh_seeds": fs}
    try:
        cv = mk(cs)(C); fs = (OVR,) if OVR else fs; r["fresh_seeds"] = fs
        fv = mk(cs[:1] if SELFTEST else fs)(F)
        m, sd = st.mean(cv), st.stdev(cv); x = fv[0]
        r.update(committed_per_seed=cv, committed_mean=m, committed_sd=sd, fresh=x)
        if kind == "count":
            r["committed_learnt"] = sum(v < 0.3 for v in cv); r["fresh_learnt"] = x < 0.3; r["pass"] = None
        else:
            if sd > 0: ok = abs(x - m) <= 2 * sd; r["z"] = (x - m) / sd; r["band"] = "2 sd"
            elif m == 0: ok = x == 0; r["band"] = "exact 0"
            else: ok = abs(x - m) <= 0.1 * abs(m); r["band"] = "10% rel (sd 0)"
            r["pass"] = bool(ok)
    except Exception as e:
        r["error"] = repr(e); r["pass"] = None
    res.append(r)
    print(f"{cid:6s} {kind:5s} pass={r.get('pass')} committed={r.get('committed_mean', float('nan')):.4f}±{r.get('committed_sd', float('nan')):.4f} fresh={r.get('fresh')} z={r.get('z')} {r.get('error','')}")
json.dump({"committed": C, "fresh": F, "claims": res}, open(OUT, "w"), indent=1, default=float)
print(sum(1 for r in res if r["pass"]), "pass /", sum(1 for r in res if r["pass"] is not None), "tested")
