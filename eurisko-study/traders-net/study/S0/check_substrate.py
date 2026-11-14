"""S0 check: runs the substrate test suite, then re-reads every headline number in SUBSTRATE.md
from its source file and fails if the file disagrees. Usage (from anywhere):
  PYTHONPATH=/home/agent/workspaces/eurisko-study/wealth-net/.pylib python3 check_substrate.py
"""
import csv, json, os, subprocess, sys
H = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(H, *a)

def rows(path):
    with open(P(path)) as f:
        return list(csv.DictReader(f))

def w1(arm, metric, fam):
    r = [x for x in rows("w1/SUMMARY-w1.csv") if (x["arm"], x["metric"], x["family"]) == (arm, metric, fam)][0]
    return float(r["mean"]), float(r["sd"])

def w3(model, metric):
    r = [x for x in rows("w3/baseline-summary.csv") if (x["model"], x["metric"]) == (model, metric)][0]
    return float(r["mean"]), float(r["sd"])

def w2(arm, fam):
    r = [x for x in rows("results/w2/audit_family_bce.csv") if (x["arm"], x["family"]) == (arm, fam)][0]
    return float(r["mean"]), float(r["sd"])

def w4(arm):
    d = json.load(open(P("w4/null_calib.json")))["mean_sd_over_seeds"][arm]
    return d

# (label, (mean, sd) from source, expected mean, expected sd, tol)  -- values quoted in SUBSTRATE.md
C = [
    ("P4 pi BCE", w1("P4", "bce", "pi"), 0.6933, 0.0004, 5e-5),
    ("P16 pi BCE", w1("P16", "bce", "pi"), 0.6957, 0.0004, 5e-5),
    ("ME pi held-out BCE", w1("ME", "bce", "pi"), 0.830, 0.021, 5e-4),
    ("ME pi in-sample BCE", w1("ME", "in_bce", "pi"), 0.562, 0.022, 5e-4),
    ("ME pi gap", w1("ME", "gap", "pi"), 0.267, 0.043, 5e-4),
    ("ME ca30 gap", w1("ME", "gap", "ca30"), 0.004, 0.004, 5e-4),
    ("P4 ca30 BCE", w1("P4", "bce", "ca30"), 0.6945, 0.0003, 5e-5),
    ("P4 lfsr16 BCE", w1("P4", "bce", "lfsr16"), 0.46, 0.40, 5e-3),
    ("P16 lfsr16 BCE", w1("P16", "bce", "lfsr16"), 0.10, 0.17, 5e-3),
    ("P4 logistic pi BCE", w1("P4", "bce_log", "pi"), 0.714, 0.0001, 5e-4),
    ("ME lfsr16 overlap49", w1("ME", "overlap49", "lfsr16"), 1.00, 0.0, 5e-3),
    ("ME prime overlap49", w1("ME", "overlap49", "prime"), 1.00, 0.0, 5e-3),
    ("wealth-net pi shift-copy share", w2("A0", "shift_copy_pi"), 1.0, 0.0, 1e-4),
    ("prng shift-copy share", w2("A1", "shift_copy_prng"), 0.0, 0.0, 1e-4),
    ("prng max corr", w2("A1", "maxcorr_prng"), 0.0096, 0.0030, 1e-4),
    ("prng BCE online logistic", w2("A1", "prng"), 0.8299, 0.0004, 1e-4),
    ("held-out BCE first 1000 after release", w2("A1", "ho_bce_first1000_after_release"), 0.0960, 0.0003, 1e-4),
    ("regime shift BCE jump", w2("A2", "shift_delta_mean"), 0.193, 0.057, 5e-4),
    ("net delay0 BCE", w3("net", "fam:delay0:bce"), 0.3765, 0.006, 5e-4),
    ("net delay64 BCE", w3("net", "fam:delay64:bce"), 0.327, 0.003, 5e-4),
    ("net delay512 BCE", w3("net", "fam:delay512:bce"), 0.442, 0.013, 5e-4),
    ("net pre taut_or viol", w3("net", "viol:pre:taut_or:mean"), 0.480, 0.184, 1e-3),
    ("net pre contra viol", w3("net", "viol:pre:contra:mean"), 0.535, 0.067, 5e-4),
    ("net pre impl viol", w3("net", "viol:pre:impl:mean"), 0.554, 0.083, 5e-4),
    ("net pre ex3 viol", w3("net", "viol:pre:ex3:mean"), 0.388, 0.148, 5e-4),
    ("logistic pre ex3 viol", w3("logistic", "viol:pre:ex3:mean"), 0.500, 0.000, 5e-4),
    ("net ex3 viol (all live steps)", w3("net", "viol:ex3:mean"), 0.132, 0.023, 5e-4),
    ("gini null z mean", tuple(w4("null")["z_mean"]), -0.19, 0.19, 5e-3),
    ("gini positive z mean", tuple(w4("positive")["z_mean"]), 2.03, 0.18, 5e-3),
]

def main():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        ["/home/agent/workspaces/eurisko-study/wealth-net/.pylib", P("substrate")]))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                       cwd=P("substrate"), env=env, capture_output=True, text=True)
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-400:]
    print("pytest:", last)
    bad = 0 if r.returncode == 0 else 1
    for lab, (m, s), em, es, tol in C:
        ok = abs(m - em) <= tol and abs(s - es) <= max(tol, 0.01 * es + 5e-4)
        bad += not ok
        print(("ok  " if ok else "BAD ") + f"{lab}: file {m:.4f} ± {s:.4f}, doc {em} ± {es}")
    print("FAIL" if bad else "PASS", f"({bad} problems)")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
