"""Skeptic audit of the smallnet rbtd8 result (2026-10-10): fast, read-only checks on the raw run files.

  PYTHONDONTWRITEBYTECODE=1 python3 audit/audit_checks.py        (from the smallnet directory; ~10 s, numpy only)

Reads results/raw/{F,W1,W2a,W2b,cal,cal2}/ only. Uses the exact posterior ONLY as the audit's evaluation metric (KL), as the
experiment did; the 'outcome-only' sections use nothing but realised outcomes.
"""
import glob, json, math, os, sys
import numpy as np

S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(os.path.join(S, "analyze.py")):
    S = "/home/agent/workspaces/eurisko-study/li-net/smallnet"
RAW = os.path.join(S, "results", "raw")
sys.path.insert(0, S)
import analyze as A  # noqa: E402  (the repo's own decision rule)

TQ = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
NS = [100, 300, 1000, 3000, 10000, 30000]
CH, K0, KC = "F-E17-rbtd8", "F-k0", "F-k0cell"


def tci(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(TQ[len(x)] * x.std(ddof=1) / math.sqrt(len(x)))


F = {}
for p in glob.glob(os.path.join(RAW, "F", "*", "*.json")):
    r = json.load(open(p))
    assert os.path.basename(p) == f"{r['tag']}-n{r['n']}-s{r['seed']}.json"
    F[(r["tag"], r["n"], r["seed"])] = r
print(f"[0] final runs: {len(F)}; every run has test nb=5000 and 5000 per-board KLs:",
      all(r["test"]["nb"] == 5000 and len(r["test_kl_per_board"]) == 5000 for r in F.values()))


def seeds(t, n):
    return [s for s in range(5) if (t, n, s) in F]


print("\n[1] TEST KL x1e3, mean +- 95% t half-width [seeds]")
for t in (K0, KC, CH):
    print(f"  {t:12s} " + " | ".join(f"{n}: {1e3*tci([F[(t,n,s)]['test']['kl'] for s in seeds(t,n)])[0]:.3f} +- "
                                       f"{1e3*tci([F[(t,n,s)]['test']['kl'] for s in seeds(t,n)])[1]:.3f} [{len(seeds(t,n))}]"
                                       for n in NS if seeds(t, n)))

print("\n[2] paired rbtd8 - base (same seed = same training boards and init), KL x1e3, 95% t; geo ratio; upper bound")
for base in (K0, KC):
    for n in NS:
        ss = [s for s in seeds(CH, n) if (base, n, s) in F]
        if len(ss) < 2:
            continue
        d = [F[(CH, n, s)]["test"]["kl"] - F[(base, n, s)]["test"]["kl"] for s in ss]
        lr = [math.log(F[(CH, n, s)]["test"]["kl"] / F[(base, n, s)]["test"]["kl"]) for s in ss]
        m, h = tci(d)
        print(f"  vs {base:8s} N={n:5d}: {1e3*m:+.3f} +- {1e3*h:.3f} (x{math.exp(np.mean(lr)):.2f}), upper {1e3*(m+h):+.3f}, seeds {ss}")

print("\n[3] outcome-only check on TEST (no posterior): paired outcome log-loss rbtd8 - k0, nats/cell x1e3")
for n in NS:
    ss = [s for s in seeds(CH, n) if (K0, n, s) in F]
    d = [F[(CH, n, s)]["test"]["logloss"] - F[(K0, n, s)]["test"]["logloss"] for s in ss]
    m, h = tci(d)
    print(f"  N={n:5d}: {1e3*m:+.3f} +- {1e3*h:.3f}; every seed negative: {all(x < 0 for x in d)}")

print("\n[4] spec rule on TEST with the repo's analyze.decision (N <= 10k)")
rows = [r for r in A.load(os.path.join(RAW, "F"), "test") if r["n"] <= 10000]
for base in (K0, KC):
    d = A.decision(rows, CH, base)
    print(f"  vs {base}: fraction of 10k boards to reach base's 10k KL = {d['frac_boards_to_reach_base10k']:.3f} "
          f"CI {[round(x, 3) for x in d['frac_ci95']]}; rule a {d['rule_a_pass']}, rule b {d['rule_b_pass']}")
for s in range(5):
    c = {n: F[(CH, n, s)]["test"]["kl"] for n in NS[:5]}
    print(f"  seed {s}: crossing fraction {A.n_to_reach(c, F[(K0, 10000, s)]['test']['kl']) / 1e4:.3f}")

print("\n[5] 10k rbtd8 vs 30k k0 (claim: 'with 10k boards it beats k0 trained on 30k')")
a = [F[(CH, 10000, s)]["test"]["kl"] for s in range(3)]
b = [F[(K0, 30000, s)]["test"]["kl"] for s in range(3)]
m, h = tci(np.array(a) - np.array(b))
print(f"  paired seeds 0-2: {1e3*m:+.3f} +- {1e3*h:.3f} x1e-3 (95% interval includes 0: {m + h > 0})")

print("\n[6] protocol per arm (cfg in each result file) and stop reasons")
keys = ("lr", "ema", "eval_every", "max_steps", "patience", "es_boards", "stages", "target", "k0w", "rbtd", "rbtd_lam", "rbtd_R",
        "teacher", "cell_q", "readout", "amp")
for t in (K0, KC, CH):
    for n in NS:
        rs = [F[(t, n, s)] for s in seeds(t, n)]
        if not rs:
            continue
        cfg = {k: sorted({str(r["cfg"].get(k)) for r in rs}) for k in keys}
        print(f"  {t:12s} N={n:5d} " + " ".join(f"{k}={v[0] if len(v) == 1 else v}" for k, v in cfg.items())
              + f" stop={sorted({r['stop_reason'] for r in rs})}")

print("\n[7] compute: FLOPs per board to the best checkpoint (analytic, fwd+bwd = 3x fwd) and total-run FLOPs, rbtd8 / k0")
for n in NS:
    fk = np.mean([F[(K0, n, s)]["flops_per_board_at_best"] for s in seeds(K0, n)])
    fc = np.mean([F[(CH, n, s)]["flops_per_board_at_best"] for s in seeds(CH, n)])
    tk = np.mean([F[(K0, n, s)]["flops_total"] for s in seeds(K0, n)])
    tc = np.mean([F[(CH, n, s)]["flops_total"] for s in seeds(CH, n)])
    print(f"  N={n:5d}: k0 {fk:.3g}, rbtd8 {fc:.3g} per board -> x{fc/fk:.1f} (to best ckpt), x{tc/tk:.1f} (whole run)")

print("\n[8] early stopping vs oracle checkpoint (val KL in each run's history, 3000 val boards; diagnostic only)")
for t in (K0, CH):
    for n in NS:
        sel, orc, last = [], [], []
        for s in seeds(t, n):
            h = F[(t, n, s)]["history"]
            sel.append([x for x in h if x["step"] == F[(t, n, s)]["best_step"]][0]["val_kl"])
            orc.append(min(x["val_kl"] for x in h))
            last.append(h[-1]["val_kl"])
        print(f"  {t:12s} N={n:5d}: early-stopped {1e3*np.mean(sel):.3f}, best anywhere {1e3*np.mean(orc):.3f}, last eval {1e3*np.mean(last):.3f}")

print("\n[9] loop selection with an OUTCOME-ONLY score (no posterior): mean over N=100,300,1k of seed-paired val log-loss minus E01-k0")
L = {}
for p in glob.glob(os.path.join(RAW, "W*", "*", "*.json")):
    r = json.load(open(p))
    assert "test" not in r, p
    L[(r["tag"], r["n"], r["seed"])] = r
print(f"  loop runs: {len(L)} (none has a TEST evaluation)")
inelig = {"E10-refpost", "E14-perm", "E22-rbtd+perm", "E25-perm+td"}
sc = []
for t in sorted({k[0] for k in L}):
    o = np.mean([np.mean([L[(t, n, s)]["val"]["logloss"] - L[("E01-k0", n, s)]["val"]["logloss"] for s in range(3)]) for n in (100, 300, 1000)])
    k = np.mean([np.mean([math.log(L[(t, n, s)]["val"]["kl"] / L[("E01-k0", n, s)]["val"]["kl"]) for s in range(3)]) for n in (100, 300, 1000)])
    sc.append((o, k, t))
sc.sort()
for i, (o, k, t) in enumerate(sc[:8], 1):
    print(f"  {i}. {t}{' (not eligible)' if t in inelig else ''}: outcome score {1e3*o:+.3f} x1e-3, S_small {k:+.3f}")
el = [x for x in sc if x[2] not in inelig]
ra, rb = np.argsort(np.argsort([x[0] for x in el])), np.argsort(np.argsort([x[1] for x in el]))
print(f"  Spearman(outcome score, S_small) over eligible arms: {np.corrcoef(ra, rb)[0, 1]:.3f}")

print("\n[10] lr/EMA calibration (cal, cal2): best setting per N by val KL (what was used) vs by val outcome log-loss")
C = {}
for p in glob.glob(os.path.join(RAW, "cal*", "*.json")):
    r = json.load(open(p))
    if r["arm"] != "k0":
        continue
    C.setdefault((r["n"], r["cfg"]["lr"], r["cfg"]["ema"]), []).append(r["val"])
for n in sorted({k[0] for k in C}):
    opts = {k[1:]: v for k, v in C.items() if k[0] == n}
    bk = min(opts, key=lambda o: np.mean([v["kl"] for v in opts[o]]))
    bl = min(opts, key=lambda o: np.mean([v["logloss"] for v in opts[o]]))
    print(f"  N={n:5d}: by KL lr={bk[0]:g} ema={bk[1]:g}; by outcome log-loss lr={bl[0]:g} ema={bl[1]:g}")
