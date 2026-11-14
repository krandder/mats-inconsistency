"""wm2 audit checks (Opus 5.5 skeptic subagent for eurisko, 2026-10-11). Independent of analyze2.py (not imported).

  PYTHONDONTWRITEBYTECODE=1 python3 audit/audit2_checks.py [--runs /tmp/wm2/runs] [--phase2]

numpy only, read-only. Sections:
  [0] TEST hygiene: no loop JSON (waves S, W1, W2, W3, RQ2) has a key containing "test" at any depth; no loop job file
      or frozen loop jobs.txt passes --test; every loop forecaster's val field scored all 3000 validation boards.
  [1] Selection recomputed from the raw JSONs (own code): D_N, G_N, S_out per tag, eligibility, champion, runner-up.
  [2] Robustness of the choice: without N=100, without N=10000, KL instead of outcome log-loss, without the RQ2
      replacements (deadline runs count as is), originals preferred always, only seeds 0-2 paired, median instead of mean.
  [3] Protocol equality: cfg (lr, ema, eval_every, max_steps, patience, es_boards, B, warmup, wd, clip) per N across all
      loop runs (forecasters, baselines, world models); every run whose stop_reason is not "patience".
  [4] Compute: FLOPs bookkeeping recomputed (all phases), ratio to k0 per N, best-checkpoint and total.
  [5] F wave (final): file list, stop reasons, best steps, cfg; the "test" fields are NOT read unless --phase2.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np

NS = [100, 300, 1000, 3000, 10000]
LOOP_WAVES = ["S", "W1", "W2", "W3", "RQ2"]
SEL_WAVES = ["W1", "W2", "W3", "RQ2"]
JOBS = "/home/agent/workspaces/eurisko-study/li-net/wm2/jobs"
TQ = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
PROTO_KEYS = ["lr", "ema", "eval_every", "max_steps", "patience", "es_boards", "B", "warmup", "wd", "clip"]


def keys_with_test(o, path=""):
    out = []
    if isinstance(o, dict):
        for k, v in o.items():
            if "test" in str(k).lower():
                out.append(path + "/" + str(k))
            out += keys_with_test(v, path + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o[:3]):
            out += keys_with_test(v, path + f"[{i}]")
    return out


def strip_test(r):
    """Remove every TEST field before anything else touches the record (phase 1 rule for wave F)."""
    for k in [k for k in r if "test" in k.lower()]:
        del r[k]
    return r


def load_all(runs, waves, task, keep_test=False):
    fcs, wms = [], []
    for w in waves:
        for p in sorted(glob.glob(os.path.join(runs, w, task, "*", "*.json"))):
            r = json.load(open(p))
            if not keep_test:
                strip_test(r)
            r["_wave"], r["_path"] = w, p
            (wms if r.get("kind") == "wm" else fcs).append(r)
    return fcs, wms


def pick(recs, mode="prereg"):
    """One record per (tag, n, seed). prereg: non-deadline beats deadline, then original wave beats RQ*.
    noreplace: original wave always wins (the RQ2 reruns are ignored where an original exists)."""
    best = {}
    for r in recs:
        k = (r.get("tag", r.get("arm")), r["n"], r["seed"])
        dl = "deadline" in str(r.get("stop_reason"))
        rq = r["_wave"].startswith("RQ")
        rank = (dl, rq) if mode == "prereg" else (rq, dl)
        if k not in best or rank < best[k][0]:
            best[k] = (rank, r)
    return {k: v[1] for k, v in best.items()}


def table(sel, metric="logloss"):
    t = defaultdict(dict)
    for (tag, n, s), r in sel.items():
        t[(tag, n)][s] = r["val"][metric]
    return t


def s_out(t, tags, ns=NS, seeds=None, stat="mean", floor=5e-5):
    """Own implementation of PREREG's S_out. Returns {tag: (S_out or None, terms, nseeds)} and the gap per N."""
    gap = {}
    for n in ns:
        a, b = t.get(("B-k0", n), {}), t.get(("B-rbtd8", n), {})
        s = sorted(set(a) & set(b) & (set(seeds) if seeds else set(a)))
        gap[n] = max(float(np.mean([a[x] - b[x] for x in s])), floor) if s else None
    out = {}
    for tag in tags:
        terms, full, ns_ = [], True, []
        for n in ns:
            a, rb, k0 = t.get((tag, n), {}), t.get(("B-rbtd8", n), {}), t.get(("B-k0", n), {})
            s = sorted(set(a) & set(rb) & (set(seeds) if seeds else set(a)))
            sk = sorted(set(a) & set(k0) & (set(seeds) if seeds else set(a)))
            if len(s) < 3 or len(sk) < 3 or gap[n] is None:
                full = False
                continue
            d = [a[x] - rb[x] for x in s]
            terms.append((float(np.median(d)) if stat == "median" else float(np.mean(d))) / gap[n])
            ns_.append(len(s))
        out[tag] = (float(np.mean(terms)) if full else None, terms, ns_)
    return out, gap


def eligible_tags(sel):
    info = defaultdict(lambda: {"rwv": True, "l0": True, "flag": ""})
    for (tag, n, s), r in sel.items():
        e = info[tag]
        e["rwv"] &= bool(r.get("real_world_valid", True))
        e["l0"] &= bool(r.get("l0", True))
        e["flag"] = e["flag"] or r.get("flag", "")
    return {t: e for t, e in info.items() if t not in ("B-k0", "B-rbtd8")}


def ranking(res, elig, key=lambda t: True):
    rows = [(v[0], t) for t, v in res.items() if v[0] is not None and t in elig and key(t)]
    return sorted(rows)


def tci(x):
    x = np.asarray(x, float)
    return float(x.mean()), (float(TQ.get(len(x), 2.0) * x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/tmp/wm2/runs")
    ap.add_argument("--phase2", action="store_true")
    args = ap.parse_args()
    R = args.runs

    # ------------------------------------------------------------------------------------------------ [0]
    print("[0] TEST hygiene of the loop")
    n_files, bad = 0, []
    for w in LOOP_WAVES:
        for p in sorted(glob.glob(os.path.join(R, w, "*", "*", "*.json"))):
            n_files += 1
            r = json.load(open(p))
            ks = keys_with_test(r)
            if ks:
                bad.append((p, ks[:5]))
    print(f"  loop JSONs scanned: {n_files}; with a key containing 'test': {len(bad)}")
    for p, ks in bad[:10]:
        print("   ", p, ks)
    loopjobs = [p for p in glob.glob(os.path.join(JOBS, "*.txt")) if re.match(r"(smoke|W\d|RQ\d)", os.path.basename(p))]
    tj = [p for p in loopjobs if "--test" in open(p).read()]
    print(f"  loop job files: {len(loopjobs)}; passing --test: {len(tj)} {tj}")
    srcs = [d for d in glob.glob("/tmp/wm2/src-*") if re.search(r"src-wm2-(smoke|w\d|rq\d)", d)]
    ts = [d for d in srcs if os.path.exists(d + "/jobs.txt") and "--test" in open(d + "/jobs.txt").read()]
    xs = []
    for d in srcs:
        x = d + "/jobs-extra.txt"
        for q in glob.glob(d + "/*.txt"):
            if q.endswith("jobs.txt"):
                continue
            if "--test" in open(q).read():
                xs.append(q)
    print(f"  frozen loop sources: {len(srcs)}; jobs.txt passing --test: {len(ts)}; other .txt with --test: {xs}")
    nb = Counter()
    for w in LOOP_WAVES:
        for p in glob.glob(os.path.join(R, w, "*", "*", "*.json")):
            r = json.load(open(p))
            if r.get("kind") != "wm":
                nb[r["val"]["nb"]] += 1
    print(f"  forecaster/baseline val.nb values: {dict(nb)}")

    # ------------------------------------------------------------------------------------------------ [1]
    print("\n[1] Selection recomputed (T1, waves W1,W2,W3,RQ2, PREREG duplicate rule)")
    fcs, wms = load_all(R, SEL_WAVES, "t1")
    sel = pick(fcs, "prereg")
    rep = [(k, sel[k]["_wave"]) for k in sel if sel[k]["_wave"] == "RQ2"]
    dups = Counter((r.get("tag"), r["n"], r["seed"]) for r in fcs)
    print(f"  forecaster/baseline records: {len(fcs)}; unique (tag,N,seed): {len(sel)}; duplicated keys: {sum(v > 1 for v in dups.values())}")
    for k in sorted(k for k, v in dups.items() if v > 1):
        cands = [(r["_wave"], r["stop_reason"], r["best_step"], round(1e3 * r["val"]["kl"], 3), round(r["val"]["logloss"], 6)) for r in fcs
                 if (r.get("tag"), r["n"], r["seed"]) == k]
        print(f"    {k}: candidates {cands} -> kept {sel[k]['_wave']}")
    elig = eligible_tags(sel)
    T = table(sel, "logloss")
    tags = sorted(elig)
    res, gap = s_out(T, tags)
    print("  G_N (k0 - rbtd8 outcome log-loss, x1e4): " + ", ".join(f"{n}: {1e4 * g:.3f}" for n, g in gap.items()))
    print("  tag | eligible | S_out | terms per N | paired seeds per N")
    for s, t in sorted([(v[0] if v[0] is not None else 9, t) for t, v in res.items()]):
        e = elig[t]
        el = e["rwv"] and e["l0"] and not e["flag"] and res[t][0] is not None
        why = "yes" if el else ("flagged" if e["flag"] else ("not L0" if not e["l0"] else "partial"))
        print(f"   {t:24s} {why:8s} {('%.4f' % res[t][0]) if res[t][0] is not None else '   -  '} "
              f"{[round(x, 3) for x in res[t][1]]} {res[t][2]}")
    elig_ok = {t for t, e in elig.items() if e["rwv"] and e["l0"] and not e["flag"]}
    rk = ranking(res, elig_ok)
    print(f"  champion: {rk[0][1]} S_out {rk[0][0]:.4f}; runner-up: {rk[1][1]} {rk[1][0]:.4f}; third: {rk[2][1]} {rk[2][0]:.4f}")
    ch = rk[0][1]
    for n in NS:
        a, b = T[(ch, n)], T[("B-rbtd8", n)]
        s = sorted(set(a) & set(b))
        d = [a[x] - b[x] for x in s]
        m, h = tci(d)
        kl_c = np.mean([sel[(ch, n, x)]["val"]["kl"] for x in s])
        kl_b = np.mean([sel[("B-rbtd8", n, x)]["val"]["kl"] for x in s])
        print(f"   N={n}: champion - rbtd8 outcome log-loss x1e4 {1e4 * m:+.3f} +- {1e4 * h:.3f} (seeds {s}); per seed {[round(1e4 * x, 2) for x in d]}; "
              f"val KL x1e3 {1e3 * kl_c:.3f} vs rbtd8 {1e3 * kl_b:.3f} (ratio {kl_c / kl_b:.2f}); runs from {[sel[(ch, n, x)]['_wave'] for x in s]}")

    # ------------------------------------------------------------------------------------------------ [2]
    print("\n[2] Robustness of the choice (report only; the pre-registered rule is [1])")
    variants = {}
    variants["prereg (as [1])"] = rk
    r2, _ = s_out(T, tags, ns=[300, 1000, 3000, 10000])
    variants["without N=100"] = ranking(r2, elig_ok)
    r2, _ = s_out(T, tags, ns=[100, 300, 1000, 3000])
    variants["without N=10000"] = ranking(r2, elig_ok)
    r2, _ = s_out(T, tags, ns=[300, 1000, 3000])
    variants["N=300-3000 only"] = ranking(r2, elig_ok)
    TK = table(sel, "kl")
    r2, _ = s_out(TK, tags, floor=1e-9)
    variants["KL instead of log-loss (same normalisation)"] = ranking(r2, elig_ok)
    # geometric mean KL ratio vs rbtd8 over 5 N (round-1 style)
    geo = {}
    for t in tags:
        lr = []
        for n in NS:
            a, b = TK.get((t, n), {}), TK.get(("B-rbtd8", n), {})
            s = sorted(set(a) & set(b))
            if len(s) >= 3:
                lr.append(np.mean([math.log(a[x] / b[x]) for x in s]))
        geo[t] = (float(np.mean(lr)) if len(lr) == 5 else None, lr, [])
    variants["geo-mean KL ratio vs rbtd8 (5 N)"] = ranking(geo, elig_ok)
    seln = pick(fcs, "noreplace")
    Tn = table(seln, "logloss")
    r2, _ = s_out(Tn, tags)
    variants["no RQ2 replacement (deadline runs as is)"] = ranking(r2, elig_ok)
    fcs_norq, _ = load_all(R, ["W1", "W2", "W3"], "t1")
    Tw = table(pick(fcs_norq, "prereg"), "logloss")
    r2, _ = s_out(Tw, tags)
    variants["waves W1-W3 only (as at 18:39Z)"] = ranking(r2, elig_ok)
    r2, _ = s_out(T, tags, stat="median")
    variants["median paired difference"] = ranking(r2, elig_ok)
    r2, _ = s_out(T, tags, floor=0.0)
    variants["no 5e-5 floor"] = ranking(r2, elig_ok)
    # drop deadline-stopped runs entirely (where not replaced)
    sel_nodl = {k: v for k, v in sel.items() if "deadline" not in str(v.get("stop_reason"))}
    r2, _ = s_out(table(sel_nodl, "logloss"), tags)
    variants["deadline-stopped runs dropped"] = ranking(r2, elig_ok)
    # leave-one-seed-out
    for drop in (0, 1, 2):
        r2, _ = s_out(T, tags, seeds=[s for s in (0, 1, 2) if s != drop])
        # with 2 seeds the >=3 rule fails; recompute with a 2-seed rule
        terms = {}
        for t in tags:
            tt, ok = [], True
            for n in NS:
                a, rb = T.get((t, n), {}), T.get(("B-rbtd8", n), {})
                k0 = T.get(("B-k0", n), {})
                s = sorted(set(a) & set(rb) & set(k0) - {drop})
                if len(s) < 2:
                    ok = False
                    break
                g = max(np.mean([k0[x] - rb[x] for x in s]), 5e-5)
                tt.append(np.mean([a[x] - rb[x] for x in s]) / g)
            terms[t] = (float(np.mean(tt)) if ok else None, tt, [])
        variants[f"leave seed {drop} out"] = ranking(terms, elig_ok)
    for name, rkv in variants.items():
        top = ", ".join(f"{t} {s:.3f}" for s, t in rkv[:4])
        print(f"  {name:48s} -> {top}")
    # seed bootstrap: resample the 3 paired seeds per N (same draw for every arm at that N), recompute S_out
    rng = np.random.default_rng(0)
    full_elig = [t for t in elig_ok if res[t][0] is not None]
    wins, margins = Counter(), []
    for _ in range(4000):
        draw = {n: rng.integers(0, 3, 3) for n in NS}
        sc = {}
        for t in full_elig:
            terms = []
            for n in NS:
                rb, k0, x = T[("B-rbtd8", n)], T[("B-k0", n)], T[(t, n)]
                g = max(np.mean([k0[s] - rb[s] for s in draw[n]]), 5e-5)
                terms.append(np.mean([x[s] - rb[s] for s in draw[n]]) / g)
            sc[t] = np.mean(terms)
        best = min(sc, key=sc.get)
        wins[best] += 1
        margins.append(sc["WMe-D-rbtd8"] - sc[ch])
    print("  seed bootstrap (4000 draws of 3 seeds with replacement per N): share of draws each arm is first: " +
          ", ".join(f"{t} {v / 4000:.3f}" for t, v in wins.most_common()))
    print(f"  S_out(WMe-D-rbtd8) - S_out({ch}) over the draws: median {np.median(margins):.3f}, 2.5-97.5% [{np.percentile(margins, 2.5):.3f}, {np.percentile(margins, 97.5):.3f}]")
    # per-N paired difference champion minus runner-up (outcome log-loss, normalised by G_N)
    for n in NS:
        a_, b_ = T[(ch, n)], T[("WMe-D-rbtd8", n)]
        d = [(a_[s] - b_[s]) / gap[n] for s in (0, 1, 2)]
        m, h = tci(d)
        print(f"   N={n}: ({ch} - WMe-D-rbtd8) / G_N = {m:+.3f} +- {h:.3f} (per seed {[round(x, 3) for x in d]})")
    # free replications: same config, same seed, original vs RQ2 rerun (outcome log-loss), in units of G_N
    reps = []
    for k, v in dups.items():
        if v > 1:
            c = [r for r in fcs if (r.get("tag"), r["n"], r["seed"]) == k]
            o = [r for r in c if r["_wave"] != "RQ2"][0]
            q = [r for r in c if r["_wave"] == "RQ2"][0]
            if "deadline" not in str(o["stop_reason"]):
                reps.append((k, (q["val"]["logloss"] - o["val"]["logloss"]) / gap[k[1]]))
    print("  same-seed reruns (RQ2 minus original, both not deadline-stopped), log-loss difference / G_N: " +
          ", ".join(f"{k[0]} N={k[1]} s{k[2]}: {d:+.3f}" for k, d in sorted(reps)))
    print(f"   sd of a single run's log-loss / G_N implied (|diff| / sqrt 2): N=3000 {np.std([d for k, d in reps if k[1] == 3000], ddof=0) / math.sqrt(2) if reps else float('nan'):.3f}; "
          f"rms over all {math.sqrt(np.mean([d * d for _, d in reps]) / 2):.3f}")

    # ------------------------------------------------------------------------------------------------ [3]
    print("\n[3] Protocol equality (cfg per N over every loop run of T1 and T2, waves W1,W2,W3,RQ2)")
    for task in ("t1", "t2"):
        f_, w_ = load_all(R, SEL_WAVES, task)
        per = defaultdict(Counter)
        for r in f_:
            c = r["cfg"]
            per[r["n"]][tuple(c.get(k) for k in PROTO_KEYS)] += 1
        for r in w_:
            c = r["cfg"]
            per[r["n"]][tuple(c.get(k) for k in PROTO_KEYS)] += 1
        for n in NS:
            print(f"  {task} N={n}: {len(per[n])} distinct ({', '.join(PROTO_KEYS)}): " + "; ".join(f"{k} x{v}" for k, v in per[n].items()))
        st = Counter()
        lst = []
        for r in f_ + w_:
            sr = str(r.get("stop_reason"))
            if sr != "patience":
                tag = r.get("tag") or r.get("wmtag")
                st[(tag, sr)] += 1
                lst.append((r["_wave"], tag, r["n"], r["seed"], sr, r.get("best_step"), r.get("steps_run"), r.get("best_step_ev"), r.get("steps_run_ev")))
        print(f"  {task}: runs whose stop_reason is not 'patience': {len(lst)}")
        for x in sorted(lst):
            print("    ", x)

    # forecasters' rho / dream settings by tag (T1 selection set)
    print("  dream settings per T1 tag (rho, dream_buf, dream_reuse, fc_arm, rbtd, k0w, stages, target, teacher):")
    seen = {}
    for r in fcs:
        c = r["cfg"]
        seen.setdefault(r.get("tag"), set()).add(tuple(c.get(k) for k in ("rho", "dream_buf", "dream_reuse", "fc_arm", "rbtd", "k0w", "stages", "target", "teacher")))
    for t, v in sorted(seen.items()):
        print(f"    {t}: {sorted(v, key=str)}")

    # ------------------------------------------------------------------------------------------------ [4]
    print("\n[4] Compute per training board (T1 selection set; FLOPs, all phases)")
    bad = 0
    for r in fcs:
        if "flops_all_at_best" not in r:
            continue
        tot = r["flops_wm_at_best"] + r["flops_dreams_at_best"] + r["flops_fc_at_best"]
        if abs(tot - r["flops_all_at_best"]) > 1e-6 * tot or abs(r["flops_per_board_at_best"] - tot / r["n"]) > 1e-6 * tot / r["n"]:
            bad += 1
    print(f"  flops_all_at_best == wm + dreams + fc and per-board = all / N: mismatches {bad}")
    fb = 2 * 25 * (128 * 3 * 128 + 128 * 128 + 2 * 128 * 256) * 4 + 2 * 2 * 25 * 25 * 128 * 4 + 2 * 25 * 128
    print(f"  forward FLOPs per 25-token pass (netdef.flops_backbone formula): {fb:.4e}")
    # per-step check: D-rbtd8 forecaster = 8 passes x 256 board-stages x 3; k0 = 1 pass x 256 x 3
    chk = []
    for r in fcs:
        if r.get("tag") in ("WMr-D-rbtd8", "B-rbtd8", "B-k0", "WMr-D-only") and r["steps_run"] > 0:
            ff = r.get("flops_fc_total", r.get("flops_total"))
            chk.append((r.get("tag"), ff / r["steps_run"] / (3 * fb * 256)))
    agg = defaultdict(list)
    for t, v in chk:
        agg[t].append(v)
    print("  forecaster FLOPs per step / (3 x pass x 256 board-stages): " + ", ".join(f"{t} {np.mean(v):.3f}" for t, v in sorted(agg.items())))
    # dream flops: n_dreams x passes x fb ; WM flops from WM JSONs
    dchk = []
    for r in fcs:
        if r.get("n_dreams", 0) and not r["cfg"].get("post_target"):
            passes = (r.get("dream_stats") or {}).get("fwd_passes_per_dream", 31)
            dchk.append(r["flops_dreams_total"] / (r["n_dreams"] * passes * fb))
    print(f"  dream FLOPs / (n_dreams x passes x pass): min {min(dchk):.4f} max {max(dchk):.4f} over {len(dchk)} runs")
    wmf = {(w["wmtag"], w["n"], w["seed"], w["_wave"]): w for w in wms}
    mism = 0
    tot_ = 0
    for r in fcs:
        wt = r.get("wm_tag")
        if not wt or "+" in wt:
            continue
        w = wmf.get((wt, r["n"], r["seed"], r["_wave"]))
        if w is None:
            continue
        tot_ += 1
        if abs(w["flops_at_best"] - r["flops_wm_at_best"]) > 1e-6 * w["flops_at_best"]:
            mism += 1
    print(f"  forecaster's flops_wm_at_best equals its WM JSON's flops_at_best: {tot_ - mism}/{tot_}")
    # ensemble: flops_wm = sum of members
    ens_bad, ens_n = 0, 0
    for r in fcs:
        wt = r.get("wm_tag") or ""
        if "+" in wt:
            ens_n += 1
            mem = [wmf.get((m, r["n"], r["seed"], r["_wave"])) for m in wt.split("+")]
            if any(m is None for m in mem) or abs(sum(m["flops_at_best"] for m in mem) - r["flops_wm_at_best"]) > 1e-6 * r["flops_wm_at_best"]:
                ens_bad += 1
    print(f"  ensemble forecasters whose flops_wm_at_best = sum of the 3 members: {ens_n - ens_bad}/{ens_n}")
    # WM flops per step check
    wchk = defaultdict(list)
    for w in wms:
        c = w["cfg"]
        if c.get("wm_split"):
            steps = w["steps_run"] + w.get("steps_run_ev", 0)
        else:
            steps = w["steps_run"]
        wchk[(w["wmtag"])].append(w["flops_total"] / max(1, steps) / (3 * fb * 64))
    print("  WM FLOPs per step / (3 x pass x 64 episodes), mean over runs: " + ", ".join(f"{t} {np.mean(v):.2f}" for t, v in sorted(wchk.items())))
    # compute table: per N, champion vs k0 and rbtd8, to best and total
    print("  FLOPs per board, seed mean (to best checkpoint | total), and ratio to B-k0 / B-rbtd8 (to best):")
    for t in ["B-k0", "B-rbtd8", ch, "WMe-D-rbtd8", "WMs-D-rbtd8", "WMr-D-only"]:
        cells = []
        for n in NS:
            v = [sel[(t, n, s)] for s in (0, 1, 2) if (t, n, s) in sel]
            k0 = [sel[("B-k0", n, s)] for s in (0, 1, 2) if ("B-k0", n, s) in sel]
            rb = [sel[("B-rbtd8", n, s)] for s in (0, 1, 2) if ("B-rbtd8", n, s) in sel]
            fbst = np.mean([r["flops_per_board_at_best"] for r in v])
            ftot = np.mean([r.get("flops_per_board_total", r["flops_total"] / r["n"]) for r in v])
            k0b = np.mean([r["flops_per_board_at_best"] for r in k0])
            rbb = np.mean([r["flops_per_board_at_best"] for r in rb])
            cells.append(f"N={n}: {fbst:.2e} | {ftot:.2e} ({fbst / k0b:.0f}x k0, {fbst / rbb:.1f}x rbtd8)")
        print(f"   {t}: " + "; ".join(cells))
    # share of each phase for the champion
    print(f"  {ch}: share of FLOPs to the best checkpoint by phase (WM / dreams / forecaster), seed mean:")
    for n in NS:
        v = [sel[(ch, n, s)] for s in (0, 1, 2) if (ch, n, s) in sel]
        sh = np.mean([[r["flops_wm_at_best"] / r["flops_all_at_best"], r["flops_dreams_at_best"] / r["flops_all_at_best"],
                       r["flops_fc_at_best"] / r["flops_all_at_best"]] for r in v], 0)
        print(f"   N={n}: {sh.round(3).tolist()}")

    # ------------------------------------------------------------------------------------------------ [4b]
    print("\n[4b] Early-stopping evaluation compute (excluded from every arm's FLOPs): eval FLOPs / training FLOPs, seed mean")
    # forecaster eval: es_boards x 4 stages forward passes per evaluation; WM board network: 25 passes per validation
    # episode; WM probe network: 6 passes per episode. Training FLOPs as recorded (3 x forward; dreams 1 x forward).
    for t in ["B-k0", "B-rbtd8", ch, "WMe-D-rbtd8", "WMs-D-only"]:
        cells = []
        for n in NS:
            v = [sel[(t, n, s)] for s in (0, 1, 2) if (t, n, s) in sel]
            ratios, fcpb = [], []
            for r in v:
                ev = len(r["history"]) * r["cfg"]["es_boards"] * 4 * fb
                tr = r.get("flops_fc_total", r["flops_total"]) + r.get("flops_dreams_total", 0.0)
                wt = r.get("wm_tag")
                if wt:
                    for m in wt.split("+"):
                        w = wmf.get((m, r["n"], r["seed"], r["_wave"]))
                        if w is None:
                            continue
                        ev += (len(w["history"]) * 3000 * 25 + len(w.get("history_ev", [])) * 3000 * 6) * fb
                        tr += w["flops_total"]
                ratios.append(ev / tr)
                fcpb.append((ev + tr) / r["n"])
            cells.append(f"N={n}: {np.mean(ratios):.2f} (all-in total {np.mean(fcpb):.2e}/board)")
        print(f"   {t}: " + "; ".join(cells))
    # WM training compute per board: WMr vs WMs (to the best checkpoint)
    wpb = defaultdict(list)
    for w in wms:
        wpb[(w["wmtag"].split("-e")[0], w["n"])].append(w["flops_per_board_at_best"])
    print("  WM FLOPs per board to the best checkpoint, WMr / WMs ratio per N: " +
          ", ".join(f"{n}: {np.mean(wpb[('WMr', n)]) / np.mean(wpb[('WMs', n)]):.1f}" for n in NS))

    # ------------------------------------------------------------------------------------------------ [6]
    print("\n[6] Validation KL x1e3 per tag (seed mean over the de-duplicated set; reported, never used to select)")
    for t in sorted({k[0] for k in sel}):
        cells = []
        for n in NS:
            v = [sel[(t, n, s)]["val"]["kl"] for s in range(5) if (t, n, s) in sel]
            cells.append(f"{1e3 * np.mean(v):.2f}({len(v)})" if v else "-")
        print(f"   {t:22s} " + " / ".join(cells))

    # ------------------------------------------------------------------------------------------------ [5]
    print("\n[5] Final wave F (non-TEST fields only unless --phase2)")
    for task in ("t1", "t2"):
        f_, w_ = load_all(R, ["F"], task, keep_test=args.phase2)
        c = Counter((r.get("tag") or r.get("wmtag"), r.get("stop_reason")) for r in f_ + w_)
        print(f"  {task}: {len(f_)} forecaster/baseline JSONs, {len(w_)} WM JSONs")
        for k, v in sorted(c.items(), key=str):
            print(f"    {k}: {v}")
        per = defaultdict(Counter)
        for r in f_ + w_:
            cc = r["cfg"]
            per[r["n"]][tuple(cc.get(k) for k in PROTO_KEYS)] += 1
        for n in NS:
            if per[n]:
                print(f"    N={n}: distinct protocol tuples {dict(per[n])}")
        cov = defaultdict(set)
        for r in f_:
            cov[(r.get("tag"), r["n"])].add(r["seed"])
        for k in sorted(cov, key=str):
            print(f"    seeds {k}: {sorted(cov[k])}")


if __name__ == "__main__":
    main()
