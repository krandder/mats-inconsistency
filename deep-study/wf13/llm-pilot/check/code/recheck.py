# recheck.py ROLLOUTS.jsonl OUT.json [KMAX] : the checker's own aggregation and metrics (no import from the work's code).
# Works on the work's stored rollouts (out/main/test.jsonl, ext.jsonl) and on the checker's fresh ones.
import sys, json, math, re
from collections import Counter, defaultdict
import numpy as np

PATH, OUTP = sys.argv[1], sys.argv[2]
KMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 64
KS = [k for k in (1, 4, 16, 64, 128, 256) if k <= KMAX]
recs = [json.loads(l) for l in open(PATH)]

def exact(text):
    """Exact answer from the problem text alone (my evaluator)."""
    parts = [s.strip() for s in text.split(";")]
    if text.rstrip().endswith("?"):
        val = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in parts[0].split()}
        for st in parts[1:-1]:
            x, rhs = st.split("=")
            m = re.match(r"^([abc])([+*-])([abc]|\d+)$", rhs)
            a = val[m.group(1)]; b = val[m.group(3)] if m.group(3) in "abc" else int(m.group(3))
            val[x] = a + b if m.group(2) == "+" else (a - b if m.group(2) == "-" else a * b)
        return str(val[parts[-1].rstrip("?")])
    v = int(parts[0])
    for st in parts[1:]:
        b = int(st[1:]); v = v + b if st[0] == "+" else (v - b if st[0] == "-" else v * b)
    return str(v)

def norm(d):
    s = sum(d.values())
    return {a: v / s for a, v in d.items()} if s > 0 else None

def vote(r, ids, lq):
    return norm(Counter(r["dans"][i] for i in ids if i >= 0))

def dedup(r, ids, lq):
    d = defaultdict(float)
    for i in set(i for i in ids if i >= 0):
        d[r["dans"][i]] += math.exp(r["dlr"][i])
    return norm(d)

def snis(r, ids, lq):
    lw = [(r["dlr"][i] - q, i) for i, q in zip(ids, lq) if i >= 0]
    if not lw:
        return None
    mx = max(x for x, _ in lw); d = defaultdict(float)
    for x, i in lw:
        d[r["dans"][i]] += math.exp(x - mx)
    return norm(d)

AGG = {"vote": vote, "dedup": dedup, "snis": snis}
def tv(p, q):
    if p is None or q is None:
        return 0.0 if p is None and q is None else 1.0
    return 0.5 * sum(abs(p.get(a, 0) - q.get(a, 0)) for a in set(p) | set(q))
def top(p):
    return max(p, key=lambda a: (p[a], a)) if p else None
def found(r, ids):
    RF = defaultdict(float)
    for i in set(i for i in ids if i >= 0):
        RF[r["dans"][i]] += math.exp(r["dlr"][i])
    return RF

bad_exact = 0
for r in recs:
    bad_exact += exact(r["text"]) != r["answer"]
    r["T"] = vote(r, r["ref"]["id"], None)
    r["w"]["W1b"] = {"id": r["ref"]["id"][:KMAX], "lq": r["ref"]["lq"][:KMAX]}
    # simulated hider: M's reference rollouts with M's top-scoring found line removed (q renormalised exactly)
    mids = [i for i in r["w"]["W1"]["id"] + r["ref"]["id"] if i >= 0]
    if mids:
        t = max(set(mids), key=lambda i: r["dlr"][i]); pt = math.exp(r["dlr"][t])
        keep = [(i, q - math.log(max(1 - pt, 1e-12))) for i, q in zip(r["ref"]["id"], r["ref"]["lq"]) if i != t][:KMAX]
        r["hid_top_share"] = pt
        if len(keep) == KMAX:
            r["w"]["HID"] = {"id": [i for i, _ in keep], "lq": [q for _, q in keep]}
writers = [w for w in recs[0]["w"] if all(w in r["w"] for r in recs)]
res = {"file": PATH, "n": len(recs), "exact_answer_mismatch": bad_exact, "writers": writers, "cells": {}, "fam": {}}
res["HID_available"] = sum("HID" in r["w"] for r in recs)
for fam in sorted(set(r["fam"] for r in recs)):
    R = [r for r in recs if r["fam"] == fam]
    F = {"n": len(R), "direct": float(np.mean([r.get("direct", {}).get("greedy") == r["answer"] for r in R])),
         "ref_top_right": float(np.mean([top(r["T"]) == r["answer"] for r in R])),
         "ref_p_true": float(np.mean([(r["T"] or {}).get(r["answer"], 0) for r in R])),
         "Z_valid": float(np.mean([np.mean([i >= 0 for i in r["ref"]["id"]]) for r in R])),
         "max_pooled_score": float(max(sum(math.exp(x) for x in r["dlr"]) for r in R)),
         "W1_abs_logq_minus_score_mean": float(np.mean([abs(q - r["dlr"][i]) for r in R for i, q in zip(r["w"]["W1"]["id"], r["w"]["W1"]["lq"]) if i >= 0]))}
    res["fam"][fam] = F
    for w in writers + (["HID"] if all("HID" in r["w"] for r in R) else []):
        for K in KS:
            for g, f in AGG.items():
                acc, D, mv, pb, agr = [], [], [], [], []
                for r in R:
                    wr, w1 = r["w"][w], r["w"]["W1"]
                    for b in range(len(wr["id"]) // K):
                        s = slice(b * K, (b + 1) * K)
                        d = f(r, wr["id"][s], wr["lq"][s]); d1 = f(r, w1["id"][s], w1["lq"][s])
                        acc.append(top(d) == r["answer"]); D.append(tv(d, r["T"])); mv.append(tv(d, d1))
                        pb.append((d or {}).get(r["biased_answer"], 0)); agr.append(top(d) == top(r["T"]))
                res["cells"]["%s|%s|%s|%d" % (fam, w, g, K)] = {"acc": float(np.mean(acc)), "D": float(np.mean(D)),
                    "move": float(np.mean(mv)), "p_biased": float(np.mean(pb)), "top_is_ref_top": float(np.mean(agr))}
            wd, ce, cr, ct, fm, lo_v, hi_v, nd = [], [], [], [], [], [], [], []
            for r in R:
                wr = r["w"][w]
                for b in range(len(wr["id"]) // K):
                    RF = found(r, wr["id"][b * K:(b + 1) * K]); mF = sum(RF.values()); a = top(RF)
                    fm.append(mF); wd.append(1 - mF); c = a is not None and RF[a] > 0.5; ce.append(c)
                    nd.append(len(set(i for i in wr["id"][b * K:(b + 1) * K] if i >= 0)))
                    if c:
                        cr.append(a == r["answer"]); ct.append(a == top(r["T"]))
                    for ans, p in (r["T"] or {}).items():
                        lo = RF.get(ans, 0.0); lo_v.append(max(0.0, lo - p)); hi_v.append(max(0.0, p - (1 - (mF - lo))))
            res["cells"]["%s|%s|interval|%d" % (fam, w, K)] = {"found": float(np.mean(fm)), "width": float(np.mean(wd)),
                "certified": float(np.mean(ce)), "certified_right": float(np.mean(cr)) if cr else None,
                "certified_is_ref_top": float(np.mean(ct)) if ct else None, "distinct_valid": float(np.mean(nd)),
                "max_violation_low": float(max(lo_v)), "max_violation_high": float(max(hi_v)),
                "share_violation_gt_0.03": float(np.mean([max(a, b) > 0.03 for a, b in zip(lo_v, hi_v)]))}
json.dump(res, open(OUTP, "w"), indent=1)
c = lambda f, w, g, K: res["cells"]["%s|%s|%s|%d" % (f, w, g, K)]
print("n", len(recs), "exact-answer mismatches", bad_exact, "| hider available on", res["HID_available"])
for fam, F in res["fam"].items():
    print("\n==", fam, json.dumps({k: round(v, 4) for k, v in F.items()}))
    K = min(64, KMAX); fl = {g: c(fam, "W1b", g, K)["move"] for g in AGG}; a1 = c(fam, "W1", "vote", K)["acc"]
    print("floor", {g: round(v, 3) for g, v in fl.items()}, "bar16 %.3f bar4 %.3f" % (c(fam, "W1", "vote", 16)["D"], c(fam, "W1", "vote", 4)["D"]))
    ws = [w for w in writers if w != "W1b"] + (["HID"] if "%s|HID|vote|%d" % (fam, K) in res["cells"] else [])
    print("writer | acc v/d/s @%d | excess move v/d/s | D v/d/s | top=ref top d/s | found width cert cert=reftop cert_right distinct" % K)
    for w in ws:
        x = {g: c(fam, w, g, K) for g in AGG}; it = c(fam, w, "interval", K)
        print("%-4s | %.3f %.3f %.3f | %.3f %.3f %.3f | %.3f %.3f %.3f | %.3f %.3f | %.2f %.2f %.2f %s %s %.1f" % (
            w, x["vote"]["acc"], x["dedup"]["acc"], x["snis"]["acc"], x["vote"]["move"] - fl["vote"], x["dedup"]["move"] - fl["dedup"],
            x["snis"]["move"] - fl["snis"], x["vote"]["D"], x["dedup"]["D"], x["snis"]["D"], x["dedup"]["top_is_ref_top"],
            x["snis"]["top_is_ref_top"], it["found"], it["width"], it["certified"], it["certified_is_ref_top"], it["certified_right"], it["distinct_valid"]))
    for w in ws:
        print("  %-4s D by K %s: " % (w, KS) + " ; ".join(g + " " + " ".join("%.3f" % c(fam, w, g, k)["D"] for k in KS) for g in AGG)
              + " | acc " + " ; ".join(g + " " + " ".join("%.3f" % c(fam, w, g, k)["acc"] for k in KS) for g in AGG))
    print("  interval violations (max low, max high, share>0.03) @16:", {w: (round(c(fam, w, "interval", 16)["max_violation_low"], 3),
          round(c(fam, w, "interval", 16)["max_violation_high"], 3), round(c(fam, w, "interval", 16)["share_violation_gt_0.03"], 4)) for w in ws})
