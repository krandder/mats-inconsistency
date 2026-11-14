# selfcheck.py OUT_DIR : an independent recomputation (fresh code, numpy only) of a few headline numbers of
# RESULT-work.json from the stored rollouts. Prints both; any disagreement beyond 1e-9 is flagged.
import sys, json, math
from collections import Counter, defaultdict
OUT = sys.argv[1]
recs = [json.loads(l) for l in open(OUT + "/test.jsonl")]
R = json.load(open(OUT + "/RESULT-work.json"))["curves"]
def dist_vote(r, ids):
    c = Counter(r["dans"][i] for i in ids if i >= 0); n = sum(c.values())
    return {a: v / n for a, v in c.items()} if n else None
def dist_dedup(r, ids):
    d = defaultdict(float)
    for i in set(i for i in ids if i >= 0):
        d[r["dans"][i]] += math.exp(r["dlr"][i])
    s = sum(d.values()); return {a: v / s for a, v in d.items()} if s else None
def dist_snis(r, ids, lq):
    d = defaultdict(float)
    for i, q in zip(ids, lq):
        if i >= 0:
            d[r["dans"][i]] += math.exp(r["dlr"][i] - q)
    s = sum(d.values()); return {a: v / s for a, v in d.items()} if s else None
def tv(p, q):
    if p is None or q is None:
        return 0.0 if p is q else 1.0
    return 0.5 * sum(abs(p.get(a, 0) - q.get(a, 0)) for a in set(p) | set(q))
bad = 0
for fam in ("arithL", "vars"):
    T = [r for r in recs if r["fam"] == fam]
    for w in ("W1", "W3", "W3s", "W4"):
        for g in ("vote", "dedup", "snis"):
            acc = d = 0.0
            for r in T:
                ids = r["w"][w]["id"]; ref = dist_vote(r, r["ref"]["id"])
                p = dist_vote(r, ids) if g == "vote" else dist_dedup(r, ids) if g == "dedup" else dist_snis(r, ids, r["w"][w]["lq"])
                top = max(p, key=lambda k: (p[k], k)) if p else None
                acc += top == r["answer"]; d += tv(p, ref)
            acc /= len(T); d /= len(T)
            c = R["%s|%s|%s|64" % (fam, w, g)]
            ok = abs(acc - c["acc"]) < 1e-9 and abs(d - c["tv_ref"]) < 1e-9
            bad += not ok
            print("%s %s %s acc %.4f (%.4f) D %.4f (%.4f) %s" % (fam, w, g, acc, c["acc"], d, c["tv_ref"], "ok" if ok else "MISMATCH"))
print("mismatches:", bad)
# the generator's exact answers, re-derived from the problem text alone by a separate evaluator
wrong = 0; biased_eq = 0
for r in recs + [json.loads(l) for l in open(OUT + "/calib.jsonl")]:
    parts = r["text"].split("; ")
    if r["fam"] == "arithL":
        v = int(parts[0])
        for p in parts[1:]:
            v = v * int(p[1:]) if p[0] == "*" else v + int(p[1:]) if p[0] == "+" else v - int(p[1:])
        ans = str(v)
    else:
        env = {k: int(x) for k, x in (t.split("=") for t in parts[0].split(" "))}
        for p in parts[1:-1]:
            x, rhs = p.split("="); env[x] = eval(rhs, {}, dict(env))
        ans = str(env[parts[-1][0]])
    wrong += ans != r["answer"]; biased_eq += r["biased_answer"] == r["answer"]
print("problems whose stored exact answer differs from the independent evaluator:", wrong,
      "| problems where the biased answer equals the right one:", biased_eq, "of", len(recs) + 300)
