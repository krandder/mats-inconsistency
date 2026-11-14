# audit.py ROLLOUTS.jsonl [CALIB.jsonl] : disjointness, counting baseline, simulated hider, trust head (checker's code).
# tasks.py (the work's problem generator) is imported only to regenerate the tuning / shot / pilot problem texts.
import sys, json, math
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, "/home/agent/workspaces/mats/work/deep-study/wf13/llm-pilot/code")
import tasks

recs = [json.loads(l) for l in open(sys.argv[1])]
cal = [json.loads(l) for l in open(sys.argv[2])] if len(sys.argv) > 2 else None
out = {}
# 1. disjointness by text, and a counting baseline (most frequent tuning answer; most frequent answer among the tuning
#    problems that share the last statement)
for fam, n in (("arithL", 3), ("vars", 4)):
    tr = tasks.gen(fam, 100, 1500, n)
    other = set(p["text"] for p in tr) | set(p["text"] for p in tasks.shots(fam, n, 4, 9001) + tasks.shots(fam, n, 2, 9002))
    other |= set(p["text"] for p in tasks.gen(fam, 7, 48, n))
    T = [r for r in recs if r["fam"] == fam]
    maj = Counter(p["answer"] for p in tr).most_common(1)[0][0]
    last = defaultdict(Counter)
    for p in tr:
        last[p["text"].split(";")[-2 if fam == "vars" else -1].strip()][p["answer"]] += 1
    hit = 0
    for r in T:
        k = r["text"].split(";")[-2 if fam == "vars" else -1].strip()
        hit += (last[k].most_common(1)[0][0] if k in last else maj) == r["answer"]
    out[fam] = {"n": len(T), "texts_seen_in_tuning_shots_pilot": sum(r["text"] in other for r in T),
                "duplicate_texts_within_set": len(T) - len(set(r["text"] for r in T)),
                "counting_majority_answer_acc": float(np.mean([r["answer"] == maj for r in T])),
                "counting_same_last_statement_acc": hit / len(T)}
if cal:
    out["calib_texts_in_test"] = len(set(r["text"] for r in cal) & set(r["text"] for r in recs))

def dedup(r, ids):
    d = defaultdict(float)
    for i in set(i for i in ids if i >= 0):
        d[r["dans"][i]] += math.exp(r["dlr"][i])
    s = sum(d.values())
    return ({a: v / s for a, v in d.items()} if s > 0 else None), s
def vote(r, ids):
    c = Counter(r["dans"][i] for i in ids if i >= 0); s = sum(c.values())
    return {a: v / s for a, v in c.items()} if s else None
def snis(r, ids, lq):
    lw = [(r["dlr"][i] - q, i) for i, q in zip(ids, lq) if i >= 0]
    if not lw:
        return None
    mx = max(x for x, _ in lw); d = defaultdict(float)
    for x, i in lw:
        d[r["dans"][i]] += math.exp(x - mx)
    s = sum(d.values())
    return {a: v / s for a, v in d.items()}
top = lambda p: max(p, key=lambda a: (p[a], a)) if p else None
tv = lambda p, q: 1.0 if (p is None or q is None) else 0.5 * sum(abs(p.get(a, 0) - q.get(a, 0)) for a in set(p) | set(q))

# 2. simulated hider at K = 16: M's own rollouts (W1 pool then reference pool) with M's top-scoring line removed.
#    Its probability of a line is M's divided by (1 - p_top): an exact rejection sampler, nothing trained.
KH = 16
for fam in ("arithL", "vars"):
    rows = []
    for r in recs:
        if r["fam"] != fam:
            continue
        T = vote(r, r["ref"]["id"])
        pool_i = r["ref"]["id"]; pool_q = r["ref"]["lq"]
        mids = [i for i in r["w"]["W1"]["id"] + pool_i if i >= 0]
        if not mids or T is None:
            continue
        t = max(set(mids), key=lambda i: r["dlr"][i]); pt = math.exp(r["dlr"][t])
        keep = [(i, q - math.log(max(1 - pt, 1e-9))) for i, q in zip(pool_i, pool_q) if i != t]
        if len(keep) < KH:
            continue
        ids = [i for i, _ in keep[:KH]]; lq = [q for _, q in keep[:KH]]
        d, mF = dedup(r, ids); s = snis(r, ids, lq)
        d1, m1 = dedup(r, r["w"]["W1"]["id"][:KH])
        RF = defaultdict(float)
        for i in set(i for i in ids if i >= 0):
            RF[r["dans"][i]] += math.exp(r["dlr"][i])
        rows.append({"p_top": pt, "top_line_answer_is_ref_top": r["dans"][t] == top(T),
                     "dedup_top_is_ref_top": top(d) == top(T), "snis_top_is_ref_top": top(s) == top(T),
                     "dedup_right": top(d) == r["answer"], "snis_right": top(s) == r["answer"], "W1_dedup_right": top(d1) == r["answer"],
                     "D_dedup": tv(d, T), "D_snis": tv(s, T), "D_W1_dedup": tv(d1, T), "found": mF, "found_W1": m1,
                     "certified": bool(RF) and max(RF.values()) > 0.5,
                     "dedup_top_prob": max(d.values()) if d else 0.0})
    m = lambda k: float(np.mean([x[k] for x in rows])) if rows else None
    out["hider|" + fam] = {"problems_with_16_non_top_rollouts": len(rows), **{k: m(k) for k in (rows[0] if rows else {})}}

# 3. trust head (my fit): logistic on [1, logit(top probability), family], fitted on calibration problems with W1, K = 16
if cal:
    K = 16
    lg = lambda p: math.log(min(max(p, 1e-4), 1 - 1e-4) / (1 - min(max(p, 1e-4), 1 - 1e-4)))
    def rows_of(R, w, g):
        X, y, raw = [], [], []
        for r in R:
            wr = r["w"][w]
            for b in range(len(wr["id"]) // K):
                ids = wr["id"][b * K:(b + 1) * K]; lq = wr["lq"][b * K:(b + 1) * K]
                d = vote(r, ids) if g == "vote" else (dedup(r, ids)[0] if g == "dedup" else snis(r, ids, lq))
                p = max(d.values()) if d else 0.0
                X.append([1.0, lg(p), float(r["fam"] == "vars")]); y.append(float(d is not None and top(d) == r["answer"])); raw.append(p)
        return np.array(X), np.array(y), np.array(raw)
    def fit(X, y, l2=1e-3):
        w = np.zeros(X.shape[1])
        for _ in range(100):
            p = 1 / (1 + np.exp(-X @ w))
            w -= np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w)), X.T @ (p - y) + l2 * w)
        return w
    out["trust"] = {}
    for g in ("vote", "dedup", "snis"):
        Xc, yc, _ = rows_of(cal, "W1", g); w = fit(Xc, yc)
        for wn in [x for x in recs[0]["w"] if x != "W1b"]:
            Xt, yt, raw = rows_of(recs, wn, g); p = 1 / (1 + np.exp(-Xt @ w))
            out["trust"]["%s|%s" % (g, wn)] = {"acc": float(yt.mean()), "brier_raw": float(((raw - yt) ** 2).mean()),
                                               "brier_trust": float(((p - yt) ** 2).mean()), "mean_conf_trust": float(p.mean())}
print(json.dumps(out, indent=1))
