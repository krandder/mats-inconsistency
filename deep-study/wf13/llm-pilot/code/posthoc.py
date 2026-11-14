# posthoc.py OUT_DIR : descriptive extras for NOTE.md (NOT pre-registered; labelled "post hoc" in the note).
# (1) one worked example; (2) token positions per arm; (3) the share of M's score that K rollouts find, and the bound
# it gives on the distance of the dedup-score aggregation to the target; (4) rank agreement for H6.
import sys, json, math
import numpy as np
from analyze import load, agg, tv, blocks

OUT = sys.argv[1]
test = load("test")
W = list(test[0]["w"].keys())
res = {"example": {}, "compute": {}, "found": {}, "h6": {}}
for r in test:
    r["ref_dist"] = agg(r, r["ref"], range(len(r["ref"]["id"])), "vote")[0]

# (1) first problem of each family whose reference probability of the right answer is between 0.4 and 0.8
for fam in ("arithL", "vars"):
    for r in test:
        p = (r["ref_dist"] or {}).get(r["answer"], 0.0)
        if r["fam"] == fam and 0.4 <= p <= 0.8:
            order = sorted(range(len(r["dlr"])), key=lambda i: -r["dlr"][i])[:5]
            ex = {"problem": r["text"], "answer": r["answer"], "biased_answer": r["biased_answer"], "ref_p_true": p,
                  "top_by_score": [{"demo": r["demos"][i], "answer": r["dans"][i], "R": math.exp(r["dlr"][i])} for i in order], "writers": {}}
            for w in ("W1", "W3", "W3s", "W4"):
                ids = [i for i in r["w"][w]["id"] if i >= 0]
                if not ids:
                    continue
                m = max(set(ids), key=ids.count); k = r["w"][w]["id"].index(m)
                ex["writers"][w] = {"most_frequent_demo": r["demos"][m], "times_of_64": ids.count(m), "answer": r["dans"][m],
                                    "R": math.exp(r["dlr"][m]), "q": math.exp(r["w"][w]["lq"][k]),
                                    "vote": agg(r, r["w"][w], range(64), "vote")[0], "dedup": agg(r, r["w"][w], range(64), "dedup")[0],
                                    "snis": agg(r, r["w"][w], range(64), "snis")[0]}
            res["example"][fam] = ex
            break

for fam in ("arithL", "vars"):
    T = [r for r in test if r["fam"] == fam]
    # (2) compute: token positions of 64 rollouts, and of M's scoring pass over that writer's distinct valid demonstrations
    for w in W:
        roll = np.mean([r["tp"][w] for r in T])
        sc = []
        for r in T:
            seen = {}
            for i, n in zip(r["w"][w]["id"], r["w"][w]["nt"]):
                if i >= 0:
                    seen[i] = n
            sc.append(sum(seen.values()))
        prompt = np.mean([r["tp"][w] - sum(r["w"][w]["nt"]) for r in T])
        res["compute"]["%s|%s" % (fam, w)] = {"rollout_positions_per_problem": float(roll), "of_which_prompt_approx": float(prompt),
                                              "score_pass_positions_distinct_valid": float(np.mean(sc)),
                                              "score_over_rollout": float(np.mean(sc) / roll)}
    res["compute"]["%s|direct_one_pass" % fam] = "one pass over the direct prompt (about the prompt length) and 1-4 answer tokens"
    # (3) found mass and the bound: TV(dedup, target) <= 1 - found / Z, Z = M's probability of a valid demonstration
    Z = [np.mean([i >= 0 for i in r["ref"]["id"]]) for r in T]
    for w in W:
        for K in (1, 4, 16, 64):
            fm, d, bound = [], [], []
            for r, z in zip(T, Z):
                for sel in blocks(64, K):
                    dist, found = agg(r, r["w"][w], sel, "dedup")
                    fm.append(found or 0.0); d.append(tv(dist, r["ref_dist"])); bound.append(max(0.0, 1 - (found or 0.0) / max(z, 1e-9)))
            res["found"]["%s|%s|%d" % (fam, w, K)] = {"found_mass": float(np.mean(fm)), "Z_valid": float(np.mean(Z)),
                                                      "D_dedup": float(np.mean(d)), "bound_1_minus_found_over_Z": float(np.mean(bound))}
    # (4) H6: rank agreement between the divergence and the distance of SNIS at K = 64
    R = json.load(open(OUT + "/RESULT-work.json"))
    xs = [R["div"]["%s|%s" % (fam, w)]["E_M[log R - log q]"] for w in W]
    ys = [R["curves"]["%s|%s|snis|64" % (fam, w)]["tv_ref"] for w in W]
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    res["h6"][fam] = {"writers": W, "divergence": xs, "D_snis_64": ys, "spearman": float(np.corrcoef(rk(xs), rk(ys))[0, 1])}
# (5) an interval from the score alone, valid for any writer. With F = the distinct valid demonstrations found,
# m_F = their total score and R_F(a) = the part of it with answer a, the target's probability of a satisfies
#   R_F(a) <= T(a) <= 1 - (m_F - R_F(a))      (because the valid mass Z obeys m_F <= Z <= 1).
# Width = 1 - m_F. An answer with R_F(a) > 1/2 is certified as the target's majority answer.
res["interval"] = {}
for fam in ("arithL", "vars"):
    T = [r for r in test if r["fam"] == fam]
    for w in W:
        for K in (4, 16, 64):
            width, cert, cert_right, cert_is_ref_top, viol, cover = [], [], [], [], [], []
            for r in T:
                reft = max(r["ref_dist"], key=lambda k: (r["ref_dist"][k], k)) if r["ref_dist"] else None
                for sel in blocks(64, K):
                    seen = set(i for i in (r["w"][w]["id"][k] for k in sel) if i >= 0)
                    RF = {}
                    for i in seen:
                        RF[r["dans"][i]] = RF.get(r["dans"][i], 0.0) + math.exp(r["dlr"][i])
                    mF = sum(RF.values()); width.append(1 - mF)
                    a = max(RF, key=lambda k: (RF[k], k)) if RF else None
                    c = a is not None and RF[a] > 0.5
                    cert.append(c)
                    if c:
                        cert_right.append(a == r["answer"]); cert_is_ref_top.append(a == reft)
                    for ans, p in (r["ref_dist"] or {}).items():
                        lo = RF.get(ans, 0.0); hi = 1 - (mF - lo)
                        v = max(0.0, lo - p, p - hi); viol.append(v); cover.append(v <= 0.03)
            res["interval"]["%s|%s|%d" % (fam, w, K)] = {
                "mean_width": float(np.mean(width)), "certified_share": float(np.mean(cert)),
                "certified_right": float(np.mean(cert_right)) if cert_right else None,
                "certified_is_reference_top": float(np.mean(cert_is_ref_top)) if cert_is_ref_top else None,
                "mean_violation_vs_reference": float(np.mean(viol)), "share_within_0.03": float(np.mean(cover))}
json.dump(res, open(OUT + "/RESULT-posthoc.json", "w"), indent=1)
print("written", OUT + "/RESULT-posthoc.json")
