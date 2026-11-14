# analyze.py OUT_DIR : evaluation code (CPU). Reads OUT_DIR/test.jsonl (and calib.jsonl for the trust head),
# builds the three aggregations from the first K rollouts of each writer, and writes OUT_DIR/RESULT-work.json.
# The exact answers are used here and nowhere else.
import sys, json, math, random, collections
import numpy as np

OUT = sys.argv[1]
SETNAME = sys.argv[2] if len(sys.argv) > 2 else "test"      # "ext" = the registered K-extension (PREREG-ADDENDUM.md)
KMAX = 256 if SETNAME == "ext" else 64
KS = [k for k in [1, 2, 4, 8, 16, 32, 64, 128, 256] if k <= KMAX]
AGG = ["vote", "dedup", "snis"]

def load(name):
    return [json.loads(l) for l in open("%s/%s.jsonl" % (OUT, name))]

def agg(rec, wr, sel, kind):
    """Answer distribution (dict) from the rollouts `sel` of writer record wr. None = abstain (no valid rollout)."""
    ids = [wr["id"][k] for k in sel]
    if kind == "vote":
        c = collections.Counter(rec["dans"][i] for i in ids if i >= 0)
        tot = sum(c.values())
        return ({a: v / tot for a, v in c.items()} if tot else None), None
    if kind == "dedup":
        seen = []
        for i in ids:
            if i >= 0 and i not in seen:
                seen.append(i)
        if not seen:
            return None, None
        m = max(rec["dlr"][i] for i in seen); d = collections.defaultdict(float)
        for i in seen:
            d[rec["dans"][i]] += math.exp(rec["dlr"][i] - m)
        tot = sum(d.values())
        return {a: v / tot for a, v in d.items()}, sum(math.exp(rec["dlr"][i]) for i in seen)  # found mass
    lw = [(rec["dlr"][i] - wr["lq"][k], i) for k, i in zip(sel, ids) if i >= 0]
    if not lw:
        return None, None
    m = max(x for x, _ in lw); d = collections.defaultdict(float); w = []
    for x, i in lw:
        e = math.exp(x - m); d[rec["dans"][i]] += e; w.append(e)
    tot = sum(w)
    return {a: v / tot for a, v in d.items()}, tot * tot / sum(x * x for x in w) / len(sel)  # ESS / K

def tv(p, q):
    if p is None and q is None:
        return 0.0
    if p is None or q is None:
        return 1.0
    return 0.5 * sum(abs(p.get(a, 0.0) - q.get(a, 0.0)) for a in set(p) | set(q))

def top(p):
    if p is None:
        return None, 0.0
    a = max(p, key=lambda k: (p[k], k))  # ties: fixed rule on the answer string, never on the exact answer
    return a, p[a]

def blocks(n, K):
    return [list(range(b * K, (b + 1) * K)) for b in range(n // K)]

def boot(vals, n=400, seed=0):
    r = np.random.default_rng(seed); v = np.asarray(vals, float)
    if len(v) == 0:
        return [float("nan")] * 2
    m = [v[r.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

def main():
    test = load(SETNAME)
    writers = list(test[0]["w"].keys())
    fams = sorted(set(r["fam"] for r in test))
    res = {"n_test": len(test), "writers": writers, "families": fams, "curves": {}, "direct": {}, "div": {}, "ref": {}}
    for r in test:
        r["ref_dist"] = agg(r, r["ref"], range(len(r["ref"]["id"])), "vote")[0]
        r["w"]["W1b"] = {k: v[:KMAX] for k, v in r["ref"].items()}   # an independent pool of M rollouts (noise floor)
    for fam in fams + ["all"]:
        T = [r for r in test if fam == "all" or r["fam"] == fam]
        # the reference target: vote of 256 independent rollouts of M
        res["ref"][fam] = {"acc": float(np.mean([top(r["ref_dist"])[0] == r["answer"] for r in T])),
                           "p_true": float(np.mean([(r["ref_dist"] or {}).get(r["answer"], 0.0) for r in T])),
                           "p_biased": float(np.mean([(r["ref_dist"] or {}).get(r["biased_answer"], 0.0) for r in T])),
                           "valid": float(np.mean([np.mean([i >= 0 for i in r["ref"]["id"]]) for r in T])),
                           "n": len(T)}
        res["direct"][fam] = {"greedy_acc": float(np.mean([r["direct"]["greedy"] == r["answer"] for r in T])),
                              "greedy_acc_ci": boot([r["direct"]["greedy"] == r["answer"] for r in T]),
                              "sample_acc": float(np.mean([np.mean([s == r["answer"] for s in r["direct"]["samples"]]) for r in T]))}
        for w in writers + ["W1b"]:
            for g in AGG:
                for K in KS:
                    acc, tvr, mv, pb, pt, aux, ab = [], [], [], [], [], [], []
                    for r in T:
                        a_, t_, m_, b_, p_, x_, n_ = [], [], [], [], [], [], []
                        for sel in blocks(len(r["w"][w]["id"]), K):
                            d, extra = agg(r, r["w"][w], sel, g)
                            d1, _ = agg(r, r["w"]["W1"], sel, g)
                            a_.append(top(d)[0] == r["answer"]); t_.append(tv(d, r["ref_dist"])); m_.append(tv(d, d1))
                            b_.append((d or {}).get(r["biased_answer"], 0.0)); p_.append((d or {}).get(r["answer"], 0.0))
                            n_.append(d is None)
                            if extra is not None:
                                x_.append(extra)
                        acc.append(np.mean(a_)); tvr.append(np.mean(t_)); mv.append(np.mean(m_)); pb.append(np.mean(b_))
                        pt.append(np.mean(p_)); ab.append(np.mean(n_))
                        if x_:
                            aux.append(np.mean(x_))
                    res["curves"]["%s|%s|%s|%d" % (fam, w, g, K)] = {
                        "acc": float(np.mean(acc)), "acc_ci": boot(acc), "tv_ref": float(np.mean(tvr)), "tv_ref_ci": boot(tvr),
                        "move_vs_W1": float(np.mean(mv)), "move_ci": boot(mv), "p_biased": float(np.mean(pb)),
                        "p_true": float(np.mean(pt)), "abstain": float(np.mean(ab)),
                        "aux": float(np.mean(aux)) if aux else None}
        # divergences between writers, valid rates, tokens
        for w in writers:
            fw, bw, nt, val, ndist = [], [], [], [], []
            for r in T:
                wr = r["w"][w]
                val.append(np.mean([i >= 0 for i in wr["id"]]))
                ndist.append(len(set(i for i in wr["id"] if i >= 0)))
                nt += [n for n, i in zip(wr["nt"], wr["id"]) if i >= 0]
                bw += [wr["lq"][k] - r["dlr"][i] for k, i in enumerate(wr["id"]) if i >= 0]          # log q/R on q's samples
                if w in r["kl"] and r["kl"][w]:
                    w1 = r["w"]["W1"]
                    fw += [r["dlr"][i] - x for x, i in zip(r["kl"][w], w1["id"]) if i >= 0 and x is not None]  # log R/q on M's
            res["div"]["%s|%s" % (fam, w)] = {
                "valid_rate": float(np.mean(val)), "distinct_valid_of_64": float(np.mean(ndist)), "tokens_per_demo": float(np.mean(nt)),
                "E_M[log R - log q]": float(np.mean(fw)) if fw else None, "E_M[log R - log q]_median": float(np.median(fw)) if fw else None,
                "E_q[log q - log R]": float(np.mean(bw)), "share_M_demos_with_logRq_gt_3": float(np.mean([x > 3 for x in fw])) if fw else None,
                "token_positions_per_problem": float(np.mean([r["tp"][w] for r in T]))}
        res["div"]["%s|Mscore_positions" % fam] = float(np.mean([r["tp"]["Mscore"] for r in T]))
    # sanity: for W1 the sampler's own log-probability must equal M's score
    d = [abs(r["w"]["W1"]["lq"][k] - r["dlr"][i]) for r in test for k, i in enumerate(r["w"]["W1"]["id"]) if i >= 0]
    res["sanity_W1_abs_logq_minus_score"] = {"mean": float(np.mean(d)), "p99": float(np.percentile(d, 99)), "max": float(np.max(d))}
    name = "/RESULT-work.json" if SETNAME == "test" else "/RESULT-%s.json" % SETNAME
    json.dump(res, open(OUT + name, "w"), indent=1)
    print("written", OUT + name)

if __name__ == "__main__":
    main()
