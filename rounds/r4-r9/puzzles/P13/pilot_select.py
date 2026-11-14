"""Phase 2A selection (PREREG spec RUNAWAY-FIX PILOT + A5), from the synced pilot evals and training logs.
  - best post-hoc control per candidate: lowest m59 log loss at 12n among raw, clip {4,8,16}, tau grid; score = v33 log loss
    at 12n with that control (mean over mazes, nats per open pixel);
  - pool per arm: v9 exact >= 0.9 and v33 exact >= arm best - 0.02; outside the pool: rank = pool size + 1;
  - mean rank over ECON and ECON-nocap, lowest wins; two lowest within 0.5 -> higher mean m59 exact;
  - extension: fewer than 3 candidates in an arm reach v9 >= 0.9 -> extend all 14 to 1.0 h first;
  - learnability gate: selected fix reaches v9 >= 0.9 and m13 >= 0.5 in ECON or ECON-nocap (else extend it to 1.0 h);
  - ladder: ECON pilot of the selected fix has frac_capped_pix (s < 1) >= 0.10 over the last 25% of its iterations -> active.
Usage: pilot_select.py [--tag 0p5|1p0]  -> results/pilot_selection_<tag>.json"""
import glob, json, os, sys
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
FIXES = ["F0", "TR1", "TR2", "LB", "HF", "FL", "IN"]
ARMS = ["econ", "nocap"]
CAL = ["raw", "clip4", "clip8", "clip16"] + [f"tau{2 ** (k / 2):.3g}" for k in range(-2, 17)]


def find(name):
    g = sorted(glob.glob(os.path.join(D, "results", "vm", "*", "out", name)))
    return g[-1] if g else None


def iters_0p5(n):
    """the pilot's 0.5 h iteration count, from its queue line (23:28Z addendum: extensions resume into the same --out, whose
    training log is cumulative, so the 0.5 h selection reads that log truncated at this count)"""
    for q in glob.glob(os.path.join(D, "gpu", "queues", "*.txt")):
        for l in open(q):
            w = l.split()
            if w and w[0] == n and "--iters" in w:
                return int(w[w.index("--iters") + 1])
    return None


def ll_at(npz, s, t):
    ts = list(npz[f"{s}/t"])
    return npz[f"{s}/ll"][ts.index(t)].astype(np.float64).mean(0)       # (NCAL,) mean over mazes


def cand(arm, fix, suffix=""):
    n = f"pil_{arm}_{fix}"
    ej, en, tj = find(f"ev_{n}{suffix}.json"), find(f"ev_{n}{suffix}.npz"), find(f"{n}.json")
    if not (ej and en):
        return {"name": n, "missing": True}
    S = json.load(open(ej))["sets"]
    z = np.load(en)
    m59 = ll_at(z, "m59", S["m59"]["T12"])
    c = int(np.argmin(m59))
    v33 = ll_at(z, "v33", S["v33"]["T12"])
    r = {"name": n, "arm": arm, "fix": fix, "ctrl": CAL[c], "m59_ll_ctrl": float(m59[c]), "m59_ll_raw": float(m59[0]),
         "score_v33_ll": float(v33[c]), "v33_ll_raw": float(v33[0])}
    for s in ["v9", "m13", "v33", "m59"]:
        r[f"{s}_exact"] = S[s]["exact@12n"]
        r[f"{s}_maxb"] = S[s]["maxb@12n"]
    if tj:
        T = json.load(open(tj))
        if suffix == "":
            n0 = iters_0p5(n)
            if n0 is not None and max(l["it"] for l in T["log"]) >= n0:
                T["log"] = [l for l in T["log"] if l["it"] < n0]
                T["train_sec"] = None
                r["log_truncated_at"] = n0
        it = max(l["it"] for l in T["log"])
        dg = [l["diag"] for l in T["log"] if "diag" in l and l["it"] >= 0.75 * it and l["diag"].get("iters")]
        r["iters"] = it + 1
        r["train_sec"] = T.get("train_sec")
        if dg:
            r["frac_capped_last25"] = float(np.mean([d["frac_capped_pix"] for d in dg]))
            r["mean_s_last25"] = float(np.mean([d["mean_s"] for d in dg]))
            r["respawns"] = dg[-1]["respawns"]
    return r


def main():
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "0p5"
    suffix = "" if tag == "0p5" else "_x"
    C = {(a, f): cand(a, f, suffix) for a in ARMS for f in FIXES}
    missing = [c["name"] for c in C.values() if c.get("missing")]
    out = {"tag": tag, "missing": missing, "cands": list(C.values())}
    if missing:
        print(json.dumps(out, indent=1)); return out
    ranks = {}
    for a in ARMS:
        best33 = max(C[(a, f)]["v33_exact"] for f in FIXES)
        pool = [f for f in FIXES if C[(a, f)]["v9_exact"] >= 0.9 and C[(a, f)]["v33_exact"] >= best33 - 0.02]
        order = sorted(pool, key=lambda f: C[(a, f)]["score_v33_ll"])
        for f in FIXES:
            C[(a, f)]["in_pool"] = f in pool
            C[(a, f)]["rank"] = order.index(f) + 1 if f in pool else len(pool) + 1
        out[f"pool_{a}"] = pool
        out[f"n_v9_ge_0.9_{a}"] = sum(C[(a, f)]["v9_exact"] >= 0.9 for f in FIXES)
    mr = {f: np.mean([C[(a, f)]["rank"] for a in ARMS]) for f in FIXES}
    m59 = {f: np.mean([C[(a, f)]["m59_exact"] for a in ARMS]) for f in FIXES}
    order = sorted(FIXES, key=lambda f: (mr[f], -m59[f]))
    sel, tie = order[0], False
    if mr[order[1]] - mr[order[0]] <= 0.5:
        tie = True
        sel = max(order[:2], key=lambda f: (m59[f], -FIXES.index(f)))
    out.update(mean_rank=mr, mean_m59_exact=m59, order=order, tie_rule_used=tie, selected=sel)
    out["extend_all"] = tag == "0p5" and min(out[f"n_v9_ge_0.9_{a}"] for a in ARMS) < 3
    out["gate"] = any(C[(a, sel)]["v9_exact"] >= 0.9 and C[(a, sel)]["m13_exact"] >= 0.5 for a in ARMS)
    e = C[("econ", sel)]
    out["econ_sel_frac_capped_last25"] = e.get("frac_capped_last25")
    out["econ_fix_active"] = (e.get("frac_capped_last25") or 0) >= 0.10
    out["F2_recorded"] = sel == "F0"
    out["post_hoc_ctrl_selected"] = {a: C[(a, sel)]["ctrl"] for a in ARMS}
    json.dump(out, open(os.path.join(D, "results", f"pilot_selection_{tag}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "cands"}, indent=1))
    for c in out["cands"]:
        print(f'{c["name"]:18s} v9 {c["v9_exact"]:.3f} m13 {c["m13_exact"]:.3f} v33 {c["v33_exact"]:.3f} m59 {c["m59_exact"]:.3f} '
              f'score {c["score_v33_ll"]:.4f} ({c["ctrl"]}) raw33 {c["v33_ll_raw"]:.3g} maxb33 {c["v33_maxb"]:.3g} '
              f'rank {c["rank"]} capped {c.get("frac_capped_last25")}')
    return out


if __name__ == "__main__":
    main()
