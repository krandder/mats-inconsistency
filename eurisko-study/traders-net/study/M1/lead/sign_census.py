"""M1 retarget (b): how often do oracle-important units carry negative gradient credit?

Read-only on existing files. Oracle = exact single-unit zero ablation dBCE on held-out data
(positive = the unit helps). Sources:
  WN  wealth-net results: decay_oracle_s*.npy (neurons, final ckpt), decay_currencies_s*.npz
      (all-time books wealth 'cum' and decayed variants), settle_currencies_s*_step<final>.npz
      (exact leave-one-out settlement, contrast), train-s*/snapshots/step-*.json (all-time wealth
      of heads and neurons at the w4 checkpoints)
  W4  M1/w4/results/s*-step*.json: held-out instantaneous income under live/v1/cp/attnlrp and
      exact ablation of neurons and heads at 8k/16k/24k/30k
  L   M1/lead/results/three-s*-step*.json (three-rules books), if present
Writes lead/results/sign_census_rows.csv (one row per seed x source x currency x unit set) and
lead/results/sign_census_summary.csv (mean +- sd over seeds).
"""
import csv, glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WN = "/home/agent/workspaces/eurisko-study/wealth-net/results"
W4 = os.path.join(HERE, "..", "w4", "results")
OUT = os.path.join(HERE, "results")
FINAL = {1: 31100, 2: 31200, 3: 31100}


def stats(credit, oracle, kind):
    credit = np.asarray(credit, float).ravel(); oracle = np.asarray(oracle, float).ravel()
    n = oracle.size
    order = np.argsort(-oracle)
    sets = {}
    if kind == "neuron":
        sets["top5pct"] = order[:max(1, int(round(0.05 * n)))]
        sets["top10pct"] = order[:max(1, int(round(0.10 * n)))]
        sets["dBCE>0.01"] = np.where(oracle > 0.01)[0]
    else:
        sets["top2"] = order[:2]
        sets["top4"] = order[:4]
    sets["all"] = np.arange(n)
    sets["harmful(dBCE<0)"] = np.where(oracle < 0)[0]
    out = {}
    for k, idx in sets.items():
        out[k] = dict(n=int(idx.size), frac_neg=float((credit[idx] < 0).mean()) if idx.size else float("nan"))
    return out


def rows_for(seed, source, currency, kind, step, credit, oracle):
    r = []
    for unit_set, v in stats(credit, oracle, kind).items():
        r.append(dict(seed=seed, source=source, currency=currency, unit=kind, step=step, unit_set=unit_set,
                      n=v["n"], frac_neg=round(v["frac_neg"], 4)))
    return r


def main():
    rows = []
    for s in (1, 2, 3):
        orc = np.load(f"{WN}/decay_oracle_s{s}.npy")
        cur = np.load(f"{WN}/decay_currencies_s{s}.npz")
        for c in ("cum", "hl1000", "hl10000", "win2000"):
            rows += rows_for(s, "wealth-net final", f"books_{c}", "neuron", FINAL[s], cur[c], orc)
        st = np.load(f"{WN}/settle_currencies_s{s}_step{FINAL[s]}.npz")
        for c in ("loo_train", "loo_heldout"):
            rows += rows_for(s, "wealth-net final", f"settle_{c}", "neuron", FINAL[s], st[c], orc)
        for fn in sorted(glob.glob(f"{W4}/s{s}-step*.json")):
            d = json.load(open(fn)); step = d["step"]
            an = np.array(d["ablation"]["neuron"]).ravel(); ah = np.array(d["ablation"]["head"]).ravel()
            fx = os.path.join(OUT, f"w4fix-s{s}-step{step}.json")   # corrected aggregation (see w4_fix.py)
            src, rr = ("w4 held-out income (re-aggregated)", json.load(open(fx))["rules"]) if os.path.exists(fx) else ("w4 held-out income (as written)", d["rules"])
            for rn in ("live", "v1", "cp", "attnlrp"):
                rows += rows_for(s, src, rn, "neuron", step, rr[rn]["neuron"], an)
                rows += rows_for(s, src, rn, "head", step, rr[rn]["head"], ah)
            snap = json.load(open(f"{WN}/train-s{s}/snapshots/step-{step:05d}.json"))
            rows += rows_for(s, "all-time books wealth", "books_cum", "neuron", step, snap["wealth"]["neuron"], an)
            rows += rows_for(s, "all-time books wealth", "books_cum", "head", step, snap["wealth"]["head"], ah)
            for tag, src in (("three", "three-rules held-out tangent"), ("three-lmsr", "three-rules held-out lmsr")):
                lf = os.path.join(OUT, f"{tag}-s{s}-step{step}.json")
                if os.path.exists(lf):
                    R = json.load(open(lf))["rules"]
                    t = R["three16"]
                    rows += rows_for(s, src, "three_total", "head", step, t["head"], ah)
                    rows += rows_for(s, src, "three_route", "head", step, t["head_route"], ah)
                    rows += rows_for(s, src, "three_value", "head", step, t["head_value"], ah)
                    rows += rows_for(s, src, "three_total", "neuron", step, t["neuron"], an)
                    rows += rows_for(s, src, "cp", "neuron", step, R["cp"]["neuron"], an)
                    rows += rows_for(s, src, "cp", "head", step, R["cp"]["head"], ah)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/sign_census_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # summary: per seed average over checkpoints first, then mean +- sd over seeds
    keys = sorted({(r["source"], r["currency"], r["unit"], r["unit_set"]) for r in rows})
    summ = []
    for k in keys:
        per_seed = []
        for s in (1, 2, 3):
            v = [r["frac_neg"] for r in rows if (r["source"], r["currency"], r["unit"], r["unit_set"]) == k and r["seed"] == s]
            v = [x for x in v if x == x]
            if v:
                per_seed.append(np.mean(v))
        ns = [r["n"] for r in rows if (r["source"], r["currency"], r["unit"], r["unit_set"]) == k]
        summ.append(dict(source=k[0], currency=k[1], unit=k[2], unit_set=k[3], n_seeds=len(per_seed),
                         n_units_mean=round(float(np.mean(ns)), 1),
                         frac_neg_mean=round(float(np.mean(per_seed)), 3),
                         frac_neg_sd=round(float(np.std(per_seed, ddof=1)), 3) if len(per_seed) > 1 else "",
                         per_seed=" ".join(f"{x:.3f}" for x in per_seed)))
    with open(f"{OUT}/sign_census_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    for r in summ:
        if r["unit_set"] in ("top5pct", "top2", "all", "dBCE>0.01"):
            print(r["source"], r["currency"], r["unit"], r["unit_set"], r["n_units_mean"], r["frac_neg_mean"], r["frac_neg_sd"])


if __name__ == "__main__":
    main()
