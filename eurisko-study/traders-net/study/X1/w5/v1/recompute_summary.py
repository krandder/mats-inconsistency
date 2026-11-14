"""X1-w5 V-b: recompute V1/w1/summary-at-common-step.json three ways and compare (rel tol 1e-6).
 (a) V1/w1 CSV rows at the step; (b) live exporter metrics.json; (c) independent reader of raw snapshots
 (does not import serve.py). Also usable for fresh seeds: --snapdir DIR --seeds 4 5 prints (c) only.
Run: python3 recompute_summary.py > recompute_summary.json"""
import argparse, csv, io, json, math, pathlib, statistics as st, urllib.request, datetime as dt
TN = pathlib.Path("/home/agent/workspaces/eurisko-study/traders-net")
V1 = TN / "study/V1"; WN = TN.parent / "wealth-net/results"
FAM = ['periodic', 'lfsr16', 'ca30', 'ca110', 'prime', 'pi', 'and', 'or']
EXPORTER = "http://[ip-removed]:8767/runs/wealth-net.train-s{}/metrics.json"

def raw(snapdir, seed, step):
    s = json.load(open(pathlib.Path(snapdir) / f"train-s{seed}/snapshots/step-{step}.json"))
    w = s["wealth"]; o = {"step": s["step"], "R_total": s["R_total"]}
    for k in ["out", "head", "neuron", "comp"]:
        o[f"wealth_{k}"] = math.fsum(w[k])
        o[f"wf_{k}_pi"] = s["wealth_by_family"][k][FAM.index("pi")]
        o[f"gini_{k}"] = s["gini_pos"][k]
        o[f"neg_{k}"] = s["n_negative"][k] / len(w[k])
        o[f"churn_{k}"] = 1 - s["wealth_rank_corr_vs_1000_steps_ago"][k]
        o[f"turnover_{k}"] = s["turnover_income_rank_corr"][k]
    return o

def from_csv_text(text, step):
    for r in csv.DictReader(io.StringIO(text)):
        if r["step"] and float(r["step"]) == step:
            return {k: float(v) for k, v in r.items() if v not in ("", None)}

def agg(per_seed, keys):
    return {k: dict(seeds=[p[k] for p in per_seed], mean=st.mean(p[k] for p in per_seed),
                    sd=st.stdev(p[k] for p in per_seed)) for k in keys}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--snapdir"); ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--step", type=int, default=23300); a = ap.parse_args()
    if a.snapdir:
        print(json.dumps({s: raw(a.snapdir, s, a.step) for s in a.seeds}, indent=1)); raise SystemExit
    ref = json.load(open(V1 / "w1/summary-at-common-step.json")); keys = list(ref)
    step = 23300
    src = {"c_raw": [raw(WN, s, step) for s in (1, 2, 3)],
           "a_v1_csv": [from_csv_text((V1 / f"w1/wealth-net.train-s{s}.csv").read_text(), step) for s in (1, 2, 3)]}
    try:
        ex = []
        for s in (1, 2, 3):
            body = urllib.request.urlopen(EXPORTER.format(s), timeout=60).read().decode()
            try:
                j = json.loads(body); rows = j if isinstance(j, list) else j.get("rows", j.get("data"))
                row = next(r for r in rows if float(r["step"]) == step)
                ex.append({k: float(v) for k, v in row.items() if v not in ("", None)})
            except (ValueError, TypeError):
                ex.append(from_csv_text(body, step))
        src["b_exporter"] = ex
    except Exception as e:
        src["b_exporter_error"] = repr(e)
    out = {"checked_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "step": step, "compare": {}}
    for name, per in src.items():
        if not isinstance(per, list): out[name] = per; continue
        missing = [k for k in keys if k not in per[0]]
        g = agg(per, [k for k in keys if k not in missing])
        worst = 0.0; worst_k = None
        for k, v in g.items():
            for x, y in zip(v["seeds"], ref[k]["seeds"]):
                rel = abs(x - y) / max(abs(y), 1e-12)
                if rel > worst: worst, worst_k = rel, k
        out["compare"][name] = dict(missing=missing, max_rel_diff_vs_V1=worst, worst_key=worst_k, pass_1e6=worst <= 1e-6)
        if name == "c_raw": out["c_raw_summary"] = g
    print(json.dumps(out, indent=1))
