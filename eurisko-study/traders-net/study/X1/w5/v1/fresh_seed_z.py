"""X1-w5 V-c: z-scores of fresh seeds 4,5 (G1 job x1-w5-wn45) against V1's 3-seed mean/sd at step 23300.
Quantities derived twice: independent raw reader (recompute_summary.raw) and V1's own serve.wn_csv (from a /tmp copy).
Verdict per quantity (PREREG): verified if |z|<=2 for both seeds, refuted if both >2, else unverified.
Run: python3 fresh_seed_z.py /tmp/traders-net/X1w5/wn45 > fresh_seed_z.json"""
import sys, json, csv, io, importlib.util, pathlib, datetime as dt
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE)); import recompute_summary as rs
snap = pathlib.Path(sys.argv[1]); STEP = 23300; SEEDS = [int(x) for x in sys.argv[2:]] or [4, 5]
ref = json.load(open(rs.V1 / "w1/summary-at-common-step.json"))
spec = importlib.util.spec_from_file_location("v1serve", "/tmp/traders-net/X1w5/v1copy/grafana/serve.py")
sv = importlib.util.module_from_spec(spec); spec.loader.exec_module(sv)
out = dict(checked_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), step=STEP, seeds={}, quantities={})
for s in SEEDS:
    r = rs.raw(snap, s, STEP)
    body = sv.wn_csv(str(snap / f"train-s{s}" / "snapshots"))
    body = body[0].decode()
    row = rs.from_csv_text(body, STEP)
    agree = max(abs(r[k] - row[k]) / max(abs(row[k]), 1e-9) for k in r if k in row and k != "step")
    out["seeds"][s] = dict(raw=r, serve_vs_raw_max_rel_diff=agree)
for k in ref:
    if k == "step": continue
    m, sd = ref[k]["mean"], ref[k]["sd"]
    zs = [(out["seeds"][s]["raw"][k] - m) / sd if sd else None for s in SEEDS]
    n_in = sum(z is not None and abs(z) <= 2 for z in zs)
    out["quantities"][k] = dict(mean3=m, sd3=sd, fresh=[out["seeds"][s]["raw"][k] for s in SEEDS],
                                z=zs, verdict="verified" if n_in == 2 else "refuted" if n_in == 0 else "unverified")
import statistics as st
for k, q in out["quantities"].items():
    five = ref[k]["seeds"] + q["fresh"]
    q["pooled5"] = dict(mean=st.mean(five), sd=st.stdev(five), sd_ratio_vs_sd3=st.stdev(five) / q["sd3"] if q["sd3"] else None,
                        same_sign_all5=all(x > 0 for x in five) or all(x < 0 for x in five))
out["tally"] = {v: sum(q["verdict"] == v for q in out["quantities"].values()) for v in ("verified", "unverified", "refuted")}
print(json.dumps(out, indent=1, default=str))
