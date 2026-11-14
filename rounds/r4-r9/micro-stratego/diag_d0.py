#!/usr/bin/env python3
"""Exploratory D0 diagnostics (not decision rules; written after the gate result): from the saved per-position arrays
(runs/s<seed>/eval/d0.npz, d0_ckpt30.npz) and the oracle file, the 0-step belief KL by turn, the share of positions above
the 0.12 bar, the posterior entropy per hidden slot for scale, and the same for the 30% checkpoint.
Writes report/d0_by_t.csv, report/d0_diag.json, report/d0_by_t.png."""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (101, 102, 103)
o = np.load(os.path.join(HERE, "data", "oracle_d1.npz"))
t = o["t"]
pf = o["pstar_full"]
pc = o["pstar_c"]
H = -(np.where(pf > 0, pf * np.log(np.clip(pf, 1e-12, 1)), 0)).sum(-1)
te = np.load(os.path.join(HERE, "data", "test.npz"))
key = {(int(g), int(tt)): i for i, (g, tt) in enumerate(zip(te["game"], te["t"]))}
rows = np.array([key[(int(g), int(tt))] for g, tt in zip(o["game"], o["t"])])
hid = te["slot_pubrev"][rows] == 0
Hpos = (H * hid).sum(1) / np.maximum(hid.sum(1), 1)
klc_or = np.where(pf > 0, pf * (np.log(np.clip(pf, 1e-12, 1)) - np.log(np.clip(pc, 1e-12, 1))), 0).sum(-1)
klc_pos = (klc_or * hid).sum(1) / np.maximum(hid.sum(1), 1)
out = {"n_positions": int(len(t)), "posterior_entropy_per_hidden_slot": float(Hpos.mean()),
       "kl_Pfull_vs_Pc_per_hidden_slot": float(klc_pos.mean()), "mean_hidden_slots": float(hid.sum(1).mean())}
rows_csv = []
for tag in ("d0", "d0_ckpt30"):
    per = []
    for s in SEEDS:
        z = np.load(os.path.join(HERE, "runs", f"s{s}", "eval", f"{tag}.npz"))
        k = z["kl0_full"]
        per.append({"seed": s, "mean_kl": float(k.mean()), "median_kl": float(np.median(k)),
                    "share_ge_0.12": float((k >= 0.12).mean()), "p90": float(np.quantile(k, 0.9)),
                    "kl_over_entropy": float(k.mean() / Hpos.mean())})
        for tt in range(4, 13):
            m = t == tt
            rows_csv.append([tag, s, tt, int(m.sum()), float(k[m].mean()), float(Hpos[m].mean()), float(klc_pos[m].mean())])
    out[tag] = per
os.makedirs(os.path.join(HERE, "report"), exist_ok=True)
with open(os.path.join(HERE, "report", "d0_by_t.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["checkpoint", "seed", "t", "n", "model_kl_to_Pfull", "Pfull_entropy_per_hidden_slot", "Pc_kl_to_Pfull"])
    w.writerows(rows_csv)
json.dump(out, open(os.path.join(HERE, "report", "d0_diag.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
try:
    sys.path.insert(0, "/tmp/ms-pylib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = np.array([[r[2], r[4], r[6]] for r in rows_csv if r[0] == "d0"], float)
    R30 = np.array([[r[2], r[4]] for r in rows_csv if r[0] == "d0_ckpt30"], float)
    ts = np.arange(4, 13)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(ts, [R[R[:, 0] == x, 1].mean() for x in ts], "o-", label="theta_real 100% (3-seed mean)")
    ax.plot(ts, [R30[R30[:, 0] == x, 1].mean() for x in ts], "s--", label="theta_real 30%")
    ax.plot(ts, [R[R[:, 0] == x, 2].mean() for x in ts], "^:", label="constraint-only oracle P*_c")
    ax.axhline(0.12, color="crimson", lw=1, label="pre-registered gap bar 0.12")
    ax.set_xlabel("turn t")
    ax.set_ylabel("KL to P*_full per hidden slot (nats)")
    ax.set_title("E1-D0: 0-step belief vs exact Bayes posterior")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "report", "d0_by_t.png"), dpi=110)
except Exception as e:
    print("plot skipped:", e)

# calibration-split fits made by train_ms.py before the gate was known (exploratory; nothing here is a test result)
cal = []
for s in SEEDS:
    tm = json.load(open(os.path.join(HERE, "runs", f"s{s}", "train_meta.json")))
    a = tm["action_scales"]
    cal.append({"seed": s, "params_theta_real": tm["real"]["params"],
                "train_minutes": round(tm["real"]["sec"] / 60, 1),
                "cal_value_ce_min": min(h["cal"][3] for h in tm["real"]["hist"]),
                "cal_value_ce_final": tm["real"]["hist"][-1]["cal"][3],
                "cal_belief_ce_final": tm["real"]["hist"][-1]["cal"][0],
                "lir_hash_unchanged": tm["lir_param_hash_before"] == tm["lir_param_hash_after"],
                "fused_hash_changed": tm["fused_param_hash_after"] != tm["lir_param_hash_after"],
                "root_pool_eta": tm["agg"]["eta"], "root_pool_u_kc": tm["agg"]["u_root"],
                "action_logloss_l0": a["kc"]["logloss_l0"], "action_logloss_kc": a["kc"]["logloss"],
                "action_logloss_mmd": a["mmd"]["logloss"], "action_scale_kc": a["kc"]["scale"],
                "action_scale_mmd": a["mmd"]["scale"]})
json.dump(cal, open(os.path.join(HERE, "report", "cal_fits.json"), "w"), indent=1)
print(json.dumps(cal, indent=1))
