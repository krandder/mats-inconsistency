"""Summarise M5 w1 results: SUMMARY-w1.csv (mean, sd over seeds) and ec_curves.png."""
import csv, glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
SOURCES = ["P4-net", "P16-net", "P4-logistic", "wn-ckpt8000", "wn-ckpt16000", "wn-ckpt30000"]
METRICS = {
    "T": lambda d: d["T"], "bce_2nd_half": lambda d: d["bce_2nd_half"], "ece_2nd_half": lambda d: d["ece_2nd_half"],
    "EC_cal_final": lambda d: d["real"]["C_cal"]["final"], "EC_cal_loglog": lambda d: d["real"]["C_cal"]["loglog"],
    "EC_cal_slope_2nd": lambda d: d["real"]["C_cal"]["lin_last_half"],
    "EC_cal_fam_final": lambda d: d["real"]["C_cal_fam"]["final"], "EC_cal_fam_loglog": lambda d: d["real"]["C_cal_fam"]["loglog"],
    "best1st_phi_eps": lambda d: d["real"]["best_first_half"]["phi_eps_2nd_half"],
    "best1st_E_2nd": lambda d: d["real"]["best_first_half"]["E_2nd_half"],
    "calon_final": lambda d: d["real"]["cal_online"]["final"], "calon_loglog": lambda d: d["real"]["cal_online"]["loglog"],
    "calon_phi_eps": lambda d: d["real"]["cal_online"]["phi_eps_2nd_half"],
    "calonfam_final": lambda d: d["real"]["cal_online_fam"]["final"],
    "calonfam_phi_eps": lambda d: d["real"]["cal_online_fam"]["phi_eps_2nd_half"],
    "calonfam_phi_per_q": lambda d: d["real"]["cal_online_fam"]["phi_eps_2nd_half"] / d["Q"],
    "c50_share_per_q_2nd": lambda d: d["real"]["c50_one_share_per_qtrade_2nd_half"],
    "null_EC_cal_final_max": lambda d: max(n["C_cal_final"] for n in d["null"]),
    "null_EC_cal_fam_final_max": lambda d: max(n["C_cal_fam_final"] for n in d["null"]),
    "null_calon_phi_max": lambda d: max(n["cal_online_phi"] for n in d["null"]),
    "null_calonfam_final_max": lambda d: max(n["cal_online_fam_final"] for n in d["null"]),
}
rows = []
for src in SOURCES:
    ds = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, f"{src}-s[0-9].json")))]
    if not ds:
        continue
    row = {"source": src, "n_seeds": len(ds), "seeds": " ".join(str(d["seed"]) for d in ds)}
    for k, fn in METRICS.items():
        v = np.array([fn(d) for d in ds], float)
        row[k + "_mean"] = float(v.mean()); row[k + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    # exploitable rule (PREREG): EC_cal final > null max in every seed AND cal-online phi > 0 in every seed
    row["exploitable_3of3"] = int(all(d["real"]["C_cal"]["final"] > max(n["C_cal_final"] for n in d["null"])
                                      and d["real"]["cal_online"]["phi_eps_2nd_half"] > 0 for d in ds) and len(ds) == 3)
    row["exploitable_fam_3of3"] = int(all(d["real"]["cal_online_fam"]["phi_eps_2nd_half"] > max(0, max(n["cal_online_phi"] for n in d["null"]))
                                          for d in ds) and len(ds) == 3)
    # top family of cal_online_fam (by mean final)
    fams = ds[0]["families"]
    byf = np.array([[d["real"]["cal_online_fam"]["by_family"][f] for f in fams] for d in ds])
    row["calonfam_by_family_mean"] = " ".join(f"{f}:{x:.0f}" for f, x in zip(fams, byf.mean(0)))
    rows.append(row)
keys = list(rows[0].keys())
for r in rows:
    for k in r:
        if k not in keys:
            keys.append(k)
with open(os.path.join(HERE, "SUMMARY-w1.csv"), "w") as f:
    w = csv.DictWriter(f, keys); w.writeheader(); [w.writerow(r) for r in rows]
for r in rows:
    print(r["source"], r["n_seeds"], "EC_cal %.1f±%.1f b=%.2f" % (r["EC_cal_final_mean"], r["EC_cal_final_sd"], r["EC_cal_loglog_mean"]),
          "calonfam phi %.4f±%.4f" % (r["calonfam_phi_eps_mean"], r["calonfam_phi_eps_sd"]), "expl", r["exploitable_3of3"], r["exploitable_fam_3of3"])
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, group in zip(axes, (["P4-net", "P16-net", "P4-logistic"], ["wn-ckpt8000", "wn-ckpt16000", "wn-ckpt30000"])):
        for i, src in enumerate(group):
            for j, fn in enumerate(sorted(glob.glob(os.path.join(R, f"{src}-s[0-9]-curve.csv")))):
                a = np.genfromtxt(fn, delimiter=",", names=True)
                ax.plot(a["t"], a["cal_online_fam"], color=f"C{i}", lw=1, label=src if j == 0 else None)
                ax.plot(a["t"], a["EC_cal_null0"], color=f"C{i}", lw=0.8, ls=":")
        ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("stream position t"); ax.set_ylabel("cumulative log gain (nats)")
        ax.legend(fontsize=8)
    axes[0].set_title("prequential markets: cal-online-fam (solid), EC_cal on null (dotted)", fontsize=9)
    axes[1].set_title("wealth-net frozen checkpoints on held-out tail", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "ec_curves.png"), dpi=110)
except Exception as e:
    print("plot skipped:", e)
