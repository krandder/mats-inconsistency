"""Aggregate M5-w4 probe runs: out/*.json -> results/summary.csv (per arm, mean +- sd over seeds),
results/family.csv (per-family phi for probe L), results/ec_curves.png."""
import glob, json, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, "out", "*.json"))) if "smoke" not in f]
groups = defaultdict(list)
for r in runs:
    arm = r["tag"].rsplit("-s", 1)[0]
    groups[arm].append(r)


def ms(x):
    x = np.asarray(x, float)
    return (float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float("nan"))


order = ["logistic-win", "preq-P4-win", "preq-P16-win"] + [f"wn-{c}-{f}" for c in ["random", "8000", "16000", "24000", "final"] for f in ["win", "hid"]]
lines = ["arm,n_seeds,n_steps,price_bce,price_bce_sd,EC_per_qstep,EC_sd,EC_phi,EC_phi_sd,EC_loglog,EC_loglog_sd,"
         "phiS,phiS_sd,phiM,phiM_sd,phiL,phiL_sd,phiP,phiP_sd,EL,EL_sd,EP,EP_sd,"
         "mem_Eclear_maxq,mem_Eclear_maxq_sd,mem_Etraded_maxq,mem_Etraded_maxq_sd,mem_share,mem_share_sd,mem_dbce,mem_dbce_sd"]
fam_lines = ["arm,family,EL,EL_sd,EP,EP_sd"]
for arm in [a for a in order if a in groups] + [a for a in groups if a not in order]:
    rs = groups[arm]
    row = [arm, len(rs), rs[0]["n_steps"]]
    row += ms([r["price_bce"] for r in rs])
    row += ms([r["E_C"]["per_qstep"] for r in rs]) + ms([r["E_C"]["phi_last_half_per_qstep"] for r in rs]) + ms([r["E_C"]["loglog"] for r in rs])
    for k in "SMLP":
        row += ms([r["probes"][k]["phi_last_half_per_qstep"] for r in rs])
    row += ms([r["probes"]["L"]["E_per_qstep"] for r in rs]) + ms([r["probes"]["P"]["E_per_qstep"] for r in rs])
    m = [r["member_L"] for r in rs]
    row += ms([x["E_clear_max_q"] for x in m]) + ms([x["E_traded_max_q"] for x in m]) + ms([x["final_share_mean"] for x in m])
    row += ms([x["market_bce"] - x["p0_bce"] for x in m])
    lines.append(",".join(str(v) if isinstance(v, (int, str)) else f"{v:.6g}" for v in row))
    for fam in rs[0]["probes"]["L"]["E_fam_per_qstep"]:
        v = ms([r["probes"]["L"]["E_fam_per_qstep"][fam] for r in rs]); p = ms([r["probes"]["P"]["E_fam_per_qstep"][fam] for r in rs])
        fam_lines.append(f"{arm},{fam},{v[0]:.6g},{v[1]:.6g},{p[0]:.6g},{p[1]:.6g}")
open(os.path.join(HERE, "results", "summary.csv"), "w").write("\n".join(lines) + "\n")
open(os.path.join(HERE, "results", "family.csv"), "w").write("\n".join(fam_lines) + "\n")
print("\n".join(lines))

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    show = [a for a in ["logistic-win", "preq-P4-win", "preq-P16-win", "wn-random-win", "wn-8000-win", "wn-final-win", "wn-final-hid"] if a in groups]
    fig, ax = plt.subplots(1, len(show), figsize=(3.2 * len(show), 3), squeeze=False)
    for j, arm in enumerate(show):
        for r in groups[arm]:
            c = np.genfromtxt(os.path.join(HERE, "out", r["tag"] + "-ec.csv"), delimiter=",", names=True)
            for k, col in zip("SMLP", ["#9ecae1", "#4292c6", "#08306b", "#e6550d"]):
                ax[0, j].plot(c["t"], c["E_" + k] / 64, color=col, lw=1, label=k if r["seed"] == 1 else None)
        ax[0, j].axhline(0, color="gray", lw=0.5); ax[0, j].set_title(arm, fontsize=9); ax[0, j].set_xlabel("step")
    ax[0, 0].set_ylabel("E_T per question (nats)"); ax[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "results", "ec_curves.png"), dpi=110)
except Exception as e:  # plotting is optional
    print("plot skipped:", e)
