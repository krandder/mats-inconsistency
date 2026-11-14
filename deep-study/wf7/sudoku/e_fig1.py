"""e_fig1.py -- figure 1: every order policy as a definition of T. Left: 4x4, exact (all 288 grids and the held-out
orbit). Right: 6x6, 1,000 held-out grids with bootstrap intervals. Reads out/e_4x4_mA4.json, the printed table of
out/e_4x4_mA4_nodp.out (the rows with 256 random orders) and out/e_trust6_mA6.json."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

b = json.load(open("out/e_4x4_mA4.json"))
r4 = {k: v for k, v in b.items() if "random" not in k}
# the rows that use 256 random orders exist only in the printed table of the first run (its json was overwritten)
for line in open("out/e_4x4_mA4_nodp.out"):
    if "256" in line and len(line.split()) > 6 and line.startswith("   ") and "M's inconsistency" not in line:
        lab = line[3:81].strip(); nums = line[81:].split()
        r4[lab] = dict(excess_all288=float(nums[0]), excess_heldout96=float(nums[1]), excess_train=float(nums[2]), valid_mass=float(nums[3]))
short = {"exact optimum of the trust MDP (DP with the true law; reference)": "exact optimum of the trust MDP\n(DP with the true law; reference)",
         "tabular trust policy (DP on the training grids; entropy order off the table)": "trust MDP solved on the training grids only\n(tabular DP)",
         "best of 256 random fixed orders (picked on all 288: oracle pick)": "best of 256 random fixed orders\n(picked with the true law)",
         "one random fixed order (mean over 256 orders)": "one random fixed order (mean of 256)",
         "learned mixture over the routes above (weights by EM on the training grids)": "learned mixture of six routes (added)",
         "pi_trust (learned, paired routes)": "pi_trust (learned on training grids)"}
labs = sorted(r4, key=lambda l: r4[l]["excess_all288"])
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2)); ax = axes[0]; y = np.arange(len(labs))[::-1]; CLIP = 0.62
va = [r4[l]["excess_all288"] for l in labs]; vh = [r4[l]["excess_heldout96"] for l in labs]
ax.barh(y + 0.2, np.minimum(va, CLIP), height=0.38, label="all 288 grids")
ax.barh(y - 0.2, np.minimum(vh, CLIP), height=0.38, label="held-out orbit (96 grids)")
for yy, v in zip(y, vh):
    if v > CLIP:
        ax.text(CLIP * 0.99, yy - 0.2, f"{v:.2f} ", ha="right", va="center", fontsize=7, color="white")
for yy, v in zip(y, va):
    ax.text(min(v, CLIP) + 0.006, yy + 0.2, f"{v:.3f}", va="center", fontsize=6)
ax.set_yticks(y); ax.set_yticklabels([short.get(l, l) for l in labs], fontsize=7); ax.set_xlim(0, CLIP)
ax.set_xlabel("excess route surprise (nats): mean of -log T(x) minus ln 288", fontsize=8)
ax.set_title("4x4, exact. Network trained on one of the two orbits", fontsize=9); ax.legend(fontsize=7, loc="upper right"); ax.grid(axis="x", alpha=.3)

r6 = json.load(open("out/e_trust6_mA6.json")); E2 = dict(r6["E2"])
E2["learned mixture of six routes (added)"] = r6["E2_mixture"]
rows = [(l, E2[l]["excess"], E2[l]["excess"] - E2[l]["lo"], E2[l]["hi"] - E2[l]["excess"]) for l in E2]
rows.append(("uniform mixture over 8 random orders\n(200 grids, estimate)", r6["mixture_excess"], 0, 0))
rows.sort(key=lambda t: t[1]); ax = axes[1]; y = np.arange(len(rows))[::-1]
ax.barh(y, [t[1] for t in rows], xerr=[[t[2] for t in rows], [t[3] for t in rows]], height=0.6, color="C2")
for yy, t in zip(y, rows):
    ax.text(t[1] + t[3] + 0.015, yy, f"{t[1]:.3f}", va="center", fontsize=7)
ax.set_yticks(y); ax.set_yticklabels([t[0].replace("pi_trust (learned on real grids)", "pi_trust (learned on real training grids)") for t in rows], fontsize=7)
ax.set_xlabel("excess route surprise (nats): mean of -log T(x) minus ln 28,200,960", fontsize=8)
ax.set_title("6x6, 1,000 grids from 9 held-out orbits (95% bootstrap intervals)", fontsize=9); ax.grid(axis="x", alpha=.3)
fig.suptitle("Every order policy turns the same atoms of M into a different law T over complete grids. Lower = closer to reality.", fontsize=10)
fig.tight_layout(); fig.savefig("fig/fig1_orders.png", dpi=150)
print("wrote fig/fig1_orders.png")
