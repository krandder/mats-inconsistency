"""Soup analogue: sh_c50 (constant-0.5 one-share shadow) from logical-soup runs/phase3 metrics.
Units: shares per question-trade (Q16 ticks / 65536 / 2^logq questions per epoch); the denominator
assumes every question live and traded every epoch, so these are lower bounds per trade."""
import csv, glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SOUP = os.path.join(HERE, "..", "..", "..", "..", "logical-soup", "runs", "phase3")
rows = []
for d in sorted(glob.glob(os.path.join(SOUP, "k*-s[0-9]"))):
    cfg = json.load(open(os.path.join(d, "config.json")))
    r = list(csv.DictReader(open(os.path.join(d, "metrics.csv"))))
    ep = np.array([int(x["epoch"]) for x in r]); c = np.array([float(x["sh_c50"]) for x in r])
    nq = 2 ** cfg["logq"]; h = len(ep) // 2
    s_all = np.polyfit(ep, c, 1)[0] / 65536 / nq
    s_2nd = np.polyfit(ep[h:], c[h:], 1)[0] / 65536 / nq
    rows.append(dict(run=os.path.basename(d), logk=cfg["logk"], epochs=int(ep[-1]), final_ticks=float(c[-1]),
                     slope_all_share_per_q=float(s_all), slope_2nd_half_share_per_q=float(s_2nd)))
json.dump(rows, open(os.path.join(HERE, "results", "soup_c50.json"), "w"), indent=1)
for x in rows: print(x)
