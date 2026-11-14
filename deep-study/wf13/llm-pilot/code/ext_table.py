# ext_table.py OUT_DIR : the registered expectations E1-E3 of PREREG-ADDENDUM.md from OUT_DIR/RESULT-ext.json.
import sys, json
OUT = sys.argv[1]
R = json.load(open(OUT + "/RESULT-ext.json")); C = R["curves"]
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
res = {}
for f in ("arithL", "vars"):
    bar = C["%s|W1|vote|16" % f]["tv_ref"]
    print("\n== %s (n=%d) bar D(W1,vote,16) = %.3f ; reference: acc %.3f p_true %.3f valid %.3f" % (
        f, R["ref"][f]["n"], bar, R["ref"][f]["acc"], R["ref"][f]["p_true"], R["ref"][f]["valid"]))
    for w in ("W1", "W1b", "W3", "W3s", "W3p", "W4"):
        row = {}
        for g in ("vote", "dedup", "snis"):
            d = [C["%s|%s|%s|%d" % (f, w, g, K)]["tv_ref"] for K in KS]
            ks = [K for K, x in zip(KS, d) if x <= bar]
            row[g] = {"D": d, "Kstar": ks[0] if ks else None, "acc256": C["%s|%s|%s|256" % (f, w, g)]["acc"],
                      "aux256": C["%s|%s|%s|256" % (f, w, g)]["aux"], "move256": C["%s|%s|%s|256" % (f, w, g)]["move_vs_W1"]}
            print("%-4s %-5s D " % (w, g) + " ".join("%.3f" % x for x in d) + " | K* %s | acc@256 %.3f | aux %s | move@256 %.3f" % (
                row[g]["Kstar"], row[g]["acc256"], "%.3f" % row[g]["aux256"] if row[g]["aux256"] is not None else "-", row[g]["move256"]))
        res["%s|%s" % (f, w)] = row
    res[f + "|bar"] = bar
json.dump(res, open(OUT + "/RESULT-ext-table.json", "w"), indent=1)
