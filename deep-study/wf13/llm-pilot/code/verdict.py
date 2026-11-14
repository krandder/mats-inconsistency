# verdict.py OUT_DIR : applies the pre-registered thresholds (PREREG.md) to OUT_DIR/RESULT-work.json, mechanically.
# Writes OUT_DIR/VERDICT.json and prints the tables used in NOTE.md.
import sys, json
OUT = sys.argv[1]
R = json.load(open(OUT + "/RESULT-work.json"))
C = R["curves"]; KS = [1, 2, 4, 8, 16, 32, 64]
FAMS = ["arithL", "vars"]; BIASED = ["W3", "W3s", "W3p"]; ALLW = ["W1", "W2a", "W2b", "W2c", "W3", "W3s", "W3p", "W4"]
c = lambda f, w, g, K: C["%s|%s|%s|%d" % (f, w, g, K)]
V = {}
for f in FAMS:
    v = {}
    d = R["direct"][f]["greedy_acc"]; a1 = c(f, "W1", "vote", 64)["acc"]
    v["H1"] = {"acc_W1_vote_64": a1, "direct_greedy": d, "gap": a1 - d, "holds": a1 - d >= 0.30}
    floor = {g: c(f, "W1b", g, 64)["move_vs_W1"] for g in ("vote", "dedup", "snis")}
    EM = lambda w, g: c(f, w, g, 64)["move_vs_W1"] - floor[g]
    v["floor"] = floor
    v["H2"] = {"EM_W3_vote": EM("W3", "vote"), "acc_W3_vote_64": c(f, "W3", "vote", 64)["acc"],
               "holds": EM("W3", "vote") >= 0.20 and c(f, "W3", "vote", 64)["acc"] <= a1 - 0.15}
    v["bites"] = [w for w in BIASED if EM(w, "vote") >= 0.10]
    bar16 = c(f, "W1", "vote", 16)["tv_ref"]; bar4 = c(f, "W1", "vote", 4)["tv_ref"]
    v["bar_D_W1_vote_16"] = bar16; v["bar_D_W1_vote_4"] = bar4
    v["writers"] = {}
    for w in ALLW:
        x = {"EM_vote": EM(w, "vote"), "EM_dedup": EM(w, "dedup"), "EM_snis": EM(w, "snis"),
             "D_vote": c(f, w, "vote", 64)["tv_ref"], "D_dedup": c(f, w, "dedup", 64)["tv_ref"], "D_snis": c(f, w, "snis", 64)["tv_ref"],
             "acc_vote": c(f, w, "vote", 64)["acc"], "acc_dedup": c(f, w, "dedup", 64)["acc"], "acc_snis": c(f, w, "snis", 64)["acc"],
             "ess_share": c(f, w, "snis", 64)["aux"], "found_mass": c(f, w, "dedup", 64)["aux"],
             "E_M_logR_minus_logq": R["div"]["%s|%s" % (f, w)]["E_M[log R - log q]"],
             "E_q_logq_minus_logR": R["div"]["%s|%s" % (f, w)]["E_q[log q - log R]"],
             "valid": R["div"]["%s|%s" % (f, w)]["valid_rate"]}
        x["H3_snis"] = x["EM_snis"] <= 0.10 and x["EM_snis"] <= x["EM_vote"] / 3
        x["H3_dedup"] = x["EM_dedup"] <= 0.10 and x["EM_dedup"] <= x["EM_vote"] / 3
        x["H4_major"] = x["D_snis"] <= bar16; x["H4_minor_bar4"] = x["D_snis"] <= bar4
        x["H5_snis"] = x["acc_snis"] >= a1 - 0.03; x["H5_dedup"] = x["acc_dedup"] >= a1 - 0.03
        ks = [K for K in KS if c(f, w, "snis", K)["tv_ref"] <= bar16]
        x["Kstar_snis"] = ks[0] if ks else None
        ks = [K for K in KS if c(f, w, "vote", K)["tv_ref"] <= bar16]
        x["Kstar_vote"] = ks[0] if ks else None
        v["writers"][w] = x
    b = v["bites"]
    v["H3_major"] = bool(b) and all(v["writers"][w]["H3_snis"] and v["writers"][w]["H3_dedup"] for w in b)
    v["H3_minor"] = v["writers"]["W3"]["H3_snis"] or v["writers"]["W3"]["H3_dedup"]
    v["H4_major"] = bool(b) and all(v["writers"][w]["H4_major"] for w in b)
    v["H4_minor"] = v["writers"]["W3"]["H4_major"] or (bool(b) and all(v["writers"][w]["H4_minor_bar4"] for w in b))
    v["H5_major"] = bool(b) and all(v["writers"][w]["H5_snis"] and v["writers"][w]["H5_dedup"] for w in b)
    v["H5_minor"] = v["writers"]["W3"]["H5_snis"] and v["writers"]["W3"]["H5_dedup"]
    V[f] = v
both = lambda k: all(V[f][k] if not isinstance(V[f][k], dict) else V[f][k]["holds"] for f in FAMS)
if not both("H2"):
    bin_ = "NULL"
elif both("H3_major") and both("H4_major") and both("H5_major"):
    bin_ = "MAJOR"
elif both("H3_minor") and both("H5_minor"):
    bin_ = "MINOR"
elif not both("H3_minor"):
    bin_ = "NEGATIVE"
else:
    bin_ = "MINOR-NOT-MET (H3 holds for W3, H5 does not)"
V["bin"] = bin_
V["summary"] = {k: {f: (V[f][k]["holds"] if isinstance(V[f][k], dict) else V[f][k]) for f in FAMS}
                for k in ("H1", "H2", "H3_major", "H3_minor", "H4_major", "H4_minor", "H5_major", "H5_minor")}
json.dump(V, open(OUT + "/VERDICT.json", "w"), indent=1)
print("BIN:", bin_); print(json.dumps(V["summary"]))
for f in FAMS:
    v = V[f]
    print("\n== %s: direct %.3f | W1 vote@64 %.3f | bites %s | floor %s | bar16 %.3f bar4 %.3f" % (
        f, v["H1"]["direct_greedy"], v["H1"]["acc_W1_vote_64"], v["bites"], {k: round(x, 3) for k, x in v["floor"].items()},
        v["bar_D_W1_vote_16"], v["bar_D_W1_vote_4"]))
    print("writer | acc vote/dedup/snis | excess move vote/dedup/snis | D vote/dedup/snis | ESS share | found | E_M[logR-logq] | E_q[logq-logR] | valid | K* snis")
    for w in ALLW:
        x = v["writers"][w]
        print("%s | %.3f %.3f %.3f | %.3f %.3f %.3f | %.3f %.3f %.3f | %.2f | %.2f | %.2f | %.2f | %.2f | %s" % (
            w, x["acc_vote"], x["acc_dedup"], x["acc_snis"], x["EM_vote"], x["EM_dedup"], x["EM_snis"], x["D_vote"], x["D_dedup"],
            x["D_snis"], x["ess_share"], x["found_mass"], x["E_M_logR_minus_logq"], x["E_q_logq_minus_logR"], x["valid"], x["Kstar_snis"]))
    for w in ("W1", "W3", "W3s", "W3p", "W4"):
        print("  %s by K:" % w)
        for g in ("vote", "dedup", "snis"):
            print("    %-5s acc " % g + " ".join("%.3f" % c(f, w, g, K)["acc"] for K in KS) + " | D " + " ".join("%.3f" % c(f, w, g, K)["tv_ref"] for K in KS)
                  + " | move " + " ".join("%.3f" % c(f, w, g, K)["move_vs_W1"] for K in KS))
    print("  noise floor (second M pool) move by K: " + " ; ".join("%s " % g + " ".join("%.3f" % c(f, "W1b", g, K)["move_vs_W1"] for K in KS) for g in ("vote", "dedup", "snis")))
