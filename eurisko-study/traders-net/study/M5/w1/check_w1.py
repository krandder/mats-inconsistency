"""Re-read EXPLOIT-w1.md headline numbers from SUMMARY-w1.csv (policy 15)."""
import csv, os, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
S = {r["source"]: r for r in csv.DictReader(open(os.path.join(HERE, "SUMMARY-w1.csv")))}
claims = [("P4-net", "calonfam_phi_eps_mean", 0.053, 0.001), ("P4-net", "EC_cal_loglog_mean", 0.53, 0.01),
          ("P16-net", "calonfam_phi_eps_mean", 0.091, 0.001), ("P4-logistic", "calonfam_phi_eps_mean", 0.644, 0.001),
          ("P4-logistic", "EC_cal_loglog_mean", 1.02, 0.01), ("wn-ckpt8000", "calonfam_phi_eps_mean", 8.07, 0.01),
          ("wn-ckpt16000", "calonfam_phi_eps_mean", 10.24, 0.01), ("wn-ckpt30000", "calonfam_phi_eps_mean", 12.27, 0.01),
          ("P4-net", "EC_cal_final_mean", 1072, 1), ("wn-ckpt30000", "ece_2nd_half_mean", 0.124, 0.001)]
bad = 0
for src, k, v, tol in claims:
    x = float(S[src][k]); ok = abs(x - v) <= tol; bad += not ok
    print("PASS" if ok else "FAIL", src, k, x, "claimed", v)
for src in S:
    ok = S[src]["exploitable_3of3"] == "1"; bad += not ok
    print("PASS" if ok else "FAIL", src, "exploitable_3of3")
print("checked_at", datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z", "problems", bad)
