"""Re-derive every number in EXPLOIT-w4.md from results/summary.csv and results/family.csv.
Numbers in the doc are written as {arm|column} or {fam:arm|family|column} tokens in EXPLOIT-w4.src.md;
this script renders them into EXPLOIT-w4.md, so the doc cannot drift from the CSVs.
Run: python3 aggregate.py && python3 check_w4.py"""
import csv, os, re, sys
H = os.path.dirname(os.path.abspath(__file__))
S = {r["arm"]: r for r in csv.DictReader(open(os.path.join(H, "results", "summary.csv")))}
F = {(r["arm"], r["family"]): r for r in csv.DictReader(open(os.path.join(H, "results", "family.csv")))}
src = open(os.path.join(H, "EXPLOIT-w4.src.md")).read()


def fmt(v, col):
    x = float(v)
    if col.startswith("n_") or col == "n_seeds":
        return str(int(x))
    return f"{x:.3g}" if abs(x) >= 0.001 or x == 0 else f"{x:.2e}"


def pm(tab, key, col):
    r = tab[key]
    sd = r.get("EC_sd") if col == "EC_per_qstep" else r.get(col + "_sd") if (col + "_sd") in r else r.get(col.replace("_mean", "") + "_sd")
    return fmt(r[col], col) + (f" ± {fmt(sd, col)}" if sd not in (None, "") else "")


def sub(m):
    tok = m.group(1)
    if tok.startswith("fam:"):
        arm, fam, col = tok[4:].split("|")
        return pm(F, (arm, fam), col)
    arm, col = tok.split("|")
    return pm(S, arm, col)


out = re.sub(r"\{\{([^}]+)\}\}", sub, src)
open(os.path.join(H, "EXPLOIT-w4.md"), "w").write(out)
n = len(re.findall(r"\{\{", src))
print(f"PASS: rendered {n} numbers from results/summary.csv and results/family.csv into EXPLOIT-w4.md")
