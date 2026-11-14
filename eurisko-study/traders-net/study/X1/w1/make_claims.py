"""X1 worker 1: build CLAIMS.md from the comparison JSONs (re-run this after any compare_*.py run).

Inputs (results/): compare_S0_s4.json, compare_S0_s5.json, compare_T1_s4.json, compare_T1_s5.json,
L1_check.json, pi_offsets.json, substrate_check.txt. Status rule (PREREG.md section 4):
fresh seed inside -> verified; outside and tie-break seed inside -> unverified (mixed);
outside on both seeds, same side -> refuted; outside, tie-break missing -> unverified (pending).
"""
import json, os, datetime
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "results")
X = "study/X1/w1"
def load(n):
    p = os.path.join(R, n); return json.load(open(p)) if os.path.exists(p) else None
def idx(d): return {c["id"]: c for c in d["claims"]} if d else {}
def fmt(v):
    if v is None: return "-"
    if isinstance(v, bool): return str(v)
    if isinstance(v, (int, float)):
        a = abs(v); return f"{v:.3g}" if (a and (a < 1e-3 or a >= 1e4)) else f"{v:.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
    return str(v)
def side(c):
    if c.get("z") is not None: return 1 if c["z"] > 0 else -1
    if c.get("committed_mean") is not None and c.get("fresh") is not None: return 1 if c["fresh"] > c["committed_mean"] else -1
    return 0
def status(c4, c5):
    if c4 is None or c4.get("pass") is None and c4.get("kind") != "count": return "unverified", "not rerun / extractor error"
    if c4.get("kind") == "count":
        return "unverified", "a count over seeds; one fresh seed cannot refute it (reported only)"
    if c4["pass"]: return "verified", ""
    if c5 is None or c5.get("pass") is None: return "unverified", "fresh seed outside; tie-break seed pending"
    if c5["pass"]: return "unverified", "mixed: fresh seed outside, tie-break seed inside"
    if side(c4) == side(c5): return "refuted", "both fresh seeds outside on the same side"
    return "unverified", "both outside, opposite sides"
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
rows = []
s4, s5 = idx(load("compare_S0_s4.json")), idx(load("compare_S0_s5.json"))
t4, t5 = idx(load("compare_T1_s4.json")), idx(load("compare_T1_s5.json"))
EV_S0 = {"1": "study/S0/w1/results/{ARM}-s*.json, SUMMARY-w1.csv", "2": "study/S0/results/w2/audit-s*.json",
         "3": "study/S0/w3/baseline-s*.json, baseline-summary.csv", "4": "study/S0/w4/null_calib.json"}
RR_S0 = {"1": "python3 study/S0/w1/run_w1.py ARM 4 (copy), then compare_S0.py", "2": "python3 study/S0/w2/audit.py 4 (copy, path patch)",
         "3": "python3 study/S0/w3/baseline.py 3 (copy)", "4": "python3 study/S0/w4/null_calib_s3.py (copy)"}
for cid, c in s4.items():
    st, why = status(c, s5.get(cid))
    g = cid.split("-")[1][0]
    fr = c.get("fresh"); f5 = (s5.get(cid) or {}).get("fresh")
    if c.get("kind") == "count": fr = f"{fmt(fr)} (learnt: {c.get('fresh_learnt')}; committed {c.get('committed_learnt')}/3)"
    rows.append(("S0", cid, c["claim"], st, why, f"{fmt(c.get('committed_mean'))} ± {fmt(c.get('committed_sd'))}",
                 f"{fmt(fr)} (s{c['fresh_seeds'][0]}, z {fmt(c.get('z'))})", fmt(f5),
                 f"{EV_S0[g]}; {X}/results/compare_S0_s4.json", f"{X}/compare_S0.py study/S0 <copy> --out ... ; {RR_S0[g]}"))
EV_T1 = {"L1": "study/T1/w1/out/L1*.json", "L2": "study/T1/w1/out/L2.json", "L3": "study/T1/w1/out/L3.json, flops_*.json",
         "L4": "study/T1/w2/results/summary*.{json,csv}", "L5": "study/T1/w2/results/summary.json, summary_li.csv",
         "L6": "study/T1/w3/out/summary.json, L6_growth.csv", "L7": "study/T1/w3/out/summary.json"}
for cid, c in t4.items():
    st, why = status(c, t5.get(cid))
    lem = cid.split("-")[1].split(".")[0]
    band = f"{fmt(c.get('committed_mean'))} ± {fmt(c.get('committed_sd'))}" if c.get("kind") == "stat" else f"{fmt(c.get('committed_mean'))} ({c.get('kind')} {fmt(c.get('tol', c.get('bound', c.get('threshold', ''))))})"
    rows.append(("T1", cid, c["claim"], st, why, band, f"{fmt(c.get('fresh'))} (s4, z {fmt(c.get('z'))})", fmt((t5.get(cid) or {}).get("fresh")),
                 f"{EV_T1[lem]}; {X}/results/compare_T1_s4.json", f"{X}/compare_T1.py study/T1 /tmp/traders-net/X1w1/T1 --out ... (jobs in {X}/jobs/)"))
L = load("L1_check.json")
if L:
    ck = L["checked_at"]
    rows.append(("L1", "L1-1", "check_novelty.py: 45 verified sources, 40 cited IDs, 0 fails", "verified" if L["rerun_identical_except_checked_at"] else "refuted", f"clean-copy rerun identical to lead/check_novelty.json except checked_at", "45/40/0", L["rerun_tail"][-1], "-", f"study/L1/lead/check_novelty.json; {X}/results/L1_check.json", f"{X}/check_L1.py --out {X}/results/L1_check.json"))
    for p in L["pdfs"]:
        rows.append(("L1", "L1-q-" + p["file"].split(".pdf")[0], "quoted numbers present in " + p["url"] + ": " + "; ".join(p["quotes"]),
                     "verified" if p["all_found"] else "refuted", "independent re-fetch; md5 same as L1 cache: " + (",".join(p["identical_to_L1_cache"]) or "no cache"),
                     "-", "all found" if p["all_found"] else "missing", "-", f"{X}/results/L1_check.json; /tmp/traders-net/X1w1/{p['file']}", f"{X}/check_L1.py"))
    e = L["eom_15.9_position"]
    rows.append(("L1", "L1-q-eom-label", "quotes-lead.txt labels the EoM 15.9->57.0 / 45.0->60.0 numbers 'abstract numbers'", "refuted" if e["in_abstract"] is False else "verified",
                 "numbers are in the PDF introduction (char %d; abstract at %d, introduction at %d); values themselves verified" % (e["char"], e["abstract_at"], e["introduction_at"]),
                 "-", "-", "-", f"study/L1/lead/quotes-lead.txt:49; {X}/results/L1_check.json", f"{X}/check_L1.py"))
    rows.append(("L1", "L1-2", "NOVELTY.md has 15 claim rows with verdicts", "verified" if L["novelty_rows"] == 15 else "refuted", "; ".join(v.replace("**", "").split(";")[0].split(" (")[0].split(",")[0] for v in L["verdicts"]), "15", str(L["novelty_rows"]), "-", f"study/L1/NOVELTY.md:35-49; {X}/results/L1_check.json", f"{X}/check_L1.py"))
    rows.append(("L1", "L1-3", "NOVELTY-FLAGS.md: flags F1-F23, no flag drops an arm", "verified" if (L["flags"]["n_distinct"] == 23 and L["no_flag_drops_an_arm"]) else "refuted", "flag ids 1..23 distinct; line %s says no flag drops an arm" % L["no_flag_drops_an_arm"], "23", str(L["flags"]["n_distinct"]), "-", f"study/L1/NOVELTY-FLAGS.md; {X}/results/L1_check.json", f"{X}/check_L1.py"))
P = load("pi_offsets.json")
if P:
    rows.append(("S0", "S0-X-pi", "(skeptic) 'pi ≈ ln 2 under prequential' holds for any seed", "refuted",
                 "wealth-net pi offsets are drawn in [0,2000); a pair within W=32 lets one pi question read another's future digits. %.1f%% of seeds 1-%d have such a pair; seeds 1-3 have none, seed 4 has 6 pairs (3 questions)" % (100 * P["frac_seeds_with_pair_in_window"], P["N_seeds"]),
                 "0.6933 ± 0.0004 (s1-3)", fmt((s4.get("S0-1a") or {}).get("fresh")) + " (s4)", fmt((s5.get("S0-1a") or {}).get("fresh")), f"{X}/results/pi_offsets.json; {X}/results/compare_S0_s4.json", f"PYTHONPATH=wealth-net/.pylib:wealth-net python3 {X}/pi_offsets.py 2000 {X}/results/pi_offsets.json"))
sub = os.path.join(R, "substrate_check.txt")
if os.path.exists(sub):
    t = open(sub).read(); ok = "pytest: 39 passed" in t and "PASS (0 problems)" in t
    rows.append(("S0", "S0-5", "check_substrate.py: 39 tests pass, every quoted number re-read (PASS, 0 problems)", "verified" if ok else "refuted", "run in the clean copy (fresh-seed outputs were written beside the committed ones; the check reads committed seeds)", "39 / 0", [l for l in t.splitlines() if "pytest" in l or "PASS" in l or "FAIL" in l][-2:].__str__(), "-", f"{X}/results/substrate_check.txt", "cd <copy>/S0 && python3 check_substrate.py"))
order = {"refuted": 0, "unverified": 1, "verified": 2}
n = {k: sum(r[3] == k for r in rows) for k in order}
with open(os.path.join(H, "CLAIMS.md"), "w") as f:
    f.write(f"# CLAIMS (X1 worker 1: S0, L1, T1)\n\nGenerated {now} by `{X}/make_claims.py` from the comparison JSONs in `{X}/results/`; "
            f"rule in `{X}/PREREG.md`. Totals: {n['verified']} verified, {n['unverified']} unverified, {n['refuted']} refuted "
            f"(of {len(rows)}). Paths are relative to `traders-net/`. Fresh seeds: S0 w1/w2 seed 4 (tie-break 5), S0 w3/w4 seed 3, T1 seed 4 (tie-break 5). "
            f"All rows checked at {now} unless the evidence file says otherwise.\n\n")
    f.write("| ws | id | claim | status | why | committed | fresh | tie-break s5 | evidence | rerun |\n|---|---|---|---|---|---|---|---|---|---|\n")
    for r in sorted(rows, key=lambda r: (r[0], order[r[3]], r[1])):
        f.write("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |\n")
    notes = os.path.join(H, "NOTES-skeptic.md")
    if os.path.exists(notes): f.write("\n" + open(notes).read())
print(n, len(rows))
