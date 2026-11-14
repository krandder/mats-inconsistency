#!/usr/bin/env python3
"""R8-E1 audit, check 8: numbers in RESULTS-R8-E1.md against the files at their printed precision.
(1) Structured checks: every cell of the sparse-root table, the strong-root table, the D3 jacobi table and the PUCT table, plus the
    TL;DR / section-3 numbers, each mapped by hand to its results.json key and compared at the printed precision.
(2) Automatic sweep: every decimal token in RESULTS is looked up among all numeric values of results.json, summary.json (3 seeds),
    numbers.txt, GPU-LOG.md and the registered constants (bars, tiers, spec numbers); "matched" = some value prints identically at
    the token's precision. The sweep is a screen; a hand-labelled sample (a8_hand_labels.json) measures its agreement.
Run: python3 a8_results_numbers.py [sample]"""
import hashlib
import json
import os
import random
import re
import sys

OUT = os.path.dirname(os.path.abspath(__file__))
R8 = "/home/agent/workspaces/eurisko-study/li-net/r8e1"
RES = f"{R8}/RESULTS-R8-E1.md"
text = open(RES).read()
lines = text.splitlines()
r = json.load(open(f"{R8}/results/main/analysis/results.json"))
st = r["stats"]
rep = {"results_sha256": hashlib.sha256(text.encode()).hexdigest(), "structured": [], "sweep": {}}


def fmt(v, d):
    s = f"{v:.{d}f}"
    return "0." + "0" * d if s in ("-0." + "0" * d,) else s


def check(where, token, key, value):
    d = len(token.split(".")[1]) if "." in token else 0
    ok = fmt(value, d) == token.replace("+", "")
    rep["structured"].append({"where": where, "printed": token, "source": key, "value": value, "match": ok})


# ---- (1a) sparse-root table (section 4)
names = {"looped_attn_T3": "looped_attn", "attn_L3-K3": "attn_K3", "jacobi_lin_T4_prec": "jacobi_prec", "kc_K3": "kc_K3",
         "looped_lin_T4 (R6)": "looped_lin", "scaled_mix": "scaled_mix", "ataraxos_mmd": "mmd", "kc_K1": "kc_K1", "mix": "mix"}
KF = (0, 1, 2, 4, 8, 16, 24, 32)
for ln in lines:
    m = re.match(r"\| \**([A-Za-z0-9_\-() ]+?)\** (?:\(anchor\) )?\| (0\.\d{4}) \|", ln)
    if m and m.group(1).strip() in names and ln.count("|") == 10:
        a = names[m.group(1).strip()]
        vals = [c.strip().strip("*") for c in ln.split("|")[2:10]]
        for K, tok in zip(KF, vals):
            check(f"sec4 sparse table {a} K{K}", tok, f"stats.fs_kl_{a}_K{K}.mean", st[f"fs_kl_{a}_K{K}"]["mean"])
# ---- (1b) strong-root table (section 4)
KS = (1, 2, 4, 8, 16, 24)
rowmap = {"M_0 = 18 (KL(0) 0.1444), **looped_attn_T3**": [("sr18", "looped_attn")], "M_0 = 18, attn_L3-K3": [("sr18", "attn_K3")],
          "M_0 = 18, ataraxos_mmd": [("sr18", "mmd")], "M_0 = 18, scaled_mix": [("sr18", "scaled_mix")], "M_0 = 18, mix": [("sr18", "mix")],
          "M_0 = 12 (KL(0) 0.1930), looped_attn_T3 / mmd": [("sr12", "looped_attn"), ("sr12", "mmd")],
          "M_0 = 6 (KL(0) 0.2452), looped_attn_T3 / mmd": [("sr6", "looped_attn"), ("sr6", "mmd")],
          "M_0 = 18, three-tier reads (reported), looped_attn_T3 / mmd": [("sr18t3", "looped_attn"), ("sr18t3", "mmd")]}
for ln in lines:
    if not ln.startswith("| M_0"):
        continue
    cells = [c.strip() for c in ln.split("|")[1:-1]]
    spec = rowmap.get(cells[0])
    if not spec:
        continue
    for K, c in zip(KS, cells[1:]):
        toks = [t.strip().strip("*") for t in c.split("/")]
        for (tg, a), tok in zip(spec, toks):
            check(f"sec4 strong table {tg} {a} K{K}", tok, f"stats.{tg}_dkl_{a}_K{K}.mean", st[f"{tg}_dkl_{a}_K{K}"]["mean"])
for tg, tok in (("sr18", "0.1444"), ("sr12", "0.1930"), ("sr6", "0.2452")):
    check(f"sec4 strong table KL(0) {tg}", tok, f"stats.{tg}_kl_looped_attn_K0.mean", st[f"{tg}_kl_looped_attn_K0"]["mean"])
# ---- (1c) D3 jacobi_prec table (section 5)
W = st["weights"]
for ln in lines:
    m = re.match(r"\| jacobi_lin_T4_prec, tau (\d\.\d) \|", ln)
    if not m:
        continue
    t = m.group(1)
    cells = [c.strip().strip("*").split(" [")[0].strip("*") for c in ln.split("|")[2:-1]]
    for N, tok in zip((1, 2, 4, 8, 16), cells[:5]):
        check(f"sec5 D3 table t{t} W({N})", tok, f"stats.weights.jacobi_prec_W_t{t}.{N}.mean", W[f"jacobi_prec_W_t{t}"][str(N)]["mean"])
    check(f"sec5 D3 table t{t} W8/W1", cells[5], f"stats.weights.jacobi_prec_W8/W1_t{t}.mean", W[f"jacobi_prec_W8/W1_t{t}"]["mean"])
    check(f"sec5 D3 table t{t} W16/W1", cells[6], f"stats.weights.jacobi_prec_W16/W1_t{t}.mean", W[f"jacobi_prec_W16/W1_t{t}"]["mean"])
for t, a, b in (("0.3", "1.409", "1.455"), ("2.5", "2.204", "2.427")):
    check(f"sec5 D3 table R7 jacobi t{t} W8/W1", a, f"stats.weights.jacobi_W8/W1_t{t}.mean", W[f"jacobi_W8/W1_t{t}"]["mean"])
    check(f"sec5 D3 table R7 jacobi t{t} W16/W1", b, f"stats.weights.jacobi_W16/W1_t{t}.mean", W[f"jacobi_W16/W1_t{t}"]["mean"])
# ---- (1d) PUCT table (section 6)
for ln in lines:
    m = re.match(r"\| \**(K[13]), (R[67])\** \|", ln)
    if not m:
        continue
    cell = f"{m.group(1)}_{m.group(2)}"
    cells = [c.strip().replace("*", "") for c in ln.split("|")[2:-1]]
    cov, rest = cells[0].split(" [")
    lo, hi = rest.rstrip("]").split(", ")
    c = st[f"puct_{cell}_coverage"]
    check(f"sec6 PUCT {cell} coverage", cov, f"stats.puct_{cell}_coverage.mean", c["mean"])
    check(f"sec6 PUCT {cell} coverage lo", lo, f"stats.puct_{cell}_coverage.lo", c["lo"])
    check(f"sec6 PUCT {cell} coverage hi", hi, f"stats.puct_{cell}_coverage.hi", c["hi"])
    check(f"sec6 PUCT {cell} wave-1 NW share", cells[1], f"stats.puct_{cell}_wave1_pref_share_NW.mean", st[f"puct_{cell}_wave1_pref_share_NW"]["mean"])
    check(f"sec6 PUCT {cell} attn_K3 JSD", cells[2], f"stats.puct_{cell}_jsd_attn_K3.mean", st[f"puct_{cell}_jsd_attn_K3"]["mean"])
    check(f"sec6 PUCT {cell} attn_K3 KL", cells[3], f"stats.puct_{cell}_kl_attn_K3.mean", st[f"puct_{cell}_kl_attn_K3"]["mean"])
# ---- (1e) TL;DR and section-3 single numbers (hand-mapped)
V = r["verdicts"]
single = [("TL;DR D0 Pearson", "0.9995", st["clause1_corr_std6"]["mean"]), ("TL;DR D0 Pearson lo", "0.9994", st["clause1_corr_std6"]["lo"]),
          ("TL;DR D0 Pearson hi", "0.9995", st["clause1_corr_std6"]["hi"]), ("sec3 Pearson", "0.99946", st["clause1_corr_std6"]["mean"]),
          ("sec3 Pearson lo", "0.99938", st["clause1_corr_std6"]["lo"]), ("sec3 Pearson hi", "0.99953", st["clause1_corr_std6"]["hi"]),
          ("TL;DR D1.1 KL32", "0.0303", st["fs_kl_looped_attn_K32"]["mean"]), ("TL;DR D1.1 KL32 lo", "0.0287", st["fs_kl_looped_attn_K32"]["lo"]),
          ("TL;DR D1.1 KL32 hi", "0.0319", st["fs_kl_looped_attn_K32"]["hi"]), ("TL;DR D1.1 JSD24", "0.00575", st["fs_jsd_looped_attn_K24"]["mean"]),
          ("TL;DR D1.1 JSD24 lo", "0.00519", st["fs_jsd_looped_attn_K24"]["lo"]), ("TL;DR D1.1 JSD24 hi", "0.00631", st["fs_jsd_looped_attn_K24"]["hi"]),
          ("TL;DR D1.1 largest step", "-0.00029", V["E1-D1.1"]["stats"]["max_step_increase"]),
          ("TL;DR D1.2 min improvement", "0.0228", -max(V["E1-D1.2"]["stats"]["looped_attn_max_increase_per_seed"].values())),
          ("TL;DR D1.2 gain16", "0.0891", V["E1-D1.2"]["stats"]["looped_attn_K0_minus_K16"]["mean"]),
          ("TL;DR D1.2 gain16 lo", "0.0883", V["E1-D1.2"]["stats"]["looped_attn_K0_minus_K16"]["lo"]),
          ("TL;DR D1.2 gain16 hi", "0.0900", V["E1-D1.2"]["stats"]["looped_attn_K0_minus_K16"]["hi"]),
          ("TL;DR D1.2 vs mmd", "0.0227", V["E1-D1.2"]["stats"]["mmd_minus_looped_attn_K16"]["mean"]),
          ("TL;DR D1.2 vs mmd lo", "0.0200", V["E1-D1.2"]["stats"]["mmd_minus_looped_attn_K16"]["lo"]),
          ("TL;DR D1.2 vs mmd hi", "0.0253", V["E1-D1.2"]["stats"]["mmd_minus_looped_attn_K16"]["hi"]),
          ("TL;DR D1.2 KL0", "0.1444", V["E1-D1.2"]["stats"]["KL0"]["mean"]), ("TL;DR D1.2 KL0 lo", "0.1440", V["E1-D1.2"]["stats"]["KL0"]["lo"]),
          ("TL;DR D1.2 KL0 hi", "0.1449", V["E1-D1.2"]["stats"]["KL0"]["hi"]),
          ("TL;DR D1.2 mmd dKL16", "-0.0665", V["E1-D1.2"]["stats"]["mmd_K16_minus_K0"]["mean"]),
          ("TL;DR D1.2 mmd lo", "-0.0700", V["E1-D1.2"]["stats"]["mmd_K16_minus_K0"]["lo"]), ("TL;DR D1.2 mmd hi", "-0.0630", V["E1-D1.2"]["stats"]["mmd_K16_minus_K0"]["hi"]),
          ("sec4 mix dKL16", "-0.0410", V["E1-D1.2"]["stats"]["mix_K16_minus_K0 (reported)"]["mean"]),
          ("TL;DR D2.1 W16/W1 0.3", "1.164", W["jacobi_prec_W16/W1_t0.3"]["mean"]), ("TL;DR D2.1 lo", "1.147", W["jacobi_prec_W16/W1_t0.3"]["lo"]),
          ("TL;DR D2.1 hi", "1.182", W["jacobi_prec_W16/W1_t0.3"]["hi"]), ("TL;DR D2.1 R7 jacobi", "1.455", W["jacobi_W16/W1_t0.3"]["mean"]),
          ("TL;DR D2.1 W8/W1 0.3", "1.150", W["jacobi_prec_W8/W1_t0.3"]["mean"]), ("TL;DR D2.1 W8/W1 1.5", "1.765", W["jacobi_prec_W8/W1_t1.5"]["mean"]),
          ("TL;DR D2.1 W8/W1 2.5", "2.501", W["jacobi_prec_W8/W1_t2.5"]["mean"]), ("TL;DR D2.1 W16/W1 2.5", "2.816", W["jacobi_prec_W16/W1_t2.5"]["mean"]),
          ("TL;DR D2.1 FS KL", "0.0414", st["fs_kl_jacobi_prec_K24"]["mean"]), ("TL;DR D2.1 FS lo", "0.0396", st["fs_kl_jacobi_prec_K24"]["lo"]),
          ("TL;DR D2.1 FS hi", "0.0432", st["fs_kl_jacobi_prec_K24"]["hi"]),
          ("sec5 vs looped_lin", "0.0263", V["E1-D2.1"]["stats"]["beats"]["looped_lin"]["kl_diff"]["mean"]),
          ("sec5 vs kc_K1", "0.0547", V["E1-D2.1"]["stats"]["beats"]["kc_K1"]["kl_diff"]["mean"]),
          ("TL;DR D2.2 W16/W1 2.5", "2.455", W["econ_D3_W16/W1_t2.5"]["mean"]), ("TL;DR D2.2 W16/W1 0.3", "1.417", W["econ_D3_W16/W1_t0.3"]["mean"]),
          ("TL;DR D2.2 lo", "1.395", W["econ_D3_W16/W1_t0.3"]["lo"]), ("TL;DR D2.2 hi", "1.439", W["econ_D3_W16/W1_t0.3"]["hi"]),
          ("TL;DR D2.2 MSE diff", "0.00178", V["E1-D2.2"]["stats"]["mse_kc_K3_minus_econ_D3_all_tiers"]["mean"]),
          ("TL;DR D2.2 MSE lo", "0.00110", V["E1-D2.2"]["stats"]["mse_kc_K3_minus_econ_D3_all_tiers"]["lo"]),
          ("TL;DR D2.2 MSE hi", "0.00247", V["E1-D2.2"]["stats"]["mse_kc_K3_minus_econ_D3_all_tiers"]["hi"]),
          ("sec1 econ_R7 W16/W1 0.3", "1.491", W["econ_R7_W16/W1_t0.3"]["mean"]), ("sec6 econ_R7 W16/W1 2.5", "2.647", W["econ_R7_W16/W1_t2.5"]["mean"]),
          ("sec5 R7 jacobi FS KL", "0.0405", st["fs_kl_jacobi_K24"]["mean"])]
for k in ("101", "102", "103"):
    pass
for nm, a in (("looped_attn_T3", "looped_attn"), ("attn_L3-K3", "attn_K3"), ("jacobi_lin_T4_prec", "jacobi_prec"), ("econ_kc_K3 (D3 fit)", "econ_D3"),
              ("kc_K3", "kc_K3"), ("kc_K1", "kc_K1"), ("ataraxos_mmd", "mmd")):
    m = re.search(re.escape(nm) + r" (0\.\d{5})", text)
    if m:
        single.append((f"sec6 D3 MSE {a}", m.group(1), st["mse_all_tiers"][a]["mean"]))
for where, tok, v in single:
    check(where, tok, "(hand-mapped)", v)
rep["structured_n"] = len(rep["structured"])
rep["structured_mismatches"] = [x for x in rep["structured"] if not x["match"]]

# ---- (2) automatic sweep
pool = []


def flat(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            flat(v, f"{p}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            flat(v, f"{p}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        pool.append((float(x), p))


flat(r, "results")
for s in (101, 102, 103):
    S = json.load(open(f"{R8}/results/main/s{s}/summary.json"))
    flat(S["test"], f"summary{s}.test")
    flat(S["fit"], f"summary{s}.fit")
for tok in re.findall(r"-?\d+\.\d+", open(f"{R8}/results/main/analysis/numbers.txt").read()):
    pool.append((float(tok), "numbers.txt"))
for tok in re.findall(r"cost_usd=([0-9.]+)", open(f"{R8}/GPU-LOG.md").read() + open(f"{R8}/../../logical-soup/GPU-LOG.md").read()):
    pool.append((float(tok), "GPU-LOG"))
CONST = {0.3: "tier", 1.5: "tier", 2.5: "tier", 0.127: "flip", 0.28: "flip", 0.15: "sensor flip", 0.033: "bar", 0.0075: "bar", 0.050: "bar",
         0.015: "bar", 0.0005: "bar", 1.25: "bar", 1.30: "bar", 2.0: "bar", 2.2: "bar", 0.040: "bar", 0.999: "bar", 0.80: "ceiling",
         0.87: "rate", 2.20: "group cap", 1.449: "spec R7", 0.9494: "spec R6", 0.9863: "spec R7", 0.035: "spec", 0.002: "R7 union KL",
         4.303: "t", 0.0: "zero", 0.62: "total", 0.36: "run1", 0.26: "run2", 1.0: "one"}
toks = []
for i, ln in enumerate(lines, 1):
    if ln.startswith("| VM") or "sha256" in ln:
        pass
    for m in re.finditer(r"(?<![\w.:/-])[-+]?\d+\.\d+(?![\w.:])", ln):
        tok = m.group(0)
        ctx = ln[max(0, m.start() - 60):m.end() + 30]
        if re.search(r"\d{4}-\d\d-\d\d|\d\d:\d\d", ln[max(0, m.start() - 3):m.end() + 3]):
            continue
        toks.append({"line": i, "token": tok, "context": ctx})
for t in toks:
    tk = t["token"].replace("+", "")
    d = len(tk.split(".")[1])
    v = float(tk)
    hits = [p for x, p in pool if fmt(x, d) == tk]
    c = [n for x, n in CONST.items() if fmt(x, d) == tk]
    t["auto"] = "data" if hits else ("constant" if c else "unmatched")
    t["hits"] = hits[:3]
    t["const"] = c[:1]
from collections import Counter
rep["sweep"]["n_tokens"] = len(toks)
rep["sweep"]["auto_counts"] = dict(Counter(t["auto"] for t in toks))
rep["sweep"]["tokens"] = [{k: t[k] for k in ("line", "token", "auto", "hits", "const", "context")} for t in toks]
if len(sys.argv) > 1 and sys.argv[1] == "sample":
    random.seed(20261017)
    idx = sorted(random.sample(range(len(toks)), 40))
    for j in idx:
        t = toks[j]
        print(f"[{j}] L{t['line']} {t['token']!r:10} auto={t['auto']:9s} hits={t['hits'][:2]} | {t['context']}")
    sys.exit(0)
hl = os.path.join(OUT, "a8_hand_labels.json")
if os.path.exists(hl):
    H = json.load(open(hl))
    agree, rows = 0, []
    for h in H:
        t = toks[h["i"]]
        assert t["token"] == h["token"], (t, h)
        auto_ok = t["auto"] in ("data", "constant")
        hand_ok = h["verdict"] in ("correct", "constant")
        agree += auto_ok == hand_ok
        rows.append({**h, "auto": t["auto"], "agree": auto_ok == hand_ok})
    rep["hand_sample"] = {"n": len(H), "agreement_with_auto_rule": agree / len(H), "hand_correct_or_constant": sum(h["verdict"] in ("correct", "constant") for h in H),
                          "hand_incorrect": [h for h in H if h["verdict"] == "incorrect"], "rows": rows}
print(json.dumps({k: v for k, v in rep.items() if k not in ("structured", "sweep")}, indent=1)[:6000])
print("structured checks:", rep["structured_n"], "mismatches:", len(rep["structured_mismatches"]))
print("sweep:", rep["sweep"]["n_tokens"], rep["sweep"]["auto_counts"])
for t in toks:
    if t["auto"] == "unmatched":
        print("  unmatched L%d %s | %s" % (t["line"], t["token"], t["context"]))
json.dump(rep, open(os.path.join(OUT, "a8_results_numbers.json"), "w"), indent=1)
