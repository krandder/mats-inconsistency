#!/usr/bin/env python3
"""R9-E1 audit, check 8 screening rule: every decimal number in the PROSE of RESULTS-R9-E1.md (table rows excluded) is looked up
as a string in the files RESULTS cites (numbers.txt, verdicts.md, GPU-LOG.md, PREREG, launch/driver logs, R8 results.json values
printed at several precisions). Tokens not found are listed for hand review; a seeded random sample of 30 found tokens is printed
for hand labelling (does the token match the quantity the sentence names?). Run: python3 check8_prose_tokens.py"""
import json
import random
import re

R9 = "/home/agent/workspaces/eurisko-study/li-net/r9e1"
txt = open(f"{R9}/RESULTS-R9-E1.md").read().splitlines()
src = ""
for p in (f"{R9}/results/main/analysis/numbers.txt", f"{R9}/results/main/analysis/verdicts.md", f"{R9}/GPU-LOG.md",
          f"{R9}/PREREG-R9-E1.md", f"{R9}/PREREG-R9-E1-AMENDMENTS.md", "/tmp/r9e1/logs/launch-r9e1-main.log",
          "/tmp/r9e1/logs/main/driver.txt", f"{R9}/results/main/setup.json"):
    src += open(p).read() + "\n"
r8 = json.load(open("/home/agent/workspaces/eurisko-study/li-net/r8e1/results/main/analysis/results.json"))
for k in ("fs_kl_looped_attn_K32", "fs_jsd_looped_attn_K24", "fs_kl_jacobi_prec_K24"):
    v = r8["stats"][k]["mean"]
    src += " ".join(f"{v:.{n}f}" for n in (3, 4, 5)) + "\n"
for k in ("jacobi_prec_W16/W1_t0.3", "jacobi_prec_W16/W1_t2.5"):
    src += f"{r8['stats']['weights'][k]['mean']:.3f}\n"
tok = re.compile(r"[+-]?\d+\.\d+")
found, missing = [], []
for i, line in enumerate(txt, 1):
    if line.startswith("|") or line.startswith("<!--"):
        continue
    for m in tok.finditer(line):
        t = m.group(0)
        core = t.lstrip("+-")
        hit = core in src
        (found if hit else missing).append((i, t, line.strip()[:150]))
print(f"prose decimal tokens: {len(found) + len(missing)}; found verbatim in cited files: {len(found)}; not found: {len(missing)}")
for i, t, l in missing:
    print(f"  NOT FOUND line {i}: {t} | {l}")
random.seed(20261017)
print("seeded sample of 30 found tokens for hand labelling:")
for i, t, l in random.sample(found, 30):
    print(f"  line {i}: {t} | {l}")


# ---- double-rounding screen: a prose token with d decimals that equals a numbers.txt value with d+1 decimals ending in 5, rounded
#      half-up (e.g. 2.195 -> 2.20). Each candidate is checked by hand against summary.json / results.json in the audit.
from decimal import Decimal, ROUND_HALF_UP  # noqa: E402

nums = set(re.findall(r"[+-]?\d+\.\d*5\b", open(f"{R9}/results/main/analysis/numbers.txt").read()))
cand = []
for i, t, l in found + missing:
    core = t.lstrip("+-")
    d = len(core.split(".")[1])
    for v in nums:
        vc = v.lstrip("+-")
        if len(vc.split(".")[1]) == d + 1 and str(Decimal(vc).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)) == core:
            cand.append((i, t, v, l[:110]))
print(f"double-rounding candidates (prose token, numbers.txt value): {len(cand)}")
for c in cand:
    print("  line", c[0], c[1], "<-", c[2], "|", c[3])
