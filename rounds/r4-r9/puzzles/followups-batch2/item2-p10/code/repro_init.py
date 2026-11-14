"""Reproduce P10's 'constant at init' count on the ORIGINAL code path (no training).
Builds M2 exactly as train_s6.m2 does for seed s, then takes P10's measurement: p1 = mean of the hard event
(phi > 0) of each learned encoder at init on the given split; P10's 9/15 counts min(p1, 1-p1) <= 0.001.
  python repro_init.py SPLIT SEEDS...    (P10_TMP = dir with s6.npz, P10_RES = dir with s6_bank_T1_seed0.json)"""
import json, sys
import torch
import train_s6 as t6

split = sys.argv[1]
out = {}
for s in map(int, sys.argv[2:]):
    torch.manual_seed(s)
    d = t6.S6(s)
    at = t6.Atoms()
    planted = json.load(open(f"{t6.RES}/s6_bank_T1_seed0.json"))["planted"]["spec"]
    model = t6.M2(1, planted, at).to(t6.DEV)
    bt = d.batch(split, idx=d.idx[split], aug=False)
    with torch.no_grad():
        p1 = [float((model.phis[k](bt["sol"], bt["mask"]) > 0).float().mean()) for k in range(t6.K_LEARN)]
    out[s] = p1
    print(s, [round(p, 4) for p in p1], flush=True)
n = sum(len(v) for v in out.values())
for thr in (0.0, 0.001, 0.01):
    c = sum(1 for v in out.values() for p in v if min(p, 1 - p) <= thr)
    print(f"constant (min(p1,1-p1) <= {thr}): {c}/{n}")
