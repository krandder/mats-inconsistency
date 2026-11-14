#!/usr/bin/env python3
"""R7-E1 audit, check 8 support: (1) the key-attribution split in the other order (pool first, then keys) and with
attn_L3-K1 as the raw-key attention pool; (2) looped_attn_T3's three-regime weight law per seed against the E1-D3 bars;
(3) looped_lin_T4 has no W(N) (not in WPOOLS); (4) kc pools' MSE ranking at every tier. From summary.json at full precision.
Usage: python3 extra_checks.py > extra-checks-output.txt"""
import json, math, time
R7 = "/home/agent/workspaces/eurisko-study/li-net/r7e1"
SEEDS = (101, 102, 103); TAUS = ("0.3", "1.5", "2.5"); NS = (1, 2, 4, 8, 16); TQ = 4.302652729911275
T = {s: json.load(open(f"{R7}/results/main/s{s}/summary.json"))["test"] for s in SEEDS}
def ci(xs):
    m = sum(xs) / 3; sd = math.sqrt(sum((x - m) ** 2 for x in xs) / 2); h = TQ * sd / math.sqrt(3)
    return f"{m:.3f} [{m - h:.3f}, {m + h:.3f}] ({' / '.join(f'{x:.3f}' for x in xs)})"
mse = lambda s, t, a: sum(T[s]["controlled"][f"t{t}_N{N}"][f"mse_{a}"] for N in NS) / 5
W = lambda s, t, N, a: T[s]["controlled"][f"t{t}_N{N}"][f"W_{a}"]
print("run at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
print("== (1) share of the kc_K1 -> attn_L3-K3 gap credited to the keys, two decomposition orders")
meas = {"FS KL": lambda s, a: T[s][f"fs_kl_{a}"], "FS JSD": lambda s, a: T[s][f"fs_jsd_{a}"],
        **{f"MSE tau {t}": (lambda s, a, t=t: mse(s, t, a)) for t in TAUS}}
for name, X in meas.items():
    keys_first = [(X(s, "kc_K1") - X(s, "kc_K3")) / (X(s, "kc_K1") - X(s, "attn_K3")) for s in SEEDS]   # RESULTS' order
    pool_first = [(X(s, "attn_K1") - X(s, "attn_K3")) / (X(s, "kc_K1") - X(s, "attn_K3")) for s in SEEDS]   # keys swapped inside attention
    print(f"  {name:10s} keys share, kc order (RESULTS): {ci(keys_first)} | keys share, attention order (attn_K1 -> attn_K3): {ci(pool_first)}")
print("== (2) looped_attn_T3 weight law per seed (E1-D3 bars: W16/W1<=1.25 at 0.3; W8/W1>=2.20, CI lo>1.90, W16/W1>=2.80 at 2.5; monotone)")
for t in TAUS:
    print(f"  tau {t}: W8/W1 {ci([W(s, t, 8, 'looped_attn') / W(s, t, 1, 'looped_attn') for s in SEEDS])}; W16/W1 {ci([W(s, t, 16, 'looped_attn') / W(s, t, 1, 'looped_attn') for s in SEEDS])}")
print("  monotone per seed:", [W(s, '0.3', 8, 'looped_attn') / W(s, '0.3', 1, 'looped_attn') < W(s, '1.5', 8, 'looped_attn') / W(s, '1.5', 1, 'looped_attn') < W(s, '2.5', 8, 'looped_attn') / W(s, '2.5', 1, 'looped_attn') for s in SEEDS])
print("  looped_attn MSE cut vs kc_K1 at tau 2.5:", ci([1 - mse(s, "2.5", "looped_attn") / mse(s, "2.5", "kc_K1") for s in SEEDS]))
print("== (3) W(N) keys present for looped_lin:", any("W_looped_lin" in T[101]["controlled"]["t0.3_N1"] for _ in [0]), "; W keys:", sorted(k for k in T[101]["controlled"]["t0.3_N1"] if k.startswith("W_")))
print("== (4) kc pool MSE ranking per tier (3-seed means)")
for t in TAUS:
    m = {a: sum(mse(s, t, a) for s in SEEDS) / 3 for a in ("kc_K1", "kc_K3", "kc_K4", "attn_K3")}
    print(f"  tau {t}: " + ", ".join(f"{a} {v:.5f}" for a, v in sorted(m.items(), key=lambda kv: kv[1])))
print("== (5) jacobi vs looped_lin vs econ: key heads differ (jacobi own K3, looped_lin own K4, econ frozen attn_L3-K3 K3)")
print("== (6) econ_K3 W ratios (one precision-crowding step), for context")
for t in TAUS:
    print(f"  tau {t}: econ W16/W1 {ci([W(s, t, 16, 'econ_K3') / W(s, t, 1, 'econ_K3') for s in SEEDS])}")
