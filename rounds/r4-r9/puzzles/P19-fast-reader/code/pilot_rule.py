"""P19 pilot rule (truth-free; seed 99 only). python pilot_rule.py PILOT_RES_DIR
Reads pilotlr_maze_s99.json and r7brep_maze_s99_r0.json (never a truth file) and prints:
  LR*        = argmin over LR_GRID of the total fast-reader codelength (all members, both contexts, mean of 2 orders)
  ratio(K,T) = FR compute per ranked item / one R7b replicate per item, where FR compute = both current readers'
               training + K stream orders x (T/64) of the whole-stream coding + training passes for base + ranked members
  (K*, T*)   = first config in PREF whose ratio <= 0.045 (10% margin under the 0.05 kill line); none -> (1, 64)."""
import json
import os
import sys

R = sys.argv[1]
d = json.load(open(os.path.join(R, "pilotlr_maze_s99.json")))
w = json.load(open(os.path.join(R, "r7brep_maze_s99_r0.json")))
N_RANK = {"empty": 23, "g2g4": 21}
CIN = {"empty": 5 + 0 + 1, "g2g4": 5 + 2 + 1}
HID, CELLS, CHUNK, REPLAY, T = 16, 121, 16, 16, 64
PREF = [(4, 64), (2, 64), (1, 64), (1, 32), (1, 16)]


def macs(cin):  # GConv(cin, hid 16, 2 layers, 1 out), per member per cell (p8lib.GConv.macs)
    return cin * HID * 9 + HID * HID * 9 + HID


ntrain = CHUNK + (T - 1) * (CHUNK + REPLAY)
fr_member = {c: (ntrain + T * CHUNK) * CELLS * macs(CIN[c]) for c in N_RANK}   # = p19.fr_run work
fr_one = sum((N_RANK[c] + 1) * fr_member[c] for c in N_RANK)
readers = sum(v["work"] for v in d["readers"].values())
n = sum(N_RANK.values())
w7 = w["work"] / n
tot = {lr: g["total_bits"] for lr, g in d["grid"].items()}
lr_star = min(tot, key=tot.get)
ratios = {f"K{k}_T{t}": (readers + k * t / T * fr_one) / n / w7 for k in (1, 2, 4) for t in (16, 32, 64)}
pick = next(((k, t) for k, t in PREF if ratios[f"K{k}_T{t}"] <= 0.045), (1, 64))
out = dict(total_bits_by_lr=tot, lr_star=float(lr_star), readers_work=readers, fr_member_work=fr_member,
           fr_one_order_work=fr_one, r7b_1rep_work_per_item=w7, readers_share_of_r7b_1rep=readers / n / w7,
           ratios={k: round(v, 4) for k, v in ratios.items()}, K_star=pick[0], T_star=pick[1],
           cpu=dict(pilotlr=d["cpu_secs"], reader=d["reader_cpu"], r7brep=w["cpu_secs"]))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(R, "pilot_rule.json"), "w"), indent=1)
