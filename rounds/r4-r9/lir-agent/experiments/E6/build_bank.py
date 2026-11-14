"""Freeze the E6 bank before any run: environment-written outcomes from simulator branches of many
policies, on boards disjoint from training (seeds 1e6*(s+1)+..), warm-start (1e4*(s+1)+..) and eval (5e7+..)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e6lib as E
preset = sys.argv[1] if len(sys.argv) > 1 else "easy"
bank = E.build_bank(preset, [70_000_000 + i for i in range(16)], per_board=16, seed=6)
calib = E.build_bank(preset, [80_000_000 + i for i in range(24)], per_board=12, seed=7)
out = {"preset": preset, "bank": bank, "calib": calib, "hash": E.bank_hash(bank), "calib_hash": E.bank_hash(calib)}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[2] if len(sys.argv) > 2 else "bank.json")
json.dump(out, open(path, "w"))
from collections import Counter
print(path, len(bank), len(calib), out["hash"], Counter(it["u"] for it in bank))
