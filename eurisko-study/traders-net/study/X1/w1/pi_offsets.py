"""X1 w1: how often do wealth-net pi questions share digits inside the model window?

Two pi-parity questions with offsets a < b = a + k read the same digit sequence shifted by k, so
question a's label at t equals question b's label at t - k, already revealed when 1 <= k <= W.
Counts, for seeds 1..N of wealth-net make_questions(Q=64), the pairs with 1 <= k <= W (W=32, the
S0 w1 window) and k = 0 (identical questions).
Usage: PYTHONPATH=<wealth-net>/.pylib:<wealth-net> python3 pi_offsets.py 2000 results/pi_offsets.json
"""
import itertools, json, sys
import streams
N, OUT, W = int(sys.argv[1]), sys.argv[2], 32
rows = []
for s in range(1, N + 1):
    b, f, d = streams.make_questions(64, 8, s)
    offs = sorted(int(x.split("=")[1]) for x, ff in zip(d, f) if ff == 5)
    diffs = [b_ - a for a, b_ in itertools.combinations(offs, 2)]
    rows.append({"seed": s, "offsets": offs, "n_pairs_in_window": sum(1 <= k <= W for k in diffs),
                 "n_identical": sum(k == 0 for k in diffs),
                 "n_questions_leaked": len({i for i, j in itertools.combinations(range(8), 2) if 1 <= offs[j] - offs[i] <= W for i in (i,)})})
frac = sum(r["n_pairs_in_window"] > 0 for r in rows) / N
res = {"W": W, "N_seeds": N, "frac_seeds_with_pair_in_window": frac,
       "seeds_1_5": [r for r in rows if r["seed"] <= 5]}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "seeds_1_5"}), [(r["seed"], r["n_pairs_in_window"], r["n_questions_leaked"]) for r in rows[:5]])
