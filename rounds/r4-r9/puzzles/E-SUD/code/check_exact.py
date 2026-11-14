"""Cross-checks of the exact ground truth (run before launch; output in results/check_exact.json).
4x4: marginals from the DFS solver == marginals from filtering the full 288-grid table (independent enumeration).
6x6: solver marginals == an independent row-permutation enumerator on 300 test boards, and every stored completion is a valid grid consistent with the givens.
Negative boards: zero completions by both methods."""
import itertools, json, sys
import numpy as np
from boards import solve_all, geom, valid_full

PERMS = list(itertools.permutations(range(1, 7)))
BOX = geom(6)[2]


def rowwise(b):
    """Independent enumerator: rows as permutations of 1..6 that match the givens, stacked with column/box checks."""
    G = b.reshape(6, 6)
    rows = [[p for p in PERMS if all(G[r, c] in (0, p[c]) for c in range(6))] for r in range(6)]
    sols = []

    def rec(r, acc):
        if r == 6:
            sols.append(np.array(acc).reshape(-1)); return
        for p in rows[r]:
            ok = True
            for c in range(6):
                for rr in range(r):
                    if acc[rr][c] == p[c] or (BOX[rr * 6 + c // 3 * 3] == BOX[r * 6 + c // 3 * 3] and p[c] in acc[rr][c // 3 * 3: c // 3 * 3 + 3]):
                        ok = False; break
                if not ok:
                    break
            if ok:
                rec(r + 1, acc + [p])
    rec(0, [])
    return sols


data = sys.argv[1]
out = {}
table = np.array(solve_all([0] * 16, 4, 10 ** 6))
assert table.shape == (288, 16)
d = np.load(f'{data}/s4_test.npz')
mx = 0.0; bad = 0
for b, s, m in zip(d['board'], d['solvable'], d['marg']):
    cons = table[((table == b) | (b == 0)).all(1)]
    if s == 0:
        bad += len(cons) != 0; continue
    mm = np.stack([(cons == k).mean(0) for k in range(1, 5)], -1)
    mx = max(mx, float(np.abs(mm - m).max()))
out['s4'] = dict(boards=int(len(d['board'])), max_abs_marg_diff=mx, negatives_with_solutions=int(bad))

d = np.load(f'{data}/s6_test.npz')
rng = np.random.default_rng(0)
nchk = 0; mx = 0.0; bad = 0; badsol = 0
for b, s, m, y in zip(d['board'], d['solvable'], d['marg'], d['outcome']):
    if s == 1:
        badsol += not (valid_full(list(y), 6) and all((y == b) | (b == 0)))
    if nchk >= 300:
        continue
    nchk += 1
    sols = rowwise(b)
    if s == 0:
        bad += len(sols) != 0; continue
    S = np.array(sols)
    mm = np.stack([(S == k).mean(0) for k in range(1, 7)], -1)
    mx = max(mx, float(np.abs(mm - m).max()))
out['s6'] = dict(boards=int(len(d['board'])), brute_checked=nchk, max_abs_marg_diff=mx, negatives_with_solutions=int(bad),
                 invalid_stored_outcomes=int(badsol))
print(json.dumps(out))
json.dump(out, open(sys.argv[2], 'w'), indent=1)
