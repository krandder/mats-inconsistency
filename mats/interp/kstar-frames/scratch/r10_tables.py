"""Markdown tables for the report from runs/*.json (net means; per-net spread where noted).
python r10_tables.py > ../logs/tables.md"""
import glob, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
R = os.path.join(HERE, '..', 'runs')
SEEDS = (400, 401, 402)
WID = (32, 64, 128)
MAIN = {32: (8, 16, 24), 16: (4, 8, 12)}


def J(name):
    f = os.path.join(R, name)
    return json.load(open(f)) if os.path.exists(f) else None


def tag(n, s, L):
    return f"n{n}_s{s}_L{L}"


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float('nan')


def f3(x):
    return '-' if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def fg(x):
    return '-' if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2g}"


print("## Table A: carrying the frame along one path (run A; O = mean cos^2 of principal angles; net means)\n")
print("| n | L | c (L') | k* | chance | push-fwd 1 | push-fwd 8 | past frame | dual 1 | dual 8 | backward 1 | new noise |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for L in (32, 16):
    for n in WID:
        rows = [J(f'A_{tag(n, s, L)}.json') for s in SEEDS]
        rows = [r for r in rows if r and 'scores' in r]
        for c in MAIN[L]:
            if not rows:
                continue
            cs = str(c)
            g = lambda nm, j=0: mean([r['scores'][nm][cs][j] for r in rows])
            print(f"| {n} | {L} | {c} ({L-c}) | {mean([r['k'][cs] for r in rows]):.1f} | {mean([r['chance'][cs] for r in rows]):.2f} | "
                  f"{f3(g('fwd_1'))} | {f3(g('fwd_8'))} | {f3(g('blv'))} | {f3(g('dual_1'))} | {f3(g('dual_8'))} | "
                  f"{f3(g('bwd_1'))} | {f3(g('noise_1'))} |")
print()
# dual split by growth of the active count, pooled over layers with L' >= 8
print("Dual transport, one step, split by whether the active count grows (pooled over layers with L' >= 8, nets):\n")
print("| n | L | O when m_c <= m_{c-1} | O when m_c > m_{c-1} | share of (path, layer) cells with growth |")
print("|---|---|---|---|---|")
for L in (32, 16):
    for n in WID:
        a, b, cnt = [], [], []
        for s in SEEDS:
            f = os.path.join(R, f'A_{tag(n, s, L)}.npz')
            if not os.path.exists(f):
                continue
            d = np.load(f)
            sc, m, pred = d['sc'], d['mact'], list(d['pred'])
            i = pred.index('dual_1')
            for c in range(2, L - 7):
                O = sc[:, c, i, 0]; gr = m[:, c] > m[:, c - 1]
                a.append(O[~gr]); b.append(O[gr]); cnt.append(gr)
        if a:
            a, b, cnt = np.concatenate(a), np.concatenate(b), np.concatenate(cnt)
            print(f"| {n} | {L} | {np.nanmean(a):.3f} | {np.nanmean(b):.3f} | {cnt.mean():.2f} |")
print()
print("## Table B: predicting a held-out input's frame at c = L/2 (CS / O, net means; T = sensitivity outside)\n")
print("| n | L | k* | chance O | marginal gates (proj) | best fixed (proj) | learned map | nearest of 16 | nearest of 256 | nearest of all A | new noise (run A) | T: true / best fixed / learned |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for L in (32, 16):
    for n in WID:
        c = L // 2; cs = str(c)
        rows = [J(f'Bres_{tag(n, s, L)}.json') for s in SEEDS]
        rows = [r for r in rows if r]
        if not rows:
            continue
        NM = 8192 if n == 128 else 16384
        mlp = [J(f'D_{tag(n, s, L)}_c{c}_N{NM}.json') for s in SEEDS]
        mlp = [m for m in mlp if m]
        A_ = [J(f'A_{tag(n, s, L)}.json') for s in SEEDS]
        A_ = [a for a in A_ if a and 'scores' in a]
        g = lambda nm, key: mean([r[cs][nm][key] for r in rows])
        Rmax = max(int(x) for x in rows[0][cs]['nearest'])
        nr = lambda Rv, key: mean([r[cs]['nearest'][str(Rv)][key] for r in rows if str(Rv) in r[cs]['nearest']])
        mm = lambda key: mean([m['mlp'][key] for m in mlp])
        print(f"| {n} | {L} | {rows[0][cs]['k']} | {g('rand_proj', 'O'):.2f} | {f3(g('Gb_proj','CS'))} / {f3(g('Gb_proj','O'))} | "
              f"{f3(g('GM_proj','CS'))} / {f3(g('GM_proj','O'))} | {f3(mm('CS'))} / {f3(mm('O'))} ({len(mlp)} nets) | "
              f"{f3(nr(16,'CS_proj'))} / {f3(nr(16,'O_proj'))} | {f3(nr(256,'CS_proj'))} / {f3(nr(256,'O_proj'))} | "
              f"{f3(nr(Rmax,'CS_proj'))} / {f3(nr(Rmax,'O_proj'))} (R = {Rmax}) | "
              f"{f3(mean([a['scores']['noise_1'][cs][1] for a in A_]))} / {f3(mean([a['scores']['noise_1'][cs][0] for a in A_]))} | "
              f"{fg(g('true','T'))} / {fg(g('GM_proj','T'))} / {fg(mm('T'))} |")
print()
print("Learned map, learning curve on net 400 at c = L/2 (CS / O):\n")
print("| n | L | N = 1024 | N = 4096 | N = max | best fixed |")
print("|---|---|---|---|---|---|")
for L in (32, 16):
    for n in WID:
        c = L // 2
        NM = 8192 if n == 128 else 16384
        v = [J(f'D_{tag(n, 400, L)}_c{c}_N{N}.json') for N in (1024, 4096, NM)]
        if not any(v):
            continue
        fx = next((x['fixed_proj'] for x in v if x), None)
        print(f"| {n} | {L} | " + " | ".join(f"{f3(x['mlp']['CS'])} / {f3(x['mlp']['O'])} (ep {x['epochs']})" if x else '-' for x in v)
              + f" | {f3(fx['CS'])} / {f3(fx['O'])} |")
print()
print("Learned map at the other main layers of L = 32 (net 400, N = max; CS / O against best fixed):\n")
for n in WID:
    NM = 8192 if n == 128 else 16384
    for c in (8, 24):
        x = J(f'D_{tag(n, 400, 32)}_c{c}_N{NM}.json')
        if x:
            print(f"- n = {n}, c = {c}: learned {f3(x['mlp']['CS'])} / {f3(x['mlp']['O'])}, fixed {f3(x['fixed_proj']['CS'])} / "
                  f"{f3(x['fixed_proj']['O'])}, T {fg(x['mlp']['T'])} vs {fg(x['fixed_proj']['T'])}")
print()
print("## Table D: atlas at c = L/2, float64 recompute (runs/Bcover2_*.json): held-out inputs against the stored inputs A\n")
print("Ball mass = share of the stored inputs whose frame, restricted to the input's active units, captures >= 1 - delta of its top-k* sensitivity.\n")
print("| n | L | k* | delta | ball mass, mean over inputs (median over nets; range) | median over inputs (median over nets) | share with no stored frame within delta (range) | all stored frames cover (range) | R99 random centres, exact form (nets) | greedy: held-out coverage (centres) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for L in (32, 16):
    for n in WID:
        rows = [J(f'Bcover2_{tag(n, s, L)}.json') for s in SEEDS]
        rows = [r for r in rows if r]
        for dl in ('0.3', '0.2', '0.1', '0.05'):
            if not rows:
                continue
            cv = [r['cover'][dl] for r in rows]
            pm = [x['p_mean'] for x in cv]
            r99 = [x['R99_exact'] for x in cv]
            r99s = ", ".join('inf' if not np.isfinite(v) else f"{v:.3g}" for v in r99)
            print(f"| {n} | {L} | {rows[0]['k']} | {dl} | {np.median(pm):.2g} ({min(pm):.2g}-{max(pm):.2g}) | "
                  f"{np.median([x['p_median'] for x in cv]):.2g} | {min(x['frac_p0'] for x in cv):.3f}-{max(x['frac_p0'] for x in cv):.3f} | "
                  f"{min(x['union_all_A'] for x in cv):.3f}-{max(x['union_all_A'] for x in cv):.3f} | {r99s} | "
                  f"{mean([x['greedy_B2_final'] for x in cv]):.3f} ({mean([x['greedy_centres_B1'] for x in cv]):.0f}) |")
