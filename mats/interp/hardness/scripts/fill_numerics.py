"""Fill the Section 6.3 placeholders in draft.md and the summary placeholders in notes/head.txt from the numerics.
usage: fill_numerics.py   (reads numerics/results/*.json through the analyze scripts' logic)"""
import glob, json, math, os, subprocess, sys
import numpy as np

base = '/home/agent/workspaces/mats/work/hardness/'
num = base + 'numerics/'
py = '/home/agent/.venvs/arc-noisy/bin/python'

k2_out = subprocess.run([py, '-I', num + 'analyze_k2s.py'], capture_output=True, text=True, cwd=num).stdout
k3_out = subprocess.run([py, '-I', num + 'analyze_k3f.py'], capture_output=True, text=True, cwd=num).stdout

lines = k2_out.splitlines()
i0 = next(i for i, l in enumerate(lines) if l.startswith('| alpha |'))
i1 = next(i for i in range(i0, len(lines)) if lines[i].strip() == '')
table_k2 = '\n'.join(lines[i0:i1])
slopes = '\n'.join(l for l in lines if l.startswith('alpha = 1/'))
j0 = next(i for i, l in enumerate(lines) if l.startswith('| width n (seeds)'))
table_depth = '\n'.join(lines[j0:])
table_k3 = k3_out.strip()

# largest width and depth present
rows = {}
for p in glob.glob(num + 'results/k2s_n*_L*_seed*.json'):
    d = json.load(open(p))
    rows.setdefault(d['n'], []).append(d)
nmax = max(rows)
dmax = max(d['Lmax'] for ds in rows.values() for d in ds)

k3rows = {}
for p in glob.glob(num + 'results/k3f_n*_seed*.json'):
    d = json.load(open(p))
    k3rows.setdefault(d['n'], []).append(d)


def k3_gain(n, frac):
    depth = n // frac
    key = f'k3f_d{depth}'
    rs = [d['runs'][key] for d in k3rows.get(n, []) if key in d['runs']]
    if not rs:
        return None, 0
    g = [(1.0 / r['N_eq']) / max(r['nmse'], 1e-12) for r in rs]
    return math.exp(np.mean(np.log(g))), len(rs)


k3_eighth = []
for n in sorted(k3rows):
    g, c = k3_gain(n, 8)
    if g is not None:
        k3_eighth.append(f"{g:.2f} at width {n}" + (f" ({c} seeds)" if c < 8 else ''))
k3_quarter = []
q_losses = []
for n in sorted(k3rows):
    g, c = k3_gain(n, 4)
    if g is not None:
        k3_quarter.append(f"{g:.3f} at width {n}")
        q_losses.append(1.0 / g)
ratios = []
for n in k3rows:
    for d in k3rows[n]:
        for frac in (8, 4):
            dep = n // frac
            a, b = d['runs'].get(f'k3f_d{dep}'), d['runs'].get(f'k2_d{dep}')
            if a and b:
                ratios.append(a['flops'] / b['flops'])
cost_txt = f"{min(ratios):.0f}-{max(ratios):.0f}"
loss_txt = f"{min(q_losses):.0f}-{max(q_losses):.0f}"

k3_par = ("**The next closure up the hierarchy.** `numerics/k3f_scaling.py` runs ARC's factorized K = 3 (and K = 2 for "
          "reference) on fresh nets at depth n/8 and n/4, with exact FLOP counts for each depth:\n\n" + table_k3 + "\n\n"
          "Factorized K = 3 costs " + cost_txt + " times more than K = 2 here. At depth = width/8 its gain over plain Monte Carlo "
          "falls with width (" + ", ".join(k3_eighth) + "); at depth = width/4 it loses by a factor " + loss_txt + " (" +
          ", ".join(k3_quarter) + "), and its error is not even below K = 2's. So the hierarchy's higher-budget points "
          "fall behind sampling at depth proportional to width, increasingly with width, which matches what ARC and the "
          "contest saw at 256 x 32 (factorized K = 3 about 11 times behind plain Monte Carlo there, our arc-noisy runs).")

d = open(base + 'draft.md').read()
d = d.replace('[TABLE_K2]', table_k2 + '\n\nFitted slopes of log nMSE_K2 against log n at fixed depth/width:\n' +
              '\n'.join('- ' + s for s in slopes.splitlines()))
d = d.replace('[TABLE_DEPTH]', table_depth)
d = d.replace('[DMAX]', str(dmax)).replace('[NMAX]', str(nmax))
d = d.replace('[K3F_PARAGRAPH]', k3_par)
open(base + 'draft.md', 'w').write(d)

h = open(base + 'notes/head.txt').read()
h = h.replace('[NMAX]', str(nmax))
h = h.replace('[K3F_SUMMARY]', 'The next closure in ARC\'s hierarchy (factorized K = 3, ' + cost_txt + ' times the cost) falls '
              'behind plain Monte Carlo as width grows: gain ' + ', '.join(k3_eighth) + ' at depth = width/8, and a loss '
              'of ' + loss_txt + 'x at depth = width/4.')
h = h.replace('[K2_SUMMARY]', 'K = 2 at its own budget gains a factor of about 1-3 over plain Monte Carlo at equal cost '
              'for depth/width 1/8 to 1/2 and widths 64-' + str(nmax) + ' (8 at width 32), with no growth in width; '
              'its error grows like depth^2 at shallow depth and then levels off (tables in 6.3). Factorized K = 3 falls '
              'behind with width (above).')
h = h.replace('[K3F_M1]', 'But the next closure (factorized K = 3) falls further behind sampling as width grows, so '
              'matching sampling over a range of budgets, which is what M1 and ARC\'s MSP ask for, is not achieved by '
              'the cumulant hierarchy at depth proportional to width.')
open(base + 'notes/head.txt', 'w').write(h)
print('filled; nmax', nmax, 'dmax', dmax, 'k3 1/8:', k3_eighth, 'k3 1/4:', k3_quarter)
