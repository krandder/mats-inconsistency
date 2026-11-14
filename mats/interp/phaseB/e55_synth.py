"""E55 plumbing (disclosed in amendment 63): a synthetic results tree in /tmp/e55_synth/results/ for e55_summ.py (run it there),
with planted relationships:
  e53_uni, e53_unif (seeds 0-11): rom = the table's prediction at c + 0.01 (large) and - 0.01 (middle) -> P215 holds;
  e38_w128ctrl, e38_w128mag (10 nets): rom_large = 0.30 - 0.05 c (Spearman -1), c around 5.0;
  e43_br0 (11 nets): rom_large independent of c (planted Spearman about 0); x_e37_ctrl / x_e37_mag (exploration): c around 3.8;
  -> P216 over two cohorts with Spearman -1, -1 and one near 0: 2 of 3 negative (< 80%) -> fails; P217 holds (5.0 > 3.8);
  traj: 4 runs with c rising and rom falling after 5,000 steps (Spearman -1), 1 run the other way -> 4 of 5 negative -> holds.
usage: python e55_synth.py; cd /tmp/e55_synth && python .../e55_summ.py
"""
import json, os
import numpy as np

R = '/tmp/e55_synth/results'
for d in ('e37', 'e38', 'e43', 'e53', 'e37_traj'):
    os.makedirs(os.path.join(R, d), exist_ok=True)
rng = np.random.default_rng(3)
grid = [0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
tab = dict(grid=grid, rom={'%.2f' % c: dict(large=0.25 / c, middle=0.15 / c ** 1.5) for c in grid})
json.dump(tab, open(os.path.join(R, 'e55_cbtable.json'), 'w'))
geom = {}


def put(tag, cohort, sub, expl, c, rl, rm, seed, fmt='ctrl'):
    geom[tag] = dict(cohort=cohort, exploration=expl, sub=sub, fmt=fmt, width=32, depth=2, seed=seed, c=c, A_cv=0.1, rho=0.05, U=0.5)
    np.savez(os.path.join(R, sub, tag + '.npz'), gap=0.01, rom_large=rl, rom_middle=rm)


for arm in ('uni', 'unif'):
    for s in range(12):
        c = 1.4 + 0.1 * rng.uniform(-1, 1)
        put('e53_%s_s%d' % (arm, s), 'e53_' + arm, 'e53', False, c, float(np.interp(c, grid, [0.25 / g for g in grid])) + 0.01,
            float(np.interp(c, grid, [0.15 / g ** 1.5 for g in grid])) - 0.01, s)
for arm in ('ctrl', 'mag'):
    for s in range(10):
        c = 5.0 + 0.5 * rng.uniform(-1, 1)
        put('e38_w128%s_s%d' % (arm, s), 'e38_w128' + arm, 'e38', False, c, 0.30 - 0.05 * c, 0.0, s, arm)
    for s in range(12):
        put('e37_%s_s%d' % (arm, s), 'x_e37_' + arm, 'e37', True, 3.8 + 0.2 * rng.uniform(-1, 1), 0.1, 0.0, s, arm)
for s in range(11):
    put('e43_br0_s%d' % s, 'e43_br0', 'e43', False, 4.0 + rng.uniform(-1, 1), float(rng.uniform(0.05, 0.15)), 0.0, s)
json.dump(geom, open(os.path.join(R, 'e55_geom.json'), 'w'))
steps = [0, 2000, 5000, 10000, 20000, 40000, 90000]
traj = {}
for k in range(5):
    tag = 'e37_ctrl_s%d' % k
    c = [2.0, 2.5, 3.0, 3.3, 3.6, 3.9, 4.2]; rl = [0.0, 0.3, 0.25, 0.2, 0.15, 0.12, 0.1]
    if k == 4:
        rl = rl[:2] + rl[2:][::-1]
    traj[tag] = dict(step=steps, c=c)
    np.savez(os.path.join(R, 'e37_traj', tag + '.npz'), step=np.array(steps), rom_large=np.array(rl))
json.dump(traj, open(os.path.join(R, 'e55_traj.json'), 'w'))
print('synthetic E55 tree written to', R)
