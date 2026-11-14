"""E72 queue helper (PREREG-B amendment 80): one analysis-queue job per cohort for Part 1 (e72_run.py -> results/e72/<tag>.json) and,
for the cohorts with pfn2's standard forward, one per cohort for Part 2 (e72_patch.py -> results/e72_patch/<cohort>.json). Acquired
nets only (registered evaluation gap < 0.10). Job names sort after E71's evaluation jobs ('e71_...'), so those run first; Part 2
jobs ('e72_0_...') before Part 1 ('e72_1_...'), registered cohorts before the descriptive depth-3 cohorts ('e72_2_...') and E71's
secondary cohorts ('e72_3_...', queued with --e71 once E71's evaluations exist).
usage: python e72_queue.py [--e71]
"""
import os, sys
import numpy as np

Q = '/tmp/s7q'
PY = '/home/agent/.venvs/aggext/bin/python'
HERE = '/home/agent/workspaces/mats/work/cm-interp-B'
MAIN = [('ctrl', 'runs/e37/e37_ctrl_s%d', 'results/e37/e37_ctrl_s%d.npz', 12, True),
        ('fix', 'runs/e30/fix_s%d', 'results/e30/fix_s%d.npz', 12, True),
        ('own', 'runs/e60/e60_own_s%d', 'results/e60/e60_own_s%d.npz', 12, False),
        ('nocons', 'runs/e64/e64_nocons_s%d', 'results/e64/e64_nocons_s%d.npz', 12, False),
        ('h4', 'runs/e69/e69_h4_s%d', 'results/e69/e69_h4_s%d.npz', 12, True),
        ('b1h4', 'runs/e70/e70_b1h4_s%d', 'results/e70/e70_b1h4_s%d.npz', 12, True),
        ('mag', 'runs/e37/e37_mag_s%d', 'results/e37/e37_mag_s%d.npz', 12, True),
        ('ownmag', 'runs/e70/e70_ownmag_s%d', 'results/e70/e70_ownmag_s%d.npz', 12, False),
        ('uni', 'runs/e53/e53_uni_s%d', 'results/e53/e53_uni_s%d.npz', 12, False)] + \
       [(c, 'runs/e38/e38_%s_s%%d' % c, 'results/e38/e38_%s_s%%d.npz' % c, 10, True) for c in ('w64ctrl', 'w128ctrl', 'w64mag', 'w128mag')]
DESC = [(c, 'runs/e38/e38_%s_s%%d' % c, 'results/e38/e38_%s_s%%d.npz' % c, 10, True) for c in ('d3ctrl', 'd3mag')]
E71 = [(c, 'runs/e71/e71_%s_s%%d' % c, 'results/e71/e71_%s_s%%d.npz' % c, 12, True) for c in ('h2w8', 'h3w16', 'h4w16')]


def specs(c, rp, ep, n):
    out = []
    for s in range(n):
        r, e = rp % s, ep % s
        if os.path.exists(os.path.join(HERE, e)) and float(np.load(os.path.join(HERE, e), allow_pickle=True)['gap']) < 0.10:
            out.append('%s_s%d=%s:%s' % (c, s, r, e))
    return out


def put(name, cmd):
    if not any(os.path.exists(os.path.join(Q, d, name)) for d in ('pending', 'running', 'done')):
        open(os.path.join(Q, 'pending', name), 'w').write(cmd + '\n')
        return 1
    return 0


def main():
    n = 0
    groups = [(MAIN, '1'), (DESC, '2')] if '--e71' not in sys.argv else [(E71, '3')]
    for i, (lst, pri) in enumerate(groups):
        for j, (c, rp, ep, cnt, std) in enumerate(lst):
            sp = ' '.join(specs(c, rp, ep, cnt))
            if not sp:
                continue
            if std:
                n += put('e72_0%s_%02d_%s' % (pri, j, c), '%s e72_patch.py results/e72_patch/%s.json %s >> results/e72_patch.log 2>&1' % (PY, c, sp))
            n += put('e72_%s_%02d_%s' % (pri, j, c), '%s e72_run.py results/e72 %s >> results/e72_run.log 2>&1' % (PY, sp))
    os.makedirs(os.path.join(HERE, 'results/e72_patch'), exist_ok=True)
    print('queued', n, 'jobs')


if __name__ == '__main__':
    main()
