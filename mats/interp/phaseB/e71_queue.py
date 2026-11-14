"""E71 queue helper (PREREG-B amendment 79): for every finished E71 net (runs/e71/<tag>/meta.json, written at the end of training)
without an evaluation output (results/e71/<tag>.npz) or a queued/running/done job, add one job to the analysis queue /tmp/s7q that
evaluates exactly that net with e71_eval.py (through a directory holding only a symlink to it), runs e71_heads.py on its model.pt,
and e71_heads.py --step N on its checkpoints (results/e71_traj). Once: the same checkpoint measures for E70's 12 b1h4 nets
(runs/e70/e70_b1h4_s*), whose trajectories have not been measured before.
usage: python e71_queue.py
"""
import glob, os

Q = '/tmp/s7q'
HERE = '/home/agent/workspaces/mats/work/cm-interp-B'
PY = '/home/agent/.venvs/aggext/bin/python'
STEPS = [0, 2000, 3000, 5000, 10000, 20000, 40000, 60000, 80000, 90000]


def queued(job):
    return any(os.path.exists(os.path.join(Q, d, job)) for d in ('pending', 'running', 'done'))


def traj(run):
    return ' && '.join('%s e71_heads.py --step %d --out results/e71_traj %s >> results/e71_traj.log 2>&1' % (PY, st, run) for st in STEPS)


def main():
    n = 0
    for mp in sorted(glob.glob(os.path.join(HERE, 'runs/e71/*/meta.json'))):
        run = os.path.dirname(mp); tag = os.path.basename(run); job = 'e71_eval_%s' % tag
        if os.path.exists(os.path.join(HERE, 'results/e71/%s.npz' % tag)) or queued(job):
            continue
        one = os.path.join(Q, 'one', tag); os.makedirs(one, exist_ok=True)
        if not os.path.exists(os.path.join(one, tag)):
            os.symlink(run, os.path.join(one, tag))
        cmd = ('%s e71_eval.py %s results/e71 0 1 >> results/e71_eval.log 2>&1 && %s e71_heads.py %s >> results/e71_heads.log 2>&1 && %s'
               % (PY, one, PY, run, traj(run)))
        open(os.path.join(Q, 'pending', job), 'w').write(cmd + '\n'); n += 1
    job = 'e71_traj_b1h4'
    if not queued(job):
        runs = sorted(glob.glob(os.path.join(HERE, 'runs/e70/e70_b1h4_s*/meta.json')))
        cmd = ' && '.join(traj(os.path.dirname(mp)) for mp in runs)
        open(os.path.join(Q, 'pending', job), 'w').write(cmd + '\n'); n += 1
    done = len(glob.glob(os.path.join(HERE, 'results/e71/e71_*.npz'))); heads = len(glob.glob(os.path.join(HERE, 'results/e71_heads/e71_*.json')))
    print('queued', n, 'jobs; evaluated', done, 'heads', heads, flush=True)


if __name__ == '__main__':
    main()
