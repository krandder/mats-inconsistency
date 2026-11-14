"""Queue helper (sixth agent): for every finished E53 net (runs/e53/<tag>/meta.json present) without an evaluation output
(results/e53/<tag>.npz) or a queued/running/done job, add an evaluation job to the priority queue /tmp/s6q (priority 20, ahead of
the remaining trainings). Each job evaluates exactly one net with e53_eval.py, through a directory that holds only a symlink to
that net (so the evaluator's sorted run list has one entry).
usage: python e53_queue_evals.py
"""
import glob, os

Q = '/tmp/s6q'
HERE = '/home/agent/workspaces/mats/work/cm-interp-B'
n = 0
for mp in sorted(glob.glob(os.path.join(HERE, 'runs/e53/*/meta.json'))):
    run = os.path.dirname(mp); tag = os.path.basename(run)
    job = '20_eval_%s' % tag
    if os.path.exists(os.path.join(HERE, 'results/e53/%s.npz' % tag)) or any(os.path.exists(os.path.join(Q, d, job)) for d in ('pending', 'running', 'done')):
        continue
    one = os.path.join(Q, 'one', tag); os.makedirs(one, exist_ok=True)
    if not os.path.exists(os.path.join(one, tag)):
        os.symlink(run, os.path.join(one, tag))
    cmd = '/home/agent/.venvs/aggext/bin/python e53_eval.py %s results/e53 0 1 >> results/e53_eval.log 2>&1' % one
    open(os.path.join(Q, 'pending', job), 'w').write(cmd + '\n'); n += 1
print('queued', n, 'evaluation jobs')
