"""Queue helper (sixth agent; generalises e53_queue_evals.py): for every finished net of an episode (runs/<EP>/<tag>/meta.json)
without an evaluation output (results/<EP>/<tag>.npz) or a queued/running/done job, add a job to the priority queue /tmp/s6q that
evaluates exactly that net with EVAL (through a directory holding only a symlink to it). PRIO is the job-name prefix.
usage: python queue_evals.py EP EVAL.py PRIO
"""
import glob, os, sys

EP, EVAL, PRIO = sys.argv[1], sys.argv[2], sys.argv[3]
Q = '/tmp/s6q'
HERE = '/home/agent/workspaces/mats/work/cm-interp-B'
n = 0
for mp in sorted(glob.glob(os.path.join(HERE, 'runs/%s/*/meta.json' % EP))):
    run = os.path.dirname(mp); tag = os.path.basename(run)
    job = '%s_eval_%s' % (PRIO, tag)
    if os.path.exists(os.path.join(HERE, 'results/%s/%s.npz' % (EP, tag))) or any(os.path.exists(os.path.join(Q, d, job)) for d in ('pending', 'running', 'done')):
        continue
    one = os.path.join(Q, 'one', tag); os.makedirs(one, exist_ok=True)
    if not os.path.exists(os.path.join(one, tag)):
        os.symlink(run, os.path.join(one, tag))
    cmd = '/home/agent/.venvs/aggext/bin/python %s %s results/%s 0 1 >> results/%s_eval.log 2>&1' % (EVAL, one, EP, EP)
    open(os.path.join(Q, 'pending', job), 'w').write(cmd + '\n'); n += 1
print('queued', n, 'evaluation jobs for', EP)
