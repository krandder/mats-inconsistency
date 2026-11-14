"""E64 queue helper (PREREG-B amendment 72): for every finished E64 net (runs/e64/<tag>/meta.json, written at the end of training) without
an evaluation output (results/e64/<tag>.npz) or a queued/running/done job, add one job to the analysis queue /tmp/s7q that evaluates
exactly that net with e64_eval.py (through a directory holding only a symlink to it) and then runs e64_heads.py on it. When all 36
nets are evaluated and have head files, queue e64_codes.py for each arm (once).
usage: python e64_queue.py
"""
import glob, os

Q = '/tmp/s7q'
HERE = '/home/agent/workspaces/mats/work/cm-interp-B'
PY = '/home/agent/.venvs/aggext/bin/python'


def queued(job):
    return any(os.path.exists(os.path.join(Q, d, job)) for d in ('pending', 'running', 'done'))


def main():
    n = 0
    for mp in sorted(glob.glob(os.path.join(HERE, 'runs/e64/*/meta.json'))):
        run = os.path.dirname(mp); tag = os.path.basename(run); job = 'e64_eval_%s' % tag
        if os.path.exists(os.path.join(HERE, 'results/e64/%s.npz' % tag)) or queued(job):
            continue
        one = os.path.join(Q, 'one', tag); os.makedirs(one, exist_ok=True)
        if not os.path.exists(os.path.join(one, tag)):
            os.symlink(run, os.path.join(one, tag))
        cmd = '%s e64_eval.py %s results/e64 0 1 >> results/e64_eval.log 2>&1 && %s e64_heads.py %s >> results/e64_heads.log 2>&1' % (PY, one, PY, run)
        open(os.path.join(Q, 'pending', job), 'w').write(cmd + '\n'); n += 1
    done = len(glob.glob(os.path.join(HERE, 'results/e64/e64_*.npz'))); heads = len(glob.glob(os.path.join(HERE, 'results/e64_heads/e64_*.json')))
    if done >= 36 and heads >= 36:
        for arm in ('nocons', 'codeln', 'both'):
            job = 'e64_codes_%s' % arm
            if not queued(job):
                open(os.path.join(Q, 'pending', job), 'w').write('%s e64_codes.py %s > results/e64_codes_%s.log 2>&1\n' % (PY, arm, arm)); n += 1
    print('queued', n, 'jobs; evaluated', done, 'heads', heads)


if __name__ == '__main__':
    main()
