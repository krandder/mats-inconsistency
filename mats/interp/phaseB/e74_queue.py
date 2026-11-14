"""E74 queue helper (PREREG-B amendment 82): for every finished S = 6 net (runs/e74/<tag>/meta.json) without an output in results/e74/S6
or a queued / running / done job, add one analysis-queue job evaluating it with e74_eval.py; once, the S = 3 references (E37's
acquired ctrl nets and E60's own nets) on results/e74_data_S3.npz.
usage: python e74_queue.py
"""
import glob, os

Q = '/tmp/s7q'
PY = '/home/agent/.venvs/aggext/bin/python'
HERE = '/home/agent/workspaces/mats/work/cm-interp-B'


def queued(job):
    return any(os.path.exists(os.path.join(Q, d, job)) for d in ('pending', 'running', 'done'))


def main():
    n = 0
    for mp in sorted(glob.glob(os.path.join(HERE, 'runs/e74/*/meta.json'))):
        run = os.path.dirname(mp); tag = os.path.basename(run); job = 'e74_eval_%s' % tag
        if os.path.exists(os.path.join(HERE, 'results/e74/S6/%s.json' % tag)) or queued(job):
            continue
        open(os.path.join(Q, 'pending', job), 'w').write('%s e74_eval.py 6 results/e74/S6 %s >> results/e74_eval.log 2>&1\n' % (PY, run)); n += 1
    if not queued('e74_eval_S3refs'):
        runs = ['runs/e37/e37_ctrl_s%d' % s for s in range(12) if s != 3] + ['runs/e60/e60_own_s%d' % s for s in range(12)]
        open(os.path.join(Q, 'pending', 'e74_eval_S3refs'), 'w').write('%s e74_eval.py 3 results/e74/S3 %s >> results/e74_eval.log 2>&1\n' % (PY, ' '.join(runs))); n += 1
    print('queued', n, 'jobs; evaluated', len(glob.glob(os.path.join(HERE, 'results/e74/S6/*.json'))), flush=True)


if __name__ == '__main__':
    main()
