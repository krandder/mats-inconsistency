"""Tiny job queue: python runq.py JOBFILE MAXPAR. Each line: NAME<TAB>command args (run with nice -n 10, 1 thread).
Skips a job if logs/NAME.done exists. Writes logs/NAME.log and logs/NAME.done on exit code 0."""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = '/home/agent/.venvs/aggext/bin/python'
env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def main():
    jobs = [l.rstrip('\n').split('\t', 1) for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
    maxpar = int(sys.argv[2])
    running = {}
    queue = [j for j in jobs if not os.path.exists(os.path.join(HERE, 'logs', j[0] + '.done'))]
    while queue or running:
        for name, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                del running[name]
                if rc == 0: open(os.path.join(HERE, 'logs', name + '.done'), 'w').write(time.strftime('%FT%TZ', time.gmtime()))
                print(time.strftime('%H:%M:%S'), 'finished', name, rc, flush=True)
        while queue and len(running) < maxpar:
            name, cmd = queue.pop(0)
            log = open(os.path.join(HERE, 'logs', name + '.log'), 'w')
            running[name] = subprocess.Popen(['nice', '-n', '10', PY] + cmd.split(), cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env)
            print(time.strftime('%H:%M:%S'), 'started', name, flush=True)
        time.sleep(5)
    print('QUEUE-DONE', flush=True)


if __name__ == '__main__':
    main()
