"""Slot-aware job queue: python runq2.py JOBFILE MAXTOTAL.
Launches the next job only while the number of python processes running in this directory (any queue, any job) is
below MAXTOTAL. Each line: NAME<TAB>script args. Skips NAME if logs/NAME.done exists or logs/NAME.log is newer than
the queue start and the process is still alive. Runs with nice -n 10 and one BLAS thread."""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = '/home/agent/.venvs/aggext/bin/python'
env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def my_python_procs():
    n = 0
    for pid in os.listdir('/proc'):
        if not pid.isdigit(): continue
        try:
            cwd = os.readlink(f'/proc/{pid}/cwd')
            cmd = open(f'/proc/{pid}/cmdline', 'rb').read().split(b'\0')
        except Exception:
            continue
        if cwd == HERE and len(cmd) > 1 and cmd[0].endswith(b'python') and not cmd[1].startswith(b'runq'):
            n += 1
    return n


def main():
    jobs = [l.rstrip('\n').split('\t', 1) for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
    maxtot = int(sys.argv[2])
    queue = [j for j in jobs if not os.path.exists(os.path.join(HERE, 'logs', j[0] + '.done'))]
    running = {}
    while queue or running:
        for name, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                del running[name]
                if rc == 0: open(os.path.join(HERE, 'logs', name + '.done'), 'w').write(time.strftime('%FT%TZ', time.gmtime()))
                print(time.strftime('%H:%M:%S'), 'finished', name, rc, flush=True)
        while queue and my_python_procs() < maxtot:
            name, cmd = queue.pop(0)
            log = open(os.path.join(HERE, 'logs', name + '.log'), 'w')
            running[name] = subprocess.Popen(['nice', '-n', '10', PY] + cmd.split(), cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env)
            print(time.strftime('%H:%M:%S'), 'started', name, flush=True)
            time.sleep(2)
        time.sleep(10)
    print('QUEUE-DONE', flush=True)


if __name__ == '__main__':
    main()
