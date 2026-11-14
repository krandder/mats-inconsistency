"""Run worker1's grid cells. usage: run_grid.py NPROC N[,N..] [T] ; skips cells whose summary.json exists."""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import econ
from multiprocessing import Pool

ROOT = os.path.join(os.path.dirname(HERE), 'runs')
CELLS = [(1, False), (64, True), (64, False), (4096, True), (4096, False)]


def rid(N, Q, nic, s, T):
    return f"w1-n{N//1000}k-q{Q}-{'on' if nic else 'off'}-s{s}" + ('' if T == 10000 else f'-t{T}')


def one(a):
    N, Q, nic, s, T = a
    r = rid(N, Q, nic, s, T)
    if os.path.exists(os.path.join(ROOT, r, 'summary.json')):
        return r, 'skip'
    d = os.path.join(ROOT, r)
    if os.path.isdir(d):                      # partial run from a killed pool: move it aside (RunLog appends)
        import subprocess
        subprocess.run(['trash-put', d], check=True)
    t = time.time()
    econ.run(N=N, Q=Q, niches=nic, seed=s, T=T, out_root=ROOT, run_id=r, verbose=True)
    return r, round(time.time() - t)


if __name__ == '__main__':
    nproc, Ns = int(sys.argv[1]), [int(x) for x in sys.argv[2].split(',')]
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    # PREREG-w1 T=5000 clause not triggered: dense path measured 0.92 s/step with null at 100k (2.6 h < 3 h)
    jobs = [(N, Q, nic, s, T) for N in Ns for (Q, nic) in CELLS for s in (1, 2, 3)]
    # expensive first (niches off, Q>1) so the pool tail is short
    jobs.sort(key=lambda j: -(j[0] * (3 if (not j[2] and j[1] > 1) else 1)))
    with Pool(nproc) as p:
        for r, dt in p.imap_unordered(one, jobs):
            print(f'DONE {r} {dt}s', flush=True)
