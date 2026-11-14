"""X1 w4: one K1 worker-1 cell (same call as K1/w1/run_grid.py one()), in the /tmp clean copy.
usage: k1_w1_one.py on|off SEED"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
C = '/tmp/traders-net/X1w4/study/K1'
sys.path.insert(0, C)
import econ
nic = sys.argv[1] == 'on'; s = int(sys.argv[2])
econ.run(N=10000, Q=64, niches=nic, seed=s, T=10000, out_root=os.path.join(C, 'runs'),
         run_id=f"w1-n10k-q64-{sys.argv[1]}-s{s}", verbose=False)
