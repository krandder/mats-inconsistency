"""Evaluate every checkpoint under a runs dir locally, in parallel processes (mech on seed 0)."""
import glob, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

root = sys.argv[1]
env = dict(os.environ, PYTHONPATH='/tmp/torchcpu', OMP_NUM_THREADS='1')


def one(d):
    args = ['python3', 'pfn.py', 'eval', '--ckpt', os.path.join(d, 'model.pt'), '--out', d]
    if d.rstrip('/').endswith('_s0'):
        args.append('--mech')
    with open(os.path.join(d, 'eval.log'), 'w') as f:
        rc = subprocess.call(args, stdout=f, stderr=subprocess.STDOUT, env=env)
    return d, rc


dirs = sorted(d for d in glob.glob(os.path.join(root, 'L*_k*_s*')) if os.path.isdir(d))
with ThreadPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 8) as ex:
    for d, rc in ex.map(one, dirs):
        print(d, rc, flush=True)
