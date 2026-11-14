"""Write a CPU-queue command file that evaluates every run under RUNDIR with SCRIPT in NPARTS chunks (the evaluators skip nets whose
output exists). usage: python mkq.py SCRIPT RUNDIR OUTDIR NPARTS CMDFILE [append]"""
import glob, math, os, sys

script, rundir, outdir, nparts, cmdfile = sys.argv[1:6]
n = len(glob.glob(os.path.join(rundir, '*/meta.json'))); k = max(1, math.ceil(n / int(nparts)))
lines = ['/home/agent/.venvs/aggext/bin/python %s %s %s %d %d' % (script, rundir, outdir, i, min(i + k, n)) for i in range(0, n, k)]
open(cmdfile, 'a' if len(sys.argv) > 6 else 'w').write('\n'.join(lines) + '\n')
print(n, 'runs,', len(lines), 'commands ->', cmdfile)
