"""Tiny job scheduler for this lane: at most MAXP heavy processes of this lane at once (counting ones started
elsewhere, by matching their script names), each job started only when its dependency files exist.
Job file lines: <dep1,dep2,...>|<command>|<logfile>   (deps relative to the lane directory; '-' for none).
nohup python sched.py jobs.txt &"""
import os, subprocess, sys, time

LANE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
MAXP = int(os.environ.get('MAXP', '3'))
SCRIPTS = ('r1_layers.py', 'r2_big.py', 'r2b_analyze.py', 'r2c_cover.py', 'r3_pairs.py', 'r4_mlp.py', 'r5_states.py')
PY = '/home/agent/.venvs/arc-noisy/bin/python'


def running():
    out = subprocess.run(['pgrep', '-af', 'python'], capture_output=True, text=True).stdout.splitlines()
    cnt = 0
    for l in out:
        if not any(s in l for s in SCRIPTS) or 'sched.py' in l or 'bash -c' in l or 'xargs' in l:
            continue
        try:
            cwd = os.readlink(f"/proc/{l.split()[0]}/cwd")
        except OSError:
            continue
        cnt += cwd.startswith(os.path.realpath(LANE))
    return cnt


jobs = []
for line in open(sys.argv[1]):
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    deps, cmd, log = line.split('|')
    jobs.append(([d for d in deps.split(',') if d and d != '-'], cmd, log))
started = set()
procs = []
while len(started) < len(jobs) or any(p.poll() is None for p in procs):
    for i, (deps, cmd, log) in enumerate(jobs):
        if i in started:
            continue
        if running() >= MAXP:
            break
        if all(os.path.exists(os.path.join(LANE, d)) for d in deps):
            full = f"cd {LANE}/scratch && OMP_NUM_THREADS=3 nice -n 19 ionice -c3 {PY} {cmd} > {LANE}/{log} 2>&1"
            procs.append(subprocess.Popen(['bash', '-c', full]))
            started.add(i)
            print(time.strftime('%H:%M:%S'), 'started', cmd, flush=True)
            time.sleep(2)
    time.sleep(20)
print(time.strftime('%H:%M:%S'), 'all jobs finished', flush=True)
