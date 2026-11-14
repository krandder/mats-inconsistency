#!/bin/bash
#SBATCH --job-name=w13-consistency
#SBATCH --partition=compute
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=00:05:00
#SBATCH --output=/cluster/home/user/wave13/consistency/bench-%j.log
cd /cluster/home/user/wave13/consistency/code
lscpu | grep -E "Model name|^CPU\(s\)|Thread"
nproc
/cluster/home/user/grpo-pilot/venv/bin/python -B - <<'PYEOF'
import sys, time, torch
sys.path.insert(0, '/cluster/home/user/wave7/probworld')
import model as MM
print('threads', torch.get_num_threads(), flush=True)
for th in (1, 2, 10):
    torch.set_num_threads(th)
    for size in ('S', 'M'):
        m = MM.build(size, 50).eval()
        tok = torch.randint(0, 3, (16384, 50))
        with torch.no_grad():
            m(tok[:512])
            t0 = time.time()
            for i in range(0, 16384, 4096):
                m(tok[i:i+4096])
        print(th, size, 'seq/s %.0f' % (16384 / (time.time() - t0)), flush=True)
PYEOF
