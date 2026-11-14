# FORECAST hawk sim, part 2: IPS tuning (smaller step, eps sweep). Reuses run() from FORECAST-sim.py
# without re-running part 1. Regret here includes the cost of the random commits.
import numpy as np, json, types, re
src = open('/home/agent/workspaces/eurisko-study/lir-agent/hawks/FORECAST-sim.py').read()
src = src.split('out = {}')[0]
m = types.ModuleType('s'); exec(src, m.__dict__)
out = {}
for param in ['tabular', 'linear']:
    for eps, lr in [(0.1, 0.05), (0.2, 0.05), (0.2, 0.02)]:
        rs = [m.run('ips', param, s, eps=eps, lr=lr) for s in range(5)]
        key = f'{param}/ips_eps{eps}_lr{lr}'
        out[key] = {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))] for k in rs[0]}
        print(key, {k: f'{v[0]:.4f}±{v[1]:.4f}' for k, v in out[key].items()}, flush=True)
    rs = [m.run('committed', param, s, lr=0.05) for s in range(5)]
    key = f'{param}/committed_lr0.05'
    out[key] = {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))] for k in rs[0]}
    print(key, {k: f'{v[0]:.4f}±{v[1]:.4f}' for k, v in out[key].items()}, flush=True)
json.dump(out, open('/home/agent/workspaces/eurisko-study/lir-agent/hawks/FORECAST-sim2.json', 'w'), indent=1)
