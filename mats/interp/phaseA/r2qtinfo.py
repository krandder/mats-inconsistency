"""Round 2 world-only check (PREREG amendment 4): do q0's six query-token features determine c up to source order?
q0 at QT reads f = (mean Q, Bayes mean, Bayes log var, labelled-only mean, labelled-only log var, sum_s beta_s Q_s),
all permutation-invariant functions of c = (Q, S). The query-token oracle O_q0 is taken to be O_c; this is exact iff f
determines c modulo source permutation. Held-out R^2 of an MLP regression f -> canonical c (sources sorted by Q, each
with its own log S). Writes r2/qtinfo.json."""
import json, sys, time
import numpy as np
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/cm-interp')
from lib import tp  # noqa: E402

torch.set_num_threads(1)
torch.manual_seed(0)
T0 = time.time()
N, NTE = 300000, 50000
d = tp.sample_episodes(N, 73000)
cs = tp.cell_stats(d)
f = np.stack([cs['ostar'].mean(1), cs['mb'], cs['lvb'], cs['ml'], cs['lvl'], (cs['beta'] * cs['ostar']).sum(1)], 1)
order = np.argsort(-cs['ostar'], 1)
Qs = np.take_along_axis(cs['ostar'], order, 1)
lS = np.take_along_axis(cs['lss'], order, 1)
Y = np.concatenate([Qs, lS], 1)
fm, fs = f[:-NTE].mean(0), f[:-NTE].std(0)
ym, ys = Y[:-NTE].mean(0), Y[:-NTE].std(0)
Xt = torch.tensor((f - fm) / fs, dtype=torch.float32)
Yt = torch.tensor((Y - ym) / ys, dtype=torch.float32)
net = torch.nn.Sequential(torch.nn.Linear(6, 256), torch.nn.SiLU(), torch.nn.Linear(256, 256), torch.nn.SiLU(),
                          torch.nn.Linear(256, 256), torch.nn.SiLU(), torch.nn.Linear(256, 6))
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
EP = 40
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
ntr = N - NTE
for ep in range(EP):
    perm = torch.randperm(ntr)
    for i in range(0, ntr, 512):
        b = perm[i:i + 512]
        loss = ((net(Xt[b]) - Yt[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
with torch.no_grad():
    P = net(Xt[-NTE:]).numpy()
Yte = Yt[-NTE:].numpy()
r2 = 1 - ((P - Yte) ** 2).mean(0) / Yte.var(0)
# control: the same regression from c itself (canonical) must reach R^2 ~ 1; and from 3 of the 6 features (must fail)
out = dict(r2_per_coord=dict(zip(['Q(1)', 'Q(2)', 'Q(3)', 'logS(1)', 'logS(2)', 'logS(3)'], map(float, r2))),
           n_train=ntr, n_test=NTE, epochs=EP, seconds=time.time() - T0,
           note='canonical c: sources sorted by query reading, each with its own log SS')
# how much of Y's variance is NOT recovered, in absolute units of log S (sd of residual)
out['resid_sd_logS'] = [float(x) for x in (np.sqrt(((P - Yte) ** 2).mean(0)) * ys)[3:]]
out['resid_sd_Q'] = [float(x) for x in (np.sqrt(((P - Yte) ** 2).mean(0)) * ys)[:3]]
json.dump(out, open('/home/agent/workspaces/mats/work/cm-interp/r2/qtinfo.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
