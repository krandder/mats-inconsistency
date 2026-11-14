import sys, json, time, torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib'); sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep-m1/lib')
import nd, ghm
torch.set_num_threads(1)
T = json.load(open('runs/truth_s103_sig0.001.json'))
net = nd.Net('plain', 103, 8, 256, 1e-3, r=T['r'])
fp = net.flops_path()
for tau in (1e-2, 1e-3, 1e-4):
    t0 = time.time()
    out, ftot = ghm.ghm_forward(net, 8, tau)
    row = []
    for c in range(1, 9):
        w, mu, C, fc = out[c]
        m = ghm.mixture_mean(w, mu)
        ta = torch.tensor(T['layers'][str(c)]['mean_h'], dtype=torch.float64)
        e = float((m - ta).pow(2).sum() / torch.tensor(T['layers'][str(c)]['cov_h']).trace())
        row.append(f"c{c}:{len(w)}comp/{fc/fp:.0f}p/err{e:.1g}")
    print(f"tau={tau:g} ({time.time()-t0:.0f}s): " + ' '.join(row), flush=True)
