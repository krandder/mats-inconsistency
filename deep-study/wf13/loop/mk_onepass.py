"""mk_onepass.py -- one pass of the world model M on the test instances: P(h_i = 1 | all checks), one call of M each.
Baseline for NOTE section (d). Uses wf7's networks and code unchanged. -> out/onepass_w{ws}.npz"""
import sys
import numpy as np
import torch
sys.dont_write_bytecode = True
import pdata
ROOT = pdata.ROOT
sys.path.insert(0, ROOT + 'wf7/probworld')
import model as MM, arms as A      # noqa: E402
PATH = {7: ('wf7/probworld/ckpt/M_w7_S.pt', 'S'), 101: ('wf7/probworld/ckpt/M_w101_M.pt', 'M'),
        102: ('wf7/probworld/ckpt/M_w102_M.pt', 'M'), 103: ('wf7/probworld/ckpt/M_w103_M.pt', 'M'),
        311: ('wf11/learn-the-score/ckpt/M_w311_M.pt', 'M'), 201: ('wf7/probworld/ckpt/Mscale_w201_M.pt', 'M')}
for ws in [int(v) for v in sys.argv[1:]]:
    tab = pdata.load_tab(ws)
    n, m = tab['loga'].shape[0], tab['checks'].shape[0]
    x, h = pdata.test_x(ws, tab, 200)
    net = MM.build(PATH[ws][1], n + m)
    net.load_state_dict(torch.load(ROOT + PATH[ws][0], map_location='cpu')['state'])
    net.eval()
    A.N_H = n
    b = A.one_pass(net, torch.tensor(x)).double().numpy()
    np.savez(pdata.HERE + 'out/onepass_w%d.npz' % ws, b=b, flops=net.flops_per_seq(), x=x)
    print('world', ws, 'instances', len(x), 'flops per call %.3g' % net.flops_per_seq(), 'log loss on the planted h %.4f'
          % float(-(h * np.log(np.clip(b, 1e-6, 1)) + (1 - h) * np.log(np.clip(1 - b, 1e-6, 1))).mean()))
