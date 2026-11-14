"""History-blind floor of the dense beta* target used in the binding arms: the best predictor of beta* that does not
use the history is E[beta* | query reports] = beta*_T0 (tower property), so its MSE on beta*/3 is
E[(beta* - beta*_T0)^2] / 9, here under stage 1 (T log-uniform in [32, 64]) and the final stage (T = 0 w.p. 0.1,
else log-uniform in [1, 64]). Also the constant-prediction MSE (variance of beta*/3) and parameter counts."""
import math, sys
WF = '/home/agent/workspaces/mats/work/world-fa'
sys.path[:0] = [WF, '/home/agent/workspaces/mats/work/worlds']
import numpy as np, torch
import fa_core as C, harness as H
torch.set_num_threads(2)
c = C.full_cfg({"prior": "f", "f_templates": [{"blocks": [[2, 0.8, 1]], "marg": 1.0}]})
for name, (lo, hi, p0) in (('stage1 T in [32,64]', (32, 64, 0.0)), ('final T 0..64, p0 .1', (0, 64, 0.1))):
    torch.manual_seed(3)
    E = 8000
    Te = C.sample_T(E, dict(c, T_train=(lo, hi), p_T0=p0))
    b = C.gen(E, 4, Te, c)
    bs = C.beta_star(b, c) / 3
    _, _, _, b0 = C.bayes_f(b, c, use_hist=False, return_post=True)
    b0 = b0 / 3
    floor = float(((bs - b0) ** 2).mean()); const = float(((bs - bs.mean()) ** 2).mean())
    print(f'{name}: MSE floor without history {floor:.4f}; constant prediction {const:.4f}; perfect 0')
for d in (2, 4):
    m = H.SetPFN(72, d, 64, 4, 'real', 4)
    print('depth', d, 'params', sum(p.numel() for p in m.parameters()))
