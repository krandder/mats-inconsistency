import sys, time, torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy/ext/mlp_cumulant_propagation/src')
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, SIMPLE, AUGMENT
torch.set_grad_enabled(False)
torch.manual_seed(0)
n, nl = 32, 5
mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=nl).double()
K_in = {1: torch.zeros(n, dtype=torch.float64), 2: torch.eye(n, dtype=torch.float64)}
x = torch.randn(1<<20, n, dtype=torch.float64)
out = mlp(x, output_acts=True)
true_act = out.act[:, -1].mean(0)
var_act = out.act[:, -1].var(0)
for k in (1, 2, 3):
    for fac in (False, True) if k == 3 else (False,):
        t = time.time()
        K = mlp_kprop(mlp, K_in, k_max=k, factor=fac, up_to_layer=f'act{nl-2}')
        est = K[1].to_tensor() if hasattr(K[1], 'to_tensor') else K[1]
        nmse = (((est - true_act) ** 2) / var_act).mean().item()
        print(k, fac, f"nmse {nmse:.3e}  (MC floor {1/(1<<20):.1e})  {time.time()-t:.1f}s")
