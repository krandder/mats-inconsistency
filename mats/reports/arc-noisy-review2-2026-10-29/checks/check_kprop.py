import sys, math, types
from pathlib import Path
import numpy as np
import torch
sys.path[:0]=[str(Path('code').resolve()),str(Path('checks/arc_source').resolve())]
from kprop_tail import kprop_all
from kprop_noisy import KH, WICK_COEF_D, add_noise_to_K2
from tl import TNet
import deduce as D
from mlp_kprop.mlp import MLP

torch.set_num_threads(1)
g=torch.Generator().manual_seed(91)
n=8; L=4; weights=[torch.randn(n,n,generator=g,dtype=torch.float64)*math.sqrt(2/n) for _ in range(L)]
net=types.SimpleNamespace(n=n,L=L,W64=weights,sigma=.01,s=[.01]*L)
th=torch.linspace(1,3,n,dtype=torch.float64)
o=kprop_all(net,[th],k_max=3)
K=KH.coerce_input({1:torch.zeros(n,dtype=torch.float64),2:torch.eye(n,dtype=torch.float64)},k_max=3,kind=KH.SIMPLE)
for l in range(L-1):
 K=KH.linear_kprop(K,weights[l],k_max=3,set_metric=None)
 K=KH.nonlin_kprop(K,nonlin_wick_coef=WICK_COEF_D['relu'],k_max=3,kind=KH.SIMPLE,use_pK=True,factor=True)
 K=add_noise_to_K2(K,.0001)
K=KH.linear_kprop(K,weights[-1],k_max=3,set_metric=None)
cs=[]
for d in (1,2,3,4):
 v=K[d].to_tensor(); cs.append(torch.stack([v[(i,)*d] for i in range(n)]))
m,v,c3,c4=cs
expected=torch.stack([m,v+m*m,c3+3*m*v+m**3,c4+4*m*c3+3*v*v+6*m*m*v+m**4])
print('moment readout max absolute discrepancy',float((expected-o['moments']).abs().max()))
print('actual fourth cumulants',c4.tolist())
print('shared/heavi/moment FLOPs',o['flops_shared'],o['flops_heavi_each'],o['flops_moments'])
# Noiseless full upstream mlp_kprop route versus tail wrapper.
mlp=MLP(input_dim=n,hidden_dim=n,output_dim=n,num_layers=L+1,nonlin=['relu']*(L-1)+['heaviside']).double()
with torch.no_grad():
 for l in range(L):
  mlp.Ws[l].weight.copy_(weights[l])
  if mlp.Ws[l].bias is not None: mlp.Ws[l].bias.zero_()
 # Bias optional by default: set explicit final threshold.
 mlp.Ws[L-1].bias=torch.nn.Parameter(-th.clone(),requires_grad=False)
net.sigma=0
ours=kprop_all(net,[th],k_max=3)['heavi'][0]
ref=KH.mlp_kprop(mlp,{1:torch.zeros(n,dtype=torch.float64),2:torch.eye(n,dtype=torch.float64)},k_max=3,kind=KH.SIMPLE,use_avg_metric=False,factor=True,up_to_layer='act3')[1].to_tensor()
print('upstream run_lpe construction agreement',float((ours-ref).abs().max()))
# Direct latent formula and sensitivity to final hidden noise.
obj=TNet.__new__(TNet)
obj.L=L; obj.n=n; obj.sigma=.01; obj.s=[.01]*L; obj.W=weights; obj.device='cpu'; obj.dtype=torch.float64
Z=torch.randn(32,L,n,generator=g,dtype=torch.float64); orig=Z.clone(); h=Z[:,0]
for l in range(L-1):h=torch.relu(h@weights[l].T)+.01*Z[:,l+1]
print('latent formula discrepancy',float((obj.z(Z)-h@weights[-1].T).abs().max()),'Z unchanged',torch.equal(Z,orig))
Z2=Z.clone();Z2[:,-1,0]+=1
print('last hidden noise discrepancy',float((obj.z(Z2)-obj.z(Z)-.01*weights[-1][:,0]).abs().max()))
