import resource
resource.setrlimit(resource.RLIMIT_CPU,(550,560))
import sys,math,types,json
from pathlib import Path
import numpy as np
import torch
S=Path('/tmp/arcn_review2/claims3_snapshot');sys.path.insert(0,str(S/'tails'))
from kprop_tail import kprop_all,kprop_moments
from kprop_noisy import KH,WICK_COEF_D,add_noise_to_K2,NamedFlopCounter
from mlp_kprop.mlp import MLP
from deduce import cumulants_from_moments

torch.set_num_threads(1);n=8;L=4;g=torch.Generator().manual_seed(91)
W=[torch.randn(n,n,generator=g,dtype=torch.float64)*math.sqrt(2/n) for _ in range(L)]
net=types.SimpleNamespace(n=n,L=L,W64=W,sigma=.001,s=[.001]*L)
th=torch.linspace(1,3,n,dtype=torch.float64)
for tag,kind in [('simple',KH.SIMPLE),('augment',KH.AUGMENT)]:
 with torch.no_grad():
  o=kprop_all(net,[th],kind=tag)
  K=KH.coerce_input({1:torch.zeros(n,dtype=torch.float64),2:torch.eye(n,dtype=torch.float64)},k_max=3,kind=kind)
  for l in range(L-1):
   K=KH.linear_kprop(K,W[l],k_max=3,set_metric=None)
   K=KH.nonlin_kprop(K,nonlin_wick_coef=WICK_COEF_D['relu'],k_max=3,kind=kind,use_pK=True,factor=True)
   K=add_noise_to_K2(K,1e-6)
  K=KH.linear_kprop(K,W[-1],k_max=3,set_metric=None)
  cs=[]
  for d in (1,2,3,4):
   a=K[d].to_tensor();cs.append(torch.stack([a[(i,)*d] for i in range(n)]))
  m,v,c3,c4=cs;expected=torch.stack([m,v+m*m,c3+3*m*v+m**3,c4+4*m*c3+3*v*v+6*m*m*v+m**4])
  print(tag,'moment difference',float(abs(expected-o['moments']).max()))
  mlp=MLP(input_dim=n,hidden_dim=n,output_dim=n,num_layers=L+1,nonlin=['relu']*(L-1)+['heaviside']).double()
  for l in range(L):
   mlp.Ws[l].weight.copy_(W[l])
   if mlp.Ws[l].bias is not None:mlp.Ws[l].bias.zero_()
  mlp.Ws[L-1].bias=torch.nn.Parameter(-th.clone(),requires_grad=False)
  orig=KH.nonlin_kprop
  def hooked(*args,**kw):
   K=orig(*args,**kw)
   return add_noise_to_K2(K,1e-6) if kw['nonlin_wick_coef'] is WICK_COEF_D['relu'] else K
  KH.nonlin_kprop=hooked
  try:
   with NamedFlopCounter() as fc:
    ref=KH.mlp_kprop(mlp,{1:torch.zeros(n,dtype=torch.float64),2:torch.eye(n,dtype=torch.float64)},k_max=3,kind=kind,use_avg_metric=False,factor=True,up_to_layer='act3')[1].to_tensor()
   print(tag,'noisy upstream heaviside difference',float(abs(ref-o['heavi'][0]).max()),'flops upstream',fc.total(),'wrapper',o['flops_shared']+o['flops_heavi_each'][0])
  finally:KH.nonlin_kprop=orig
  M,_=kprop_moments(net,kind=tag,powers=(1,2,3,4,5,6));mu=np.vstack([np.ones(n),M.numpy()]);k=np.zeros_like(mu)
  for j in range(1,7):k[j]=mu[j]-sum(math.comb(j-1,i-1)*k[i]*mu[j-i] for i in range(1,j))
  print(tag,'k5 maxabs',abs(k[5]).max(),'k6+10k3sq maxabs',abs(k[6]+10*k[3]**2).max())
rt=json.load(open('/home/agent/workspaces/mats/work/arc-noisy/rtable.json'))
print('fresh net rtable entries',sum(f'256_32_{s}_0.001' in rt for s in range(44,64)),'of20')
print('DONE')
