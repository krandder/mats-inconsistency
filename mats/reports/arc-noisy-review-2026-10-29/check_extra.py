import sys, ast, json
from pathlib import Path
import numpy as np, torch
P=Path('/home/agent/workspaces/mats/work/arc-noisy');sys.path.insert(0,str(P));torch.set_num_threads(1)
import anet,truthlib
ns={'torch':torch,'np':np,'anet':anet,'n':256};t=ast.parse((P/'gpu_samplers.py').read_text());exec(compile(ast.Module(body=[x for x in t.body if isinstance(x,ast.FunctionDef)],type_ignores=[]),'gpu','exec'),ns)
z=torch.ones(2,2);WT=[torch.eye(2),2*torch.eye(2),3*torch.eye(2)]
print('sigma0 result expected6 actual',ns['forward_from_z1'](z,WT,[0,0,0],3,None,[3])[3][0].tolist())
T=truthlib.load(100,.001);net=anet.Net(100,sigma=.001)
print('truth100 N',T['N'],'checksum diff',T['chk']-anet.weights_checksum(net.W),'r maxdiff',np.max(abs(T['r']-net.r.numpy())))
print('float64 vs float32 weights maxdiff',max(float((w-w.float().double()).abs().max()) for w in net.W))
for L in [8,32]:
 fs=[P/f'results/kprop_100_s0.001_L{L}_{v}.json' for v in ['k1','k2','k2a','k3f']]
 print('stored costs',L,{f.stem:json.load(open(f))['flops'] for f in fs if f.exists()})
 p=P/f'samp/samp_100_s0.001_L{L}.npz'
 if p.exists():
  d=np.load(p);N=d['N_mub'];S=d[f'sum_mub_{L}'];e=np.array([np.mean([truthlib.nmse(S[r,j]/N[j],T,L) for r in range(S.shape[0])]) for j in range(len(N))]);print('dev100 N',N.tolist(),'nMSE*N',(N*e).tolist())
# Active-gradient trace is not variance share: Gaussian x, f(x)=x1+0.1 He10(x2)/sqrt(10!)
print('AGOP counterexample: derivative shares',1/(1+.1),.1/(1+.1),'variance shares',1/1.01,.01/1.01)
