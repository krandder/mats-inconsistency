import os,sys,ast,json,math
from pathlib import Path
import numpy as np,torch
P=Path('/home/agent/workspaces/mats/work/arc-noisy');sys.path.insert(0,str(P));torch.set_num_threads(1)
import anet,samplers
src=(P/'kprop_noisy.py').read_text().replace("'ext', 'mlp_cumulant_propagation', 'src'", "'/tmp/arcn_review/ext/src'")
ns={'__file__':str(P/'kprop_noisy.py')};exec(compile(src,'kprop_noisy.py','exec'),ns)
Base=ns['NamedFlopCounter']; counters=[]
def factory():
 c=Base();counters.append(c);return c
ns['NamedFlopCounter']=factory
net=anet.Net(100,sigma=.001)
for k,fac,kind,L in [(2,False,'simple',8),(2,False,'augment',8),(3,True,'simple',8)]:
 _,fl,_=ns['kprop_noisy'](net,L,k,factor=fac,kind=kind)
 c=counters[-1]; print('KPROP',k,kind,L,'ARC',fl,'raw',c.raw_total(),'raw/ARC',c.raw_total()/fl,flush=True)
 print('raw by region',c.raw_flop_dict(),flush=True)
# Pure-function regression checks, without executing the GPU CLI or creating files.
tree=ast.parse((P/'gpu_samplers.py').read_text()); fs=[x for x in tree.body if isinstance(x,ast.FunctionDef)]
gpu={'torch':torch,'np':np,'anet':anet,'n':256,'MUB_BLOCK':66048,'MUB_SETUP':6*256**3+16*256**2};exec(compile(ast.Module(body=fs,type_ignores=[]),'gpu_samplers.py','exec'),gpu)
a=gpu['forward_from_z1'](torch.ones(2,3),[torch.eye(3),2*torch.eye(3),3*torch.eye(3)],[0.,0.],3,None,[3])[3]
print('ZERO SIGMA',a.tolist())
# Quadratic radial polynomial counterexample to the claimed exact Gaussian 5-design.
rng=np.random.default_rng(123)
rr=np.concatenate([samplers.chi_radii(256,256,rng,'strat') for _ in range(129)])
print('full-block Gaussian polynomial ||x||^2:',float(np.mean(rr**2)),'exact Gaussian moment',256)
print('full-block Gaussian polynomial ||x||^4:',float(np.mean(rr**4)),'exact Gaussian moment',256*258)
