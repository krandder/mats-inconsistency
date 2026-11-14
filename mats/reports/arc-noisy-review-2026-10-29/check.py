import os, sys, ast, math, json, shutil
os.environ['OMP_NUM_THREADS']='1'
import numpy as np, torch
from pathlib import Path
P=Path('/home/agent/workspaces/mats/work/arc-noisy')
sys.path.insert(0,str(P)); torch.set_num_threads(1)
import anet, samplers, kerdock
shutil.copytree(P/'ext/mlp_cumulant_propagation/src','/tmp/arcn_review/ext/src',dirs_exist_ok=True)
src=(P/'kprop_noisy.py').read_text().replace("'ext', 'mlp_cumulant_propagation', 'src'", "'/tmp/arcn_review/ext/src'")
ns={'__file__':str(P/'kprop_noisy.py')}; exec(compile(src,'kprop_noisy.py','exec'),ns)
KH=ns['KH']; HT=ns['HTensor']
for k,fac,kind in [(1,False,'simple'),(2,False,'simple'),(2,False,'augment'),(3,True,'simple'),(3,True,'augment')]:
 net=anet.Net(3,n=8,L=3,sigma=.1,r=[1]*3)
 ke={'simple':KH.SIMPLE,'augment':KH.AUGMENT}[kind]
 K=KH.coerce_input({1:torch.zeros(8,dtype=torch.float64),2:torch.eye(8,dtype=torch.float64)},k_max=k,kind=ke)
 for l in range(3):
  K=KH.linear_kprop(K,net.W[l],k_max=k,set_metric=2.)
  K=KH.nonlin_kprop(K,ns['WICK_COEF_D']['relu'],k_max=k,kind=ke,factor=fac,use_pK=True)
  old=K[2].to_tensor().clone(); K=ns['add_noise_to_K2'](K,.01)
  assert torch.allclose(K[2].to_tensor()-old,.01*torch.eye(8,dtype=torch.float64),atol=1e-12)
 net0=anet.Net(3,n=8,L=3)
 ours,_,_=ns['kprop_noisy'](net0,3,k,factor=fac,kind=kind,count_flops=False)
 ref=KH.mlp_kprop(ns['arc_mlp'](net0,3),{1:torch.zeros(8,dtype=torch.float64),2:torch.eye(8,dtype=torch.float64)},k_max=k,kind=ke,factor=fac,up_to_layer='act2')[1].to_tensor()
 print('kprop',k,kind,'noise covariance exact; upstream max diff',float((ours-ref).abs().max()),flush=True)
# Compile only pure functions from GPU CLI (no top-level execution).
tree=ast.parse((P/'gpu_samplers.py').read_text()); funcs=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))]
gpu={'torch':torch,'np':np,'anet':anet,'n':256}; exec(compile(ast.Module(body=funcs,type_ignores=[]),'gpu_samplers.py','exec'),gpu)
# deterministic noise-zero paths expose h update bug, checkpoint test uses positive s but prescribed deterministic mock.
def f(z,*args,**kwargs):return {2:z}
gpu['forward_from_z1']=f
z=torch.arange(33,dtype=torch.float64).reshape(11,3); cps=np.array([1,4,5,8,11]); sums={2:dict(run=torch.zeros(3,dtype=torch.float64),out=torch.zeros(5,3,dtype=torch.float64))}; idx=[0]
for st in range(0,11,4):gpu['run_points'](z[st:st+4],None,None,2,None,[2],sums,idx,cps,st)
print('checkpoints maxdiff',float((sums[2]['out']-torch.stack([z[:x].sum(0) for x in cps])).abs().max()),flush=True)
B=np.load(P/'kerdock129.npy').astype(np.float64)
print('MUB all pairs',kerdock.check(B),flush=True)
for L in [8,32]:
 net=anet.Net(100,L=L,sigma=.001,r=[1]*L); c=gpu['path_cost'](net,L,'mub'); N=int(1e9/c)//2*2
 # charged matvec already accounts basis multiply; only rotation setup should be extra
 blocks=math.ceil(N/(129*512)); setup=blocks*(6*256**3+16*256**2)
 print('MUB budget',L,N,c,'missing rotation setup',setup,'ratio',setup/1e9,flush=True)
for m in [2,4,8,12,16,24,32]:
 net=anet.Net(3,n=8,L=m,sigma=.001,r=[1]*m)
 _,cost,_=ns['kprop_noisy'](net,m,3,factor=True)
 print('small width cost',m,cost,flush=True)
for L in [8,32]:
 f=P/f'results/kprop_100_s0.001_L{L}_k3f.json'
 if f.exists():print('stored width256 cost',L,json.load(open(f))['flops'])
