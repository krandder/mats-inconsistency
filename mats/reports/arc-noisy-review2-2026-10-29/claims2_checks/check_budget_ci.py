import resource
resource.setrlimit(resource.RLIMIT_CPU,(550,560))
import sys,json,pickle
from pathlib import Path
import numpy as np
R=Path('/tmp/arcn_review2');sys.path.insert(0,str(R/'claims2_snapshot'))
from common import load_truth,c_move,c_path,run_evals
DATA=pickle.load(open(R/'claims2_checks/raw_recomputed.pkl','rb'));B=Path('/home/agent/workspaces/mats/work/arc-noisy/tails/runs')
pick=np.random.default_rng(12345).integers(0,20,(10000,20))
def pooled(V,mask):
 a=np.where(mask,V,np.nan)[pick].reshape(10000,-1);n=(~np.isnan(a)).sum(1);a.sort(axis=1);bs=a[np.arange(10000),(n-1)//2]
 return [np.quantile(V[mask],.5,method='inverted_cdf'),*np.quantile(bs,[.025,.975],method='inverted_cdf')]
for sg in (.001,.003,.01):
 for L in (4,8,32):
  out,meth=DATA[sg,L];P=np.array(out['p']);V=~np.array(out['prov']);kp=json.load(open(B/'kp'/f'kp_0_L{L}_s{sg:g}_k3f.json'));arc=kp['flops_shared']+kp['flops_heavi_each'][0]
  for bud in (arc,3e11):
   mc=np.sqrt((1-P)/(P*bud/c_path(L)));gs=[]
   for s in range(20):
    T=load_truth(B/'truth_eval'/f'truth_{s}_s{sg:g}.json')[L];pil=run_evals(500,T['nlev'])*c_move(L);run=run_evals(2000,T['nlev'])*c_move(L)
    gs.append(np.sqrt(T['rv1']*run/np.maximum(1e-200,bud-pil)))
   for valid in (False,True):
    mask=V if valid else np.ones_like(V);ans={}
    for name,x in [('MC',mc),('GS',np.array(gs))]:
     z=[pooled(x[:,:,k],mask[:,:,k]) for k in range(5)];ans[name]=''.join('Y' if m<=.9 and hi<1 else 'n' for m,lo,hi in z)
    print('BUDGET corrected',sg,L,bud,valid,ans)
  if L==32 and sg==.001:print('K3f single vector',arc,'five vectors',kp['flops_shared']+sum(kp['flops_heavi_each']))
  if sg==.001:
   for name in ('U3','C-GS','C-MC','C-K3f'):
    lst=meth[name]
    if name=='C-K3f':bs=np.array([np.where(d['rel']<=.5,d['costam'],np.inf) for d in lst])
    else:bs=np.array([d['b50am'] for d in lst])
    for k in range(5):
     m,lo,hi=pooled(bs[:,:,k],np.ones((20,8),bool));print('POOLED CI',L,name,k,m,lo,hi)
print('DONE')
