import sys,json,itertools
from pathlib import Path
import numpy as np
P=Path('/home/agent/workspaces/mats/work/arc-noisy');sys.path.insert(0,str(P))
import analyze as A,truthlib
rng=np.random.default_rng(234);ix=rng.integers(0,24,(10000,24))
for L in (8,32):
 nets=A.collect(L,.001)
 for v in ('k2','k2a','k3f','k3fa'):
  for a,b in itertools.combinations((.003,.01,.03,.1),2):
   vals=[]; costs=[]
   w=(b*b-.001**2)/(b*b-a*a)
   for s in range(24):
    ka=A.kprop(s,a,L,v);kb=A.kprop(s,b,L,v)
    if ka is None or kb is None:break
    vals.append(truthlib.nmse(w*ka[2][L-1]+(1-w)*kb[2][L-1],A.truth(s,.001),L));costs.append(ka[1]+kb[1])
   if len(vals)!=24 or max(costs)>3e11:continue
   B=next(x for x in A.BUDGETS if x>=max(costs));bi=list(A.BUDGETS).index(B)
   bv=A.best_of(nets,[f'kprop-{k}' for k in A.KPROP],bi)[0]
   ratio=np.array([nets[s][bv][bi] for s in range(24)])/vals
   ci=np.quantile(np.median(ratio[ix],axis=1),[.025,.975]);cm=np.quantile(np.mean(ratio[ix],axis=1),[.025,.975])
   print(L,v,a,b,'w',round(w,4),'B',B,'baseline',bv,'nMSE',float(np.median(vals)),'gain',float(np.median(ratio)),ci.tolist(),float(ratio.mean()),cm.tolist(),flush=True)
# Recompute the mean shifts from all 24 final evaluation truths.
for L in (8,32):
 for sg in (.003,.01,.03,.1):
  vals=[]
  for s in range(24):
   base=A.truth(s,.001); T=A.truth(s,sg)
   vals.append(((T['mean'][L-1]-base['mean'][L-1])**2).mean()/base['var'][L-1].mean()-1/base['N']-T['var'][L-1].mean()/base['var'][L-1].mean()/T['N'])
  print('SHIFT',L,sg,float(np.median(vals)),flush=True)
