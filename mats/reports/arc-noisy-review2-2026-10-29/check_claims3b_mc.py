import json,sys,resource
from pathlib import Path
import numpy as np
resource.setrlimit(resource.RLIMIT_CPU,(590,600))
root=Path('/home/agent/workspaces/mats/work/arc-noisy/tails');sys.path.insert(0,str(root))
import deduce as D
from common import load_truth
rng=np.random.default_rng(12345)
reached=[]
for tag in ('m20','m22'):
 errors=[]
 for s in range(20):
  t=load_truth(root/f'runs/truth_eval/truth_{s}_s0.001.json')[32]; p=np.exp(t['lp'])
  mc=json.loads((root/f'runs/arcn-tdev1/mc/mc_{tag}_{s}_s0.001.json').read_text())
  est=[]
  for rep in mc['reps']:
   mom=np.array(rep['moments']['32']); e=np.empty((8,5))
   for i in range(8):
    kk=D.cumulants_from_moments(mom[:,i])
    for k in range(5):
     v=D.saddle(t['thr'][i,k],kk,4)
     if not np.isfinite(v): v=D.saddle(t['thr'][i,k],kk,3)
     if not np.isfinite(v): v=D.gauss(t['thr'][i,k],kk)
     e[i,k]=v if np.isfinite(v) else 0.
   est.append(e)
  errors.append(np.sqrt(np.mean((np.array(est)/p-1)**2,axis=0)))
 errors=np.array(errors); reached.append(errors<=.5)
 print(tag,'fraction relRMSE <= .5',np.mean(errors<=.5,axis=(0,1)),flush=True)
 for k in (0,1):
  vals=errors[:,:,k]; pick=rng.integers(0,20,(10000,20)); bm=np.quantile(vals[pick].reshape(10000,160),.5,axis=1,method='inverted_cdf')
  print('level',k,'median',np.quantile(vals,.5,method='inverted_cdf'),'95% CI',np.quantile(bm,[.025,.975],method='inverted_cdf'),flush=True)
print('U3 combined reached fractions',np.any(reached,axis=0).mean((0,1)))
print('CPU seconds',resource.getrusage(resource.RUSAGE_SELF).ru_utime)
