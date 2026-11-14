import json, sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('code').resolve()))
import common, deduce as D
root=Path('/home/agent/workspaces/mats/work/arc-noisy/tails/runs/arcn-tdev1')
for L in (4,8,32):
 a=json.loads(Path(f'kp_sample/kp_102_L{L}_s0.001_k3f.json').read_text())
 sh,hv,mo=a['flops_shared'],a['flops_heavi_each'][0],a['flops_moments']
 print('L',L,'shared/heavi/mom',sh,hv,mo,'U2/C-K3f per-event',(sh+mo)/(sh+hv),'amortized',(sh+mo)/(sh+5*hv))
 missing=[]
 for seed in range(102,106):
  a=json.loads(Path(f'kp_sample/kp_{seed}_L{L}_s0.001_k3f.json').read_text()); T=common.load_truth(f'dev_truth/truth_{seed}_s0.001.json')[L]
  for b in range(8):
   k=D.cumulants_from_moments(np.array(a['moments'])[:,b])
   for j,t in enumerate(T['thr'][b]):
    roots=np.roots([k[3]/6,k[2]/2,k[1],k[0]-t]); valid=[]
    for r in roots:
     if abs(r.imag)<1e-8 and r.real>0 and k[1]+k[2]*r.real+k[3]*r.real**2/2>0:
      th=float(r.real); K=sum(k[q]*th**(q+1)/math.factorial(q+1) for q in range(4))
      if th*t-K>0:valid.append(th)
    if valid and not np.isfinite(D.saddle(t,k)):missing.append((seed,b,j,min(valid)))
 print('missed real convex roots with positive deviance',len(missing),missing[:4])

for L in (32,):
 stats=[]
 for s in range(102,106):
  mo=json.loads((root/f'mode/mode_{s}_L{L}_s0.001.json').read_text());T=common.load_truth(f'dev_truth/truth_{s}_s0.001.json')[L]
  for b in range(8):
   ru=mo['runs']['up' if mo['c'][b]>0 else 'down']; E=np.array(ru['est'])[:,b,2]; p=np.exp(T['lp'][b,2]);rmse=np.mean((E/p-1)**2)
   stats.append(rmse)
 print('mode exact <1 normalized residuals',sum(x<1 for x in stats),'code <.999',sum(x<.999 for x in stats),'out of',len(stats))
 print('rMSE in [.999,1)',[v for v in stats if .999<=v<1])

# Seed namespace checks in the registered main/truth/MC roles (pilot/frozen consume disjoint stream segments).
seeds=list(range(20))+list(range(100,108)); sigmas=[.001,.003,.01]
roles={}
for seed in seeds:
 for L in (4,8,32):
  roles.setdefault(50000+97*seed+L,[]).append(('threshold',seed,L))
  for sigma in sigmas:
   for role,start in [('truth',60000),('xcheck',70000)]:roles.setdefault(start+97*seed+7*L+round(1e4*sigma),[]).append((role,seed,L,sigma))
print('threshold/truth/xcheck seed collisions',[(k,v) for k,v in roles.items() if len(v)>1])
