import resource
resource.setrlimit(resource.RLIMIT_CPU,(550,560))
import os,sys,json,math,pickle,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/tmp/arcn_review2'); SRC=ROOT/'claims2_snapshot'
sys.path.insert(0,str(SRC))
BASE=Path('/home/agent/workspaces/mats/work/arc-noisy/tails')
RD=BASE/'runs'; MC=RD/'arcn-tdev1/mc'; KD=RD/'arcn-tdev1/kd'; MODE=RD/'arcn-tdev1/mode'
sys.argv=['analyze_tails.py',str(RD/'truth_eval'),str(RD/'kp'),str(MC),str(KD),'--mode',str(MODE)]
ns={'__file__':str(SRC/'analyze_tails.py')}
exec((SRC/'analyze_tails.py').read_text().split('summary = {}')[0],ns)
# Radial readouts are separately checked. They do not affect the primary U3 or baseline computations.
ns['D'].radial=lambda *a,**kw: float('nan')
DATA={}
for sg in (.001,.003,.01):
 S=json.load(open(SRC/f'summary_s{sg:g}.json'))
 for L in (4,8,32):
  out,meth=ns['ev'](L,sg); DATA[sg,L]=(out,meth)
  P=np.array(out['p']); V=~np.array(out['prov']); fl=np.array(out['flags'])
  print('CELL',sg,L,'prov',(~V).sum((0,1)).tolist(),'flags',[(i,int(((fl&i)>0).sum())) for i in (1,2,4,8,16)])
  mism=[]
  for name,lst in meth.items():
   if name in ('U2r','U2ar','U3r') or '@' in name or len(lst)!=20:continue
   for valid in ('all','valid'):
    mask=V if valid=='valid' else np.ones_like(V)
    for am in ('ev','am'):
     key=f'{sg}|{L}|{name}|{am}|{valid}'
     if key not in S:continue
     B=np.array([ns['b50_of'](d,am=='am') for d in lst])
     for k in range(5):
      old=S[key].get(str(ns['KS'][k])); vals=B[:,:,k][mask[:,:,k]]
      if old is None:continue
      med=np.quantile(vals,.5,method='inverted_cdf'); fr=np.isfinite(vals).mean()
      if not (med==old['b50'] and fr==old['reached']):mism.append((key,k,med,fr,old))
  print('RAW TABLE MISMATCHES',mism)
  for valid in ('all','valid'):
   mask=V if valid=='valid' else np.ones_like(V)
   for k in range(5):
    rec=S[f'{sg}|{L}|win|U3|am|{valid}'][str(ns['KS'][k])]; comp=rec['best']
    B=np.array([ns['b50_of'](d,True) for d in meth['U3']])[:,:,k]; C=np.array([ns['b50_of'](d,True) for d in meth[comp]])[:,:,k]
    for convention in ('lower','midpoint'):
     def med(x):
      s=np.sort(x); n=len(x)
      return s[(n-1)//2] if convention=='lower' else (s[(n-1)//2]+s[n//2])/2
     bo=np.array([med(B[j][mask[j,:,k]]) for j in range(20)]); bc=np.array([med(C[j][mask[j,:,k]]) for j in range(20)])
     with np.errstate(invalid='ignore'):ratio=np.where(np.isinf(bo)&np.isinf(bc),1,bc/bo)
     rng=np.random.default_rng(12345); r=ratio[rng.integers(0,20,(10000,20))]; r.sort(axis=1)
     bs=r[:,9] if convention=='lower' else (r[:,9]+r[:,10])/2
     ci=np.quantile(bs,[.025,.975],method='inverted_cdf')
     if valid=='all' or rec['lo']>1:
      print('WIN',valid,ns['KS'][k],comp,convention,med(ratio),ci.tolist(),'finite nets',np.isfinite(bo).sum(),'paper',rec['lo']>1)
  # Test fixed m20 label versus cheapest successful tested budget.
  for name in ('U3','C-MCx-edge'):
   b20=np.array([ns['b50_of'](d,True) for d in meth[name+'@m20']]); b=np.array([ns['b50_of'](d,True) for d in meth[name]])
   print('BUDGET',name,'m20_success',np.isfinite(b20).sum((0,1)).tolist(),'selected_success',np.isfinite(b).sum((0,1)).tolist())
  # Fraction a single frozen U1 run actually reaches target.
  if L==32:
   rm=np.array([d['rmse'] for d in meth['U1']]);print('U1 actual <=.25', (rm<=.25).mean((0,1)).tolist())
  if sg==.001:
   est=np.array([d['est'] for d in meth['C-K3f']]);print('ARC negative fraction',(est<0).mean((0,1)).tolist())
  for tag in ('m20','m22'):
   rm=np.array([d['rmse'] for d in meth['U3@'+tag]]);print('U3 RMSE',tag,'median',np.median(np.sqrt(rm),axis=(0,1)).tolist())
  print(flush=True)
pickle.dump(DATA,open(ROOT/'claims2_checks/raw_recomputed.pkl','wb'))
# Noise probability changes, with and without provisional events.
for L in (4,8,32):
 a=np.array(DATA[.001,L][0]['p']);va=~np.array(DATA[.001,L][0]['prov'])
 for sg in (.003,.01):
  b=np.array(DATA[sg,L][0]['p']);vb=~np.array(DATA[sg,L][0]['prov']);dif=np.log10(b/a)
  print('NOISE',L,sg,'median',np.median(dif,axis=(0,1)).tolist(),'k12_q',np.quantile(dif[:,:,4],[0,.05,.95,1]).tolist(),'max_valid',np.max(dif[:,:,4][(va&vb)[:,:,4]]))
print('hash',hashlib.sha256((SRC/'PREREG-TAILS.md').read_bytes()).hexdigest())
