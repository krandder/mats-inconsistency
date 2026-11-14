import resource,pickle,json
resource.setrlimit(resource.RLIMIT_CPU,(550,560))
import numpy as np
from pathlib import Path
R=Path('/tmp/arcn_review2');data=pickle.load(open(R/'claims2_checks/raw_recomputed.pkl','rb'))
rng=np.random.default_rng(7);q=lambda x,p:np.quantile(x,p,method='inverted_cdf')
lines=[]
def boot(v,kind):
 a=v[rng.integers(0,20,(4000,20))]
 if kind=='mean':m=v.mean(); bs=a.mean(1)
 elif kind=='lower':m=q(v,.5);bs=np.sort(a,axis=1)[:,9]
 else:m=np.median(v);bs=np.median(a,axis=1)
 return m,*q(bs,[.025,.975])
for sg in (.001,.003,.01):
 for L in (4,8):
  out,meth=data[sg,L];B=np.array([d['b50am'] for d in meth['U3']]);GS=np.array([d['b50am'] for d in meth['C-GS']])
  reach={'U3':np.isfinite(B).mean(1),'U3 fixed 2^20':np.array([d['rmse']<=.25 for d in meth['U3@m20']]).mean(1),'C-K3f':np.array([d['rel']<=.5 for d in meth['C-K3f']]).mean(1),'U2':np.array([d['rel']<=.5 for d in meth['U2']]).mean(1)}
  lines.append(f'\n=== sigma {sg}, L = {L}')
  for name,vals in reach.items():
   row=[]
   for k in range(5):
    m,lo,hi=boot(vals[:,k],'mean');row.append(f'{m:.2f} [{lo:.2f}, {hi:.2f}]')
   lines.append(f'  fraction within 50% {name:>14}: '+ ' | '.join(row))
  for conv,kind in [('lower median','lower'),('midpoint median','mid')]:
   row=[]
   for k in range(5):
    if kind=='lower':bo=np.sort(B[:,:,k],axis=1)[:,3];bc=np.sort(GS[:,:,k],axis=1)[:,3]
    else:bo=np.median(B[:,:,k],axis=1);bc=np.median(GS[:,:,k],axis=1)
    v=bc/bo;m,lo,hi=boot(v,kind);row.append(f'{m:.3g} [{lo:.3g}, {hi:.3g}]'+(' WIN' if lo>1 else ''))
   lines.append(f'  U3 / splitting B50 ratio ({conv}): '+ ' | '.join(row))
actual='\n'.join(lines).strip();expected=(R/'claims2_snapshot/sensitivity.txt').read_text().strip()
print('EXACT SENSITIVITY TEXT MATCH',actual==expected)
if actual!=expected:
 import difflib
 print('\n'.join(difflib.unified_diff(expected.splitlines(),actual.splitlines())))
# Paired noise, both references validated, 10000 net resamples.
picks=np.random.default_rng(12345).integers(0,20,(10000,20))
p=np.array(data[.001,32][0]['p']);v=~np.array(data[.001,32][0]['prov'])
for sg in (.003,.01):
 p1=np.array(data[sg,32][0]['p']);v1=~np.array(data[sg,32][0]['prov']);d=np.log10(p1/p)
 for k in range(5):
  mask=(v&v1)[:,:,k];a=np.where(mask,d[:,:,k],np.nan)[picks].reshape(10000,-1)
  print('PAIRED VALID',sg,'k',(4,6,8,10,12)[k],'n',mask.sum(),'median',np.median(d[:,:,k][mask]),'CI',np.quantile(np.nanmedian(a,axis=1),[.025,.975]).tolist())
print('DONE')
