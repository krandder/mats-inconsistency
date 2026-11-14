import sys,json,pickle,math
from pathlib import Path
import numpy as np
from scipy.stats import chi
from scipy.special import ndtr
ROOT=Path('/tmp/arcn_review2');SRC=ROOT/'claims_snapshot';sys.path.insert(0,str(SRC))
import deduce as D
DATA=pickle.load(open(ROOT/'claims_checks/raw_recomputed.pkl','rb'))
BASE=Path('/home/agent/workspaces/mats/work/arc-noisy/tails/runs')
r=np.linspace(chi.ppf(1e-9,256),chi.ppf(1-1e-12,256),161);w=chi.pdf(r,256);w/=np.trapezoid(w,r)
def radial(t,mom):
 # Set y = theta*r. Then K_r(theta)=K_g(y) and K'_g(y)=t/r.
 kg=np.array(D.cumulants_from_moments([mom[j]/D.chi_moment(256,j+1) for j in range(4)]));t0=t/r
 if np.any(t0<=kg[0]):return np.nan
 coef=np.array([np.full(161,kg[3]/6),np.full(161,kg[2]/2),np.full(161,kg[1]),kg[0]-t0]).T
 while np.all(coef[:,0]==0):coef=coef[:,1:]
 n=coef.shape[1]-1;a=np.zeros((161,n,n));a[:,0,:]=-coef[:,1:]/coef[:,:1]
 for j in range(n-1):a[:,j+1,j]=1
 roots=np.linalg.eigvals(a);real=roots.real
 kp2=kg[1]+kg[2]*real+kg[3]*real**2/2
 good=(np.abs(roots.imag)<=1e-9*np.maximum(1,np.abs(real)))&(real>0)&(kp2>0)
 th=np.min(np.where(good,real,np.inf),axis=1)
 if not np.isfinite(th).all():return np.nan
 kval=kg[0]*th+kg[1]*th**2/2+kg[2]*th**3/6+kg[3]*th**4/24
 d=th*t0-kval
 if np.any(d<=0):return np.nan
 ww=np.sqrt(2*d);u=th*np.sqrt(kg[1]+kg[2]*th+kg[3]*th**2/2)
 vals=ndtr(-ww)+np.exp(-ww**2/2)/np.sqrt(2*np.pi)*(1/u-1/ww)
 return np.trapezoid(w*vals,r)
validation=[]
for L in (4,8,32):
 kp=json.load(open(BASE/'kp'/f'kp_0_L{L}_s0.001_k3f.json'));T=DATA[.001,L][0]
 from common import load_truth
 th=load_truth(BASE/'truth_eval'/'truth_0_s0.001.json')[L]['thr']
 for i in range(2):
  for k in (0,4):
   mom=np.array(kp['moments'])[:,i];a=radial(th[i,k],mom);b=D.radial(th[i,k],mom)
   assert (np.isnan(a) and np.isnan(b)) or np.isclose(a,b,rtol=1e-9,atol=1e-25),(a,b)
   validation.append((L,i,k,a,b))
print('vector vs original scalar validated',validation,flush=True)
for sg in (.001,.003,.01):
 S=json.load(open(SRC/f'summary_s{sg:g}.json'))
 for L in (4,8,32):
  out,meth=DATA[sg,L];P=np.array(out['p']);V=~np.array(out['prov']);results={}
  for tag in ('m20','m22'):
   estimates=[]
   for s in range(20):
    m=json.load(open(BASE/'arcn-tdev1/mc'/f'mc_{tag}_{s}_s{sg:g}.json'));T=load_truth(BASE/'truth_eval'/f'truth_{s}_s{sg:g}.json')[L]
    reps=[]
    for rep in m['reps']:
     mom=np.array(rep['moments'][str(L)]);reps.append([[radial(t,mom[:,i]) for t in T['thr'][i]] for i in range(8)])
    estimates.append(np.nan_to_num(reps,nan=0.0))
   E=np.array(estimates);rm=np.mean((E/P[:,None]-1)**2,axis=1)
   results[tag]=rm
   cost=meth['U3r@'+tag][0]['costam'];results[tag+'_b']=np.where(rm<=.25,cost,np.inf)
  B=np.minimum(results['m20_b'],results['m22_b']);errors=[]
  for valid in ('all','valid'):
   mask=V if valid=='valid' else np.ones_like(V)
   for k in range(5):
    expected=S[f'{sg}|{L}|U3r|am|{valid}'][str((4,6,8,10,12)[k])];v=B[:,:,k][mask[:,:,k]];med=np.quantile(v,.5,method='inverted_cdf');fr=np.isfinite(v).mean()
    if not (med==expected['b50'] and fr==expected['reached']):errors.append((valid,k,med,fr,expected))
  print('U3r RAW CHECK',sg,L,'errors',errors,flush=True)
  if sg==.001 and L==8:
   print('U3r L8 m20 successes', (results['m20']<=.25).sum((0,1)).tolist(),'combined',np.isfinite(B).sum((0,1)).tolist(),flush=True)
   for k in (1,2):
    comp=S[f'{sg}|{L}|win|U3r|am|all'][str((4,6,8,10,12)[k])]['best'];C=np.array([d['b50am'] for d in meth[comp]])[:,:,k]
    for lower in (True,False):
     med=lambda a: np.quantile(a,.5,method='inverted_cdf') if lower else np.median(a)
     bo=np.array([med(a) for a in B[:,:,k]]);bc=np.array([med(a) for a in C]);rr=bc/bo;pick=np.random.default_rng(12345).integers(0,20,(10000,20));bs=np.quantile(rr[pick],.5,axis=1,method='inverted_cdf') if lower else np.median(rr[pick],axis=1)
     print('RADIALWIN',k,lower,med(rr),np.quantile(bs,[.025,.975],method='inverted_cdf').tolist(),flush=True)
print('DONE')
