import resource
resource.setrlimit(resource.RLIMIT_CPU,(550,560))
import json,pickle,sys,math
from pathlib import Path
import numpy as np
from scipy.special import gammaln,ndtr
from scipy.stats import chi
S=Path('/tmp/arcn_review2/claims3_snapshot/tails'); O=Path('/tmp/arcn_review2/claims3_checks')
sys.path.insert(0,str(S)); import deduce as D
rng=np.random.default_rng(20261029); pick=rng.integers(0,20,(10000,20))
def ci(a):return np.quantile(a,[.025,.975],axis=0,method='inverted_cdf')
def radial(t,m,nodes=161,lo=1e-9,hi=1-1e-12):
 r=np.linspace(chi.ppf(lo,256),chi.ppf(hi,256),nodes);w=chi.pdf(r,256);w/=np.trapezoid(w,r)
 mg=m/np.exp(np.arange(1,5)/2*np.log(2)+gammaln((256+np.arange(1,5))/2)-gammaln(128))
 a,b,c,d=mg;k=np.array([a,b-a*a,c-3*a*b+2*a**3,d-4*a*c-3*b*b+12*a*a*b-6*a**4])
 # Solve for the angular saddle directly, then use t/r. Equivalent change of variables, independent implementation.
 vals=[]
 for rr in r:
  z=t/rr
  roots=np.roots([k[3]/6,k[2]/2,k[1],k[0]-z]);ok=[]
  for x in roots:
   if abs(x.imag)<1e-9*max(1,abs(x.real)) and x.real>0:
    th=x.real;Kpp=k[1]+k[2]*th+k[3]*th*th/2
    if Kpp>0:ok.append((th,Kpp))
  if not ok or z<=k[0]:return np.nan
  th,Kpp=min(ok);A=th*z-sum(k[j]*th**(j+1)/math.factorial(j+1) for j in range(4))
  if A<=0:return np.nan
  v=np.sqrt(2*A);u=th*np.sqrt(Kpp);vals.append(ndtr(-v)+np.exp(-v*v/2)/np.sqrt(2*np.pi)*(1/u-1/v))
 return np.trapezoid(w*np.array(vals),r)
for L in (8,4):
 cache=pickle.load(open(O/f'cache_{L}_20.pkl','rb'));ps=[];pr=[];rse=[];half=[];xc=[];est=[];raw=[];cost=[]
 for s in range(44,64):
  t=json.load(open(S/f'runs/fresh3/truth/truth_{s}_s0.001.json'))['by_L'][str(L)]
  th=json.load(open(S/f'runs/fresh3/thr/thr_{s}.json'));assert np.array_equal(t['thr'],th['thr'][str(L)])
  assert t['N']==1000 and t['R']==8 and np.shape(t['logp'])==(8,5,8)
  assert np.shape(t['xcheck']['logp'])==(2,5,4)
  pp=np.exp(t['logp']);p=pp.mean(-1);se=pp.std(-1,ddof=1)/np.sqrt(8)/p;h=np.log2(pp[:,:,:4].mean(-1)/pp[:,:,4:].mean(-1))
  xx=np.exp(t['xcheck']['logp']);xp=xx.mean(-1);xse=xx.std(-1,ddof=1)/2/xp;xb=abs(np.log(xp/p[:2]))>3*np.sqrt(se[:2]**2+xse**2)
  prov=(se>.25)|(abs(h)>1);prov[:2]|=xb
  ka=json.load(open(S/f'runs/kp_fresh3/kp_{s}_L{L}_s0.001_k3fa.json'));kf=json.load(open(S/f'runs/kp_fresh3/kp_{s}_L{L}_s0.001_k3f.json'))
  m=np.array(ka['moments']);ee=np.array([[radial(t['thr'][i][j],m[:,i]) for j in range(5)]for i in range(8)])
  estimates=np.array([ee,ka['heavi'],kf['heavi']]);rel=np.abs(estimates/p-1)
  assert np.max(abs(rel[0]-cache[s-44]['rel']['P3A']))<1e-6
  assert np.array_equal(prov,cache[s-44]['prov'])
  ps.append(p);pr.append(prov);rse.append(se);half.append(h);xc.append(xb);est.append(estimates);raw.append(pp)
  cost.append([ka['flops_shared'],ka['flops_moments'],ka['flops_heavi_each'][0],kf['flops_shared']+kf['flops_heavi_each'][0],ka['secs']])
 P=np.array(ps);V=~np.array(pr);E=np.array(est);R=np.abs(E/P[:,None]-1);raw=np.array(raw);C=np.array(cost)
 np.savez(O/f'independent_L{L}.npz',P=P,V=V,E=E,raw=raw,R=R)
 print('L',L,'PROV',(~V).sum((0,1)).tolist(),'RSE_MED',np.median(rse,axis=(0,1)).tolist(),'RSE_MAX',np.max(rse,axis=(0,1)).tolist(),'XFAIL',np.sum(xc,axis=(0,1)).tolist(),flush=True)
 print('COST shared,mom,heavi,K3f,secs med',np.median(C,axis=0).tolist(),'range_secs',C[:,-1].min(),C[:,-1].max())
 print('COST_RATIOS 1,5vsaug,1vsK3f',np.median((C[:,0]+C[:,1])/(C[:,0]+C[:,2])),np.median((C[:,0]+C[:,1])/(C[:,0]+5*C[:,2])),np.median((C[:,0]+C[:,1])/C[:,3]))
 for valid in (False,True):
  mask=V if valid else np.ones_like(V)
  for comp in (1,2):
   dif=((R[:,0]<=.5)*mask).sum(1)/mask.sum(1)-((R[:,comp]<=.5)*mask).sum(1)/mask.sum(1)
   print('GAIN',valid,comp,dif.mean(0).tolist(),'CI',ci(dif[pick].mean(1)).tolist())
  for meth in range(3):
   meds=[];ends=[];mids=[]
   for k in range(5):
    a=np.where(mask[:,:,k],R[:,meth,:,k],np.nan);vals=a[np.isfinite(a)];meds.append(np.quantile(vals,.5,method='inverted_cdf'));mids.append(np.median(vals))
    b=a[pick].reshape(len(pick),-1);count=np.isfinite(b).sum(1);b.sort(1);b=b[np.arange(len(pick)),(count-1)//2];ends.append(ci(b).tolist())
   print('BTZ',valid,meth,'med',meds,'CI',ends,'midpoint',mids)
 # Sensitivity to stochastic reference estimates, not a replacement for prereg scoring.
 for k in (2,4):
  trials=10000;ix=rng.integers(0,8,(trials,20,8,8));refs=np.take_along_axis(np.broadcast_to(raw[:,:,k,:],ix.shape),ix,-1).mean(-1)
  rel=np.abs(E[None,:,:,:,k]/refs[:,:,None,:]-1)
  delta=(rel[:,:,0]<=.5).mean(-1)-(rel[:,:,1]<=.5).mean(-1)
  print('TRUTH_RESAMPLE',k,'gain mean',delta.mean(),'CI',ci(delta.mean(-1)).tolist(),'JOINT',ci(np.take_along_axis(delta,pick,1).mean(1)).tolist())
 print('VALID_COUNTS',V.sum((0,1)).tolist(),flush=True)
for L in (4,8):
 for var in ('k3f','k3fa'):
  aa=[];bb=[]
  for s in (100,101):
   a=np.array(json.load(open(S/f'runs/kp_x1/kp_{s}_L{L}_s0.001_{var}.json'))['moments'])
   b=np.array(json.load(open(S/f'runs/arcn-tdev1/x3/x3_{s}_s0.001.json'))['moments'][str(L)])
   aa.append(np.array(D.cumulants_from_moments(a)));bb.append(np.array(D.cumulants_from_moments(b)))
  a=np.concatenate(aa,axis=1);b=np.concatenate(bb,axis=1);sk=a[2]/a[1]**1.5;skt=b[2]/b[1]**1.5;ku=a[3]/a[1]**2;kut=b[3]/b[1]**2
  print('MOMENTS',L,var,'all512 skew corr',np.corrcoef(sk,skt)[0,1],'exkurt corr',np.corrcoef(ku,kut)[0,1],'rawk4 corr',np.corrcoef(a[3],b[3])[0,1],'exkurt mederr',np.median(abs(ku-kut)),'exkurt range',np.min(ku),np.max(ku),'rawk4 range',np.min(a[3]),np.max(a[3]))
  for s in (100,101):
   print('PER_NET',L,var,s, 'skew corr',np.corrcoef(sk[(s-100)*256:(s-99)*256],skt[(s-100)*256:(s-99)*256])[0,1],'exkurt corr',np.corrcoef(ku[(s-100)*256:(s-99)*256],kut[(s-100)*256:(s-99)*256])[0,1])
print('DONE',flush=True)
