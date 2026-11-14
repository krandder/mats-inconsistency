import os,sys,json,math,resource
resource.setrlimit(resource.RLIMIT_CPU,(550,560))
from pathlib import Path
import numpy as np
from scipy.special import ndtr
ROOT=Path('/tmp/arcn_review2'); S=ROOT/'claims2_snapshot'; B=Path('/home/agent/workspaces/mats/work/arc-noisy/tails/runs')
sys.path.insert(0,str(S)); import deduce as D
rng=np.random.default_rng(12345); picks=rng.integers(0,20,(10000,20))
q=lambda v,p:np.quantile(v,p,method='inverted_cdf')
def meanvar(logs):
 w=np.exp(np.array(logs)); p=w.mean(-1); se=w.std(axis=-1,ddof=1)/np.sqrt(w.shape[-1]);return p,se/p

def readout(t,m):
 a,b,c,d=m; v=b-a*a; c3=c-3*b*a+2*a**3;c4=d-4*c*a-3*b*b+12*b*a*a-6*a**4
 for order in (4,3):
  e4=c4 if order==4 else 0
  co=np.trim_zeros([e4/6,c3/2,v,a-t],'f'); roots=np.roots(co)
  roots=sorted(z.real for z in roots if abs(z.imag)<=1e-9*max(1,abs(z.real)) and z.real>0 and v+c3*z.real+e4*z.real**2/2>0)
  if roots:
   z=roots[0]; dev=z*t-(a*z+v*z*z/2+c3*z**3/6+e4*z**4/24)
   if dev>0:
    w=np.sqrt(2*dev);u=z*np.sqrt(v+c3*z+e4*z*z/2)
    return ndtr(-w)+np.exp(-w*w/2)/np.sqrt(2*np.pi)*(1/u-1/w)
 return ndtr(-(t-a)/np.sqrt(v))

for L in (4,8):
 ps=[]; rs=[]; cs=[]; flags=[]; deltas=[]; costs=[];agree=[]
 for seed in range(24,44):
  tr=json.load(open(B/'truth_conf'/f'truth_{seed}_s0.001.json')); t=tr['by_L'][str(L)]
  kp=json.load(open(B/'kp_conf'/f'kp_{seed}_L{L}_s0.001_k3f.json'))
  mc=json.load(open(B/'arcn-tconf/mc'/f'mc_xc_{seed}_s0.001.json'))
  th=json.load(open(B/'thr_conf'/f'thr_{seed}.json'))
  assert tr['seed']==kp['seed']==mc['seed']==th['seed']==seed
  assert kp['L']==L and kp['variant']=='k3f' and t['R']==16 and t['N']==2000 and mc['N']==2**23
  assert np.array_equal(th['thr'][str(L)],t['thr'])
  logs=np.array(t['logp']);assert logs.shape==(8,5,16)
  p,rse=meanvar(logs); h1=np.exp(logs[:,:,:8]).mean(-1); h2=np.exp(logs[:,:,8:]).mean(-1)
  fl=(rse>.25).astype(int);fl|=(np.abs(np.log2(h1/h2))>1)*2
  cnt=np.array(mc['reps'][0]['counts'][str(L)])[:,0]; pm=cnt/mc['N'];serm=np.sqrt((1-pm)/cnt)
  fl[:,0]|=(np.abs(np.log(pm/p[:,0]))>3*np.sqrt(rse[:,0]**2+serm**2))*4
  px,rsx=meanvar(t['xcheck']['logp']); fl[:2]|=(np.abs(np.log(px/p[:2]))>3*np.sqrt(rsx**2+rse[:2]**2))*8
  m=np.array(kp['moments']);u=np.array([[readout(thr,m[:,i]) for thr in t['thr'][i]] for i in range(8)])
  authored=np.array([[D.saddle(thr,D.cumulants_from_moments(m[:,i]),4) for thr in t['thr'][i]] for i in range(8)])
  agree.append(float(np.max(np.abs(u-authored)/np.maximum(np.abs(u),1e-300))))
  ps.append(p);rs.append(np.abs(u/p-1));cs.append(np.abs(np.array(kp['heavi'])/p-1)); flags.append(fl)
  shared=kp['flops_shared'];term=np.array(kp['flops_heavi_each'])[0];mom=kp['flops_moments'];ro=D.F_SADDLE+D.F_CUMULANTS
  costs.append(((shared+mom+ro)/(shared+term),((shared+mom)/1280+ro)/((shared+5*term)/1280)))
 P=np.array(ps);U=np.array(rs);C=np.array(cs);F=np.array(flags);V=F==0
 print('L',L,'provisional',np.sum(~V,axis=(0,1)).tolist(),'p medlog',np.median(np.log10(P),axis=(0,1)).tolist(),'independent readout max relative disagreement',max(agree),'costs',costs[0])
 for valid in (False,True):
  mask=V if valid else np.ones_like(V)
  for k in range(5):
   d=np.array([((U[j,:,k]<=.5).astype(float)-(C[j,:,k]<=.5))[mask[j,:,k]].mean() for j in range(20)])
   ci=q(d[picks].mean(1),[.025,.975]);print('GAP',valid,(4,6,8,10,12)[k],d.mean(),ci.tolist(),'bar passes',d.mean()>=.2 and ci[0]>0)
  for name,R in [('U2',U),('C-K3f',C)]:
   fl=[]
   for k in range(5):
    vals=R[:,:,k][mask[:,:,k]];m=q(vals,.5)
    ar=np.where(mask[:,:,k],R[:,:,k],np.nan)[picks].reshape(10000,-1);count=(~np.isnan(ar)).sum(1);ar.sort(axis=1)
    bs=ar[np.arange(10000),(count-1)//2];ci=q(bs,[.025,.975]);fl.append(m<=.9 and ci[1]<1)
    print('ERR',valid,name,(4,6,8,10,12)[k],m,ci.tolist())
   print('BETTERZERO',valid,name,fl)
 print('WITHIN50 U2 C-K3f',(U<=.5).mean((0,1)).tolist(),(C<=.5).mean((0,1)).tolist())
print('DONE')
