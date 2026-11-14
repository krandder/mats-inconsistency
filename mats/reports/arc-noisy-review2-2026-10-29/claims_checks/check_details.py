import sys,pickle,json,math,resource
resource.setrlimit(resource.RLIMIT_CPU,(550,560))
from pathlib import Path
import numpy as np
ROOT=Path('/tmp/arcn_review2'); SRC=ROOT/'claims_snapshot'; sys.path.insert(0,str(SRC))
import deduce as D
from common import load_truth,c_path,c_move,run_evals
DATA=pickle.load(open(ROOT/'claims_checks/raw_recomputed.pkl','rb'))
BASE=Path('/home/agent/workspaces/mats/work/arc-noisy/tails'); RD=BASE/'runs'
rng=np.random.default_rng(12345); pick=rng.integers(0,20,(10000,20))
for sg in (.001,.003,.01):
 for L in (4,8,32):
  out,meth=DATA[sg,L]; P=np.array(out['p']);mask=~np.array(out['prov']); S=json.load(open(SRC/f'summary_s{sg:g}.json'))
  print('CELL',sg,L)
  # independently compute all better-zero median CIs from raw-estimate errors and explicit budgets.
  for valid in (False,True):
   mk=mask if valid else np.ones_like(mask); ans=[]
   for name,lst in meth.items():
    if name in ('U2r','U2ar') or name.startswith('U3r') or name in ('C-MC','C-GS','C-Kd','C-MCsim','U3','C-MCx-edge','C-MCx-gauss') or len(lst)!=20:continue
    re=np.array([d['rel'] if d['kind']=='det' else np.sqrt(d['rmse']) for d in lst]);fl=[]
    for k in range(5):
     v=re[:,:,k]; vals=[v[j][mk[j,:,k]] for j in range(20)];q=lambda x: np.quantile(x,.5,method='inverted_cdf'); med=q(np.concatenate(vals))
     arr=np.where(mk[:,:,k],v,np.nan)[pick].reshape(10000,-1);count=np.sum(~np.isnan(arr),axis=1);arr.sort(axis=1);bs=arr[np.arange(10000),(count-1)//2];hi=np.quantile(bs,.975,method='inverted_cdf')
     fl.append('Y' if med<=.9 and hi<1 else 'n')
    ans.append((name,''.join(fl)))
   print('BETTERZERO',valid,ans)
  if sg==.001:
   kp=json.load(open(RD/'kp'/f'kp_0_L{L}_s0.001_k3f.json'));print('ARCcost',kp['flops_shared'],kp['flops_heavi_each'],kp['flops_moments'])
   print('M20cost',meth['U3@m20'][0]['cost'],meth['U3@m20'][0]['costam'])
   if L==4:
    # Bias statements from per-replicate mean predictions, limited to events failing rMSE<=.25 at m20.
    E=[]
    for s in range(20):
     m=json.load(open(RD/'arcn-tdev1/mc'/f'mc_m20_{s}_s0.001.json'));T=load_truth(RD/'truth_eval'/f'truth_{s}_s0.001.json')[L]
     e=[]
     for rep in m['reps']:
      mo=np.array(rep['moments'][str(L)]); row=[]
      for i in range(8):
       kk=D.cumulants_from_moments(mo[:,i]);row.append([D.saddle(t,kk) for t in T['thr'][i]])
      e.append(row)
     E.append(np.mean(e,axis=0))
    rat=np.array(E)/P; rm=np.array([d['rmse'] for d in meth['U3@m20']]);bad=rm>.25
    for ks in ([1,2],[3,4]):
     vals=rat[:,:,ks][bad[:,:,ks]];print('U3miss',ks,len(vals),'overfrac',np.mean(vals>1.5),'quantiles',np.quantile(vals,[.1,.5,.9]).tolist())
  if L==32:
   print('GSvalidB50', [S[f'{sg}|32|C-GS|ev|valid'][str(k)]['b50'] for k in (4,6,8,10,12)])
   for sg1 in (.003,.01) if sg==.001 else ():
    P1=np.array(DATA[sg1,L][0]['p']);V1=~np.array(DATA[sg1,L][0]['prov']);dif=np.log10(P1/P)
    for k in range(5):
     bs=np.median(dif[pick,:,k].reshape(10000,-1),axis=1);print('PAIRED_NOISE',sg1,k,'median',np.median(dif[:,:,k]),'CI',np.quantile(bs,[.025,.975]).tolist())
    ix=np.unravel_index(np.argmax(dif[:,:,4]),(20,8));print('MAXNOISE',sg1,ix,'factor',10**dif[ix+(4,)],'prov_both',not(mask&V1)[ix+(4,)])
print('DONE',flush=True)
