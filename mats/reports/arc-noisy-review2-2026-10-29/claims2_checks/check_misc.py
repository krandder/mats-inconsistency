import resource
resource.setrlimit(resource.RLIMIT_CPU,(550,560))
import sys,json,pickle,math
from pathlib import Path
import numpy as np
R=Path('/tmp/arcn_review2');sys.path.insert(0,str(R/'claims2_snapshot'))
from common import load_truth,c_move,c_path,run_evals,gs_b50
import deduce as D
DATA=pickle.load(open(R/'claims2_checks/raw_recomputed.pkl','rb'));B=Path('/home/agent/workspaces/mats/work/arc-noisy/tails/runs');pick=np.random.default_rng(12345).integers(0,20,(10000,20))
for L in (4,8):
 out,methods=DATA[.001,L]
 for name,base in [('U2','C-K3f'),('U2a','C-K3fa')]:
  eo=np.array([d['rel'] for d in methods[name]]);ec=np.array([d['rel'] for d in methods[base]]);diff=((eo<=.5).mean(1)-(ec<=.5).mean(1))
  print('POSTHOC',L,name,'diff',diff.mean(0).tolist(),'CI',np.quantile(diff[pick].mean(1),[.025,.975],axis=0,method='inverted_cdf').tolist())
# U1 kill with corrected evaluator costs (hidden recomputation, fits, both directions).
counts=np.zeros(3,int)
for s in (102,103,104,105):
 T=load_truth(B/'arcn-tdev1/truth'/f'truth_{s}_s0.001.json')[32];mo=json.load(open(B/'arcn-tdev1/mode'/f'mode_{s}_L32_s0.001.json'));cm=c_move(32);n=256
 setup=mo['M0']*c_path(32)+2*mo['M0']*n*n+10*n**3+sum(ru['pilot_n_eval']*cm for ru in mo['runs'].values());per=0
 for ru in mo['runs'].values():
  K=len(ru['levels']);per+=run_evals(2000,K+1)*cm+(K+1)*2000*((32-1)*(n*(2*n-1)+3*n))+(K+1)*2000*(n*(2*n-1)+n*5)
 for i in range(8):
  E=np.array(mo['runs']['up' if mo['c'][i]>0 else 'down']['est'])[:,i,2];p=np.exp(T['lp'][i,2]);rm=np.mean((E/p-1)**2);cost=(setup+per*rm/.25)/(5*256);gs=gs_b50(32,T['nlev'][i,2],T['rv1'][i,2])
  counts += np.array([rm<=1,rm<1,rm<=.81])&(cost<gs)
print('U1 corrected costs kill counts',counts.tolist())
# Bias wording: means across four replicates, per k.
P=np.array(DATA[.001,4][0]['p']);Est=[]
for s in range(20):
 T=load_truth(B/'truth_eval'/f'truth_{s}_s0.001.json')[4];M=json.load(open(B/'arcn-tdev1/mc'/f'mc_m20_{s}_s0.001.json'));Est.append(np.array([[[D.saddle(t,D.cumulants_from_moments(np.array(rep['moments']['4'])[:,i])) for t in T['thr'][i]] for i in range(8)] for rep in M['reps']]))
E=np.array(Est);err=np.mean((E/P[:,None]-1)**2,axis=1);rat=E.mean(1)/P
for k in (1,2,3,4):
 v=rat[:,:,k][err[:,:,k]>.25];print('MISSES',k,len(v),np.quantile(v,[.1,.5,.9]).tolist())
files=list((B/'kp').glob('kp_*.json'));evalfiles=[f for f in files if int(f.name.split('_')[1])<20]
print('KPROP eval files',len(evalfiles))
print('DONE')
