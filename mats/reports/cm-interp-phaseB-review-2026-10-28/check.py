import sys,json,math,numpy as np
from scipy.stats import norm
sys.path.insert(0,'/home/agent/workspaces/mats/work/cm-interp-B')
import nnoise as N, world as W
from pfn_diag import SimMean
from pfn_rare import prior
from dec import interp_p
base='/home/agent/workspaces/mats/work/cm-interp-B'
out={}
rng=np.random.default_rng(5); d=W.sample_np(65536,rng); _,_,mom,_=W.references(d); mb=mom['bayes'][0]
qs=np.r_[np.linspace(1/16,15/16,15),1-2.**-np.arange(5,14)]
lab=np.searchsorted(np.quantile(mb,np.sort(qs)),mb); counts=np.bincount(lab); k=np.maximum(256,counts)
out['cell_weights']={'counts':counts.tolist(),'original_last':float(counts[-1]/counts.sum()),'refit_last':float(k[-1]/k.sum()),'inflation_bits':float(np.log2(k[-1]/k.sum()/(counts[-1]/counts.sum()))),'components':len(counts)}
out['interp_beyond_support']=interp_p([(0,.1),(1,.01)],[1.1]).tolist()
D=json.load(open(base+'/runs/L2_s0.05/diag.json')); out['diag_keys']=list(D['diag']);out['truth_last_log2_reps']=np.log2(np.array(D['p_reps'])[:,-1]).tolist()
model,meta=N.load_run(base+'/runs/L2_s0.1'); sim=SimMean(model,meta)
D=json.load(open(base+'/runs/L2_s0.1/diag.json')); i=int(np.argmin(np.abs(np.log2(D['p_true'])+19)));t=D['t'][i];p=D['p_true'][i]
out['is_target']={'t':t,'p':p,'dim':sim.dim}
# Learn a single proposal direction from black-box score correlations, no Bayes formula.
g=np.random.default_rng(12345);u,v=prior(sim,4096,g);s=sim.score_batched(u,v)
corr=(u.T@(s-s.mean()))/len(s);j=int(np.argmax(np.abs(corr)));m=t*np.sign(corr[j]);out['blackbox_selected_coordinate']=j
out['is_runs']=[]
for method,jj,shift in [('query',W.Q-1,t),('blackbox_sparse',j,m)]:
 for r in range(3):
  g=np.random.default_rng(9000+r);u,v=prior(sim,32768,g);u[:,jj]+=shift;s=sim.score_batched(u,v);vals=np.exp(-shift*u[:,jj].astype(float)+shift**2/2)*(s>t)
  row={'method':method,'rep':r,'estimate':float(vals.mean()),'ratio':float(vals.mean()/p),'hits':int((s>t).sum()),'se':float(vals.std()/np.sqrt(len(vals)))};out['is_runs'].append(row);print(row,flush=True)
json.dump(out,open('/tmp/pbrev/check.json','w'),indent=2)
print(json.dumps(out,indent=2))
