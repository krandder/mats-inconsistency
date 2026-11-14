import sys, json, glob, math, hashlib, time
sys.path.insert(0,'/home/agent/workspaces/mats/work/cm-interp-R')
import numpy as np, torch
import rnet as R, rare_gs as GS, rare as RA
from pathlib import Path
base=Path('/home/agent/workspaces/mats/work/cm-interp-R'); l2=math.log(2)
print('checked',time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),flush=True)
for f in sorted(base.glob('results/rare_*.json')):
 r=json.load(open(f)); z=r['levels'][-1]; tr=z.get('truth_hi',z['truth']); ch=z['chain']
 print(f.name,'last k',z['k'],'truth bits',round(tr['logp']/l2,2),'SE',round(tr['rel_se'],3),'bulk gap',round((z['crosscheck']['bulk_logp']-tr['logp'])/l2,2),'chain gap',round((ch['log_KqLm1']-tr['logp'])/l2,2),'chain SD',ch.get('rel_sd_KqLm1'),flush=True)
es=[json.load(open(f)) for f in base.glob('results/eout_d256*.json')]
print('eout default advantages',min(e['k2']['mc_nmse_at_equal_flops']/e['k2']['nmse'] for e in es),max(e['k2']['mc_nmse_at_equal_flops']/e['k2']['nmse'] for e in es),flush=True)
s=[json.load(open(f)) for f in base.glob('results/surr*.json')]
print('DV ranges',[(k,min(r['dv'][k] for r in s),max(r['dv'][k] for r in s)) for k in ['width_ratio_y_median','width_ratio_F2_median']],flush=True)
gains=[r['flops_sample']/(r['flops_sample']+r['L']*(2*r['d']**2+2*r['d'])+16*r['d'])/r['eout']['rho_stat'] for r in s if r['act']=='tanh' and r['g'] in [.25,.75]]
print('CV asymptotic gains',min(gains),max(gains),flush=True)
for g in [.75,3]:
 r=json.load(open(base/f'results/score_d32_L16_tanh{g:g}_s0.3_n0.json'))
 v=[x['rel_KLc'] for x in r['layers']['G']]; print('depth',g,min(v),max(v),flush=True)
r=json.load(open(base/'results/rare_d8_L4_tanh0.75_s0.3_n0.json')); net=R.RNet(8,4)
for k in [10,60]:
 z=next(z for z in r['levels'] if z['k']==k); t=z['t']; lv=[v for v in r['levels_curve'] if v<t-1e-12]
 for moves in [5,20]:
  gen=torch.Generator().manual_seed(981+k+moves)
  v=np.array([GS.splitting(net,t,2048,gen,levels=lv,n_moves=moves)[0] for _ in range(8)])
  w=np.exp(v-v.max()); lp=v.max()+np.log(w.mean())
  print('RERUN',k,moves,'log2p',lp/l2,'SE',w.std(ddof=1)/w.mean()/np.sqrt(8),'published',z['truth']['logp']/l2,flush=True)
