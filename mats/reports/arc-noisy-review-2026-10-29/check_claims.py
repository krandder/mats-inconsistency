import sys, os, json, math, re
from pathlib import Path
import numpy as np
ROOT=Path('/home/agent/workspaces/mats/work/arc-noisy')
sys.path.insert(0,str(ROOT))
import analyze as A, combine as H, truthlib, anet
rng=np.random.default_rng(20261029)
ix=rng.integers(0,24,(10000,24))
def stats(v):
 v=np.asarray(v); return [float(np.median(v)),*np.quantile(np.median(v[ix],axis=1),[.025,.975]).tolist(),float(v.mean()),*np.quantile(v[ix].mean(axis=1),[.025,.975]).tolist()]
allrows=[]
for L in (8,32):
 for sg in A.SIGMAS:
  nets=A.collect(L,sg)
  ds=sg if (ROOT/'samp_v2'/f'samp_100_s{sg:g}_L{L}.npz').exists() else .001
  b2={}; costs={}; VN=[]
  for v in H.VARIANTS:
   es=[]
   for s in H.DEV:
    k=H.kprop(s,ds,L,v)
    if k is not None:
     es.append(truthlib.nmse(k[0],A.truth(s,ds),L)); costs[v]=k[1]
   if len(es)>=3:b2[v]=float(np.median(es))
  for s in H.DEV:
   kd=H.kerd(s,ds,L,'samp_v2')
   if kd is not None:
    N,C,S,T=kd
    VN.append([np.mean([truthlib.nmse(S[r,j]/N[j],T,L) for r in range(len(S))])*N[j] for j in range(len(N))])
  V=np.median(VN,axis=0)
  vals=[]; spends=[]
  for s in range(24):
   N,C,S,T=H.kerd(s,sg,L,'samp_v2'); rr=[]; sp=[]
   for B in A.BUDGETS:
    jk=np.searchsorted(C,B,side='right')-1
    best=(V[jk]/N[jk],None,jk)
    for v in b2:
     j=np.searchsorted(C,B-costs[v],side='right')-1
     if j>=0:
      pred=b2[v]*(V[j]/N[j])/(b2[v]+V[j]/N[j])
      if pred<best[0]:best=(pred,v,j)
    _,v,j=best
    w=(V[j]/N[j])/(b2[v]+V[j]/N[j]) if v else 0
    mu=H.kprop(s,sg,L,v)[0] if v else 0
    rr.append(np.mean([truthlib.nmse(w*mu+(1-w)*sr[j]/N[j],T,L) for sr in S]))
    sp.append([C[j]+(costs[v] if v else 0),C[jk]])
   vals.append(rr); spends.append(sp)
  vals=np.array(vals)
  for bi,B in enumerate(A.BUDGETS):
   bs=A.best_of(nets,['mc','mcrb','sobol','mub'],bi)[0]
   sf=A.best_of(nets,[f'kprop-{v}' for v in A.KPROP],bi)[0]
   base=np.array([nets[s][bs][bi] for s in range(24)])
   sfv=np.array([nets[s][sf][bi] for s in range(24)])
   st=stats(base/vals[:,bi]); bt=stats(np.minimum(base,sfv)/vals[:,bi])
   row=dict(L=L,sigma=sg,B=float(B),best_sampler=bs,best_sf=sf,H2_median=float(np.median(vals[:,bi])),ratio=st,better_ratio=bt,win=st[1]>1 and st[4]>1,cost_fraction=np.asarray(spends)[0,bi].tolist())
   allrows.append(row)
   if L==32 or sg==.001: print(json.dumps(row),flush=True)
Path('/tmp/arcn_review/h2_independent.json').write_text(json.dumps(allrows,indent=2))
print('MULTILEVEL')
for m,G in [(8,1.15),(16,2.1),(24,5.2)]:
 print(m,G,(math.sqrt(32/G)+math.sqrt(m*(1-1/G)))**2,'perfect coarse',32/G)
print('SURVIVAL')
lines=(ROOT/'logs/survival_100.log').read_text().splitlines()
vals=[np.array([float(x) for x in re.findall(r'energy kept ([\d.e+-]+)',l)]) for l in lines]
print('change .03 versus .001:', vals[2]/vals[0]-1, 'reduction .1:', vals[0]/vals[3])
print('CHECKPOINT / COST VERIFICATION')
count=0; methods={}; shape={}; mismatches=[]
for p in (ROOT/'samp_v2').glob('*.npz'):
 d=np.load(p);L=int(d['L']);sg=float(d['sigma']);net=anet.Net(int(d['seed']),sigma=sg).truncate(L)
 for key in [x[2:] for x in d.files if x.startswith('N_')]:
  N=d['N_'+key]; C=d['costs_'+key]; c=net.flops_path(L)
  if key in ('mub','mubanti'):c=c-256*16-anet.mm_flops(1,256,256)+256**2+256+100
  if key=='mubanti':c*=2
  if key=='sobol':c=c-256*16+256*100
  expect=N*c+(np.ceil(N/66048)*(6*256**3+16*256**2) if key in ('mub','mubanti') else 0)
  if not np.array_equal(C,expect):mismatches.append(str(p))
  methods[key]=methods.get(key,0)+1
  shape[str(d[f'sum_{key}_{L}'].shape[:2])]=shape.get(str(d[f'sum_{key}_{L}'].shape[:2]),0)+1
 count+=1
print('files',count,'methods',methods,'shapes',shape,'mismatches',mismatches)
