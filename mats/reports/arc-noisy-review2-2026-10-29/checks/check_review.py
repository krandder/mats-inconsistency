import ast, json, math, sys, types
from pathlib import Path
import numpy as np
import torch
from scipy.stats import norm
sys.path.insert(0, str(Path('code').resolve()))
import bgs, gs, common, deduce as D

torch.set_num_threads(1)
print('SCORING')
mod=ast.parse(Path('code/analyze_tails.py').read_text())
ns=dict(np=np,os=__import__('os'),json=json,math=math,D=D,INF=np.inf,NN=8,KS=(4,6,8,10,12),A=types.SimpleNamespace(boot=10000),rng=np.random.default_rng(12345))
ns.update({k:getattr(common,k) for k in ['load_truth','c_path','c_move','run_evals','logmeanexp']})
exec(compile(ast.Module(body=[n for n in mod.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'scorer-functions','exec'),ns)
with np.errstate(invalid='ignore'):
 print('bootstrap all inf:',ns['boot_median']([np.inf]*20))
a=np.array([[1]*5+[100]*3,[10]*8]*10)
print('pooled median versus reported:',np.median(a),np.median(np.median(a,axis=1)))
try: print('all-dead rel_var',common.rel_var([-np.inf]*16))
except Exception as e: print('all-dead rel_var:',type(e).__name__,str(e))
base=Path('checks/fixtures'); (base/'truth').mkdir(parents=True,exist_ok=True); (base/'kd').mkdir(exist_ok=True)
(base/'truth/truth_102_s0.001.json').write_text(Path('dev_truth/truth_102_s0.001.json').read_text())
zero=np.zeros((16,8,5)).tolist()
(base/'kd/kd_102_s0.001.json').write_text(json.dumps(dict(N=65536,R=16,**{key:dict(setup_flops=16e8,counts={str(L):zero for L in (4,8,32)}) for key in ('kerdock','mc')})))
ns['A']=types.SimpleNamespace(boot=10000,truth=str(base/'truth'),kp=str(base/'absent'),mc=str(base/'absent'),kd=str(base/'kd'))
ns['SEEDS']=[102]
out,meth=ns['ev'](4,.001)
print('all-zero KD rMSE:',np.unique(meth['C-Kd'][0]['rmse']),'finite B50 first two levels:',np.isfinite(meth['C-Kd'][0]['b50'][:,:2]).sum())
print('missing MC cross-check flags:',out['flags'][0][:,0].tolist())
print('SADDLE')
k=[0,1,-1,-1]; boundary=math.sqrt(3)-1
t=boundary-boundary**2/2-boundary**3/6-1e-7
roots=np.roots([k[3]/6,k[2]/2,k[1],k[0]-t]); valid=[float(r.real) for r in roots if abs(r.imag)<1e-8 and r.real>0 and k[1]+k[2]*r.real+k[3]*r.real**2/2>0]
print('k',k,'t',t,'valid roots',valid,'saddle',D.saddle(t,k))
th=min(valid); K=th**2/2-th**3/6-th**4/24; w=math.sqrt(2*(th*t-K)); u=th*math.sqrt(1-th-th**2/2)
print('direct LR',norm.sf(w)+norm.pdf(w)*(1/u-1/w))
print('Gaussian saddle error:',max(abs(D.saddle(t,[0,1,0,0])-norm.sf(t)) for t in (1,3,5,7)))
print('BGS')
net=types.SimpleNamespace(device='cpu',dtype=torch.float64,L=1,n=2)
B=3000; N=128; gen=torch.Generator().manual_seed(7744)
levels=torch.tensor([0.,1.,1.,2.],dtype=torch.float64).repeat(B,1); bs=torch.full((B,3,3),.5,dtype=torch.float64)
# clone models real score functions, which return fresh tensors, not latent views.
sf=lambda z:z[:,:,0,0].clone()
def readout(z,y,g):
 band=y<=g[:,None]
 return torch.stack([band.double().mean(1),((z[:,:,0,1]>0)&band).double().mean(1),((y>2.4)&band).double().mean(1)],1)
o=bgs.frozen(net,sf,N,gen,levels,bs,n_moves=3,readout=readout)
P=o['logp'].exp().numpy(); E=o['est'].numpy()
for j,t in enumerate([0,1,1,2]): print('survival',t,'mean/se/true',P[:,j].mean(),P[:,j].std(ddof=1)/math.sqrt(B),norm.sf(t))
for j,v in enumerate([1,.5,norm.sf(2.4)]):print('FK',j,'mean/se/true',E[:,j].mean(),E[:,j].std(ddof=1)/math.sqrt(B),v)
print('repeat exact:',np.array_equal(P[:,1],P[:,2]))
levels[:]=100
o=bgs.frozen(net,sf,32,gen,levels,bs,n_moves=3,readout=readout)
print('dead logs -inf:',torch.isneginf(o['logp']).all().item(),'FK finite:',torch.isfinite(o['est']).all().item())
print('GS')
vals=[]
for r in range(500):
 draw=lambda n:torch.randn(n,2,generator=gen,dtype=torch.float64)
 q=gs.gs_run(lambda z:z[:,0].clone(),draw,128,gen,2.,levels=[0.,1.,1.],bs=[[.5]*3]*3,n_moves=3)
 vals.append(math.exp(sum(q['logfrac'])))
print('scalar survival mean/se/true',np.mean(vals),np.std(vals,ddof=1)/math.sqrt(len(vals)),norm.sf(2))
print('COSTS')
N=65536;n=256
print('KD basis products already in setup',128*2*n**3,'also charged per-point',N*n*n)
print('mode omitted hidden cost per run L32 K23',24*2000*31*(n*(2*n-1)+3*n))
print('TRUTH')
for L in (4,8,32):
 Ts=[common.load_truth(f'dev_truth/truth_{s}_s0.001.json')[L] for s in range(102,106)]
 print('L',L,'max rse',max(x['rse'].max() for x in Ts),'rv1 >=1',sum((x['rv1']>=1).sum() for x in Ts),'provisional without MC',sum(((x['rse']>.25)|(abs(x['half'])>1)).sum() for x in Ts))
