import contextlib, io, json, os, sys, resource
from pathlib import Path
import numpy as np
resource.setrlimit(resource.RLIMIT_CPU, (590, 600))
root = Path('/home/agent/workspaces/mats/work/arc-noisy/tails')
out = Path('/tmp/arcn_review2')
sys.path.insert(0, str(root)); os.chdir(root)
src = (root/'p3b_score.py').read_text()
# Record intervals without changing the scorer's RNG or printed output.
src = src.replace("hi = q(np.array(bm), 0.975)", "hi = q(np.array(bm), 0.975)\n            cis.append((valid, name, k, med, q(np.array(bm), .025), hi))")
ns = {'__file__':str(root/'p3b_score.py'), 'cis':[]}
buf = io.StringIO()
with contextlib.redirect_stdout(buf): exec(compile(src, str(root/'p3b_score.py'), 'exec'), ns)
(out/'p3b_reproduced.txt').write_text(buf.getvalue())
print('SCORER BYTE MATCH', buf.getvalue() == (root/'results/p3b.txt').read_text())
print('MEDIAN CIs', [v for v in ns['cis'] if v[2] == 0])
np.savez(out/'p3b_arrays.npz', **{name:np.array([n['rel'][name] for n in ns['nets']]) for name in ns['nets'][0]['rel']}, prov=np.array([n['prov'] for n in ns['nets']]))
from scipy.special import logsumexp
counts = {name:np.zeros(5,int) for name in ns['nets'][0]['rel']}
secs=[]; stamps=[]; nonpos=[0,0]
for n in ns['nets']:
 s=n['seed']; t=json.loads((root/f'runs/truth_eval/truth_{s}_s0.001.json').read_text())['by_L']['32']
 th=json.loads((root/f'runs/thr_eval/thr_{s}.json').read_text())
 assert t['R']==32 and t['N']==2000
 assert np.array_equal(t['thr'],th['thr']['32'])
 logs=np.array(t['logp']); assert logs.shape==(8,5,32)
 lp=logsumexp(logs,axis=-1)-np.log(32); p=np.exp(lp)
 assert np.allclose(p,n['p'],rtol=1e-13)
 w=np.exp(logs-logs.max(-1,keepdims=True)); se=np.std(w,axis=-1,ddof=1)/w.mean(-1)/np.sqrt(32)
 half=(logsumexp(logs[:,:,:16],axis=-1)-logsumexp(logs[:,:,16:],axis=-1))/np.log(2)
 prov=(se>.25)|(abs(half)>1)
 xc=np.array(t['xcheck']['logp']); xlp=logsumexp(xc,axis=-1)-np.log(xc.shape[-1])
 xw=np.exp(xc-xc.max(-1,keepdims=True)); xse=xw.std(-1,ddof=1)/xw.mean(-1)/np.sqrt(xc.shape[-1])
 prov[:2] |= abs(xlp-lp[:2])>3*np.sqrt(se[:2]**2+xse**2)
 mc=json.loads((root/f'runs/arcn-tdev1/mc/mc_xc_{s}_s0.001.json').read_text()); N=mc['N']
 cnt=np.array(mc['reps'][0]['counts']['32'])[:,0]
 for i,c in enumerate(cnt):
  prov[i,0] |= (p[i,0]>3/N) if c==0 else abs(np.log(c/N)-lp[i,0])>3*np.sqrt(se[i,0]**2+(1-c/N)/c)
 assert np.array_equal(prov,n['prov'])
 for j,(v,dr) in enumerate([('k3fa','kp32a'),('k3f','kp')]):
  path=root/f'runs/{dr}/kp_{s}_L32_s0.001_{v}.json'; d=json.loads(path.read_text())
  assert (d['seed'],d['sigma'],d['L'],d['variant'])==(s,.001,32,v)
  assert np.array(d['heavi']).shape==(8,5) and np.array(d['moments']).shape==(4,256)
  nonpos[j]+=int((np.array(d['heavi'])<=0).sum())
  if j==0: secs.append(d['secs']); stamps.append(path.stat().st_mtime)
 for name in counts: counts[name]+=(n['rel'][name]<=.5).sum(0)
print('RAW CHECKS passed: 20 nets, metadata, thresholds, truth means and independent provisional masks')
print('SUCCESS COUNTS', {k:v.tolist() for k,v in counts.items()})
print('NONPOS',nonpos,'of 800 each')
print('SECONDS min median max sum',min(secs),np.median(secs),max(secs),sum(secs))
from datetime import datetime,timezone
print('COMPLETION MTIMES UTC',*[datetime.fromtimestamp(v,timezone.utc).isoformat() for v in (min(stamps),max(stamps))])
print('COSTS',ns['c'])
print('CPU seconds',resource.getrusage(resource.RUSAGE_SELF).ru_utime)
