import sys,json,math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('code').resolve()))
from common import load_truth,c_path,c_move,run_evals,gs_b50
root=Path('/home/agent/workspaces/mats/work/arc-noisy/tails/runs/arcn-tdev1')
wins=0; winscost=0; N=2000;L=32;n=256
for s in range(102,106):
 mo=json.loads((root/f'mode/mode_{s}_L32_s0.001.json').read_text()); T=load_truth(f'dev_truth/truth_{s}_s0.001.json')[32]
 setup=mo['M0']*c_path(L)+sum(ru['pilot_n_eval']*c_move(L) for ru in mo['runs'].values())
 for b in range(8):
  name='up' if mo['c'][b]>0 else 'down';ru=mo['runs'][name];K=len(ru['levels']);E=np.array(ru['est'])[:,b,2];p=np.exp(T['lp'][b,2]);r=np.mean((E/p-1)**2)
  per=run_evals(N,K+1)*c_move(L)+(K+1)*N*(n*(2*n-1)+n*5)
  b50=(setup+2*per*r/.25)/(5*n) if r<1 else np.inf
  hidden=(K+1)*N*(L-1)*(n*(2*n-1)+3*n)
  corrected=(setup+2*(per+hidden)*r/.25)/(5*n) if r<1 else np.inf
  gs=gs_b50(L,T['nlev'][b,2],T['rv1'][b,2]);wins+=b50<gs;winscost+=corrected<gs
  if .999<=r<1:print('newly admitted',s,b,'rMSE',r,'est/p',E/p,'U1 B50',b50,'C-GS B50',gs,'U1 incl hidden recomputation',corrected)
print('exact strict wins',wins,'/32','after adding hidden recomputation',winscost,'/32')
