from pathlib import Path
src=Path('/tmp/arcn_review2/claims3_checks/check_independent.py').read_text()
exec(src.split('for L in (8,4):')[0])
for L in (8,4):
 dat=np.load(O/f'independent_L{L}.npz');rat=dat['E'][:,0]/dat['P'];ids=list(np.argsort(np.minimum(abs(rat-.5),abs(rat-1.5)).ravel())[:8])
 ids+=list(np.argsort(-rat[:,:,4].ravel())[:4]*5+4)
 dif=[];flips=[]
 for ix in ids:
  si,i,j=np.unravel_index(ix,rat.shape);s=si+44
  t=json.load(open(S/f'runs/fresh3/truth/truth_{s}_s0.001.json'))['by_L'][str(L)]['thr'][i][j]
  mom=np.array(json.load(open(S/f'runs/kp_fresh3/kp_{s}_L{L}_s0.001_k3fa.json'))['moments'])[:,i]
  new=radial(t,mom,nodes=641,lo=1e-12,hi=1-1e-15);old=dat['E'][si,0,i,j]
  dif.append(abs(new/old-1));flips.append((abs(new/dat['P'][si,i,j]-1)<=.5)!=(abs(old/dat['P'][si,i,j]-1)<=.5))
 print('QUADRATURE',L,len(ids),'max relative change',max(dif),'classification flips',sum(flips))
import hashlib,datetime
man=json.load(open(S.parent/'manifest.json'));changed=[]
for f,entry in man['files'].items():
 if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=entry['sha256']:changed.append(f)
print('SOURCE_CHANGES_SINCE_SNAPSHOT',changed)
print('CHECKED_AT',datetime.datetime.now(datetime.timezone.utc).isoformat())
