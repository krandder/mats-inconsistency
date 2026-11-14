import sys,math,json
from pathlib import Path
import numpy as np
from scipy.stats import norm,chi
sys.path.insert(0,str(Path('code').resolve()))
import deduce as D,common
k=[0.,1.,-.1,-.01];t=3.9871647423554393
rs=np.roots([k[3]/6,k[2]/2,k[1],-t]);r=min(x.real for x in rs if abs(x.imag)<1e-8 and x.real>0 and k[1]+k[2]*x.real+k[3]*x.real*x.real/2>0)
K=r*r/2+k[2]*r**3/6+k[3]*r**4/24;w=math.sqrt(2*(r*t-K));u=r*math.sqrt(k[1]+k[2]*r+k[3]*r*r/2)
print('saddle valid root repro: k,t,theta,Kpp,returned,LR',k,t,r,k[1]+k[2]*r+k[3]*r*r/2,D.saddle(t,k),norm.sf(w)+norm.pdf(w)*(1/u-1/w))
for L in (4,8):
 T=common.load_truth('dev_truth/truth_102_s0.001.json')[L];M=np.array(json.loads(Path(f'kp_sample/kp_102_L{L}_s0.001_k3f.json').read_text())['moments'])
 for b in (0,1):
  t=T['thr'][b,-1];mom=M[:,b];v=D.radial(t,mom);v4=D.radial(t,mom,nodes=641)
  mg=[mom[j]/D.chi_moment(256,j+1) for j in range(4)];kg=D.cumulants_from_moments(mg)
  r=np.linspace(chi.ppf(1e-14,256),chi.ppf(1-1e-15,256),641);f=np.array([D.saddle(t,[kg[j]*ri**(j+1) for j in range(4)]) for ri in r]);wide=np.trapezoid(chi.pdf(r,256)*f,r)
  print('radial L/output default/dense/wider',L,b,v,v4,wide,'relative dense delta',v4/v-1,'wider delta',wide/v-1)
