import sys
from pathlib import Path
import torch
P=Path('/home/agent/workspaces/mats/work/arc-noisy');sys.path.insert(0,str(P));torch.set_num_threads(1)
import anet
ns={'__file__':str(P/'kprop_noisy.py')};exec(compile((P/'kprop_noisy.py').read_text().replace("'ext', 'mlp_cumulant_propagation', 'src'","'/tmp/arcn_review/ext/src'"),'kprop_noisy.py','exec'),ns)
net=anet.Net(100,sigma=.001)
for L in [4,12]:
 _,f,t=ns['kprop_noisy'](net,L,3,factor=True)
 print(L,f,'scaled',12624186712*(L/8)**2,'seconds',t,flush=True)
