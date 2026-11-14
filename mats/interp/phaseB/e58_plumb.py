"""E58 plumbing (disclosed in amendment 66; no registered quantity): 600-step CPU runs of train_frz.py (seed 0; unif with the c = 2
embedding, learned pool with the c = 4 embedding) in /tmp/e58_smoke: at step 0 the input embedding equals the designed one and every
other parameter equals E53 unif s0's / E37 ctrl s0's step 0; at step 600 the embedding is unchanged and the other parameters moved.
usage: python e58_plumb.py
"""
import os
import numpy as np
import torch

SM = '/tmp/e58_smoke'
for tag, base, emb in (('unif_c2', 'runs/e53/e53_unif_s0', '/tmp/e58_plumb_emb/emb_c2.npz'), ('lrn_c4', 'runs/e37/e37_ctrl_s0', '/tmp/e58_plumb_emb/emb_c4.npz')):
    a0 = torch.load(os.path.join(SM, tag, 'ckpt/step_000000.pt')); a1 = torch.load(os.path.join(SM, tag, 'model_final.pt'))
    b0 = torch.load(os.path.join(base, 'ckpt/step_000000.pt')); e = np.load(emb)
    emb0 = np.allclose(a0['inp.weight'].numpy(), e['W']) and np.allclose(a0['inp.bias'].numpy(), e['b'])
    emb1 = torch.equal(a0['inp.weight'], a1['inp.weight']) and torch.equal(a0['inp.bias'], a1['inp.bias'])
    others = all(torch.equal(a0[k], b0[k]) for k in a0 if not k.startswith('inp.'))
    moved = all(not torch.equal(a0[k], a1[k]) for k in a0 if not k.startswith('inp.') and 'ln' not in k)
    print(tag, '| step 0: embedding = design:', emb0, '| other parameters = %s step 0:' % base, others,
          '| step 600: embedding unchanged:', emb1, '| other weights moved:', moved)
