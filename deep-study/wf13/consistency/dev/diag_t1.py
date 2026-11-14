"""Diagnostic (no exact inference): real log loss of fixed predictors on t1, by context size. Usage: diag_t1.py t1.pt"""
import sys, torch, numpy as np
import torch.nn.functional as F
d = torch.load(sys.argv[1])
x, mask = d['x'].float(), d['mask']
lu, lv, la, lq = d['lu'].float(), d['lv'].float(), d['la'], d['lq']
sgn = (2 * x - 1)[:, None, :]
ell = F.logsigmoid(lu * sgn) * mask[:, None, :].float()
se = ell.sum(-1)
size = mask.sum(1)
def ll(p):
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(x * p.log() + (1 - x) * (1 - p).log())
preds = {'half': torch.full_like(x, 0.5), 'one pass': d['p_one']}
for k in ('p_one_M', 'p_one_L'):
    if k in d: preds[k] = d[k]
for K in (1, 4, 8):
    w = torch.softmax((se + la - lq)[:, :K].double(), 1).float()[:, :, None]
    u, v = torch.sigmoid(lu[:, :K]), torch.sigmoid(lv[:, :K])
    preds['meanu K%d' % K] = u.mean(1)
    preds['meanv K%d' % K] = v.mean(1)
    preds['exu K%d' % K] = (w * u).sum(1)
    preds['exuv K%d' % K] = (w * v).sum(1)
    preds['mix(one,meanv) K%d' % K] = 0.5 * d['p_one'] + 0.5 * v.mean(1)
print('contradicted context checks per trajectory (mean), by size stratum:', [round(float(((ell < -0.6931).float().sum(-1))[(size >= lo) & (size <= hi)].mean()), 2) for lo, hi in ((0, 9), (10, 19), (20, 29))])
for nm, p in preds.items():
    L = ll(p)
    print('%-22s' % nm, ' '.join('%.4f' % float(L[(~mask) & ((size >= lo) & (size <= hi))[:, None]].mean()) for lo, hi in ((0, 9), (10, 19), (20, 29))))
