"""Saved check of the M_{2,2} closed form against 2^24-sample truths at several sigma (review 1, finding 9)."""
import json, torch
import az, exact22
torch.set_num_threads(1)
res = {}
for s in ['0.001', '0.003', '0.01', '0.03', '0.1']:
    W, r, sg = az.load_npz(f'runs/ft_2_2_2_s{s}.npz')
    p, fl = exact22.error_rate(W, r, sg); b, fl2 = exact22.flip_bound(W, r, sg)
    m = az.model_from(W, r=r, sigma=sg); acc = az.accuracy(m, 2 ** 24, noise=True, seed=91)
    res[s] = dict(closed_form=p, flops=fl, certified_upper=p + b, mc=1 - acc['acc'], mc_se=acc['se'], z=(p - (1 - acc['acc'])) / acc['se'])
    print(s, res[s], flush=True)
json.dump(res, open('results/exact22_check.json', 'w'), indent=1)
