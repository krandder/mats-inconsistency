"""Amendment 22's determinism check for the checkpointed replays: validation curves to 1e-9 and model files tensor by tensor.
usage: python e16_determinism.py -> results/e16_determinism.txt
"""
import json
import torch

lines = []
for s in (5, 6):
    o = json.load(open(f'runs/base_mix4_s{s}/meta.json')); r = json.load(open(f'runs/dyn_base_s{s}/meta.json'))
    co, cr = dict((int(a), b) for a, b in o['curve']), dict((int(a), b) for a, b in r['curve'])
    dmax = max(abs(co[k] - cr[k]) for k in co)
    same_keys = sorted(co) == sorted(cr)
    res = [f's{s}: curve steps equal {same_keys} ({len(co)} points), max |val difference| {dmax:.3e}; '
           f"best step original {o['best_step']} replay {r['best_step']}; phase-1 steps {o['phase1_steps']} / {r['phase1_steps']}"]
    for f in ('model.pt', 'model_phase1.pt'):
        a = torch.load(f'runs/base_mix4_s{s}/{f}', map_location='cpu'); b = torch.load(f'runs/dyn_base_s{s}/{f}', map_location='cpu')
        eq = all(torch.equal(a[k], b[k]) for k in a) and set(a) == set(b)
        mx = max(float((a[k] - b[k]).abs().max()) for k in a)
        res.append(f'  {f}: bitwise equal {eq}; max |difference| {mx:.3e}')
    ck = torch.load(f"runs/dyn_base_s{s}/ckpt/step_{o['best_step']:06d}.pt", map_location='cpu'); a = torch.load(f'runs/base_mix4_s{s}/model.pt', map_location='cpu')
    res.append(f"  checkpoint at the best step equals the original model.pt: {all(torch.equal(a[k], ck[k]) for k in a)}")
    lines += res
open('results/e16_determinism.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
