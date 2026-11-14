"""ARC-counter FLOPs of kprop variants run to depth m (shape-determined, so one net suffices):
results/kprop_costs.json {variant: {m: flops}} for m in 2, 4, 6, 8, 12, 16, 24, 32."""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anet, kprop_noisy as KN

torch.set_num_threads(1)
out_p = os.path.join(anet.HERE, 'results', 'kprop_costs.json')
out = json.load(open(out_p)) if os.path.exists(out_p) else {}
net = anet.Net(100, sigma=0.001)
spec = dict(k1=(1, False, 'simple'), k2=(2, False, 'simple'), k2a=(2, False, 'augment'), k3f=(3, True, 'simple'),
            k3fa=(3, True, 'augment'))
for v, (k, fac, kind) in spec.items():
    out.setdefault(v, {})
    for m in (2, 4, 6, 8, 12, 16, 24, 32):
        if str(m) in out[v] or (v == 'k3fa' and m > 8):
            continue
        _, fl, sec = KN.kprop_noisy(net, m, k, factor=fac, kind=kind)
        out[v][str(m)] = fl
        print(v, m, f'{fl:.4e}', f'{sec:.0f}s', flush=True)
        json.dump(out, open(out_p, 'w'), indent=1)
