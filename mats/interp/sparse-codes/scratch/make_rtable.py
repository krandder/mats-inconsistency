"""Noise scales r_l (ARC pilot, seeded; part of the net definition) for this lane's ARC nets -> ext/rtable_sc.json."""
import json, os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ext'))
import anet_copy as A
torch.set_num_threads(2)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ext', 'rtable_sc.json')
tab = json.load(open(out)) if os.path.exists(out) else {}
for n in (64, 128, 256):
    for seed in (64, 65, 66, 67):
        W = A.make_weights(seed, n, 32)
        for sigma in (0.01, 0.001):
            key = f"{n}_32_{seed}_{sigma:g}"
            if key not in tab:
                tab[key] = A.pilot_rms(W, sigma, seed=seed)
                print(key, [round(v, 4) for v in tab[key][::8]], flush=True)
json.dump(tab, open(out, 'w'))
