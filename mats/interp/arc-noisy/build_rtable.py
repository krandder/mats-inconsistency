"""Fix r_l (noise scale) for each net and sigma: rtable_part_<tag>.json, merged into rtable.json by merge mode.
usage: build_rtable.py <tag> <seed> [<seed> ...]   |   build_rtable.py merge"""
import json, glob, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anet
torch.set_num_threads(1)
if sys.argv[1] == 'merge':
    tab = {}
    for f in sorted(glob.glob(os.path.join(anet.HERE, 'rtable_part_*.json'))):
        tab.update(json.load(open(f)))
    json.dump(tab, open(anet.RTABLE, 'w'), indent=0)
    print(len(tab), 'entries'); sys.exit()
tag = sys.argv[1]; out = {}
for seed in map(int, sys.argv[2:]):
    W = anet.make_weights(seed)
    out[f"chk_{seed}"] = anet.weights_checksum(W)
    for sigma in anet.SIGMAS:
        t = time.time()
        out[f"256_32_{seed}_{sigma:g}"] = anet.pilot_rms(W, sigma, seed=seed)
        print(seed, sigma, f"r32={out[f'256_32_{seed}_{sigma:g}'][-1]:.4f} {time.time()-t:.0f}s", flush=True)
    json.dump(out, open(os.path.join(anet.HERE, f'rtable_part_{tag}.json'), 'w'))
