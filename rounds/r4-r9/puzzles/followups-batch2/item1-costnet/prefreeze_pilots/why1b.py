# Exploratory (not pre-registered): 1b failures vs the learned wall/open cost ratio and the solution path length.
import sys, json, numpy as np, torch
sys.path.insert(0, '.')
import bansal as b
out = {}
for seed in (200, 201):
    ck = torch.load(f'../vmout/runs1b/bansal_s{seed}.pt', weights_only=False)
    m = b.BCostNet(32); m.load_state_dict(ck['state']); m.eval()
    ev = np.load(f'../vmout/eval1b/test/bansal_s{seed}.npz')
    for st in ('e2h_maze_33', 'e2h_maze_59'):
        x, y = b.load_raw(st); xb = torch.from_numpy(b.blocks(x).copy())
        L = (y[:, 3:-3:2, 3:-3:2] > 0.5).reshape(len(y), -1).sum(1)
        with torch.no_grad():
            c = m(xb[:200]).numpy()
        wall = xb[:200].max(1).values.numpy() < 0.5
        r = np.array([np.median(c[i][wall[i]]) / np.median(c[i][~wall[i]]) for i in range(200)])
        ex = ev[f'{st}/exact']
        bins = [0, 100, 200, 300, 500, 1000, 10000]
        tab = {f'{lo}-{hi}': [int(((L >= lo) & (L < hi)).sum()), float(ex[(L >= lo) & (L < hi)].mean()) if ((L >= lo) & (L < hi)).any() else None] for lo, hi in zip(bins[:-1], bins[1:])}
        out[f's{seed}/{st}'] = {'ratio_median': float(np.median(r)), 'ratio_p10_p90': [float(np.percentile(r, 10)), float(np.percentile(r, 90))],
                                'exact': float(ex.mean()), 'exact_by_path_len[n, acc]': tab}
        print(seed, st, json.dumps(out[f's{seed}/{st}']), flush=True)
json.dump(out, open('../why1b_exploratory.json', 'w'), indent=1)
