"""EXPLORATORY (not registered; fifth agent, 2026-10-30, while E48 trains): do simple features of a net's INITIAL weights go with
its final A/B type? Existing runs only (E37 ctrl, E43 newinit, E44 hybrids, E46 perturbed; ctrl format, fixed schedule, width 32,
depth 2). Any later registered test of these features would use new seeds and disclose this exploration.
Features of the step-0 net on the base episodes of E28's first 1,000 bases (ctrl format), means over heads and episodes:
  own1   block-1 attention of each source's query-question (summary) token to its own source's 4 labelled tokens (uniform 4/15);
  oth1   the same to the other sources' 8 labelled tokens;
  xq     E27's block-1 cross-source same-question attention;
  lab2   block-2 query-row attention on the 12 labelled tokens (E47's lab at step 0);
  odd1   slope of the summary's attention weight on an own labelled token on that token's signed error r (pooled regression
         over tokens and episodes, per head; reported per head and as the mean over heads);
  even1  the same on |r| - mean |r|;
  oddsym product of the two heads' odd slopes (negative = opposite signs, the two heads could cancel the odd part);
  valodd norm of the summary's block-1 attention output change per unit r of one own labelled token (finite difference of
         +-0.5 in r, others fixed), odd part; valeven its even (second-difference) part.
usage: python e49x_initfeat.py -> results/e49x_initfeat.json
"""
import glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e30_fmt as F  # noqa: E402
import pfn2 as P2  # noqa: E402
import world as W  # noqa: E402

torch.set_num_threads(1)
NB = 1000
LAB = [[5 * s + q for q in range(4)] for s in range(3)]
SUMM = [4, 9, 14]


def runs():
    out = []
    for f in sorted(glob.glob('results/e37/e37_ctrl_s*.npz')):
        tag = os.path.basename(f)[:-4]; out.append((tag, 'runs/e37/' + tag, f, 'indep', int(tag.rsplit('_s', 1)[1])))
    for f in sorted(glob.glob('results/e43/e43_newinit_s*.npz')):
        tag = os.path.basename(f)[:-4]; out.append((tag, 'runs/e43/' + tag, f, 'indep', 1000 + int(tag.rsplit('_s', 1)[1])))
    for f in sorted(glob.glob('results/e44/*.npz')):
        tag = os.path.basename(f)[:-4]; out.append((tag, 'runs/e44/' + tag, f, 'hybrid', -1))
    for f in sorted(glob.glob('results/e46/*.npz')):
        tag = os.path.basename(f)[:-4]; out.append((tag, 'runs/e46/' + tag, f, 'pert', int(tag.rsplit('_s', 1)[1])))
    return out


def feats(model, X, valid, R):
    """R (n, 3, 4): signed labelled errors."""
    with torch.no_grad():
        out = model(torch.from_numpy(X), torch.from_numpy(valid), want=True)
    a1 = out[-1][0].numpy(); a2 = out[-1][1].numpy()                       # (n, h, T, T)
    f = {}
    own = np.stack([a1[:, :, SUMM[s]][:, :, LAB[s]] for s in range(3)], 2)    # (n, h, 3, 4)
    oth = np.stack([a1[:, :, SUMM[s]][:, :, sum([LAB[u] for u in range(3) if u != s], [])].sum(-1) for s in range(3)], 2)
    f['own1'] = float(own.sum(-1).mean()); f['oth1'] = float(oth.mean())
    t = 0.0
    for s in range(3):
        for q in range(4):
            t += a1[:, :, 5 * s + q, [5 * u + q for u in range(3) if u != s]].sum(-1).mean()
    f['xq'] = t / 12
    f['lab2'] = float(a2[:, :, -1][:, :, sum(LAB, [])].sum(-1).mean())
    h = a1.shape[1]; r = R.reshape(len(R), 1, 3, 4)
    odd, even = [], []
    for j in range(h):
        y = own[:, j].reshape(-1); x = np.broadcast_to(r, (len(R), 1, 3, 4)).reshape(-1)
        xa = np.abs(x) - np.abs(x).mean()
        odd.append(float(np.cov(y, x)[0, 1] / x.var())); even.append(float(np.cov(y, xa)[0, 1] / xa.var()))
    f['odd1'] = float(np.mean(odd)); f['even1'] = float(np.mean(even)); f['oddsym'] = float(np.prod(odd))
    for j in range(h):
        f['odd1_h%d' % j] = odd[j]; f['even1_h%d' % j] = even[j]
    return f


def val_sens(model, o, y, semb, qemb):
    """Block-1 attention output at each summary token: finite differences in one own labelled token's error (q = 0)."""
    blk = model.blocks[0]
    outs = []
    for dl in (-0.5, 0.0, 0.5):
        o2 = o.copy()
        for s in range(3):
            o2[:, s, 0] = o[:, s, 0] + dl
        X, v = F.make_X(o2, y, semb, qemb, 'ctrl')
        with torch.no_grad():
            x = model.inp(torch.from_numpy(X)); B, T, d = x.shape
            qi = T - 1; vv = torch.from_numpy(v); ctx = vv.clone(); ctx[:, qi] = False
            allow = ctx[:, None, :].expand(B, T, T).clone(); allow[:, qi, qi] = True
            q, k, val = blk.qkv(blk.ln1(x)).split(blk.h * blk.dh, -1)
            sp = lambda t: t.view(B, T, blk.h, blk.dh).transpose(1, 2)
            q, k, val = sp(q), sp(k), sp(val)
            sc = (q @ k.transpose(-1, -2) / np.sqrt(blk.dh)).masked_fill(~allow[:, None], float('-inf'))
            ao = blk.o((sc.softmax(-1) @ val).transpose(1, 2).reshape(B, T, blk.h * blk.dh))
        outs.append(ao[:, SUMM].numpy())
    oddv = (outs[2] - outs[0]) / 1.0; evv = (outs[2] + outs[0] - 2 * outs[1]) / 0.25
    return float(np.linalg.norm(oddv, axis=-1).mean()), float(np.linalg.norm(evv, axis=-1).mean())


def main():
    z = np.load('results/e28_data.npz')
    o, y, semb, qemb = z['o'][:NB, 0].astype(np.float64), z['y'][:NB].astype(np.float64), z['semb'][:NB], z['qemb'][:NB]
    X, valid = F.make_X(o, y, semb, qemb, 'ctrl')
    R = o[:, :, :4] - y[:, None, :4]
    res = {}
    for tag, run, fres, kind, iseed in runs():
        ck = os.path.join(run, 'ckpt/step_000000.pt')
        if not os.path.exists(ck):
            continue
        fz = np.load(fres)
        meta = json.load(open(os.path.join(run, 'meta.json')))
        P2.fin_of = lambda ds: 2 * ds + 4
        model = P2.build(meta); model.load_state_dict(torch.load(ck, map_location='cpu')); model.eval()
        f = feats(model, X, valid, R)
        f['valodd'], f['valeven'] = val_sens(model, o, y, semb, qemb)
        f.update(kind=kind, gap=float(fz['gap']), B=bool(float(fz['rom_middle']) > 0), acquired=bool(float(fz['gap']) < 0.10), iseed=iseed)
        res[tag] = f
    json.dump(res, open("results/e49x_initfeat.json", "w"), indent=1, default=float)
    print(len(res), 'runs')


if __name__ == '__main__':
    main()
