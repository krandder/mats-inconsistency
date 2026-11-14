"""Patching controls (reviewer finding 2): after layer l of the m-copy run, replace with the m=1 run's states
  - 'copies': every row of source s0 (original + copies) <- the m=1 state of s0's row
  - 'rest':   every other source's row <- its own m=1 state
  - 'all':    both
  - 'random': copies get a random perturbation with the same per-token norm as the 'copies' patch would apply
Pull toward the duplicated value for each, m in {2, 4}. If the copies alone carry the discount, 'rest' does nothing.

usage: PYTHONPATH=... python mech4.py RUNS_DIR OUT_JSON [n]
"""
import glob, json, os, sys
import numpy as np, torch
import mech2, world as W
from pfn import PFN, DEV


def controls(model, keys, base, m):
    d, s0 = base[0], base[4]
    n = len(s0)
    X1, v1, src1 = mech2.tokens_with_copies(base, 1, keys)
    Xm, vm, srcm = mech2.tokens_with_copies(base, m, keys)
    with torch.no_grad():
        mu1, _, hs1, _ = mech2.run(model, torch.tensor(X1, device=DEV), torch.tensor(v1, device=DEV))
        mum, _, _, _ = mech2.run(model, torch.tensor(Xm, device=DEV), torch.tensor(vm, device=DEV))
    odup = d['o'][np.arange(n), s0, W.QL]
    mu1 = mu1.cpu().numpy()
    pull = lambda mu: mech2.slope(mu - mu1, odup - mu1)
    # (episode, row in m-run, row in m=1 run) for copies and for the rest
    bi, rr = np.nonzero(np.ones_like(srcm, dtype=bool))
    is_copy = srcm[bi, rr] == s0[bi]
    src_row = np.where(is_copy, s0[bi], rr)            # rows 0..S-1 are the m=1 rows in the same order
    sets = {'copies': is_copy, 'rest': ~is_copy, 'all': np.ones_like(is_copy)}
    out = {'unpatched': pull(mum.cpu().numpy())}
    g = torch.Generator().manual_seed(0)
    for li in range(len(model.blocks) - 1):            # after the last layer nothing reads context tokens
        res = {}
        for name in ('copies', 'rest', 'all', 'random'):
            sel = sets['copies' if name == 'random' else name]
            b, r, s = bi[sel], rr[sel], src_row[sel]
            b = np.repeat(b, W.Q); q = np.tile(np.arange(W.Q), len(r))
            dst = torch.tensor(np.repeat(r, W.Q) * W.Q + q); srt = torch.tensor(np.repeat(s, W.Q) * W.Q + q)
            bt = torch.tensor(b)

            def patch(l, x, li=li, name=name):
                if l != li:
                    return x
                x = x.clone()
                new = hs1[li + 1][bt, srt]
                if name == 'random':
                    delta = new - x[bt, dst]
                    rnd = torch.randn(delta.shape, generator=g)
                    rnd = rnd / rnd.norm(dim=-1, keepdim=True) * delta.norm(dim=-1, keepdim=True)
                    new = x[bt, dst] + rnd
                x[bt, dst] = new
                return x
            with torch.no_grad():
                mup, _, _, _ = mech2.run(model, torch.tensor(Xm, device=DEV), torch.tensor(vm, device=DEV), patch)
            res[name] = pull(mup.cpu().numpy())
        out[li] = res
    return out


def main(runs, outp, n=600):
    base = mech2.base_episodes(n, np.random.default_rng(mech2.SWEEP_SEED))
    res = []
    for f in sorted(glob.glob(os.path.join(runs, 'L*_k*_s*/meta.json'))):
        meta = json.load(open(f))
        if meta['depth'] == 1:
            continue
        torch.manual_seed(meta['seed'])
        model = PFN(meta['fin'], meta['depth']).to(DEV)
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(f), 'model.pt'), map_location=DEV))
        model.eval()
        r = {m: controls(model, meta['keys'], base, m) for m in (2, 4)}
        res.append({'depth': meta['depth'], 'keys': meta['keys'], 'seed': meta['seed'], 'ctl': r})
        print(meta['depth'], meta['keys'], meta['seed'], json.dumps({m: {k: (round(v, 3) if not isinstance(v, dict) else
              {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in r[m].items() if k in ('unpatched', 0)} for m in r}), flush=True)
    json.dump(res, open(outp, 'w'))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 600)
