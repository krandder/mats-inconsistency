"""One valuation harness for every arm.

  python eval.py design --split test|tune|pilot [--n_per N]    -> results/eval/design_<split>.json (seeds + exact CMI)
  python eval.py run --split S --arm A [--run RUN] [--max_n N]  -> results/eval/<split>/<arm>.npz

Every arm sees rows 1..H+k only (information rule). Per episode we store the prequential value V^preq_k at KS and,
where the arm supports it, the settlement gain on rows H+k+1..H+k+16 given rows 1..H+k at SK.
Arms: C1, C4 (forward, reverse and tuned combination), C5, S1, S2, L (L-all and L-train), FR:<run>, FRD:<run> (FR-direct
prediction at each k), CONS (S2 consumer gain on a fresh 256-row block), E1 (TabPFN v2; see e1.py; k <= 16).
params: arm hyperparameters, plus 'types' (list) and 'per' (episodes per type x stratum) to subsample, 'tag' (file suffix).
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import gen
from gen import H, ALL_TYPES, ID_TYPES, HP_TYPES, HN_TYPES, LEAK_TYPES, make_structure, sample_rows, ybits, rng_for
from oracle import cmi_batch, best_device

KS = (4, 8, 16, 32, 64, 128)
SK = (4, 8, 16, 32, 64)
T_ROWS = H + 128
W_TEST = 6
OUT = os.environ.get('P16_OUT', 'results')
POS_RETRY, NEG_RETRY = gen.POS_RETRY, gen.NEG_RETRY
SPLIT_TYPES = {'test': ALL_TYPES, 'tune': ID_TYPES, 'pilot': HP_TYPES + HN_TYPES}


def design(split, n_per, dev):
    """Deterministic list of episodes with the retry rules applied on the GPU oracle (batched per try)."""
    items = gen.test_design(n_per, types=SPLIT_TYPES[split], ns=split, w=W_TEST)
    out = []
    t0 = time.time()
    pend = [(idx, t, m, 1) for idx, t, m in items]
    while pend:
        nxt = []
        by_m = {}
        for it in pend:
            by_m.setdefault(it[2], []).append(it)
        for m, its in by_m.items():
            cs = min(256, max(4, 2 ** (17 - m)))                       # fp64 joint is (cs, 2^(m+1), 2^w): bound memory
            for c0 in range(0, len(its), cs):
                chunk = its[c0:c0 + cs]
                eps = [make_structure(rng_for(split, idx, tr), split, idx, t, m, W_TEST) for idx, t, m, tr in chunk]
                cm = cmi_batch(eps, dev, torch.float64).cpu().numpy()
                for (idx, t, m_, tr), c in zip(chunk, cm):
                    c = float(c)
                    redo = (t in POS_RETRY and c < .05) or (t in NEG_RETRY and c >= 1e-12)
                    if redo and tr < 30:
                        nxt.append((idx, t, m_, tr + 1))
                    else:
                        out.append(dict(idx=idx, type=t, m=m_, tries=tr, cmi=c, retry_failed=bool(redo)))
        pend = nxt
    out.sort(key=lambda d: d['idx'])
    os.makedirs(f'{OUT}/eval', exist_ok=True)
    json.dump(dict(split=split, n_per=n_per, w=W_TEST, items=out, wall_s=time.time() - t0), open(f'{OUT}/eval/design_{split}.json', 'w'))
    print(split, len(out), 'episodes', f'{time.time()-t0:.0f}s', 'retry_failed', sum(d['retry_failed'] for d in out))


def load_design(split):
    return json.load(open(f'{OUT}/eval/design_{split}.json'))['items']


def episode(split, d):
    ep = make_structure(rng_for(split, d['idx'], d['tries']), split, d['idx'], d['type'], d['m'], W_TEST)
    ep.cmi = d['cmi']
    return ep


def rows_of(split, ep, idx, T=T_ROWS):
    return sample_rows(ep, T, rng_for(split, idx, 1000))


def batches(split, items, E):
    by_m = {}
    for d in items:
        by_m.setdefault(d['m'], []).append(d)
    for m in sorted(by_m):
        its = by_m[m]
        for c0 in range(0, len(its), E):
            chunk = its[c0:c0 + E]
            eps = [episode(split, d) for d in chunk]
            rows = [rows_of(split, ep, d['idx']) for ep, d in zip(eps, chunk)]
            yield chunk, eps, rows


def tens(rows, dev):
    V = torch.as_tensor(np.stack([r[2] for r in rows]), dtype=torch.long, device=dev)
    Y = torch.as_tensor(np.stack([ybits(r[1], W_TEST) for r in rows]), dtype=torch.long, device=dev)
    return V, Y


def at_ks(g):
    """g (E, K) per-row gain -> (E, len(KS)) cumulative at KS."""
    c = g.cumsum(1)
    return torch.stack([c[:, k - 1] for k in KS], 1).cpu().numpy()


def run_arm(split, arm, run=None, max_n=0, dev='cuda', params=None):
    items = load_design(split)
    if max_n:
        rng = np.random.default_rng(0)
        sel = []
        by = {}
        for d in items:
            by.setdefault((d['type'], d['m']), []).append(d)
        per = max(1, max_n // len(by))
        for k in sorted(by):
            sel += by[k][:per]
        items = sel
    params = params or {}
    if params.get('types') or params.get('per'):
        by = {}
        for d in items:
            if not params.get('types') or d['type'] in params['types']:
                by.setdefault((d['type'], d['m']), []).append(d)
        items = [d for k in sorted(by) for d in by[k][:params.get('per', 10 ** 9)]]
    res = {'idx': [], 'preq': [], 'settle': [], 'extra': {}}
    def add_extra(name, arr):
        res['extra'].setdefault(name, []).append(arr)
    t0 = time.time()
    model = None
    if arm.startswith('FR'):
        from reader import FR
        ck = torch.load(f'{OUT}/runs/{run}/final.pt', map_location='cpu', weights_only=False)
        mode = {'FR': 'joint', 'FR-small': 'joint', 'FR-fact': 'fact', 'FR-direct': 'direct'}[ck['meta']['arm']]
        model = FR(mode=mode).to(dev).eval()
        model.load_state_dict(ck['model'])
    E = params.get('E', {'S1': 64, 'S2': 64, 'L': 1, 'E1': 1, 'C4': 16, 'C5': 16, 'CONS': 128}.get(arm, 32))
    for chunk, eps, rows in batches(split, items, E):
        V, Y = tens(rows, dev)
        Vn = V.clone(); Vn[:, :, -1] = -1
        m = V.shape[2] - 1
        res['idx'] += [d['idx'] for d in chunk]
        with torch.no_grad():
            if arm == 'C1':
                from baselines import c1_counts
                res['preq'].append(at_ks(c1_counts(V, Y, H)))
            elif arm in ('C4', 'C5'):
                from baselines import bma_forward, bma_reverse, c5_betting
                al, rho = params.get('alpha', 2.), params.get('rho', .5)
                f = bma_forward(V, Y, alpha=al, rho=rho, flip=True)
                gf = (f['with'] - f['without'])[:, H:H + 128]
                rv, phat = bma_reverse(V[:, :H + 128], Y[:, :H + 128], H)
                if arm == 'C4':
                    lam = params.get('lam', 0.0)
                    res['preq'].append(at_ks(gf + lam * rv))
                    add_extra('fwd', at_ks(gf)); add_extra('rev', at_ks(rv))
                    st = []
                    for k in SK:
                        s = bma_forward(V[:, :H + k + 16], Y[:, :H + k + 16], alpha=al, rho=rho, flip=False, settle=(H + k, 16))
                        st.append((s['set_with'] - s['set_without']).sum(1).cpu().numpy())
                    res['settle'].append(np.stack(st, 1))
                else:
                    g = torch.Generator().manual_seed(int(chunk[0]['idx']))
                    lw = c5_betting(f['with'][:, H:H + 128], f['with_flip'][:, H:H + 128], V[:, H:H + 128, m], phat, g,
                                    n_twins=8, scale=params.get('scale', 1.0))
                    res['preq'].append(torch.stack([lw[:, k - 1] for k in KS], 1).cpu().numpy())
            elif arm == 'L':
                from larms import l_scores
                ep, r = eps[0], rows[0]
                sc = l_scores(ep, r, 128, dev)
                res['preq'].append(np.stack([sc['L-all'].numpy()[[k - 1 for k in KS]]])[None][0])
                add_extra('ltrain', np.stack([sc['L-train'].numpy()[[k - 1 for k in KS]]]))
            elif arm.startswith('FR') and model.mode != 'direct':
                g = torch.Generator().manual_seed(int(chunk[0]['idx']) + 17)
                order = torch.randperm(W_TEST, generator=g).to(dev)[None].expand(len(chunk), -1).contiguous()
                lw = model.row_logp(V, Y, order); lo = model.row_logp(Vn, Y, order)
                res['preq'].append(at_ks((lw - lo)[:, H:H + 128].double()))
                add_extra('pre_birth_maxdiff', (lw[:, :H] - lo[:, :H]).abs().amax(1).cpu().numpy())
                st = []
                for k in SK:
                    T2 = H + k + 16
                    a = model.row_logp(V[:, :T2], Y[:, :T2], order, ctx=H + k)
                    b = model.row_logp(Vn[:, :T2], Y[:, :T2], order, ctx=H + k)
                    st.append((a - b)[:, H + k:].sum(1).cpu().numpy())
                res['settle'].append(np.stack(st, 1))
            elif arm.startswith('FR'):
                preds, logits = [], []
                for k in KS:
                    p, l = model.direct(V[:, :H + k], Y[:, :H + k])
                    preds.append(p.float().cpu().numpy()); logits.append(l.float().cpu().numpy())
                res['preq'].append(np.stack(preds, 1))
                add_extra('logit', np.stack(logits, 1))
        if arm == 'E1':
            import e1
            kk = params.get('k', 16)
            g = e1.e1_gain(V[0].cpu().numpy(), Y[0].cpu().numpy(), H, kk)
            c = np.cumsum(g)
            res['preq'].append(np.array([[c[k - 1] if k <= kk and k % 4 == 0 else np.nan for k in KS]]))
            add_extra('calls_s', np.array([e1.CALLS['s']])); add_extra('calls_n', np.array([e1.CALLS['n']]))
        if arm in ('S1', 'S2'):
            from baselines import sgd_reader
            g = sgd_reader(V, Y, H, 128, mode=arm, width=params.get('width', 64), lr=params.get('lr', 3e-3),
                           wd=params.get('wd', 1e-4), E_steps=params.get('E_steps', 20), pre_steps=params.get('pre_steps', 300),
                           seed=int(chunk[0]['idx']))
            res['preq'].append(at_ks(g))
        elif arm == 'CONS':
            from baselines import consumer_gain
            Vf, Yf = [], []
            for ep, d in zip(eps, chunk):
                _, yf, vf = sample_rows(ep, 256, rng_for(split, d['idx'], 2000), birth=0)
                Vf.append(vf); Yf.append(ybits(yf, W_TEST))
            Vf = torch.as_tensor(np.stack(Vf), dtype=torch.long, device=dev)
            Yf = torch.as_tensor(np.stack(Yf), dtype=torch.long, device=dev)
            cg = consumer_gain(V, Y, Vf, Yf, seed=int(chunk[0]['idx']))
            res['preq'].append(cg.cpu().numpy()[:, None])
        n = len(res['idx'])
        if n % 512 < len(chunk):
            print(f'{split} {arm} {run or ""} {n}/{len(items)} {time.time()-t0:.0f}s', flush=True)
    out = dict(idx=np.array(res['idx']), preq=np.concatenate(res['preq']), wall_s=time.time() - t0,
               params=json.dumps(params))
    if res['settle']:
        out['settle'] = np.concatenate(res['settle'])
    for k, v in res['extra'].items():
        out['x_' + k] = np.concatenate(v)
    name = arm if run is None else f'{arm}:{run}'
    if params.get('tag'):
        name += '@' + params['tag']
    os.makedirs(f'{OUT}/eval/{split}', exist_ok=True)
    np.savez_compressed(f'{OUT}/eval/{split}/{name.replace(":", "__")}.npz', **out)
    print('done', split, name, len(out['idx']), f'{time.time()-t0:.0f}s', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('--split', default='test')
    ap.add_argument('--arm', default='C4')
    ap.add_argument('--run', default=None)
    ap.add_argument('--n_per', type=int, default=100)
    ap.add_argument('--max_n', type=int, default=0)
    ap.add_argument('--params', default='{}')
    a = ap.parse_args()
    dev = best_device()
    if a.cmd == 'design':
        design(a.split, a.n_per, dev)
    elif a.cmd == 'run':
        run_arm(a.split, a.arm, a.run, a.max_n, dev, json.loads(a.params))


if __name__ == '__main__':
    main()
