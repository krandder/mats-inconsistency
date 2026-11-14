"""M1 (theory note 6.3; Amendment 3): interaction profile of a trained net across a cut, W0.
For episode x = (a, b) split by cut C into C's errors a and the rest b:
  L(x) = log q(y1 | x) - log q(y2 | x), y1, y2 = Bayes mean -/+ one Bayes sd of the base episode;
  Delta = L(a,b) - L(a',b) - L(a,b') + L(a',b'), main effects A = L(a,b) - L(a',b), B = L(a,b) - L(a,b');
  share = Var(Delta) / (Var(A) + Var(B)).
Cuts: 'src' (one source vs the rest: a' = a fresh source, new variance level and errors; b' = fresh other sources);
'split' (source c's labelled questions 1-2 vs the rest: a' = new errors for those two given v_c; b' = fresh other
sources and new errors for c's remaining questions given v_c). Labels, y* and ids are kept.
Bayes reference: exactly 0 for 'src' (log-odds add over sources given y*); > 0 for 'split'.
usage: python m1prof.py RUN [RUN ...] --n 3000 --out results/m1.jsonl"""
import argparse, json, math, os
import numpy as np
import torch
import sfworld as W
import sftrain as T

torch.set_num_threads(1)


def load(run):
    meta = json.load(open(os.path.join(run, 'meta.json')))
    a = argparse.Namespace(**meta)
    for k, v in dict(gid=False, attn='', cut='', cut_k='', nocross=False, m1=16, inner='sum', pool='sum',
                     h=64, aux=0.3).items():
        if not hasattr(a, k):
            setattr(a, k, v)
    model = T.build_model(a, a.world)
    model.load_state_dict(torch.load(os.path.join(run, 'model.pt'), map_location='cpu')); model.eval()
    return model, a


def net_logq(model, a, d, ys, idseed):
    rng = np.random.default_rng(idseed)               # same ids for every variant of an episode batch
    inp = T.build_inputs(d, a.arch, rng, 'cpu', want_group=a.gid)
    with torch.no_grad():
        mu, ls, lg = [t.double().numpy() for t in model(**inp)]
    out = []
    for y in ys:
        lw = lg - np.logaddexp.reduce(lg, axis=1, keepdims=True)
        lp = lw - ls - 0.5 * math.log(2 * math.pi) - 0.5 * ((y[:, None] - mu) / np.exp(ls)) ** 2
        out.append(np.logaddexp.reduce(lp, axis=1))
    return out


def bayes_logq(d, ys):
    """Unnormalised log posterior at the given points (normaliser cancels in L)."""
    return [-0.5 * y ** 2 + W.log_lik_total(d, y[:, None])[:, 0] for y in ys]


def variants(d, cut, rng):
    """Return the four episodes (a,b), (a',b), (a,b'), (a',b') as dicts (copies of d with errors replaced)."""
    n, N = d['o'].shape[:2]
    y = d['y']
    e = d['o'] - y[:, None, :]
    v = d['v'].copy()
    fresh_v = W.VLEV[rng.integers(0, 3, (n, N))]
    fresh_e = rng.standard_normal((n, N, W.Q)) * np.sqrt(fresh_v)[:, :, None]
    c = 0                                               # the cut source (sources are exchangeable)
    maskA = np.zeros((N, W.Q), bool)
    if cut == 'src':
        maskA[c, :] = True
        eA = e.copy(); eA[:, c] = fresh_e[:, c]; vA = v.copy(); vA[:, c] = fresh_v[:, c]
        eB = e.copy(); eB[:, 1:] = fresh_e[:, 1:]; vB = v.copy(); vB[:, 1:] = fresh_v[:, 1:]
    else:
        maskA[c, :2] = True
        same_v = rng.standard_normal((n, W.Q)) * np.sqrt(v[:, c])[:, None]
        eA = e.copy(); eA[:, c, :2] = same_v[:, :2]; vA = v
        eB = e.copy(); eB[:, 1:] = fresh_e[:, 1:]; eB[:, c, 2:] = same_v[:, 2:]
        vB = v.copy(); vB[:, 1:] = fresh_v[:, 1:]
    # (a', b'): A-part from eA, B-part from eB
    eAB = np.where(maskA[None, :, :], eA, eB)
    vAB = vB.copy()
    if cut == 'src':
        vAB[:, c] = vA[:, c]
    out = []
    for ee, vv in ((e, v), (eA, vA), (eB, vB), (eAB, vAB)):
        dd = dict(d); dd['o'] = y[:, None, :] + ee; dd['v'] = vv
        out.append(dd)
    return out


def profile(run, n, seed=0):
    model, a = load(run)
    rng = np.random.default_rng([seed, 31337, a.N])
    d = W.sample(n, a.N, rng)
    B = W.bayes(d)
    sd = np.sqrt(B['var'])
    ys = (B['mean'] - sd, B['mean'] + sd)
    res = {'run': run, 'arch': a.arch, 'N': a.N, 'm': getattr(a, 'm', None), 'n': n}
    for cut in ('src', 'split'):
        V = variants(d, cut, rng)
        allL = {}
        for who in ('net', 'bayes'):
            Ls = []
            for dd in V:
                lq = net_logq(model, a, dd, ys, 7) if who == 'net' else bayes_logq(dd, ys)
                Ls.append(lq[0] - lq[1])
            allL[who] = Ls
        allL['dev'] = [x - y for x, y in zip(allL['net'], allL['bayes'])]   # the net's deviation from Bayes
        for who, Ls in allL.items():
            Lab, Lapb, Labp, Lapbp = Ls
            delta = Lab - Lapb - Labp + Lapbp
            A_, B_ = Lab - Lapb, Lab - Labp
            share = float(delta.var() / (A_.var() + B_.var()))
            res[f'{cut}_{who}_share'] = share
            res[f'{cut}_{who}_rms_delta'] = float(np.sqrt((delta ** 2).mean()))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+'); ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--out', default='results/m1.jsonl')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    for r in args.runs:
        res = profile(r, args.n)
        print(json.dumps(res), flush=True)
        with open(args.out, 'a') as f:
            f.write(json.dumps(res) + '\n')
