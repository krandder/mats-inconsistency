"""Meta-train FR (prequential joint NLL), FR-fact, FR-small or FR-direct (exact-CMI regression + useful/useless BCE).

Episodes come from gen.py's 'train' namespace on the fly (no data files). Batch b of run seed s is a pure function of
(s, b): a batch shares (m, w); every column gets a random birth row or a 30% drop (FR/FR-fact), or the candidate is born
at H with random k (FR-direct). Compute is counted with torch's FlopCounterMode (once per batch shape) and training
stops at F_meta; the learning-rate schedule is cosine in the fraction of F_meta used.
Checkpoints every ~10 min (model, optimizer, step, counted FLOPs, torch/numpy/python RNG, data position), last 2 kept.
"""
import argparse, json, math, os, random, time, glob
import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.flop_counter import FlopCounterMode
import gen
from gen import H, ID_TYPES, LEAK_TYPES, make_structure, sample_rows, ybits, rng_for
from reader import FR, random_order, n_params

T_ROWS = H + 128
ZERO_TYPES = {'T0', 'T3a', 'T3b', 'HN1', 'HN3', 'HN4'}


def batch_size(m, w, base=32):
    return max(8, int(base * 14 / (m + 1 + w)))


def train_batch(seed, bi, mode, base_bs=32):
    """-> dict(V (B,T,m+1) int64, Y (B,T,w) int64, [cmi-free fields]) as numpy; deterministic in (seed, bi)."""
    rng = np.random.default_rng([seed, bi, 7])
    m = int(rng.integers(2, 13)); w = int(rng.integers(4, 7))
    B = batch_size(m, w, base_bs)
    Vs, Ys, types, k_used = [], [], [], []
    for e in range(B):
        idx = int(bi) * 64 + e + int(seed) * 10 ** 9
        r0 = rng_for('train', idx, 0)
        ctype = LEAK_TYPES[int(r0.integers(4))] if r0.random() < .1 else ID_TYPES[int(r0.integers(8))]
        ep = make_structure(rng_for('train', idx, 1), 'train', idx, ctype, m, w)
        rr = rng_for('train', idx, 2)
        if mode == 'direct':
            k = int(np.exp(rr.uniform(np.log(4), np.log(128))))
            _, y, V = sample_rows(ep, H + k, rr, birth=H)
            V = np.concatenate([V, np.full((T_ROWS - H - k, m + 1), -1, np.int8)], 0)
            y = np.concatenate([y, np.zeros(T_ROWS - H - k, y.dtype)])
            k_used.append(k)
            Vs.append(V); Ys.append(ybits(y, w)); types.append(ep)
            continue
        _, y, V = sample_rows(ep, T_ROWS, rr, birth=0)
        for c in range(m + 1):
            u = rr.random()
            if u < .3:
                V[:, c] = -1
            elif u < .65:
                V[:int(rr.integers(1, T_ROWS)), c] = -1
        Vs.append(V); Ys.append(ybits(y, w))
    out = dict(V=np.stack(Vs).astype(np.int64), Y=np.stack(Ys).astype(np.int64), m=m, w=w)
    if mode == 'direct':
        out['eps'] = types; out['k'] = np.array(k_used)
    return out


class Batches(torch.utils.data.IterableDataset):
    def __init__(self, seed, start, mode, base_bs):
        self.seed, self.start, self.mode, self.base_bs = seed, start, mode, base_bs

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        nw, k = (1, 0) if wi is None else (wi.num_workers, wi.id)
        bi = self.start + k
        while True:
            yield bi, train_batch(self.seed, bi, self.mode, self.base_bs)
            bi += nw


def direct_targets(b, dev):
    """exact CMI labels (bits) on the GPU for the batch's episodes (same m, w)."""
    from oracle import cmi_batch
    eps = b['eps']
    cmi = cmi_batch(eps, dev, torch.float64).float()
    zero = torch.tensor([ep.ctype in ZERO_TYPES for ep in eps], device=dev)
    cmi = torch.where(zero, torch.zeros_like(cmi), cmi)
    t3b = torch.tensor([ep.ctype == 'T3b' for ep in eps], device=dev)
    useful = (cmi >= .05).float()
    valid = ((cmi >= .05) | ((cmi < 1e-12) & ~t3b)).float()
    return cmi, useful, valid


def loss_fn(model, b, dev, gen_t, ret_parts=False):
    V = torch.as_tensor(b['V'], device=dev); Y = torch.as_tensor(b['Y'], device=dev)
    if model.mode == 'direct':
        # truncate to the longest context in the batch; padded rows are all-absent and after every real row, so they
        # are masked for real rows by causality; the pooled head sees them though, so mask them out by truncation per k
        K = torch.as_tensor(b['k'], device=dev)
        Tm = int(H + K.max())
        V = V[:, :Tm].clone(); Y = Y[:, :Tm]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
            pred, logit = model.direct(V, Y, lengths=H + K)
        cmi, useful, valid = direct_targets(b, dev)
        mse = Fn.mse_loss(pred.float(), cmi)
        bce = (Fn.binary_cross_entropy_with_logits(logit.float(), useful, reduction='none') * valid).sum() / valid.sum().clamp(min=1)
        loss = mse * 10 + bce
        return (loss, dict(mse=float(mse), bce=float(bce))) if ret_parts else loss
    order = random_order(V.shape[0], Y.shape[2], gen_t, dev)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
        lp = model.row_logp(V, Y, order)
    loss = -lp.mean() / Y.shape[2]
    return (loss, dict(nll_bit=float(loss))) if ret_parts else loss


def val_nll(model, vb, dev):
    model.eval()
    g = torch.Generator().manual_seed(123)
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in vb:
            l, parts = loss_fn(model, b, dev, g, True)
            tot += float(l) * len(b['V']); n += len(b['V'])
    model.train()
    return tot / n


def save_ckpt(path_dir, state):
    os.makedirs(path_dir, exist_ok=True)
    p = os.path.join(path_dir, f"ckpt_{state['step']:07d}.pt")
    torch.save(state, p + '.tmp'); os.replace(p + '.tmp', p)
    for old in sorted(glob.glob(os.path.join(path_dir, 'ckpt_*.pt')))[:-2]:
        os.remove(old)


def load_last(path_dir):
    cs = sorted(glob.glob(os.path.join(path_dir, 'ckpt_*.pt')))
    for c in reversed(cs):
        try:
            return torch.load(c, map_location='cpu', weights_only=False)
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='FR', choices=['FR', 'FR-fact', 'FR-small', 'FR-direct'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fmeta', type=float, default=2e16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--out', default='/root/out/runs')
    ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--ckpt_min', type=float, default=10)
    ap.add_argument('--val_every', type=int, default=250)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    mode = {'FR': 'joint', 'FR-small': 'joint', 'FR-fact': 'fact', 'FR-direct': 'direct'}[a.arm]
    fmeta = a.fmeta / 10 if a.arm == 'FR-small' else a.fmeta
    run = f"{a.arm}_s{a.seed}"
    rdir = os.path.join(a.out, run); os.makedirs(rdir, exist_ok=True)
    if os.path.exists(os.path.join(rdir, 'final.pt')):
        print('already done', run); return
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    model = FR(mode=mode).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(.9, .98))
    gen_t = torch.Generator().manual_seed(1000 + a.seed)
    step, flops, hist, flop_cache = 0, 0.0, [], {}
    st = load_last(rdir)
    if st is not None:
        model.load_state_dict(st['model']); opt.load_state_dict(st['opt'])
        step, flops, hist, flop_cache = st['step'], st['flops'], st['hist'], st['flop_cache']
        torch.set_rng_state(st['rng_torch']); np.random.set_state(st['rng_np']); random.setstate(st['rng_py'])
        gen_t.set_state(st['rng_gen'])
        print(f'resumed {run} at step {step} flops {flops:.3e}', flush=True)
    vseed = 999_000 + (7 if mode == 'direct' else 0)
    vb = [train_batch(vseed, i, mode, a.bs) for i in range(16)]
    dl = torch.utils.data.DataLoader(Batches(a.seed * 1000 + (500 if mode == 'direct' else 0), step, mode, a.bs),
                                     batch_size=None, num_workers=a.workers, prefetch_factor=4 if a.workers else None,
                                     persistent_workers=False)
    t_ck = time.time(); t0 = time.time(); ema = None
    meta = dict(arm=a.arm, seed=a.seed, fmeta=fmeta, params=n_params(model), lr=a.lr, wd=a.wd, bs=a.bs)
    json.dump(meta, open(os.path.join(rdir, 'meta.json'), 'w'))
    it = iter(dl)
    while flops < fmeta and (a.max_steps == 0 or step < a.max_steps):
        bi, b = next(it)
        assert bi == step, (bi, step)
        frac = flops / fmeta
        lr = a.lr * min(1.0, (step + 1) / 300) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for gp in opt.param_groups: gp['lr'] = lr
        key = f"{len(b['V'])}_{b['V'].shape[1]}_{b['m']}_{b['w']}" + (f"_{int(b['k'].max())}" if mode == 'direct' else '')
        if key not in flop_cache:
            fc = FlopCounterMode(display=False)
            with fc:
                loss = loss_fn(model, b, dev, gen_t)
                loss.backward()
            flop_cache[key] = float(fc.get_total_flops())
        else:
            loss = loss_fn(model, b, dev, gen_t)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        flops += flop_cache[key]; step += 1
        lv = float(loss.detach()); ema = lv if ema is None else .98 * ema + .02 * lv
        if step % 50 == 0:
            el = time.time() - t0
            print(f"{run} step {step} loss {lv:.4f} ema {ema:.4f} lr {lr:.2e} flops {flops:.3e} ({flops/fmeta:.1%}) "
                  f"eff_tflops {flops/1e12/max(el,1):.2f}", flush=True)
        if step % a.val_every == 0 or flops >= fmeta:
            v = val_nll(model, vb, dev)
            hist.append(dict(step=step, flops=flops, val=v, train_ema=ema, t=time.time()))
            print(f"{run} VAL step {step} val {v:.4f}", flush=True)
        if time.time() - t_ck > a.ckpt_min * 60:
            save_ckpt(rdir, dict(model=model.state_dict(), opt=opt.state_dict(), step=step, flops=flops, hist=hist,
                                 flop_cache=flop_cache, rng_torch=torch.get_rng_state(), rng_np=np.random.get_state(),
                                 rng_py=random.getstate(), rng_gen=gen_t.get_state(), meta=meta))
            t_ck = time.time()
    torch.save(dict(model=model.state_dict(), step=step, flops=flops, hist=hist, meta=meta), os.path.join(rdir, 'final.pt'))
    json.dump(dict(meta, steps=step, flops=flops, hist=hist, wall_s=time.time() - t0), open(os.path.join(rdir, 'train.json'), 'w'))
    for old in glob.glob(os.path.join(rdir, 'ckpt_*.pt')):
        os.remove(old)
    print('done', run, step, flops, flush=True)


if __name__ == '__main__':
    main()
