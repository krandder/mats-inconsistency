"""W2b training (PREREG-W2.md, W2b amendment): GPU data path (w2fast), curriculum, optional dense per-item Bayes
supervision, learning curves (val gap, N=128 gap, C4 and C7 readings) every --eval-every steps.
Checkpoints and meta.json are harness-compatible (python harness.py eval --run OUT ...).

usage: python w2_train.py --depth 4 --dense 0|1 [--stepping 1] --out runs/Q4 [--steps 150000 --hours 5.0]
       [--es DIR]   (pilot evalset root holding es_main, es_c4_96, es_c7_96 for the learning curves)
"""
import argparse, json, math, os, time
import numpy as np
import torch
import harness as Hn
import world2_market as W
import w2fast as F

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
GH_X, GH_W = np.polynomial.hermite.hermgauss(10)


def forward_multi(model, X, mask, pos=None):
    """SetPFN forward (no sink) returning the head at the last position and, if pos is given, at positions pos."""
    ctx = mask.clone(); ctx[:, -1] = False
    empty = ~ctx.any(1); ctx[empty, -1] = True
    km = ctx[:, None, None, :]
    x = model.inp(X)
    for blk in model.blocks:
        x, _ = blk(x, km, False)

    def head(h):
        o = model.out(model.ln(h)).float()
        lg, mu, ls = o.split(model.mix, -1)
        return lg.log_softmax(-1), mu, ls.clamp(-7, 4)
    q = head(x[:, -1])
    return q, (head(x[:, pos]) if pos is not None else None)


def mix_logp(out, y):
    lw, mu, ls = out
    return torch.logsumexp(lw - ls - 0.5 * math.log(2 * math.pi) - 0.5 * ((y[..., None] - mu) / ls.exp()) ** 2, -1)


def dense_loss(out, mean, var, valid):
    """Cross-entropy E_{x ~ N(mean, var)}[-log q(x)] per item (Gauss-Hermite), averaged over valid items."""
    xs = torch.as_tensor(GH_X, device=mean.device, dtype=torch.float32)
    ws = torch.as_tensor(GH_W / math.sqrt(math.pi), device=mean.device, dtype=torch.float32)
    x = mean.float()[..., None] + torch.sqrt(2 * var.float().clamp_min(1e-10))[..., None] * xs   # (E, K, G)
    lw, mu, ls = out
    lp = torch.logsumexp(lw[:, :, None] - ls[:, :, None] - 0.5 * math.log(2 * math.pi)
                         - 0.5 * ((x[..., None] - mu[:, :, None]) / ls.exp()[:, :, None]) ** 2, -1)
    ce = -(lp * ws).sum(-1)
    v = valid.float()
    return ((ce * v).sum(1) / v.sum(1)).mean()


# ------------------------------------------------------------------------------------------------ curves
class Curves:
    """Fixed sets from the pilot evalsets: N=128 dir (gap), C4 N=96 (closure, shift), C7 N=96 (log-sd ratios)."""

    def __init__(self, es_root):
        self.world = W.make_world()
        self.sets = {}
        for name, es, pol, N in (('n128', 'es_main', 'dir', 128), ('c4', 'es_c4_96', 'c4', 96), ('c7', 'es_c7_96', 'c7', 96)):
            idx = json.load(open(os.path.join(es_root, es, 'index.json')))
            c = [c for c in idx['cells'] if c['policy'] == pol and c['N'] == N][0]
            d = dict(np.load(os.path.join(es_root, es, c['file'])))
            E = min(c['E'], 500) if name == 'n128' else c['E']
            ep = self.world.sample(np.random.default_rng(c['seed']), c['E'], N, pol)
            assert np.allclose(ep['target'], d['target'])
            self.sets[name] = dict(X=ep['X'][:E], M=ep['mask'][:E], y=ep['target'][:E], d=d, E=E)
        d = self.sets['c4']['d']
        pb = Hn._unragged(d)
        self.c4_pb = pb
        g = (np.ones((len(d['target']), 1)), d['mean_freq_only'][:, None], d['var_freq_only'][:, None])
        self.c4_klf = Hn.kl_bayes_model(pb, g)
        mem = d['meta_member']; self.i0 = np.flatnonzero(mem == 0); self.i1 = np.flatnonzero(mem == 1)
        self.shiftB = d['mean_bayes'][self.i0] - d['mean_bayes'][self.i1]
        d7 = self.sets['c7']['d']; arm = d7['meta_arm'].astype(int)
        self.c7_pairs = {}
        for a, b in (('D4', 'T4'), ('HR4', 'HI4'), ('D', 'T')):
            ia = np.flatnonzero(arm == W.C7_ARMS.index(a)); ib = np.flatnonzero(arm == W.C7_ARMS.index(b))
            rB = 0.5 * np.log(d7['var_bayes'][ia] / d7['var_bayes'][ib]).mean()
            self.c7_pairs[a + '/' + b] = (ia, ib, rB)

    def __call__(self, model):
        out = {}
        for name, s in self.sets.items():
            s['pred'] = Hn.predict(model, s['X'], s['M'])
        s = self.sets['n128']
        out['gap_n128'] = float((Hn.nll(s['pred'], s['y']) - s['d']['nll_bayes'][:s['E']]).mean())
        p = self.sets['c4']['pred']
        kl = Hn.kl_bayes_model(self.c4_pb, p)
        clos = 1 - kl / self.c4_klf
        mean, var = Hn.moments(p)
        out['c4_frac_closure'] = float((clos >= 0.5).mean())
        out['c4_shift_frac'] = float(np.median((mean[self.i0] - mean[self.i1]) / self.shiftB))
        mean, var = Hn.moments(self.sets['c7']['pred'])
        for k, (ia, ib, rB) in self.c7_pairs.items():
            out['c7_dedup_' + k] = float(0.5 * np.log(var[ia] / var[ib]).mean() / rB)
        return out


# ------------------------------------------------------------------------------------------------ train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=int, default=4); ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--heads', type=int, default=4); ap.add_argument('--mix', type=int, default=4)
    ap.add_argument('--dense', type=int, default=0); ap.add_argument('--dense-weight', type=float, default=1.0)
    ap.add_argument('--stepping', type=int, default=0); ap.add_argument('--curriculum', type=int, default=1)
    ap.add_argument('--steps', type=int, default=150000); ap.add_argument('--hours', type=float, default=5.0)
    ap.add_argument('--bs', type=int, default=256); ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--warmup', type=int, default=2000); ap.add_argument('--eval-every', type=int, default=5000)
    ap.add_argument('--nval', type=int, default=2048); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--es', default=''); ap.add_argument('--out', required=True)
    ap.add_argument('--train-mix', dest='train_mix', default='old', choices=['old', 'rich']); ap.add_argument('--nmax', type=int, default=128)
    ap.add_argument('--batch-n', dest='batch_n', type=int, default=0, help='1: one N range per batch (W2c)')
    ap.add_argument('--dense-nmax', dest='dense_nmax', type=int, default=10 ** 9)
    ap.add_argument('--init', default=''); ap.add_argument('--deadline', type=float, default=0, help='unix time; stop')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    base = dict(F.STEPPING) if a.stepping else {}
    if a.train_mix == 'rich':
        base.update(F.RICH)
    base = base or None
    world = W.make_world()
    model = Hn.SetPFN(W.FEAT, a.depth, a.width, a.heads, 'real', a.mix).to(DEV)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=DEV))
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    amp = dict(device_type='cuda', dtype=torch.bfloat16) if DEV == 'cuda' else dict(device_type='cpu', enabled=False)
    # validation set: final-stage distribution, fixed seed, exact Bayes
    g_state = torch.get_rng_state(); torch.manual_seed(10_000)
    vcfg = F.stage_cfg(1.0, F.STEPPING if a.stepping else None, True)     # val: W2 training distribution, N<=128
    vb = F.gen(a.nval, vcfg, DEV)
    torch.set_rng_state(g_state)
    vX, vM, _ = F.tokens(vb, vcfg['nmax'] + 1)
    _, _, (qw, qm, qv) = F.bayes(vb)
    vy = vb['v'].gather(1, vb['query'][:, None])[:, 0]
    vbayes = float(-mix_logp((torch.log(qw.clamp_min(1e-300)), qm, 0.5 * torch.log(qv)), vy).mean())
    curves = Curves(a.es) if (a.es and not a.stepping) else None

    def step_once(frac, train=True):
        cfg = F.stage_cfg(frac, base, bool(a.curriculum))
        if a.batch_n:
            hi = int(math.exp(math.log(16) + np.random.rand() * (math.log(a.nmax + 1) - math.log(16))))
            cfg = dict(cfg, nmin=max(8, hi // 2), nmax=max(8, min(hi, a.nmax)))
        dense = bool(a.dense) and cfg['nmax'] <= a.dense_nmax
        b = F.gen(a.bs, cfg, DEV)
        X, M, ipos = F.tokens(b, cfg['nmax'] + 1, item_queries=dense)
        y = b['v'].gather(1, b['query'][:, None])[:, 0].float()
        with torch.autocast(**amp):
            q, it = forward_multi(model, X, M, ipos)
        loss = -mix_logp(q, y).mean()
        if dense:
            im, iv, _ = F.bayes(b)
            loss = loss + a.dense_weight * dense_loss(it, im, iv, b['vk'])
        if train:
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        return float(loss)

    # speed probe at full size (no optimizer step) -> total steps within the time budget
    t0 = time.time()
    nprobe = 40 if a.batch_n else 20
    for _ in range(nprobe):
        step_once(1.0, train=False)
    if DEV == 'cuda': torch.cuda.synchronize()
    ms_full = (time.time() - t0) / nprobe * 1e3
    budget_s = a.hours * 3600 if not a.deadline else a.deadline - time.time()
    steps = int(min(a.steps, budget_s * 1e3 / (1.15 * ms_full)))
    print(f'[{os.path.basename(a.out)}] params {nparam} probe {ms_full:.1f} ms/step at full size -> steps {steps} '
          f'(val Bayes NLL {vbayes:.4f})', flush=True)
    curve, best, best_step = [], 1e9, -1
    t0 = time.time()
    for step in range(1, steps + 1):
        lr = a.lr * min(1.0, step / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for gp in opt.param_groups: gp['lr'] = lr
        model.train()
        loss = step_once(step / steps)
        stop = bool(a.deadline) and time.time() > a.deadline
        if step % a.eval_every == 0 or step == steps or stop:
            model.eval()
            with torch.no_grad():
                pm = Hn.predict(model, vX.cpu().numpy(), vM.cpu().numpy())
            v = float(Hn.nll(pm, vy.cpu().numpy()).mean())
            row = dict(step=step, train_loss=loss, val=v, val_gap=v - vbayes, secs=time.time() - t0,
                       stage=[f for f, _ in F.CURRICULUM if step / steps >= f][-1] if a.curriculum else None)
            if curves is not None:
                row.update(curves(model))
            curve.append(row)
            if v < best:
                best, best_step = v, step
                torch.save(model.state_dict(), os.path.join(a.out, 'best.pt'))
            if step % (a.eval_every * 5) == 0:
                torch.save(model.state_dict(), os.path.join(a.out, f'ckpt_{step}.pt'))
            print(f'[{os.path.basename(a.out)}] ' + ' '.join(f'{k} {v:.4f}' if isinstance(v, float) else f'{k} {v}'
                                                            for k, v in row.items()), flush=True)
            json.dump(curve, open(os.path.join(a.out, 'curve.json'), 'w'), indent=1)
        if stop:
            print(f'[{os.path.basename(a.out)}] deadline reached at step {step} of {steps}', flush=True)
            break
    torch.save(model.state_dict(), os.path.join(a.out, 'model.pt'))
    tail = [c['val'] for c in curve if c['step'] >= 0.75 * steps]; head = [c['val'] for c in curve if c['step'] < 0.75 * steps]
    args = dict(world='world2_market', depth=a.depth, width=a.width, heads=a.heads, mix=a.mix, mlp_ratio=4, sink=0,
                dense=a.dense, stepping=a.stepping, curriculum=a.curriculum, steps=steps, steps_done=step, bs=a.bs,
                lr=a.lr, seed=a.seed, train_mix=a.train_mix, nmax=a.nmax, batch_n=a.batch_n, init=a.init)
    meta = dict(args=args, world_cfg=world.cfg, n_params=nparam, val_bayes_nll=vbayes, best_val=best,
                best_step=best_step, last_quarter_improvement=(min(head) - min(tail)) if head and tail else None,
                curve=curve, seconds=time.time() - t0, ms_per_step=(time.time() - t0) / step * 1e3,
                probe_ms_full=ms_full, device=DEV)
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
