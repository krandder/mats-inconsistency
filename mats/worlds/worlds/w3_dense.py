"""Dense-supervision trainer for W3 (C5-type arm; PREREG-W3 amendment A4). Same models (harness SetPFN or
looped.LoopedSetPFN, optional bf16 wrapper), same world, same validation and outputs as harness.train, so
`python looped.py eval --run DIR ...` evaluates the result unchanged. The only difference: every training episode
also carries K extra query tokens (nodes at nearest-report distance <= 4, i.e. inside the training range), placed
before the real query with mask = False, so they are never attended to (no other token can see them); each reads
the context like the real query and is trained on its node's true bit. Evaluation uses single-query episodes.

usage: python w3_dense.py --dense-k 16 [--loop-block 2 --loop-iters 3 | --depth 4] [harness train args] --out DIR
"""
import argparse, json, math, os, sys, time, zlib
import numpy as np
import torch
import torch.nn.functional as F
import harness as H
import looped as LP
import world3_chains as W3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', default='world3_chains'); ap.add_argument('--world-cfg', dest='world_cfg', default='{}')
    ap.add_argument('--depth', type=int, default=2); ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--heads', type=int, default=4); ap.add_argument('--mix', type=int, default=1)
    ap.add_argument('--mlp-ratio', dest='mlp_ratio', type=int, default=4); ap.add_argument('--sink', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--bs', type=int, default=256); ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--warmup', type=int, default=1000); ap.add_argument('--eval-every', dest='eval_every', type=int, default=2000)
    ap.add_argument('--nval', type=int, default=2048); ap.add_argument('--nmin', type=int, default=0)
    ap.add_argument('--nmax', type=int, default=4); ap.add_argument('--out', required=True)
    ap.add_argument('--loop-block', dest='loop_block', type=int, default=2)
    ap.add_argument('--loop-iters', dest='loop_iters', type=int, default=0)
    ap.add_argument('--loop-no-inject', dest='loop_no_inject', action='store_true')
    ap.add_argument('--amp-bf16', dest='amp_bf16', action='store_true')
    ap.add_argument('--dense-k', dest='dense_k', type=int, default=16)
    a = ap.parse_args()
    if a.loop_iters:
        a.depth = a.loop_block * a.loop_iters
    a.trainer = 'w3_dense'
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    DEV = H.DEV
    world = H.load_world(a.world, json.loads(a.world_cfg))
    torch.manual_seed(a.seed)
    rng = np.random.default_rng([a.seed, zlib.crc32(a.out.encode())])
    args = vars(a)
    model = LP.build_model(world, args).to(DEV)
    inner = model.m if isinstance(model, LP.AmpWrap) else model
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    vep = world.sample_train(np.random.default_rng(H.VAL_SEED), a.nval, a.nmin, a.nmax)
    vb = H.nll(world.exact_bayes(vep), vep['target'], world.target_kind).mean()
    os.makedirs(a.out, exist_ok=True)
    curve, best, best_step = [], 1e9, -1
    t_data = t_comp = 0.0; t0 = time.time()
    K = a.dense_k
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups: g['lr'] = lr
        ta = time.time()
        ep = world.sample_train(rng, a.bs, a.nmin, a.nmax)
        Xq, yq, _ = W3.dense_queries(ep, rng, world.wcfg, K)
        X = np.concatenate([ep['X'][:, :-1], Xq, ep['X'][:, -1:]], 1)
        Mk = np.concatenate([ep['mask'][:, :-1], np.zeros(Xq.shape[:2], bool), ep['mask'][:, -1:]], 1)
        Y = np.concatenate([yq, ep['target'][:, None]], 1)                       # (B, K+1)
        X, Mk, Y = (torch.as_tensor(t, device=DEV) for t in (X, Mk, Y))
        tb = time.time()
        hs = model(X, Mk, want_h=True)
        T = X.shape[1]
        h = hs[-1][:, T - 1 - K:].float()                                       # K extra queries + real query
        logits = inner.out(inner.ln(h))
        loss_all = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1), reduction='none').view(Y.shape)
        loss = loss_all.mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if DEV == 'cuda': torch.cuda.synchronize()
        tc = time.time(); t_data += tb - ta; t_comp += tc - tb
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            v = float(H.nll(H.predict(model, vep['X'], vep['mask']), vep['target'], world.target_kind).mean())
            model.train()
            curve.append((step, v, v - vb))
            if v < best:
                best, best_step = v, step
                torch.save(model.state_dict(), os.path.join(a.out, 'best.pt'))
            print(f'[{os.path.basename(a.out)}] step {step} train {loss.item():.4f} (query {loss_all[:, -1].mean().item():.4f}) '
                  f'val {v:.4f} gap {v - vb:.4f} best {best:.4f}@{best_step} ({time.time() - t0:.0f}s; '
                  f'data {t_data / step * 1e3:.1f}ms comp {t_comp / step * 1e3:.1f}ms per step)', flush=True)
    torch.save(model.state_dict(), os.path.join(a.out, 'model.pt'))
    tail = [c[1] for c in curve if c[0] >= 0.75 * a.steps]; head = [c[1] for c in curve if c[0] < 0.75 * a.steps]
    meta = dict(args=args, world_cfg=world.cfg, n_params=nparam, val_bayes_nll=float(vb), best_val=best,
                best_step=best_step, last_quarter_improvement=(min(head) - min(tail)) if head and tail else None,
                curve=curve, seconds=time.time() - t0, ms_per_step=(t_data + t_comp) / a.steps * 1e3,
                ms_data_per_step=t_data / a.steps * 1e3, ms_compute_per_step=t_comp / a.steps * 1e3, device=DEV)
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
