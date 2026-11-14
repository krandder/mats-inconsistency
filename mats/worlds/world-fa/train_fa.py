"""World FA trainer for the binding arms (PREREG-FA.md Amendment A3).

Differences from `harness.py train`: one history length T per batch (drawn from the curriculum stage, so batches
carry no padding), an optional dense per-trader auxiliary target (beta* = E[Sigma^-1 1 | data], read from each
query-report token by its own LayerNorm + Linear head, MSE on beta* / 3, weight --dense), a wall-clock budget
(--hours), TF32 matmuls on GPU, and learning curves with the binding read-outs (gap to Bayes at T = 0 and T = 32,
fraction of the history value recovered at T = 32, own/other per-token attention ratio by layer).
A5 options: --pairaux W (bilinear head on the query-report tokens predicting every pair's empirical error
correlation, MSE weight W, batches with T >= 2) and --ncurr (trader curriculum, e.g. 0:2,0.2:3,0.4:4); token modes
'qcol' / 'qcov' (binding given) come from the world cfg.
Outputs are harness-compatible: model.pt (final), best.pt, meta.json (load with harness.load_run; eval with
harness.py eval or diag_fa.py); the auxiliary heads are saved separately as aux.pt.

usage: python train_fa.py --cfg '<world cfg json>' --N 4 --depth 4 --width 64 --heads 4 --mix 4 --steps 60000
       --bs 256 --dense 1.0 --curr 0:32:64:0,0.3:1:64:0.1 --seed 0 --hours 3.0 --out runs/X
       (--curr: comma list of stage_start_fraction:Tmin:Tmax:p_T0; T log-uniform in [max(Tmin, 1), Tmax], 0 w.p. p_T0)
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import harness as H
import fa_core as C
import world_fa as WF
from diag_fa import attention_binding

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEV == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
BETA_SCALE = 3.0


def parse_curr(s, steps):
    out = []
    for part in s.split(','):
        f, lo, hi, p0 = part.split(':')
        out.append((int(round(float(f) * steps)), int(lo), int(hi), float(p0)))
    return out


def draw_T(rng, st):
    _, lo, hi, p0 = st
    if rng.random() < p0:
        return 0
    lo = max(lo, 1)
    return int(min(hi, max(lo, math.floor(math.exp(rng.uniform(math.log(lo), math.log(hi + 1)))))))


def forward_all(model, X, M):
    """SetPFN forward (no sink) -> (query-head output, final residual stream (B, L, d))."""
    ctx = M.clone(); ctx[:, -1] = False
    empty = ~ctx.any(1); ctx[empty, -1] = True
    km = ctx[:, None, None, :]
    x = model.inp(X)
    for blk in model.blocks:
        x, _ = blk(x, km, False)
    o = model.out(model.ln(x[:, -1]))
    lw, mu, ls = o.split(model.mix, -1)
    return (lw.log_softmax(-1), mu, ls.clamp(-7, 4)), x


def make_val(c, N, T, E, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        b = C.gen(E, N, T, c, 'cpu')
    refs = C.references(b, c)
    X, M = C.tokens(b, c)
    y = b['v'].numpy()
    return dict(X=X.to(DEV), M=M.to(DEV), y=y, T=T, nb=float(C.nll_t(refs['bayes'], b['v']).mean()),
                n0=float(C.nll_t(refs['bayes_T0'], b['v']).mean()), sigma=b['Sigma'])


@torch.no_grad()
def evaluate(model, vals, N, cfg):
    model.eval()
    r = {}
    for name, v in vals.items():
        pm = H.predict(model, v['X'], v['M'])
        nm = float(H.nll(pm, v['y']).mean())
        r[name] = dict(nll=nm, gap=nm - v['nb'],
                       frac=(v['n0'] - nm) / (v['n0'] - v['nb']) if abs(v['n0'] - v['nb']) > 1e-4 else None)
        if v['T'] > 0:
            att = attention_binding(model, v['X'][:256], v['M'][:256], N, v['T'], cfg=cfg, sigma=v['sigma'][:256])
            r[name]['attn_ratio'] = att.get('ratio_by_layer')
            r[name]['partner_ratio'] = att.get('partner_ratio_by_layer')
    model.train()
    return r


def pair_corr(b, N):
    """Empirical error correlation of every trader pair from the history (B, N, N); valid for T >= 2."""
    _, S = C.hist_stats(b)
    S = S[:, :N, :N]
    d = torch.sqrt(torch.diagonal(S, dim1=1, dim2=2).clamp(min=1e-12))
    return S / d[:, :, None] / d[:, None, :]


def parse_ncurr(s, steps):
    return [(int(round(float(f) * steps)), int(n)) for f, n in (part.split(':') for part in s.split(','))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True); ap.add_argument('--N', type=int, default=4)
    ap.add_argument('--depth', type=int, default=4); ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--heads', type=int, default=4); ap.add_argument('--mix', type=int, default=4)
    ap.add_argument('--steps', type=int, default=60000); ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--warmup', type=int, default=0)
    ap.add_argument('--dense', type=float, default=0.0); ap.add_argument('--curr', default='0:32:64:0,0.3:1:64:0.1')
    ap.add_argument('--pairaux', type=float, default=0.0, help='weight of the pair-correlation auxiliary target (A5 arm 3)')
    ap.add_argument('--ncurr', default='', help='trader curriculum: comma list of stage_start_fraction:N (A5 arm 4)')
    ap.add_argument('--nrange', default='', help='Nmin:Nmax: one N per batch, log-uniform (A7 part 2); val at --N')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--hours', type=float, default=1e9)
    ap.add_argument('--eval-every', dest='eval_every', type=int, default=2000)
    ap.add_argument('--nval', type=int, default=1024); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    world = WF.make_world(json.loads(a.cfg))
    c = world.c
    assert c['prior'] == 'f' or a.dense == 0, 'the dense target is implemented for prior (f) here'
    curr = parse_curr(a.curr, a.steps)
    warmup = a.warmup or max(1, a.steps // 60)
    torch.manual_seed(a.seed)
    model = H.SetPFN(world.feat_dim, a.depth, a.width, a.heads, 'real', a.mix).to(DEV)
    aux = nn.Sequential(nn.LayerNorm(a.width), nn.Linear(a.width, 1)).to(DEV)        # created after the model
    pa_ln, pa_A, pa_B = nn.LayerNorm(a.width).to(DEV), nn.Linear(a.width, 16).to(DEV), nn.Linear(a.width, 16).to(DEV)
    params = list(model.parameters()) + (list(aux.parameters()) if a.dense > 0 else [])
    if a.pairaux > 0:
        params += list(pa_ln.parameters()) + list(pa_A.parameters()) + list(pa_B.parameters())
    ncurr = parse_ncurr(a.ncurr, a.steps) if a.ncurr else [(0, a.N)]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    vals = {f'T{T}': make_val(c, a.N, T, a.nval, 20000 + T) for T in (0, 32)}
    rng = np.random.default_rng([a.seed, 777])
    args = dict(vars(a), world='world_fa', world_cfg_json=a.cfg, nmin=a.N, nmax=a.N, mlp_ratio=4, sink=0, sum_head=0,
                warmup=warmup, trainer='train_fa.py')
    curve, best, best_step = [], 1e9, -1
    t0 = time.time(); t_data = t_comp = 0.0; aux_acc = []; pair_base = []; stopped = False; step = 0
    for step in range(1, a.steps + 1):
        if time.time() - t0 > a.hours * 3600:
            stopped = True; step -= 1
            break
        lr = a.lr * min(1.0, step / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups:
            g['lr'] = lr
        ta = time.time()
        st = [x for x in curr if step - 1 >= x[0]][-1]
        T = draw_T(rng, st)
        Nb = [n for f, n in ncurr if step - 1 >= f][-1]
        if a.nrange:
            lo, hi = (int(x) for x in a.nrange.split(':'))
            Nb = int(min(hi, max(lo, math.floor(math.exp(rng.uniform(math.log(lo), math.log(hi + 1)))))))
        torch.manual_seed(int(rng.integers(0, 2 ** 62)))
        with torch.no_grad():
            b = C.gen(a.bs, Nb, T, dict(c, _part='train'), DEV)
            X, M = C.tokens(b, c)
            y = b['v'].float()
            beta = (C.beta_star(b, c).float() / BETA_SCALE) if a.dense > 0 else None
            rho = pair_corr(b, Nb).float() if (a.pairaux > 0 and T >= 2) else None
        tb = time.time()
        out, x = forward_all(model, X, M)
        loss = H.loss_fn(out, y, 'real').mean()
        qpos = C.query_positions(c, Nb, T).to(DEV)
        if a.dense > 0:
            pred = aux(x[:, qpos])[..., 0]
            la = ((pred - beta) ** 2).mean()
            loss = loss + a.dense * la
            aux_acc.append(la.detach())
        if rho is not None:
            hq = pa_ln(x[:, qpos])
            pr = (pa_A(hq) @ pa_B(hq).transpose(1, 2)) / 4.0                          # (B, N, N)
            off = ~torch.eye(Nb, dtype=torch.bool, device=DEV)
            la = ((pr - rho) ** 2)[:, off].mean()
            loss = loss + a.pairaux * la
            aux_acc.append(la.detach())
            pair_base.append(((rho[:, off]) ** 2).mean())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if DEV == 'cuda':
            torch.cuda.synchronize()
        tc = time.time(); t_data += tb - ta; t_comp += tc - tb
        if step % a.eval_every == 0 or step == a.steps:
            r = evaluate(model, vals, a.N, c)
            score = r['T0']['nll'] + r['T32']['nll']
            row = dict(step=step, examples=step * a.bs, N=Nb, gapT0=r['T0']['gap'], gapT32=r['T32']['gap'],
                       fracT32=r['T32']['frac'], attn_ratio=r['T32']['attn_ratio'],
                       partner_ratio=r['T32']['partner_ratio'],
                       aux_mse=float(torch.stack(aux_acc).mean()) if aux_acc else None,
                       aux_mse_zero_baseline=float(torch.stack(pair_base).mean()) if pair_base else None,
                       seconds=time.time() - t0)
            aux_acc = []; pair_base = []
            curve.append(row)
            if score < best:
                best, best_step = score, step
                torch.save(model.state_dict(), os.path.join(a.out, 'best.pt'))
            fr = 'na' if row['fracT32'] is None else f'{row["fracT32"]:.3f}'
            rd = lambda v: None if v is None else [round(x, 2) for x in v]
            print(f'[{os.path.basename(a.out)}] step {step} N {Nb} T0gap {row["gapT0"]:.4f} T32gap {row["gapT32"]:.4f} '
                  f'frac32 {fr} attn {rd(row["attn_ratio"])} partner {rd(row["partner_ratio"])} '
                  f'aux {row["aux_mse"]} ({row["seconds"]:.0f}s; data {t_data / step * 1e3:.1f}ms '
                  f'comp {t_comp / step * 1e3:.1f}ms per step)', flush=True)
    if stopped and (not curve or curve[-1]['step'] != step) and step > 0:     # final evaluation at the stop point
        r = evaluate(model, vals, a.N, c)
        curve.append(dict(step=step, examples=step * a.bs, gapT0=r['T0']['gap'], gapT32=r['T32']['gap'],
                          fracT32=r['T32']['frac'], attn_ratio=r['T32']['attn_ratio'],
                          partner_ratio=r['T32']['partner_ratio'], aux_mse=None, seconds=time.time() - t0))
        if r['T0']['nll'] + r['T32']['nll'] < best:
            best, best_step = r['T0']['nll'] + r['T32']['nll'], step
            torch.save(model.state_dict(), os.path.join(a.out, 'best.pt'))
    torch.save(model.state_dict(), os.path.join(a.out, 'model.pt'))
    torch.save(dict(beta=aux.state_dict(), pair=[m.state_dict() for m in (pa_ln, pa_A, pa_B)]), os.path.join(a.out, 'aux.pt'))
    meta = dict(args=args, world_cfg=world.cfg, n_params=sum(p.numel() for p in model.parameters()),
                best_val=best, best_step=best_step, curve=curve, steps_done=step, stopped_on_time=stopped,
                seconds=time.time() - t0, ms_data_per_step=t_data / max(step, 1) * 1e3,
                ms_compute_per_step=t_comp / max(step, 1) * 1e3, device=DEV,
                val_ref=dict(T0=dict(nb=vals['T0']['nb'], n0=vals['T0']['n0']), T32=dict(nb=vals['T32']['nb'], n0=vals['T32']['n0'])))
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
