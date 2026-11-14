"""Pilot / binding-arm read-outs for a trained World FA run (PREREG-FA.md section 10, Amendment A3).

usage: python diag_fa.py --run RUN [--ckpt best.pt] [--E 2000] [--Ts 0,8,32,64,128] --out RUN/diag.json

Per T: NLL of model, bayes, bayes_T0, diag_bayes; fraction of the history value recovered
(NLL_T0 - NLL_model) / (NLL_T0 - NLL_bayes); KL(bayes || model).
Relabelling invariance: the same episodes with the trader-id vectors permuted among traders (consistently in
history and query): mean KL(model || model'); and with fresh id vectors.
Misbinding: history columns shuffled among traders (shuffle_history): model NLL against bayes_T0 and against Bayes
on the shuffled input; functional binding response (astra's ID-intervention analogue): with dm, db the changes of
the model's and Bayes's predictive means, RMS ratio RMS(dm)/RMS(db), slope sum(dm db)/sum(db^2), correlation, and
the distribution-change ratio mean J(model) / mean J(Bayes), J = half the sum of forward and reverse KL.
Binding attention (astra's diagnostic): attention from trader i's query-report token to its OWN history tokens vs
OTHER traders' history tokens, per token (own mass / T vs other mass / ((N - 1) T)); ratio of means pooled over
episodes, traders and heads, per layer (layer 1 is astra's primary), plus per head. 1 = no binding.
Partner ratio (A5): the same for an anti-pair member's attention to its PARTNER's tokens over non-partner traders'
tokens (history tokens in report mode; query-report tokens in qcol / qcov)."""
import argparse, json, math
import numpy as np
import torch
import harness as H
import fa_core as C


def _rows_attention(model, X, M, qpos):
    """Yield, per layer, attention probabilities (B, heads, len(qpos), L) from the rows qpos to all keys."""
    x = model.inp(X)
    ctx = M.clone(); ctx[:, -1] = False
    empty = ~ctx.any(1); ctx[empty, -1] = True
    km = ctx[:, None, None, :]
    B, L = X.shape[:2]
    out = []
    for blk in model.blocks:
        d = x.shape[-1]; dh = d // blk.h
        q, k, _ = blk.qkv(blk.ln1(x)).split(d, -1)
        q = q[:, qpos].view(B, len(qpos), blk.h, dh).transpose(1, 2)
        k = k.view(B, L, blk.h, dh).transpose(1, 2)
        out.append(((q @ k.transpose(-1, -2)) / math.sqrt(dh)).masked_fill(~km, float('-inf')).softmax(-1).double())
        x, _ = blk(x, km, False)
    return out


@torch.no_grad()
def attention_binding(model, X, M, N, T, chunk=256, cfg=None, sigma=None):
    """Attention diagnostics from the N query-report tokens (one N and T for the batch).
    own/other (report tokens only; astra's diagnostic): per-token attention to the trader's OWN history tokens over
      OTHER traders' history tokens, ratio of means pooled over heads, traders and episodes, per layer and per head.
    partner (needs sigma, the true Sigma (B, N, N); anti-pair members only): per-token attention to the PARTNER's
      tokens (history tokens in report mode; the partner's query-report token in qcol / qcov) over the non-partner
      traders' tokens."""
    dev = next(model.parameters()).device
    assert not model.use_sink
    mode = (cfg or {}).get('tok_mode', 'report')
    L = X.shape[1]
    qpos = (torch.arange(N) + T * N if mode == 'report' else torch.arange(N)).to(dev)
    tok_trader = torch.full((L,), -1, dtype=torch.long, device=dev)          # trader owning each candidate key
    if mode == 'report':
        for t in range(T):
            tok_trader[t * N:(t + 1) * N] = torch.arange(N, device=dev)
    else:
        tok_trader[:N] = torch.arange(N, device=dev)
    nl, nh = len(model.blocks), model.blocks[0].h
    z = lambda: torch.zeros(nl, nh, dtype=torch.float64, device=dev)
    own, oth, hist, pa, npa = z(), z(), z(), z(), z()
    n = 0; npair = 0
    ii = torch.arange(N, device=dev)
    for i0 in range(0, len(X), chunk):
        A = _rows_attention(model, X[i0:i0 + chunk], M[i0:i0 + chunk], qpos)
        B = len(A[0])
        if mode == 'report' and T > 0:
            own_m = (tok_trader[None, :] == ii[:, None]).double()                     # (N, L)
            oth_m = ((tok_trader[None, :] >= 0) & (tok_trader[None, :] != ii[:, None])).double()
            for li, a in enumerate(A):
                own[li] += (a * own_m).sum(-1).sum((0, 2)) / T
                oth[li] += (a * oth_m).sum(-1).sum((0, 2)) / ((N - 1) * T)
                hist[li] += (a * (own_m + oth_m)).sum(-1).sum((0, 2))
            n += B * N
        if sigma is not None and N >= 3 and (mode != 'report' or T > 0):
            S = sigma[i0:i0 + chunk, :N, :N].to(dev)
            offd = (S.abs() > 0.1) & ~torch.eye(N, dtype=torch.bool, device=dev)
            has = offd.any(2)                                                         # (B, N) pair members
            partner = offd.float().argmax(2)                                          # (B, N)
            kt = tok_trader[None, None, :].expand(B, N, L)
            pm = (kt == partner[:, :, None]) & has[:, :, None]
            nm = (kt >= 0) & (kt != partner[:, :, None]) & (kt != ii[None, :, None]) & has[:, :, None]
            cnt_p = pm.sum(-1).clamp(min=1).double(); cnt_n = nm.sum(-1).clamp(min=1).double()
            for li, a in enumerate(A):
                pa[li] += ((a * pm[:, None].double()).sum(-1) / cnt_p[:, None]).sum((0, 2))
                npa[li] += ((a * nm[:, None].double()).sum(-1) / cnt_n[:, None]).sum((0, 2))
            npair += int(has.sum())
    res = {}
    if n:
        own, oth, hist = own / n, oth / n, hist / n
        res.update(ratio_by_layer=(own.mean(1) / oth.mean(1)).tolist(), ratio_by_layer_head=(own / oth).tolist(),
                   own_per_token_by_layer=own.mean(1).tolist(), other_per_token_by_layer=oth.mean(1).tolist(),
                   history_mass_by_layer=hist.mean(1).tolist())
    if npair:
        pa, npa = pa / npair, npa / npair
        res.update(partner_ratio_by_layer=(pa.mean(1) / npa.mean(1)).tolist(),
                   partner_ratio_by_layer_head=(pa / npa).tolist())
    return res


def binding_response(pm, pm2, pb, pb2):
    """Functional binding response to an input change: model (pm -> pm2) against Bayes (pb -> pb2)."""
    mm, _ = H.moments(pm); mm2, _ = H.moments(pm2); mb, _ = H.moments(pb); mb2, _ = H.moments(pb2)
    dm, db = mm2 - mm, mb2 - mb
    J = lambda p, q: 0.5 * (H.kl_bayes_model(p, q) + H.kl_bayes_model(q, p))
    jm, jb = J(pm, pm2), J(pb, pb2)
    return dict(rms_ratio=float(np.sqrt((dm ** 2).mean() / (db ** 2).mean())), slope=float((dm * db).sum() / (db ** 2).sum()),
                corr=float(np.corrcoef(dm, db)[0, 1]), skl_ratio=float(jm.mean() / jb.mean()),
                skl_model=float(jm.mean()), skl_bayes=float(jb.mean()))


def run(a):
    model, world, meta = H.load_run(a.run, a.ckpt)
    c = world.c
    N = int(meta['args']['nmax'])
    dev = next(model.parameters()).device
    out = dict(run=a.run, ckpt=a.ckpt, N=N, cells={})
    np_ = lambda p: tuple(x.detach().cpu().numpy().astype(np.float64) for x in p)
    Ts = [int(x) for x in a.Ts.split(',')]
    if c.get('tok_mode') in ('qcol', 'qcov'):
        Ts = [T for T in Ts if T <= c['tok_col']]                      # the error column holds tok_col rounds
    for T in Ts:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1000 + T)
            b = C.gen(a.E, N, T, c, 'cpu')
            perm = torch.argsort(torch.rand(a.E, N), 1)
            b_rel = dict(b); b_rel['uid'] = torch.gather(b['uid'], 1, perm[..., None].expand_as(b['uid']))
            b_new = dict(b)
            if c.get('id_mode') == 'slot':                         # fresh labelling = another slot permutation
                p2 = torch.argsort(torch.rand(a.E, N), 1)
                b_new['uid'] = torch.gather(b['uid'], 1, p2[..., None].expand_as(b['uid']))
            else:
                b_new['uid'] = C._ortho_ids(a.E, N, c['d_id'], 'cpu') if c.get('id_ortho') else C._unit(b['uid'].shape, 'cpu')
            b_shuf = C.shuffle_history(b)
        y = b['v'].numpy()
        pred = lambda bb: H.predict(model, *[x.numpy() for x in C.tokens(bb, c)])
        pm, pr, pn, ps = pred(b), pred(b_rel), pred(b_new), pred(b_shuf)
        refs = {k: np_(v) for k, v in C.references(b, c).items()}
        nb = H.nll(refs['bayes'], y).mean(); n0 = H.nll(refs['bayes_T0'], y).mean(); nm = H.nll(pm, y).mean()
        bs_seen = np_(C.bayes(b_shuf, c))
        cell = dict(T=T, nll_model=float(nm), nll_bayes=float(nb), nll_bayes_T0=float(n0),
                    nll_diag_bayes=float(H.nll(refs['diag_bayes'], y).mean()), nll_prior=float(H.nll(refs['prior'], y).mean()),
                    history_value=float(n0 - nb),
                    frac_history_recovered=float((n0 - nm) / (n0 - nb)) if abs(n0 - nb) > 1e-4 else None,
                    gap_model=float(nm - nb), se_gap=float((H.nll(pm, y) - H.nll(refs['bayes'], y)).std() / math.sqrt(a.E)),
                    kl_bayes_model=float(H.kl_bayes_model(refs['bayes'], pm).mean()),
                    kl_relabel=float(H.kl_bayes_model(pm, pr).mean()), kl_fresh_ids=float(H.kl_bayes_model(pm, pn).mean()),
                    shuf_nll_model=float(H.nll(ps, y).mean()), shuf_nll_bayes_seen=float(H.nll(bs_seen, y).mean()),
                    shuf_gain_over_T0=float(n0 - H.nll(ps, y).mean()))
        if T > 0:
            cell['binding_response_shuffle'] = binding_response(pm, ps, refs['bayes'], bs_seen)
            if c.get('tok_mode', 'report') in ('report', 'qcol', 'qcov'):
                X, M = C.tokens(b, c)
                cell['attention'] = attention_binding(model, X.to(dev), M.to(dev), N, T, cfg=c, sigma=b['Sigma'])
        out['cells'][str(T)] = cell
        att = cell.get('attention', {})
        print(json.dumps({k: v for k, v in cell.items() if k != 'attention'}
                         | {k: att[k] for k in ('ratio_by_layer', 'partner_ratio_by_layer') if k in att}), flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--ckpt', default='best.pt')
    ap.add_argument('--E', type=int, default=2000); ap.add_argument('--Ts', default='0,8,32,64,128')
    ap.add_argument('--out', required=True)
    run(ap.parse_args())
