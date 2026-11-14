"""One backbone pass per batch to T_cap.

rollout() keeps the logit of every step (fp16, on device) and computes, causally at each
step t, the scalar observations the simple rules may read and the head outputs at the
head decision times. Features are also saved at requested training decision times.
eval_arrays() computes label-dependent arrays from the stored logits AFTER the pass;
only scoring and the oracle read them.
"""
import math
import torch
from heads import features, kl_bern, ent_bern, clampp, logq, HORIZONS, horizon_index

LOG_HALF = math.log(0.5)


def _flat(p):
    return p.flatten(1)


@torch.no_grad()
def rollout(model, x, Tc, feat_times=(), head_times=(), heads=None, qact_logc=None, keep_all_feats=False):
    """heads: dict name -> (module, kind). Returns dict:
      Z (Tc+1, B, *S) fp16 logits (Z[0] = 0, i.e. p_0 = 0.5)
      obs: kl1, kl2, conf_sum, conf_min, ent  (B, Tc+1) float32 (index 0 = before any step)
      hout: name -> (B, len(head_times), nout) float32 ; qbits for kind 'qact': (B, len, K) bool
      F: dict t -> features (B, C+5, *S) fp16 at feat_times (all t if keep_all_feats)
    """
    feat_times = set(feat_times)
    head_times = list(head_times)
    hidx = {t: i for i, t in enumerate(head_times)}
    st = model.init_state(x)
    B = x.shape[0]
    Z, F = None, {}
    obs = {k: torch.zeros(B, Tc + 1, device=x.device) for k in ("kl1", "kl2", "conf_sum", "conf_min", "ent")}
    hout = {}
    ps = []   # last three p (float) for features
    for t in range(1, Tc + 1):
        st, logit, h = model.step_once(st)
        logit = logit.float()
        if Z is None:
            Z = torch.zeros((Tc + 1,) + tuple(logit.shape), dtype=torch.float16, device=x.device)
            p0 = torch.full_like(logit, 0.5)
            ps = [p0, p0]
            npix = logit[0].numel()
            obs["ent"][:, 0] = npix * math.log(2)
            obs["conf_sum"][:, 0] = npix * LOG_HALF
            obs["conf_min"][:, 0] = 0.5
        Z[t] = logit.half()
        p = torch.sigmoid(logit)
        pf = _flat(p)
        obs["kl1"][:, t] = kl_bern(pf, _flat(ps[-1])).sum(1)
        obs["kl2"][:, t] = kl_bern(pf, _flat(ps[-2])).sum(1)
        conf = torch.maximum(clampp(pf), 1 - clampp(pf))
        obs["conf_sum"][:, t] = torch.log(conf).sum(1)
        obs["conf_min"][:, t] = conf.min(1).values
        obs["ent"][:, t] = ent_bern(pf).sum(1)
        need_f = keep_all_feats or t in feat_times or t in hidx
        if need_f:
            f = features(h, p, ps[-1], ps[-2])
            if keep_all_feats or t in feat_times:
                F[t] = f.half()
            if t in hidx and heads:
                i = hidx[t]
                for name, (mod, kind) in heads.items():
                    tot, pooled = mod.totals(f)
                    if kind == "qact":
                        npos = p[0].numel()
                        A = mod.qact_adv(pooled, npos, qact_logc) + tot[:, 1:2]      # (B, K)
                        if name not in hout:
                            hout[name] = torch.zeros(B, len(head_times), 1, device=x.device)
                            hout[name + "_bits"] = torch.zeros(B, len(head_times), len(qact_logc),
                                                               dtype=torch.bool, device=x.device)
                        hout[name][:, i] = tot[:, :1]
                        hout[name + "_bits"][:, i] = A <= 0
                    else:
                        if name not in hout:
                            hout[name] = torch.zeros(B, len(head_times), tot.shape[1], device=x.device)
                        hout[name][:, i] = tot
        ps = [ps[-1], p]
    return {"Z": Z, "obs": obs, "hout": hout, "F": F}


def probs(Z, t):
    return torch.sigmoid(Z[t].float())


@torch.no_grad()
def eval_arrays(Z, y, emask, Tc, kgrid):
    """y (B,*S) labels; emask (B,*S) bool positions where a predicted 1 counts (maze open
    pixels; all positions for 1D tasks). Returns S, Sp, exact, cons (B, Tc+1) and
    K (B, len(kgrid), 8) realized KL at each horizon (incl. H=1 for diagnostics)."""
    B = y.shape[0]
    yf = _flat(y)
    em = _flat(emask).bool()
    pc = _flat(probs(Z, Tc))
    yhat = (pc > 0.5).float()
    S = torch.zeros(B, Tc + 1, device=y.device)
    Sp = torch.zeros(B, Tc + 1, device=y.device)
    exact = torch.zeros(B, Tc + 1, dtype=torch.bool, device=y.device)
    cons = torch.zeros(B, Tc + 1, dtype=torch.bool, device=y.device)
    for t in range(Tc + 1):
        p = _flat(probs(Z, t))
        S[:, t] = (logq(p, yf) - LOG_HALF).sum(1)
        Sp[:, t] = (logq(p, yhat) - LOG_HALF).sum(1)
        pred = (p > 0.5) & em
        exact[:, t] = (pred.float() == yf).all(1)
        cons[:, t] = ((p > 0.5) == (pc > 0.5)).all(1)
    K = torch.zeros(B, len(kgrid), len(HORIZONS), device=y.device)
    for i, t in enumerate(kgrid):
        pt = _flat(probs(Z, t))
        for j, H in enumerate(HORIZONS):
            K[:, i, j] = kl_bern(_flat(probs(Z, horizon_index(t, H, Tc))), pt).sum(1)
    return {"S": S, "Sp": Sp, "exact": exact, "cons": cons, "K": K}


@torch.no_grad()
def first_cons(Z, Tc):
    """Label-free: first t >= 1 at which thresholded b_t equals thresholded b_cap (for R0L)."""
    pc = _flat(probs(Z, Tc)) > 0.5
    B = pc.shape[0]
    out = torch.full((B,), Tc, dtype=torch.long, device=pc.device)
    done = torch.zeros(B, dtype=torch.bool, device=pc.device)
    for t in range(1, Tc + 1):
        c = ((_flat(probs(Z, t)) > 0.5) == pc).all(1) & ~done
        out[c] = t
        done |= c
    return out
