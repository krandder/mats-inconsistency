"""T2 hawk toy: does a goal-blind, stop-gradient (prequential) self-model show desire-shaped bias?
Self-model p = P(I take the careful step). Policy reads p (visible) or a neutral 0.5 (hidden):
  pi(p) = sigmoid(b + w*(2p-1)); careful pays 1.0, careless pays 0.4 (+N(0,0.1)); REINFORCE on b, w.
Arms: fused (self-model also ascends dE[r]/dp through the policy), protected-visible (log loss on own
realised action only, stop-gradient), protected-counterfactual (report hidden on 50% of episodes,
self-model trained on hidden episodes only), frozen-policy control (policy fixed; pure tracking).
Measures (exact, analytic per step): signed error p - pi_visible(p); performativity pi_visible(p) - pi_hidden.
"""
import json, sys, numpy as np
sig = lambda z: 1/(1+np.exp(-z))
def run(arm, seed, T=4000, lr_pol=0.05, lr_self=0.05, lam=1.0, freeze_b=False):
    rng = np.random.default_rng(seed)
    b, w, q = -1.0, 0.0, 0.0          # policy bias, report weight, self-model logit
    errs, perf = [], []
    for t in range(T):
        p = sig(q)
        hidden = (arm == "cf") and rng.random() < 0.5
        x = 0.0 if hidden else (2*p - 1)
        pi = sig(b + w*x)
        a = rng.random() < pi
        r = (1.0 if a else 0.4) + rng.normal(0, 0.1)
        base = 0.4 + 0.6*pi
        if arm != "frozen":
            g = (r - base) * ((1.0 if a else 0.0) - pi)   # REINFORCE grad of log pi wrt logit
            b += 0.0 if freeze_b else lr_pol*g; w += lr_pol*g*x
        if arm != "cf" or hidden:
            q += lr_self*((1.0 if a else 0.0) - p)          # log-loss on own realised action, stop-grad
        if arm == "fused":
            piv = sig(b + w*(2*p-1))
            q += lam*lr_self*0.6*piv*(1-piv)*w*2*p*(1-p)   # dE[r]/dq through the policy reading p
        p2 = sig(q); piv = sig(b + w*(2*p2-1)); pih = sig(b)
        errs.append(p2 - piv); perf.append(piv - pih)
    errs, perf = np.array(errs), np.array(perf)
    return dict(mean_signed_err=float(errs.mean()), mean_signed_err_first1000=float(errs[:1000].mean()),
                final_signed_err=float(errs[-200:].mean()), final_perf=float(perf[-200:].mean()),
                final_b=b, final_w=w, final_p=float(sig(q)))
out = {}
cfgs = {"base": {}, "lam10": {"lam": 10.0}, "lam50": {"lam": 50.0},
        "hardaction": {"freeze_b": True}, "hardaction_lam10": {"freeze_b": True, "lam": 10.0}}
for cname, cfg in cfgs.items():
    for arm in ["fused", "protected", "cf", "frozen"]:
        if arm in ("protected", "cf", "frozen") and "lam" in cfg: continue   # lam only affects fused
        rs = [run(arm, s, **cfg) for s in range(5)]
        out[cname + "/" + arm] = {k: {"mean": float(np.mean([r[k] for r in rs])), "sd": float(np.std([r[k] for r in rs], ddof=1))} for k in rs[0]}
json.dump(out, open(sys.argv[1], "w"), indent=1)
for arm, d in out.items():
    print(arm, " ".join(f"{k}={v['mean']:+.4f}±{v['sd']:.4f}" for k, v in d.items()))
