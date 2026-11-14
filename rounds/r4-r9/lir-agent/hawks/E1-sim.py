"""Hawk E1 toy: does fused GRPO distort stated confidence, and does a claim mask matter,
depending on whether the reward reads the claim?  Tabular softmax over 10 confidence bins per
context (true success prob p in {0.2,0.5,0.8}); success y ~ Bern(p) independent of the claim.
GRPO: group of G samples per context, advantage (r-mean)/(std+1e-6), REINFORCE on the claim logits.
reward modes:  env  r = y                      (claim has no causal effect on reward)
               appr r = y + 0.5*1[c>=0.7]      (harness `approval_reward` w_conf-style channel)
arms: fused        claim tokens get the reward advantage
      fused_brier  claim tokens get advantage of r - (c-y)^2   (RLCR-style signal, shared objective)
      masked       claim tokens get advantage of -(c-y)^2 only (PPG/DCPO/LIR claim protection)
norm=True is GRPO group-std normalisation; norm=False is mean-baseline only (Dr. GRPO style).
Metrics after T steps: ECE over contexts (expected |E[c]-p| weighted), signed optimism E[c]-p,
mean policy entropy. 5 seeds."""
import json, sys
import numpy as np
BINS = np.linspace(0.05, 0.95, 10); PS = np.array([0.2, 0.5, 0.8])
def run(mode, arm, seed, T=3000, G=8, lr=0.5, norm=True):
    rng = np.random.default_rng(seed)
    L = np.stack([-(BINS - p) ** 2 / 0.02 for p in PS])  # start roughly calibrated
    for _ in range(T):
        for k, p in enumerate(PS):
            pi = np.exp(L[k] - L[k].max()); pi /= pi.sum()
            idx = rng.choice(10, size=G, p=pi); c = BINS[idx]; y = (rng.random(G) < p).astype(float)
            r = y + (0.5 * (c >= 0.7) if mode == "appr" else 0.0)
            br = -(c - y) ** 2
            s = {"fused": r, "fused_brier": r + br, "masked": br}[arm]
            A = (s - s.mean()) / (s.std() + 1e-6) if norm else (s - s.mean())
            g = np.zeros(10)
            for i, a in zip(idx, A):
                g += a * (np.eye(10)[i] - pi)
            L[k] += lr * g / G
    out = {}
    pis = np.exp(L - L.max(1, keepdims=True)); pis /= pis.sum(1, keepdims=True)
    Ec = pis @ BINS
    out["signed_optimism"] = float(np.mean(Ec - PS)); out["ece"] = float(np.mean(np.abs(Ec - PS)))
    out["entropy"] = float(np.mean(-(pis * np.log(pis + 1e-12)).sum(1)))
    out["Ec_per_p"] = Ec.tolist()
    return out
res = {}
for mode in ["env", "appr"]:
  for norm in [True, False]:
    for arm in ["fused", "fused_brier", "masked"]:
        rs = [run(mode, arm, s, norm=norm) for s in range(5)]
        agg = {m: [float(np.mean([r[m] for r in rs])), float(np.std([r[m] for r in rs]))]
               for m in ["ece", "signed_optimism", "entropy"]}
        agg["Ec_per_p_seed0"] = rs[0]["Ec_per_p"]
        res[f"{mode}/{arm}/{'std' if norm else 'nostd'}"] = agg
        print(f"{mode:5s} {'std  ' if norm else 'nostd'} {arm:12s} ECE {agg['ece'][0]:.3f}+-{agg['ece'][1]:.3f}  "
              f"optimism {agg['signed_optimism'][0]:+.3f}+-{agg['signed_optimism'][1]:.3f}  "
              f"entropy {agg['entropy'][0]:.2f}  E[c]|p=.2,.5,.8 {np.round(rs[0]['Ec_per_p'],2)}")
init = np.stack([-(BINS - p) ** 2 / 0.02 for p in PS]); pi0 = np.exp(init - init.max(1, keepdims=True)); pi0 /= pi0.sum(1, keepdims=True)
res["init"] = {"ece": float(np.mean(np.abs(pi0 @ BINS - PS))), "entropy": float(np.mean(-(pi0*np.log(pi0)).sum(1)))}
print("init ECE %.3f entropy %.2f" % (res["init"]["ece"], res["init"]["entropy"]))
json.dump(res, open(sys.argv[1], "w"), indent=1)
