"""e_posthoc2.py -- post hoc (addendum A8/A9): (1) the pair check on T's own invalid draws; (2) the exact sign test
for KT against the pre-registered own-confidence rule on HARD; (3) the fraction of test inputs seen in training.
Usage: python -u e_posthoc2.py"""
import json, math
import numpy as np, torch
import core
import oracle as O, eval_lib as E          # scoring only

torch.set_num_threads(1); K, N = 6, 36
M = core.Net(core.load_model("ckpt/mA6.pt")[0], K, "mA6")
ck = torch.load("ckpt/trust_mA6.pt", map_location="cpu"); m = core.RelFormer(**ck["cfg"]); m.load_state_dict(ck["state"]); tA = core.TrustPolicy(m, K)
theta = json.load(open("out/pair_meta.json"))["theta"]; tau = json.load(open("out/trust_meta_mA6.json"))["tau"]
RES = {}
# (1) the same 2,000 draws from T as in e_trust6.py [f] (same seed)
gs = torch.Generator().manual_seed(41)
smp = core.rollout(M, tA, torch.full((2000, N), -1, dtype=torch.long), gen=gs)
ok = O.is_valid_batch(smp["s"].numpy(), K); bad = torch.from_numpy(~ok)
ps, _ = core.pair_min(M, smp["s"][bad]); pv, _ = core.pair_min(M, smp["s"][~bad][:300])
print(f"(1) draws from T by pi_trust: {int(bad.sum())} invalid of 2000; accepted by the route threshold: {int((smp['logp'][bad] >= tau).sum())}; "
      f"accepted by the pair check: {int((ps >= theta).sum())} (their pair score: max {ps.max():.2f}, median {ps.median():.2f}); "
      f"300 valid draws: pair score min {pv.min():.3f}, rejected {int((pv < theta).sum())}")
RES["draws"] = dict(n_invalid=int(bad.sum()), route_accepts=int((smp['logp'][bad] >= tau).sum()), pair_accepts=int((ps >= theta).sum()), pair_max=float(ps.max()))
# (1b) every wrong board that the pre-registered route check accepted in any search arm: does the pair check reject it?
import os
wa = []
for f in sorted(os.listdir("out")):
    if f.startswith("search_") and f.endswith(".pt") and "_T2" not in f:
        r = torch.load("out/" + f)
        if len(r["wrong_acc_boards"]):
            wa.append(r["wrong_acc_boards"])
        nwrong = int(((r["acc_idx"] >= 0) & ~(r["answer"] == r["sol"]).all(1)).sum())
        print(f"    {f}: wrong candidates accepted {len(r['wrong_acc_boards'])}; puzzles with a wrong accepted answer {nwrong} of {len(r['sol'])}")
if wa:
    WA = torch.unique(torch.cat(wa), dim=0); pa, _ = core.pair_min(M, WA)
    print(f"(1b) distinct wrong boards accepted by the route check over all arms: {len(WA)}; accepted by the pair check: {int((pa >= theta).sum())} (pair score max {pa.max():.2f})")
    RES["route_accepted_wrong"] = dict(n=len(WA), pair_accepts=int((pa >= theta).sum()), pair_max=float(pa.max()))
# (2) KT against K1 on HARD
b = torch.load("out/base_HARD.pt"); kt = b["KT"]
for rule in ("entropy", "maxprob", "margin"):
    k1 = b[f"K1_mB6_{rule}"]; a_only = int((k1 & ~kt).sum()); b_only = int((kt & ~k1).sum())
    p = E.mcnemar_p(a_only, b_only)
    print(f"(2) HARD: KT {kt.float().mean():.4f} against K1 ({rule}) {k1.float().mean():.4f}: only K1 right {a_only}, only KT right {b_only}, exact sign test p = {p:.4f}")
    RES[f"KT_vs_{rule}"] = dict(kt=float(kt.float().mean()), k1=float(k1.float().mean()), k1_only=a_only, kt_only=b_only, p=p)
# (3) memorisation audit: are test puzzles' solutions, or any symmetric image of them, in the training sets?
d = np.load("data/k6_base.npz"); bs = d["base_split"]; ids = d["base_orbit"]
print(f"(3) orbits: train {sorted(set(ids[bs == 0].tolist()))[:5]}... ({len(set(ids[bs == 0].tolist()))}), validation {len(set(ids[bs == 1].tolist()))}, "
      f"test {len(set(ids[bs == 2].tolist()))}; shared between train and test: {len(set(ids[bs == 0].tolist()) & set(ids[bs == 2].tolist()))}")
tr = np.load("data/k6_train_min.npz")["sol"]; te = np.load("data/k6_test_min.npz")["sol"]
key = lambda X: set(map(bytes, np.asarray(X, dtype=np.int8)))
print(f"    test-puzzle solutions that appear among the 12,000 training-puzzle solutions: {len(key(tr) & key(te))}")
json.dump(RES, open("out/e_posthoc2.json", "w"), indent=1)
