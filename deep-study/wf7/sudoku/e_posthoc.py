"""e_posthoc.py -- NOT pre-registered (addendum A8). Why does T_A at the pre-set tau accept some wrong candidates,
and does a check that reads M only at one-cell contexts fix it?
  [1] the wrong boards that the search arms' check accepted: how many rule violations they contain, and where the
      trust route loses (or fails to lose) probability on them
  [2] trade-off of the route threshold tau: false accepts on wrong candidates against false rejects on valid grids
  [3] the pair check: min over ordered pairs of log q(x_j | x_i alone); threshold theta from real validation grids
Usage: python -u e_posthoc.py"""
import os, json, math, time
import numpy as np, torch
import core, data
import oracle as O            # scoring: counting rule violations, validity

torch.set_num_threads(1); K, N = 6, 36; T0 = time.time()
d = np.load("data/k6_base.npz"); B6 = d["B"]; bs = d["base_split"]
va = data.GridStream(B6[bs == 1], K, 202); te = data.GridStream(B6[bs == 2], K, 303)
XT = te.sample(2000)[:500]; XV = va.sample(2000)[:1000]                      # the same grids as e_trust6.py
M = core.Net(core.load_model("ckpt/mA6.pt")[0], K, "mA6")
ck = torch.load("ckpt/trust_mA6.pt", map_location="cpu"); m = core.RelFormer(**ck["cfg"]); m.load_state_dict(ck["state"]); tA = core.TrustPolicy(m, K)
meta = json.load(open("out/trust_meta_mA6.json")); tau = meta["tau"]
P = O.peers(K)


def log(*a):
    print(*a, flush=True)


wa, wb = [], []
for f in sorted(os.listdir("out")):
    if f.startswith("search_") and f.endswith(".pt"):
        r = torch.load("out/" + f)
        if len(r["wrong_acc_boards"]):
            wa.append(r["wrong_acc_boards"])
        if len(r["wrong_boards"]):
            wb.append(r["wrong_boards"])
        log(f"   {f}: wrong candidates accepted {len(r['wrong_acc_boards'])}; sample of wrong candidates {len(r['wrong_boards'])}")
WA = torch.unique(torch.cat(wa), dim=0) if wa else torch.zeros(0, N, dtype=torch.long)
WB = torch.unique(torch.cat(wb), dim=0); WB = WB[torch.randperm(len(WB), generator=torch.Generator().manual_seed(1))[:1500]]
log(f"distinct wrong boards accepted by the pre-set check: {len(WA)}; sample of distinct wrong candidates: {len(WB)}")


def n_conf(X):
    X = X.numpy(); eq = (X[:, :, None] == X[:, None, :]) & P[None]
    return eq.sum((1, 2)) // 2


# ---- [1] anatomy of the accepted wrong boards
lv = core.route_logT(M, tA, XV)["logp"].numpy().astype(np.float64)
lt = core.route_logT(M, tA, XT)["logp"].numpy().astype(np.float64)
rb = core.rollout(M, tA, torch.full_like(WB, -1), X=WB, record=True); lwb = rb["logp"].numpy().astype(np.float64)
log(f"\n[1] route log T: validation grids min {lv.min():.2f}, 1% quantile {np.quantile(lv, .01):.2f}, median {np.median(lv):.2f}; "
    f"wrong candidates (sample): median {np.median(lwb):.2f}, 99th percentile {np.quantile(lwb, .99):.2f}, max {lwb.max():.2f}")
log(f"    rule violations (pairs of equal digits in a row, column or box) per wrong candidate: median {np.median(n_conf(WB)):.0f}, min {n_conf(WB).min()}")
if len(WA):
    ra = core.rollout(M, tA, torch.full_like(WA, -1), X=WA, record=True); nc = n_conf(WA)
    worst = ra["lps"].min(1).values
    log(f"    accepted wrong boards: violations per board: median {np.median(nc):.0f}, min {nc.min()}, max {nc.max()}; log T median {ra['logp'].median():.2f}; "
        f"smallest atom on the route: median {worst.median():.2f} nats, i.e. probability {worst.median().exp():.3f}; "
        f"boards whose smallest atom is above 1%: {(worst > math.log(0.01)).float().mean():.3f}")
    wr = rb["lps"].min(1).values
    log(f"    all wrong candidates (sample): smallest atom on the route: median {wr.median():.2f} nats; share with smallest atom above 1%: {(wr > math.log(0.01)).float().mean():.4f}")

# ---- [2] the route threshold: trade-off
log("\n[2] route threshold tau: false accepts on wrong candidates against false rejects on held-out valid grids")
RES = {"tau_tradeoff": []}
for lab, t in (("pre-set: validation minimum - 2.0", tau), ("validation minimum", lv.min()), ("validation 0.5% quantile", np.quantile(lv, .005)),
               ("validation 1% quantile", np.quantile(lv, .01)), ("validation 5% quantile", np.quantile(lv, .05)), ("validation 20% quantile", np.quantile(lv, .2))):
    fa = float((lwb >= t).mean()); fr = float((lt < t).mean())
    RES["tau_tradeoff"].append(dict(rule=lab, tau=float(t), false_accept=fa, false_reject=fr))
    log(f"    tau = {t:7.2f} ({lab:34s}): wrong candidates accepted {fa:.4f}; valid grids rejected {fr:.4f}")

# ---- [3] the pair check (one-cell contexts)
log("\n[3] pair check: min over ordered pairs (i, j) of log q(x_j | only x_i revealed)")
pv, _ = core.pair_min(M, XV); pt, _ = core.pair_min(M, XT); pw, usedw = core.pair_min(M, WB)
theta = float(pv.min() - 2.0)
log(f"    real validation grids: min {pv.min():.3f}, median {pv.median():.3f} -> theta = {theta:.3f} (validation minimum minus 2.0)")
log(f"    held-out valid grids rejected: {(pt < theta).float().mean():.4f} (min {pt.min():.3f}); wrong candidates accepted: {(pw >= theta).float().mean():.5f} "
    f"({int((pw >= theta).sum())} of {len(pw)}); their score: max {pw.max():.2f}, median {pw.median():.2f}")
gap = float(pv.min() - pw.max())
log(f"    gap between the lowest valid score and the highest wrong-candidate score: {gap:.2f} nats")
RES.update(theta=theta, pair_val_min=float(pv.min()), pair_fr=float((pt < theta).float().mean()), pair_fa=float((pw >= theta).float().mean()),
           pair_wrong_max=float(pw.max()), n_wrong=len(pw), gap=gap)
if len(WA):
    pa, _ = core.pair_min(M, WA)
    log(f"    the {len(WA)} wrong boards that the route check accepted: accepted by the pair check: {int((pa >= theta).sum())} (their score: max {pa.max():.2f})")
    RES["pair_fa_on_route_accepted"] = int((pa >= theta).sum()); RES["n_route_accepted_wrong"] = len(WA)
_, ua = core.pair_min(M, WB, abort_below=theta)
log(f"    calls of M per wrong candidate with early stop: mean {ua.float().mean():.1f} of {N}")
rng = np.random.default_rng(5)
import eval_lib as E
for kind in ("cell", "swap"):
    Xn = torch.from_numpy(E.near_miss(XT.numpy(), rng, kind, K)); pn, _ = core.pair_min(M, Xn)
    log(f"    near-miss boards ({kind}): accepted {(pn >= theta).float().mean():.4f}; max score {pn.max():.2f}")
    RES[f"pair_near_{kind}"] = float((pn >= theta).float().mean())
json.dump(RES, open("out/e_posthoc.json", "w"), indent=1)
json.dump(dict(theta=theta, val_min=float(pv.min())), open("out/pair_meta.json", "w"))
log(f"total time {time.time()-T0:.0f}s")
