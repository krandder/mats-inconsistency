"""e_trust6.py -- 6x6, the grids-only world model M_A: experiments E1, E2, E3 and the constructed part of E5.

  [a] route surprise on held-out grids for every order policy that needs no learning; M's inconsistency
  [b] the trust policy: paired-route policy iteration on real training grids; selection on real validation grids
  [c] E2 table (excess route surprise on 2,000 held-out test grids, paired bootstrap)
  [d] E1: per-step surprise by context type (exact conditionals from the counter: scoring only)
  [e] what order the trust policy learned
  [f] E3: T on complete boards (acceptance threshold from real validation grids; valid mass; near-miss boards)
  [g] E5 (constructed): states where an atom of M is confidently wrong; what each order rule picks there
Usage: python -u e_trust6.py [MODEL=mA6]
"""
import sys, time, math, json
import numpy as np, torch
import core, data
import oracle as O, eval_lib as E            # scoring only

name = sys.argv[1] if len(sys.argv) > 1 else "mA6"
MODE = sys.argv[2] if len(sys.argv) > 2 else "fit"      # fit: [b] only, saves pi_trust and tau; analyse: everything else
torch.set_num_threads(1); torch.manual_seed(0)
K, N = 6, 36; T0 = time.time(); HREAL = math.log(28200960)
d = np.load("data/k6_base.npz"); B6 = d["B"]; bs = d["base_split"]
tr = data.GridStream(B6[bs == 0], K, 101); va = data.GridStream(B6[bs == 1], K, 202); te = data.GridStream(B6[bs == 2], K, 303)
model, meta = core.load_model(f"ckpt/{name}.pt"); M = core.Net(model, K, name)
C6 = O.Counter(B6, K)
OUT = {}


def log(*a):
    print(*a, flush=True)


XT = te.sample(2000); XV = va.sample(2000); XVsel = va.sample(512)
assert O.is_valid_batch(XT.numpy(), K).all()
log(f"== {name}: {M.params} parameters. 2,000 held-out test grids, 2,000 validation grids ({time.time()-T0:.0f}s)")

POL = {"own confidence: lowest entropy": core.pol_entropy, "own confidence: top probability": core.pol_maxprob,
       "own confidence: top-1 minus top-2": core.pol_margin, "least confident first": core.pol_anti,
       "raster order": core.pol_raster}
conf_labs = [l for l in POL if l.startswith("own confidence")]

if MODE == "fit":
    # ------------------------------------------------------------------ [b] trust policy from real training grids
    log("[b] trust policy: paired routes on real training grids")
    gen = torch.Generator().manual_seed(5)

    def sample_real(n):
        x = tr.sample(n); start = torch.full_like(x, -1)
        sub_ = torch.rand(n, generator=gen) < 0.4
        r = torch.floor(torch.rand(n, generator=gen) * (N - 1)).long()
        rank = torch.rand(n, N, generator=gen).argsort(1).argsort(1)
        return x, torch.where((rank < r[:, None]) & sub_[:, None], x, start), None

    policies, hist = core.fit_trust(M, K, sample_real, core.pol_entropy, iters=4, boards_per_iter=1024, n_dev=6, seed=3, log=log)
    sel = [float(-core.route_logT(M, p, XVsel)["logp"].mean()) for p in policies]
    best = int(np.argmin(sel)); trust = policies[best]
    log(f"   validation route surprise of the iterates: {[round(v - HREAL, 4) for v in sel]} (excess) -> pi_trust = iterate {best+1}")
    torch.save(dict(cfg=trust.m.cfg, state=trust.m.state_dict()), f"ckpt/trust_{name}.pt")
    for j, p in enumerate(policies):
        torch.save(dict(cfg=p.m.cfg, state=p.m.state_dict()), f"ckpt/trust_{name}_it{j+1}.pt")
    valc = {l: float(-core.route_logT(M, POL[l], XVsel)["logp"].mean()) for l in conf_labs}
    conf_star = min(valc, key=valc.get)
    log(f"   validation excess of the own-confidence rules: { {l: round(v - HREAL, 4) for l, v in valc.items()} } -> conf* = {conf_star}")
    lv = core.route_logT(M, trust, XV)["logp"].numpy().astype(np.float64); tau = float(lv.min() - 2.0)
    lvc = core.route_logT(M, POL[conf_star], XV)["logp"].numpy().astype(np.float64)
    log(f"   tau = {tau:.4f} (minimum of log T over 2,000 real validation grids {lv.min():.4f}, minus 2.0; mean {lv.mean():.4f})")
    json.dump(dict(tau=tau, val_min=float(lv.min()), val_mean=float(lv.mean()), best_iter=best, sel=sel, hist=hist, conf_star=conf_star,
                   val_conf=valc, tau_conf=float(lvc.min() - 2.0), val_min_conf=float(lvc.min()), val_mean_conf=float(lvc.mean())),
              open(f"out/trust_meta_{name}.json", "w"), indent=1)
    log(f"calls of M: {M.calls}; total time {time.time()-T0:.0f}s")
    sys.exit(0)

tm = json.load(open(f"out/trust_meta_{name}.json")); best = tm["best_iter"]; hist = tm["hist"]; sel = tm["sel"]; conf_star = tm["conf_star"]
ck = torch.load(f"ckpt/trust_{name}.pt", map_location="cpu"); _m = core.RelFormer(**ck["cfg"]); _m.load_state_dict(ck["state"])
trust = core.TrustPolicy(_m, K)
policies = []
for j in range(len(sel)):
    ck = torch.load(f"ckpt/trust_{name}_it{j+1}.pt", map_location="cpu"); _m = core.RelFormer(**ck["cfg"]); _m.load_state_dict(ck["state"])
    policies.append(core.TrustPolicy(_m, K))

# ------------------------------------------------------------------ [a] policies that need no learning
NLL = {}; NLLV = {}
XT2 = XT[:1000]                                      # addendum A7: the E2 table uses the first 1,000 test grids
for lab, p in POL.items():
    NLL[lab] = -core.route_logT(M, p, XT2)["logp"].numpy().astype(np.float64)
    log(f"   [a] {lab:40s} test excess {NLL[lab].mean()-HREAL:8.4f}  validation excess (512 grids, from the fit run) "
        f"{tm['val_conf'].get(lab, float('nan'))-HREAL:8.4f} ({time.time()-T0:.0f}s)")
R = 8; sub = XT[:200]; lts = []
for r in range(R):
    lts.append(core.route_logT(M, core.RandomOrder(1000 + r), sub)["logp"].numpy().astype(np.float64))
lts = np.array(lts)                                        # [32, 500]; each row: a fresh random order per board
mix = -(np.logaddexp.reduce(lts, axis=0) - math.log(R))
NLL["one random order"] = -core.route_logT(M, core.RandomOrder(7), XT2)["logp"].numpy().astype(np.float64)
OUT["inconsistency_std_over_orders"] = float(lts.std(0).mean())
OUT["mixture_excess"] = float(mix.mean() - HREAL); OUT["random_order_excess"] = float((-lts).mean() - HREAL)
log(f"   [a] M's inconsistency: std over 8 random orders of log q_sigma(x), mean over 200 held-out grids: {lts.std(0).mean():.4f} nats; "
    f"range (max - min over the 8 orders), mean {np.ptp(lts, axis=0).mean():.4f}")
log(f"   [a] one random order: excess {(-lts).mean()-HREAL:.4f}; uniform mixture over 8 random orders (200 grids): excess {mix.mean()-HREAL:.4f} ({time.time()-T0:.0f}s)")

NLL["pi_trust (learned on real grids)"] = -core.route_logT(M, trust, XT2)["logp"].numpy().astype(np.float64)

# ------------------------------------------------------------------ [c] E2 table
log(f"\n[c] E2. excess route surprise = mean_x[-log T_pi(x)] - ln 28,200,960, on 1,000 held-out test grids (nats)")
log(f"   conf* (own-confidence rule with the lowest validation surprise) = {conf_star}")
log(f"   {'order policy':44s} {'excess':>8s} {'95% interval':>20s} {'minus pi_trust (paired)':>32s}")
base = NLL["pi_trust (learned on real grids)"]; E2 = {}
for lab, v in NLL.items():
    m, lo, hi = E.boot_ci(v - HREAL); dm, dlo, dhi = E.boot_ci(v - base[:len(v)], seed=1)
    E2[lab.strip()] = dict(excess=m, lo=lo, hi=hi, diff_vs_trust=dm, dlo=dlo, dhi=dhi)
    log(f"   {lab:44s} {m:8.4f} [{lo:8.4f},{hi:8.4f}] {dm:+10.4f} [{dlo:+8.4f},{dhi:+8.4f}]")
log(f"   {'uniform mixture over 8 random orders (200)':44s} {mix.mean()-HREAL:8.4f}")
dm, dlo, dhi = E.boot_ci(NLL[conf_star] - base, seed=2)
ex_t, ex_c = base.mean() - HREAL, NLL[conf_star].mean() - HREAL
if dlo > 0:
    verdict = "major" if ex_t <= 0.5 * ex_c else "minor"
elif dhi < 0:
    verdict = "negative"
else:
    verdict = "null"
log(f"   HEADLINE E2 (pre-registered bins): excess(pi_trust) = {ex_t:.4f}, excess(conf*) = {ex_c:.4f}, "
    f"conf* - pi_trust = {dm:+.4f} [{dlo:+.4f}, {dhi:+.4f}] -> {verdict.upper()}")
OUT["E2"] = E2; OUT["E2_verdict"] = verdict; OUT["conf_star"] = conf_star; OUT["trust_hist"] = hist; OUT["trust_sel"] = sel
# added by addendum A6: a learned mixture over routes. Weights by EM on 512 real validation grids; T_mix is a convex
# combination of normalised laws, so it is normalised, and its density costs one route per component.
mix_pols = [(l, POL[l]) for l in POL] + [("pi_trust (learned on real grids)", trust)]
LV = np.stack([core.route_logT(M, p, XVsel[:256])["logp"].numpy().astype(np.float64) for l, p in mix_pols], 1)
wmix = E.fit_mixture(LV); LT_ = np.stack([-NLL[l] for l, p in mix_pols], 1); lmix = E.mixture_logT(LT_, wmix)
m, lo, hi = E.boot_ci(-lmix - HREAL); dm, dlo, dhi = E.boot_ci(-lmix - base, seed=3)
log("   learned mixture over the six order policies (weights fit on 256 real validation grids): " + ", ".join(f"{l.split(':')[-1].strip()[:24]} {w:.3f}" for (l, p), w in zip(mix_pols, wmix)))
log(f"   {'learned mixture over six routes':44s} {m:8.4f} [{lo:8.4f},{hi:8.4f}] {dm:+10.4f} [{dlo:+8.4f},{dhi:+8.4f}]")
OUT["E2_mixture"] = dict(excess=m, lo=lo, hi=hi, diff_vs_trust=dm, dlo=dlo, dhi=dhi, weights={l: float(w) for (l, p), w in zip(mix_pols, wmix)})
# added by addendum A4 (after the 4x4 result): the same table on FRESH grids of the TRAINING orbits (in-distribution
# for M and for the trust learner; these 1,000 grids were not drawn during training with probability ~1)
XF = data.GridStream(B6[bs == 0], K, 909).sample(500); E2f = {}
log(f"   same table on 500 fresh grids of the training orbits (in distribution): excess, and paired difference to pi_trust")
basef = -core.route_logT(M, trust, XF)["logp"].numpy().astype(np.float64)
for lab, p in [(l, POL[l]) for l in conf_labs] + [("pi_trust (learned on real grids)", trust)]:
    v = basef if p is trust else -core.route_logT(M, p, XF)["logp"].numpy().astype(np.float64)
    m, lo, hi = E.boot_ci(v - HREAL); dm, dlo, dhi = E.boot_ci(v - basef, seed=1)
    E2f[lab] = dict(excess=m, lo=lo, hi=hi, diff_vs_trust=dm, dlo=dlo, dhi=dhi)
    log(f"   {lab:44s} {m:8.4f} [{lo:8.4f},{hi:8.4f}] {dm:+10.4f} [{dlo:+8.4f},{dhi:+8.4f}]")
OUT["E2_train_orbits"] = E2f

# ------------------------------------------------------------------ [d] E1: per-step surprise by context type
log("\n[d] E1. per-step held-out surprise by context type (exact conditionals; 1,200 states on routes of test grids)")
gsel = torch.Generator().manual_seed(11); S1, X1, src = [], [], []
for lab, p in (("random", core.RandomOrder(5)), ("own confidence", POL[conf_star]), ("trust", trust)):
    out = core.rollout(M, p, torch.full_like(XT[:400], -1), X=XT[:400], record="full")
    k = torch.randint(0, N, (400,), generator=gsel)
    S1.append(out["states"][torch.arange(400), k]); X1.append(XT[:400]); src += [lab] * 400
S1 = torch.cat(S1); X1 = torch.cat(X1)
LP1 = M.logp(S1)
A = E.atom_stats(S1.numpy(), X1.numpy(), LP1.numpy(), C6, K); typ = E.ctx_type(A); c = E.COLS
names = ["(a) locally forced", "(b) forced, not locally", "(c) free"]
log(f"   atoms scored: {len(A)}")
log(f"   {'context type':26s} {'share':>7s} {'surprise':>9s} {'entropy':>8s} {'excess':>8s} {'leak: rule-breaking':>20s} {'leak: no completion':>20s}")
E1 = {}
for t in range(3):
    s = typ == t
    E1[names[t]] = dict(share=float(s.mean()), surprise=float(A[s, c["surprise"]].mean()), entropy=float(A[s, c["H"]].mean()),
                        excess=float(A[s, c["excess"]].mean()), leak_direct=float(A[s, c["leak_direct"]].mean()),
                        leak_dead=float(A[s, c["leak_dead"]].mean()))
    e = E1[names[t]]
    log(f"   {names[t]:26s} {e['share']:7.3f} {e['surprise']:9.4f} {e['entropy']:8.4f} {e['excess']:8.4f} {e['leak_direct']:20.2e} {e['leak_dead']:20.2e}")
ld = A[:, c["leak_direct"]]
log(f"   mass on directly rule-breaking values per atom: mean {ld.mean():.2e}, 99th percentile {np.quantile(ld, .99):.2e}, max {ld.max():.2e}")
OUT["E1_rule"] = "sharp" if ld.mean() < 1e-3 else ("partial" if ld.mean() < 1e-2 else "no")
log(f"   pre-registered reading of 'M encodes the rule': {OUT['E1_rule'].upper()}")
log(f"   by number of filled cells:   {'filled':>8s} {'atoms':>7s} {'surprise':>9s} {'entropy':>8s} {'excess':>8s} {'leak rule':>10s} {'leak dead':>10s} {'share forced-not-local':>23s}")
E1d = []
for lo_, hi_ in ((0, 5), (6, 11), (12, 17), (18, 23), (24, 29), (30, 35)):
    s = (A[:, c["depth"]] >= lo_) & (A[:, c["depth"]] <= hi_)
    if s.sum():
        E1d.append(dict(lo=lo_, hi=hi_, n=int(s.sum()), surprise=float(A[s, c["surprise"]].mean()), entropy=float(A[s, c["H"]].mean()),
                        excess=float(A[s, c["excess"]].mean()), leak_direct=float(A[s, c["leak_direct"]].mean()),
                        leak_dead=float(A[s, c["leak_dead"]].mean()), share_b=float((typ[s] == 1).mean())))
        e = E1d[-1]
        log(f"   {'':28s} {lo_:3d}-{hi_:<3d} {e['n']:7d} {e['surprise']:9.4f} {e['entropy']:8.4f} {e['excess']:8.4f} {e['leak_direct']:10.2e} {e['leak_dead']:10.2e} {e['share_b']:23.3f}")
# calibration of free atoms: M's top probability against the exact probability of that value
fr = typ == 2; qt = A[fr, c["q_top"]]; pt = A[fr, c["p_top"]]
log("   calibration on free atoms (M's top probability vs exact probability of the same value):")
CAL = []
for lo_, hi_ in ((0, .3), (.3, .5), (.5, .7), (.7, .9), (.9, .99), (.99, 1.01)):
    s = (qt >= lo_) & (qt < hi_)
    if s.sum():
        CAL.append(dict(lo=lo_, hi=hi_, n=int(s.sum()), q=float(qt[s].mean()), p=float(pt[s].mean())))
        log(f"      M says {lo_:.2f}-{min(hi_,1):.2f}: n {int(s.sum()):6d}  mean M {qt[s].mean():.3f}  exact {pt[s].mean():.3f}")
OUT["E1"] = E1; OUT["E1_depth"] = E1d; OUT["E1_cal"] = CAL
np.save("out/e_trust6_atoms.npy", A)

# ------------------------------------------------------------------ [e] what order did the trust policy learn
log("\n[e] the order each policy takes on 100 held-out grids (exact quantities of the chosen cell)")
ORD = {}
for lab, p in (("pi_trust", trust), ("conf*: " + conf_star, POL[conf_star]), ("random order", core.RandomOrder(9))):
    out = core.rollout(M, p, torch.full_like(XT[:100], -1), X=XT[:100], record="full")
    St = out["states"].reshape(-1, N); At = out["atoms"].reshape(-1, N, K); Xr = XT[:100].repeat_interleave(N, 0)
    Aall = E.atom_stats(St.numpy(), Xr.numpy(), At.numpy(), C6, K); ty = E.ctx_type(Aall)
    chosen = out["cells"].reshape(-1).numpy()
    stt = Aall[:, c["state"]].astype(int); cel = Aall[:, c["cell"]].astype(int)
    is_ch = cel == chosen[stt]
    # per state: availability of each type, type of the chosen cell, excess of chosen vs mean / min of available
    ns = len(St); av = np.zeros((ns, 3), bool); ch_t = np.zeros(ns, int); ch_ex = np.zeros(ns); mean_ex = np.zeros(ns); min_ex = np.zeros(ns)
    order = np.argsort(stt, kind="stable"); bounds = np.searchsorted(stt[order], np.arange(ns + 1))
    for si in range(ns):
        ix = order[bounds[si]:bounds[si + 1]]
        for t in range(3):
            av[si, t] = (ty[ix] == t).any()
        j = ix[is_ch[ix]][0]; ch_t[si] = ty[j]; ch_ex[si] = Aall[j, c["excess"]]
        mean_ex[si] = Aall[ix, c["excess"]].mean(); min_ex[si] = Aall[ix, c["excess"]].min()
    depth = np.tile(np.arange(N), 100)
    r = dict(pick_a_when_avail=float((ch_t[av[:, 0]] == 0).mean()), share_states_a_avail=float(av[:, 0].mean()),
             pick_b_when_avail_no_a=float((ch_t[av[:, 1] & ~av[:, 0]] == 1).mean()) if (av[:, 1] & ~av[:, 0]).any() else float("nan"),
             share_steps=[float((ch_t == t).mean()) for t in range(3)],
             excess_chosen=float(ch_ex.sum() / 100), excess_mean_avail=float(mean_ex.sum() / 100), excess_min_avail=float(min_ex.sum() / 100),
             free_excess_per_step=float(ch_ex[ch_t == 2].mean()), forced_nl_excess_per_step=float(ch_ex[ch_t == 1].mean()) if (ch_t == 1).any() else float("nan"),
             mean_depth_free=float(depth[ch_t == 2].mean()), route_surprise=float(-out["logp"].mean()),
             first_cells=[int(x) for x in np.bincount(out["cells"][:, 0].numpy(), minlength=N).argsort()[::-1][:3]])
    ORD[lab] = r
    log(f"   {lab}")
    log(f"      steps by type of the chosen cell: locally forced {r['share_steps'][0]:.3f}, forced-not-locally {r['share_steps'][1]:.3f}, free {r['share_steps'][2]:.3f}")
    log(f"      when a locally forced cell exists ({r['share_states_a_avail']:.3f} of states) it is chosen {r['pick_a_when_avail']:.3f} of the time")
    log(f"      exact excess summed along the route: chosen cells {r['excess_chosen']:.4f}; "
        f"if every step had the mean available cell {r['excess_mean_avail']:.4f}; the per-step minimum {r['excess_min_avail']:.4f}")
    log(f"      exact excess per step at free choices {r['free_excess_per_step']:.4f}; at forced-not-locally cells {r['forced_nl_excess_per_step']:.4f}; "
        f"mean position of free choices in the route {r['mean_depth_free']:.1f} of 36")
OUT["order"] = ORD

# ------------------------------------------------------------------ [f] E3: T on complete boards
log("\n[f] E3. T on complete boards")
E3 = {}
rng = np.random.default_rng(31)
for lab, p in (("pi_trust", trust), ("conf*", POL[conf_star])):
    tau = tm["tau"] if lab == "pi_trust" else tm["tau_conf"]
    vmin, vmean = (tm["val_min"], tm["val_mean"]) if lab == "pi_trust" else (tm["val_min_conf"], tm["val_mean_conf"])
    lt = -NLL["pi_trust (learned on real grids)"] if lab == "pi_trust" else -NLL[conf_star]
    r = dict(tau=tau, val_min=vmin, val_mean=vmean, false_reject_test=float((lt < tau).mean()),
             test_min=float(lt.min()))
    nn_ = 1000 if lab == "pi_trust" else 500
    for kind in ("cell", "swap"):
        Xn = E.near_miss(XT[:nn_].numpy(), rng, kind, K); assert not O.is_valid_batch(Xn, K).any()
        ln = core.route_logT(M, p, torch.from_numpy(Xn))["logp"].numpy().astype(np.float64)
        r[f"near_{kind}_false_accept"] = float((ln >= tau).mean()); r[f"near_{kind}_max"] = float(ln.max()); r[f"near_{kind}_median"] = float(np.median(ln))
    gs = torch.Generator().manual_seed(41)
    ns_ = 2000 if lab == "pi_trust" else 1000
    smp = core.rollout(M, p, torch.full((ns_, N), -1, dtype=torch.long), gen=gs)
    ok = O.is_valid_batch(smp["s"].numpy(), K); pv, lo, hi = E.wilson(int(ok.sum()), ns_)
    r.update(valid_mass=pv, valid_lo=lo, valid_hi=hi, sampled_invalid_accepted=int(((smp["logp"].numpy() >= tau) & ~ok).sum()),
             sampled_invalid=int((~ok).sum()), sampled_invalid_max_logT=float(smp["logp"].numpy()[~ok].max()) if (~ok).any() else float("nan"))
    E3[lab] = r
    log(f"   T with order {lab}: tau = {tau:.3f} (validation minimum {vmin:.3f} minus 2.0; validation mean {vmean:.3f})")
    log(f"      held-out valid grids: false rejects {r['false_reject_test']:.4f} (minimum log T {lt.min():.3f})")
    log(f"      near-miss invalid boards: one cell changed: accepted {r['near_cell_false_accept']:.4f}, max log T {r['near_cell_max']:.2f}, median {r['near_cell_median']:.2f}; "
        f"two cells of a row swapped: accepted {r['near_swap_false_accept']:.4f}, max {r['near_swap_max']:.2f}, median {r['near_swap_median']:.2f}")
    log(f"      mass on valid grids ({ns_} draws from T): {pv:.4f} [{lo:.4f}, {hi:.4f}]; invalid draws {r['sampled_invalid']}, of which accepted by tau: {r['sampled_invalid_accepted']}"
        f" (largest log T among invalid draws {r['sampled_invalid_max_logT']:.2f})")
t3 = E3["pi_trust"]
fa = max(t3["near_cell_false_accept"], t3["near_swap_false_accept"], t3["sampled_invalid_accepted"] / max(1, t3["sampled_invalid"]))
OUT["E3_verdict_partial"] = "pass" if (t3["false_reject_test"] <= 0.01 and fa <= 1e-3) else ("partial" if (t3["false_reject_test"] <= 0.05 and fa <= 1e-2) else "fail")
log(f"   E3 bins so far (the search candidates of E4 are added in e_search6.py): {OUT['E3_verdict_partial'].upper()}")
OUT["E3"] = E3

# ------------------------------------------------------------------ [g] E5 constructed: confidently wrong atoms
log("\n[g] E5 (constructed). 10,000 states on real test grids, screened for confidently wrong atoms of M")
Xs = te.sample(10000); gg = torch.Generator().manual_seed(51)
r_ = torch.floor(torch.rand(10000, generator=gg) * (N - 1)).long()
rank = torch.rand(10000, N, generator=gg).argsort(1).argsort(1)
Ss = torch.where(rank < r_[:, None], Xs, torch.full_like(Xs, -1))
LPs = M.logp(Ss); emp = Ss < 0
top_q, top_v = LPs.exp().max(-1)
cand_state = ((top_q >= 0.9) & emp).any(1)                                    # states with some confident atom
log(f"   states screened 20,000; states where some empty cell has top probability >= 0.9: {int(cand_state.sum())}")
# exact probability of M's top value, only for states with confident atoms (counter)
idx = torch.where(cand_state)[0].numpy()
Aa = E.atom_stats(Ss[idx].numpy(), Xs[idx].numpy(), LPs[idx].numpy(), C6, K)
conf = Aa[:, c["q_top"]] >= 0.9; wrong = conf & (Aa[:, c["p_top"]] <= 0.1)
log(f"   confident atoms (top probability >= 0.9): {int(conf.sum())}; of these confidently wrong (exact probability of that value <= 0.1): {int(wrong.sum())} "
    f"({wrong.sum()/max(1,conf.sum()):.2e}); exact probability exactly 0: {int((conf & (Aa[:, c['p_top']] == 0)).sum())}")
log(f"   mean exact probability of the top value among confident atoms: {Aa[conf, c['p_top']].mean():.5f} (M's mean top probability {Aa[conf, c['q_top']].mean():.5f})")
CW = dict(n_conf=int(conf.sum()), n_wrong=int(wrong.sum()))
wstates = np.unique(Aa[wrong, c["state"]].astype(int))
log(f"   states containing a confidently wrong atom: {len(wstates)}")
if len(wstates):
    Sw = Ss[idx[wstates]]; LPw = LPs[idx[wstates]]; empw = Sw < 0
    wrong_cells = {}
    for a_, cell_ in zip(Aa[wrong, c["state"]].astype(int), Aa[wrong, c["cell"]].astype(int)):
        wrong_cells.setdefault(a_, set()).add(cell_)
    exc = {}
    for a_, cell_, e_ in zip(Aa[:, c["state"]].astype(int), Aa[:, c["cell"]].astype(int), Aa[:, c["excess"]]):
        exc[(a_, cell_)] = e_
    for lab, p in (("pi_trust", trust), ("own confidence: top probability", core.pol_maxprob), ("own confidence: top-1 minus top-2", core.pol_margin),
                   ("own confidence: lowest entropy", core.pol_entropy), ("random order", core.RandomOrder(3))):
        cell = p(Sw, LPw, empw).masked_fill(~empw, float("inf")).argmin(1).numpy()
        picks_wrong = np.mean([cell[j] in wrong_cells[w] for j, w in enumerate(wstates)])
        ex = np.mean([exc[(w, cell[j])] for j, w in enumerate(wstates)])
        CW[lab] = dict(picks_wrong=float(picks_wrong), excess_picked=float(ex))
        log(f"      {lab:36s} picks a confidently wrong cell {picks_wrong:.3f} of the time; mean exact excess of the picked cell {ex:.4f}")
OUT["E5_constructed"] = CW

OUT["tau"] = tm["tau"]; OUT["best_iter"] = best
json.dump(OUT, open(f"out/e_trust6_{name}.json", "w"), indent=1, default=float)
np.savez(f"out/e_trust6_{name}_nll.npz", **{k.strip().replace(" ", "_").replace(":", "").replace("(", "").replace(")", "").replace(",", ""): v for k, v in NLL.items()})
log(f"\ncalls of M: {M.calls}; total time {time.time()-T0:.0f}s")
