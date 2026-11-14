"""e_search6.py -- 6x6: search corrected against T_A, the no-check baselines, puzzles with several solutions.

  python -u e_search6.py arm ARM SET [LIMIT]   ARM in S1..S5; SET in ALL, HARD, T12       -> out/search_ARM_SET.pt
  python -u e_search6.py base SET              K0, K1 (rules, sizes), K2, KT, found contexts -> out/base_SET.pt
  python -u e_search6.py multi                 E4(d): importance weights on puzzles with 2-12 solutions
The arms are built from core.py only. The real solution of a test puzzle is read only in the scoring lines marked
SCORING; no arm receives it.
"""
import sys, time, math, json, os
import numpy as np, torch
import core

torch.set_num_threads(1); K, N = 6, 36; T0 = time.time()
mode = sys.argv[1]


def log(*a):
    print(*a, flush=True)


def load_trust(path, name):
    ck = torch.load(path, map_location="cpu"); m = core.RelFormer(**ck["cfg"]); m.load_state_dict(ck["state"])
    return core.TrustPolicy(m, K, name)


def get_set(name):
    if name in ("ALL", "HARD"):
        d = np.load("data/k6_test_min.npz"); rating = d["rating"]
        idx = np.arange(500) if name == "ALL" else np.where(rating == 2)[0][:300]
    elif name == "T12":
        d = np.load("data/k6_test_12p.npz"); rating = d["rating"]; idx = np.arange(500)
    elif name == "VAL":
        d = np.load("data/k6_val_min.npz"); rating = d["rating"]; idx = np.arange(500)
    return torch.from_numpy(d["puz"][idx]), torch.from_numpy(d["sol"][idx]), rating[idx]


meta = json.load(open("out/trust_meta_mA6.json")); TAU = meta["tau"]
NETS = {}
for nm in ("mA6", "mB6", "mB6t", "mB6L", "D6"):
    if os.path.exists(f"ckpt/{nm}.pt"):
        NETS[nm] = core.Net(core.load_model(f"ckpt/{nm}.pt")[0], K, nm)
tA = load_trust("ckpt/trust_mA6.pt", "trustA")
tB = load_trust("ckpt/trust_mB6.pt", "trustB") if os.path.exists("ckpt/trust_mB6.pt") else None
ORD = {"margin": core.pol_margin, "maxprob": core.pol_maxprob, "entropy": core.pol_entropy, "anti": core.pol_anti,
       "trustA": tA, "trustB": tB}
FL = {nm: n.flops_per_call for nm, n in NETS.items()}; FL["trustA"] = tA.flops_per_call
if tB is not None:
    FL["trustB"] = tB.flops_per_call
ARMS = {"S1": dict(prop="mA6", order="margin", floor=0.0, greedy_first=True, flag=False),
        "S2": dict(prop="mA6", order="anti", floor=0.05, greedy_first=False, flag=False),
        "S3": dict(prop="D6", order="margin", floor=0.0, greedy_first=True, flag=True),
        "S4": dict(prop="mB6", order="margin", floor=0.0, greedy_first=True, flag=True),
        "S5": dict(prop="mA6", order="trustA", floor=0.0, greedy_first=True, flag=False)}
CAP = 256

if mode == "arm":
    arm, setname = sys.argv[2], sys.argv[3]; limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
    CHK = sys.argv[5] if len(sys.argv) > 5 else "T"        # "T2" (addendum A8): pair check, then the route threshold
    cfg = ARMS[arm]; puz, sol, rating = get_set(setname)
    if limit:
        puz, sol, rating = puz[:limit], sol[:limit], rating[:limit]
    prop = NETS[cfg["prop"]]; order = ORD[cfg["order"]]
    checker = core.make_checker(NETS["mA6"], tA, TAU)
    if CHK == "T2":
        THETA = json.load(open("out/pair_meta.json"))["theta"]
        checker = core.make_checker2(NETS["mA6"], tA, TAU, THETA)
    gen = torch.Generator().manual_seed(900 + int(arm[1]))
    P = len(puz); log(f"== arm {arm} on {setname}: {P} puzzles; proposal {cfg['prop']} in order {cfg['order']}, floor {cfg['floor']}; tau {TAU:.3f}")
    R = dict(answer=[], acc_idx=[], prop_steps=[], chk_steps=[], first=[], n_cand=[], n_wrong=[], n_wrong_acc=[],
             wrong_boards=[], wrong_acc_boards=[], n_right_rej=[])
    for a in range(0, P, 50):
        pz = puz[a:a + 50]; fl = (pz >= 0) if cfg["flag"] else None
        r = core.search_first_accept(prop, order, pz, fl, checker, CAP, gen, floor=cfg["floor"], greedy_first=cfg["greedy_first"])
        for k in ("answer", "acc_idx", "prop_steps", "chk_steps", "first"):
            R[k].append(r[k])
        for j in range(len(pz)):                                             # SCORING: candidates against the solution
            X = torch.cat([c[0] for c in r["cands"][j]]); acc = torch.cat([c[3] for c in r["cands"][j]])
            wrong = (X != sol[a + j]).any(1)
            R["n_cand"].append(len(X)); R["n_wrong"].append(int(wrong.sum())); R["n_wrong_acc"].append(int((wrong & acc).sum()))
            R["n_right_rej"].append(int((~wrong & ~acc).sum()))
            if (wrong & acc).any():
                R["wrong_acc_boards"].append(X[wrong & acc])
            if wrong.any() and sum(len(w) for w in R["wrong_boards"]) < 4000:
                R["wrong_boards"].append(X[wrong][:8])
        done = torch.cat(R["acc_idx"]) >= 0
        log(f"   {min(a+50, P)}/{P}: accepted {done.float().mean():.3f}; candidates checked {sum(R['n_cand'])}, wrong {sum(R['n_wrong'])}, "
            f"wrong accepted {sum(R['n_wrong_acc'])} ({time.time()-T0:.0f}s)")
    out = {k: torch.cat(R[k]) for k in ("answer", "acc_idx", "prop_steps", "chk_steps", "first")}
    out.update(n_cand=torch.tensor(R["n_cand"]), n_wrong=torch.tensor(R["n_wrong"]), n_wrong_acc=torch.tensor(R["n_wrong_acc"]),
               n_right_rej=torch.tensor(R["n_right_rej"]),
               wrong_boards=torch.cat(R["wrong_boards"]) if R["wrong_boards"] else torch.zeros(0, N, dtype=torch.long),
               wrong_acc_boards=torch.cat(R["wrong_acc_boards"]) if R["wrong_acc_boards"] else torch.zeros(0, N, dtype=torch.long),
               nfree=(puz < 0).sum(1), sol=sol, rating=torch.from_numpy(rating), nclue=(puz >= 0).sum(1),
               fl_prop=FL[cfg["prop"]] + (FL[cfg["order"]] if cfg["order"].startswith("trust") else 0),
               fl_chk=FL["mA6"] + FL["trustA"], cfg=cfg, tau=TAU)
    out["chk"] = CHK; out["chk_steps_are_mixed"] = CHK == "T2"
    torch.save(out, f"out/search_{arm}_{setname}{'_T2' if CHK == 'T2' else ''}.pt")
    solved = out["acc_idx"] >= 0; correct = (out["answer"] == sol).all(1)                       # SCORING
    flops = out["prop_steps"] * out["fl_prop"] + out["chk_steps"] * out["fl_chk"]
    log(f"   accepted {solved.float().mean():.4f}; accepted answers correct {correct[solved].float().mean():.4f}; "
        f"first candidate (unchecked) correct {(out['first'] == sol).all(1).float().mean():.4f}; "
        f"FLOPs to accept: mean {flops[solved].float().mean():.3e}, median {flops[solved].float().median():.3e}")
    log(f"   total time {time.time()-T0:.0f}s")

elif mode == "base":
    setname = sys.argv[2]; puz, sol, rating = get_set(setname); clue = puz >= 0; nfree = (puz < 0).sum(1)
    OUT = dict(nfree=nfree, rating=torch.from_numpy(rating), nclue=clue.sum(1), FL=FL)
    log(f"== baselines on {setname}: {len(puz)} puzzles")
    # K0: one pass of M_A, own confidence
    g = core.rollout(NETS["mA6"], core.pol_margin, puz, greedy=True)
    OUT["K0"] = (g["s"] == sol).all(1)                                                          # SCORING
    log(f"   K0  one pass of M_A (grids only), own confidence (margin): accuracy {OUT['K0'].float().mean():.4f}")
    # K1: one pass of M_B, own confidence; rules and sizes
    for nm in ("mB6", "mB6t", "mB6L", "D6"):
        if nm not in NETS:
            continue
        for rule in ("margin", "maxprob", "entropy"):
            g = core.rollout(NETS[nm], ORD[rule], puz, flag=clue, greedy=True)
            OUT[f"K1_{nm}_{rule}"] = (g["s"] == sol).all(1)                                     # SCORING
            log(f"   K1  one pass of {nm:5s} ({NETS[nm].params} parameters), own confidence ({rule:7s}): accuracy "
                f"{OUT[f'K1_{nm}_{rule}'].float().mean():.4f}; FLOPs per puzzle {(nfree * FL[nm]).float().mean():.3e}")
    # KT: one pass of M_B in the order of pi_trust^B; its own route density (for the T_B > 1/2 rule)
    if tB is not None:
        g = core.rollout(NETS["mB6"], tB, puz, flag=clue, greedy=True, record="full")
        OUT["KT"] = (g["s"] == sol).all(1); OUT["KT_logT"] = g["logp"]                          # SCORING (first)
        acc = g["logp"] > math.log(0.5)
        log(f"   KT  one pass of mB6 in the order of pi_trust^B: accuracy {OUT['KT'].float().mean():.4f}; "
            f"FLOPs per puzzle {(nfree * (FL['mB6'] + FL['trustB'])).float().mean():.3e}")
        log(f"   E4(e) rule 'T_B(x|c) > 1/2': an accepted answer exists for {acc.float().mean():.4f} of puzzles; "
            f"accepted answers that are wrong: {(acc & ~OUT['KT']).sum().item()} of {acc.sum().item()}")
        KTrec = g
    # K2: best of k passes of M_B by its own route log-probability
    gen = torch.Generator().manual_seed(555); KMAX = 32 if setname == "HARD" else 16
    g0 = core.rollout(NETS["mB6"], core.pol_margin, puz, flag=clue, greedy=True, record="full")
    rep = torch.arange(len(puz)).repeat_interleave(KMAX - 1)
    gs = core.rollout(NETS["mB6"], core.pol_margin, puz[rep], flag=clue[rep], gen=gen)
    Xc = torch.cat([g0["s"][:, None], gs["s"].view(len(puz), KMAX - 1, N)], 1)
    lq = torch.cat([g0["logp"][:, None], gs["logp"].view(len(puz), KMAX - 1)], 1)
    right = (Xc == sol[:, None]).all(2)                                                         # SCORING
    OUT["K2_k"] = [k for k in (1, 2, 4, 8, 16, 32) if k <= KMAX]; OUT["K2_acc"] = []
    for k in OUT["K2_k"]:
        pick = lq[:, :k].argmax(1); OUT["K2_acc"].append(right[torch.arange(len(puz)), pick])
        log(f"   K2  best of {k:2d} passes of mB6 by its own route log-probability: accuracy {OUT['K2_acc'][-1].float().mean():.4f}; "
            f"solution among the {k} candidates {right[:, :k].any(1).float().mean():.4f}; FLOPs {(k * nfree * FL['mB6']).float().mean():.3e}")
    OUT["K2_any"] = [right[:, :k].any(1) for k in OUT["K2_k"]]

    # E5 (found contexts): first wrong step of K1 (mB6, margin); what pi_trust^B picks at that state
    def first_error(rec, label, other, other_label):
        wrongcell = torch.gather(rec["s"], 1, rec["cells"].clamp(min=0)) != torch.gather(sol, 1, rec["cells"].clamp(min=0))
        wrongcell &= rec["cells"] >= 0
        fail = wrongcell.any(1); t = wrongcell.float().argmax(1); idx = torch.where(fail)[0]
        if len(idx) == 0:
            log(f"   E5 found contexts: {label} makes no error on {setname}"); return None
        st = rec["states"][idx, t[idx]]; at = rec["atoms"][idx, t[idx]]; emp = st < 0; ar = torch.arange(len(idx))
        cell = rec["cells"][idx, t[idx]]; truth = sol[idx]
        p_wrong = rec["lps"][idx, t[idx]].exp(); sur_true = -at[ar, cell, truth[ar, cell]]
        argmax_wrong = (at.argmax(-1) != truth) & emp                              # cells where M_B's top value is wrong
        cell2 = other(st, at, emp).masked_fill(~emp, float("inf")).argmin(1)
        ok2 = at[ar, cell2].argmax(-1) == truth[ar, cell2]; sur2 = -at[ar, cell2, truth[ar, cell2]]
        res = dict(n_fail=len(idx), n=len(puz), p_wrong_mean=float(p_wrong.mean()), p_wrong_median=float(p_wrong.median()),
                   sur_true_mean=float(sur_true.mean()), frac_cells_wrong=float((argmax_wrong.sum(1) / emp.sum(1)).mean()),
                   n_empty=float(emp.sum(1).float().mean()), other_same_cell=float((cell2 == cell).float().mean()),
                   other_ok=float(ok2.float().mean()), other_sur=float(sur2.mean()), other_ptop=float(at[ar, cell2].max(-1).values.exp().mean()))
        log(f"   E5 found contexts ({label}): fails on {len(idx)} of {len(puz)} puzzles. At the first wrong step: M_B's probability on the "
            f"wrong value mean {res['p_wrong_mean']:.3f} (median {res['p_wrong_median']:.3f}); surprise at the true value {res['sur_true_mean']:.2f} nats; "
            f"empty cells {res['n_empty']:.1f}, of which M_B's top value is wrong at {res['frac_cells_wrong']:.3f}")
        log(f"      at the same state {other_label} picks the same cell {res['other_same_cell']:.3f} of the time; M_B's top value at its cell is right "
            f"{res['other_ok']:.3f} of the time (top probability there {res['other_ptop']:.3f}; surprise at the truth {res['other_sur']:.2f} nats)")
        return res

    if tB is not None:
        OUT["E5_K1"] = first_error(g0, "K1: own confidence, margin", tB, "pi_trust^B")
        OUT["E5_KT"] = first_error(KTrec, "KT: trust order", core.pol_margin, "own confidence (margin)")
        both = dict(k1_only=int((OUT["K1_mB6_margin"] & ~OUT["KT"]).sum()), kt_only=int((OUT["KT"] & ~OUT["K1_mB6_margin"]).sum()))
        OUT["E5_disc"] = both
        log(f"   KT against K1 (margin): puzzles only K1 solves {both['k1_only']}, only KT solves {both['kt_only']}")
    torch.save(OUT, f"out/base_{setname}.pt")
    log(f"   total time {time.time()-T0:.0f}s")

elif mode == "multi":
    import oracle as O                                                           # SCORING only: the true solution sets
    d = np.load("data/k6_test_multi.npz")
    NP_ = int(sys.argv[2]) if len(sys.argv) > 2 else 100; NC = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    puz = torch.from_numpy(d["puz"][:NP_]); nsol = d["nsol"][:NP_]
    C6 = O.Counter(np.load("data/k6_base.npz")["B"], K)
    MA = NETS["mA6"]; checker = core.make_checker(MA, tA, TAU)
    log(f"== E4(d): {NP_} puzzles with 2-12 solutions; {NC} draws per search policy; weights T_A(x)/proposal(x)")
    EST = {}
    for arm in ("S1", "S2", "S4"):
        cfg = ARMS[arm]; gen = torch.Generator().manual_seed(1700 + int(arm[1]))
        rep = torch.arange(NP_).repeat_interleave(NC); fl = (puz >= 0)[rep] if cfg["flag"] else None
        o = core.rollout(NETS[cfg["prop"]], ORD[cfg["order"]], puz[rep], flag=fl, floor=cfg["floor"], gen=gen)
        acc, logT, used = checker(o["s"])
        w = torch.where(acc, (logT - o["logprop"]).double().exp(), torch.zeros(len(acc), dtype=torch.double))
        EST[arm] = (o["s"].view(NP_, NC, N), w.view(NP_, NC), acc.view(NP_, NC))
        log(f"   {arm}: draws accepted {acc.float().mean():.3f} ({time.time()-T0:.0f}s)")
    rows = {a: [] for a in EST}; pair = {}
    for p in range(NP_):
        sols = C6.solutions(puz[p].numpy()); assert len(sols) == nsol[p]                         # SCORING
        lt = core.route_logT(MA, tA, torch.from_numpy(sols))["logp"].double()
        exact = torch.softmax(lt, 0).numpy(); dist = {}
        for a, (X, w, acc) in EST.items():
            est = np.zeros(len(sols)); other = 0.0; wsum = float(w[p].sum())
            for j in range(NC):
                if w[p, j] > 0:
                    m = np.where((sols == X[p, j].numpy()).all(1))[0]
                    if len(m):
                        est[m[0]] += float(w[p, j])
                    else:
                        other += float(w[p, j])
            if wsum > 0:
                est /= wsum; other /= wsum
            dist[a] = est
            ess = float(w[p].sum() ** 2 / (w[p] ** 2).sum()) if wsum > 0 else 0.0
            rows[a].append((0.5 * (np.abs(est - exact).sum() + other), 0.5 * (np.abs(est - 1 / len(sols)).sum() + other),
                            float((est > 0).mean()), other, ess, len(sols)))
        for a in EST:
            for b in EST:
                if a < b:
                    pair.setdefault((a, b), []).append(0.5 * np.abs(dist[a] - dist[b]).sum())
        if p == 0:
            ex_unif = []
        ex_unif.append(0.5 * np.abs(exact - 1 / len(sols)).sum())
    log(f"   exact conditional of T_A on the true solution set against the uniform law over solutions: TV mean {np.mean(ex_unif):.4f}, median {np.median(ex_unif):.4f}, max {np.max(ex_unif):.4f}")
    res = dict(exact_vs_uniform=float(np.mean(ex_unif)))
    for a, r in rows.items():
        r = np.array(r)
        res[a] = dict(tv_exact_mean=float(r[:, 0].mean()), tv_exact_median=float(np.median(r[:, 0])), tv_uniform_mean=float(r[:, 1].mean()),
                      found=float(r[:, 2].mean()), mass_nonsolutions=float(r[:, 3].mean()), ess=float(r[:, 4].mean()))
        log(f"   {a}: TV(estimate, exact target) mean {r[:,0].mean():.4f} median {np.median(r[:,0]):.4f}; TV to uniform {r[:,1].mean():.4f}; "
            f"solutions found {r[:,2].mean():.3f}; weight on non-solutions {r[:,3].mean():.2e}; effective sample size {r[:,4].mean():.1f} of {NC}")
    for (a, b), v in pair.items():
        res[f"{a}-{b}"] = dict(mean=float(np.mean(v)), median=float(np.median(v)))
        log(f"   between {a} and {b}: TV mean {np.mean(v):.4f}, median {np.median(v):.4f}")
    json.dump(res, open("out/e_multi.json", "w"), indent=1)
    log(f"   total time {time.time()-T0:.0f}s")
