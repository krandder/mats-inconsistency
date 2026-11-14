"""Item 3 (P8 at N = 4096) jobs. python jobs3.py JOB SEED [REP]. Writes $P8_RES/<name>.json.
Sizes come from jobs.py (P8_N sets the train/price stream length; P8_TRUTH_TAG the truth-train draw).
  b2gate SEED            G1 baseline: C2's selected B2 (8-layer conv, h64, lr 1e-3) on the truth-train bank
  truthrep SEED REP      deployed truth replicate (P8 job_truth; REP changes only the training randomness)
  pricerep SEED REP      one paired price replicate (P8 price_rep, R7a group off)
  pricemerge SEED        mean over the KREP price replicates -> price_maze_s{SEED} (P8 job_price format)
  r7brep SEED REP        one R7b 80/20 replicate (P8 job_r7b loop body)
  r7bmerge SEED          -> r7b_maze_s{SEED} (P8 job_r7b format)"""
import glob

from jobs import *


def tseed(seed, rep):
    return seed + 1000 * rep


def job_b2gate(seed):
    t0 = cpu_secs()
    T, evals = truth_banks("maze", seed)
    r = list(run_b2(T, evals, seed, [(64, 1e-3)]).values())[0]
    dump(f"b2_s{seed}", dict(seed=seed, N=T.N, config="h64_lr0.001", B2=strip(r), cpu_secs=cpu_secs() - t0))


def job_truthrep(seed, rep):
    """P8 job_truth (maze) with every training seed replaced by tseed(seed, rep); banks depend on seed only.
    rep 0 reproduces P8's training randomness. Also stores held per-example bits for base|empty, empty:g2, empty:nz05."""
    kind = "maze"
    ts = tseed(seed, rep)
    t0 = cpu_secs()
    torch.manual_seed(seed_int("truth", kind, ts))
    X = xs(kind)
    ctx = ctx_of(kind)
    its = items(kind)
    bank, evals = truth_banks(kind, seed)
    work = 0.0
    work += crossfit(kind, bank, evals, [("ts:g2", X, "g2")] + [(f"tz:empty:{z}", X, z) for c, z in its if c == "empty"],
                     ts, "A")
    work += crossfit(kind, bank, evals, [("ts:g4", X + ["ts:g2"], "g4")], ts, "B")
    Shat = {"empty": [], "g2g4": ["ts:g2", "ts:g4"]}
    c2 = [c for c in ctx if c != "empty"][0]
    work += crossfit(kind, bank, evals, [(f"tz:{c2}:{z}", X + Shat[c2], z) for c, z in its if c == c2], ts, "C")
    inits = (0, 1)
    ymem = []
    for c in ctx:
        for it in inits:
            ymem.append((f"base|{c}|{it}", X + Shat[c], it))
    for c, z in its:
        for it in inits:
            ymem.append((f"{c}:{z}|{it}", X + Shat[c] + [f"tz:{c}:{z}"], it))
    cin = max(len(m[1]) for m in ymem)
    mY = new_model(kind, len(ymem), cin, [ts * 10 + m[2] for m in ymem])
    bY = member_builder(bank, [m[1] for m in ymem], ["Y"] * len(ymem), cin)
    resY, wY, _ = fit_eval(mY, bY, bank, evals, ts, "Y")
    cmem = [(c, it) for c in ctx for it in inits]
    cb = len(X) + max(len(S) for S in Shat.values())
    mC = GStack(len(cmem), cb, 1, "conv", init_seeds=[ts * 10 + it for _, it in cmem])
    bC = member_builder(bank, [X + Shat[c] for c, _ in cmem], ["Y"] * len(cmem), cb)
    resC, wC, _ = fit_eval(mC, bC, bank, evals, ts, "C")
    work += wY + wC
    yk = {m[0]: i for i, m in enumerate(ymem)}
    ck = {m: i for i, m in enumerate(cmem)}
    out = dict(kind=kind, seed=seed, rep=rep, tseed=ts, N=bank.N, items={}, controls={}, work=work)
    for e, en in enumerate(["held", "f15"]):
        bitsY, accY = resY[e]
        bitsC, accC = resC[e]
        for c in ctx:
            ctl = np.mean([bitsC[:, ck[(c, it)]] for it in inits], 0)
            base = np.mean([bitsY[:, yk[f"base|{c}|{it}"]] for it in inits], 0)
            out["controls"][f"{en}:{c}"] = dict(control_bits=ctl.mean(), base_bits=base.mean(),
                                                control_acc=np.mean([accC[:, ck[(c, it)]].mean() for it in inits]),
                                                base_acc=np.mean([accY[:, yk[f'base|{c}|{it}']].mean() for it in inits]))
        for c, z in its:
            ctl = np.mean([bitsC[:, ck[(c, it)]] for it in inits], 0)
            pipe = np.mean([bitsY[:, yk[f"{c}:{z}|{it}"]] for it in inits], 0)
            m, h = paired_ci(ctl - pipe)
            base = np.mean([bitsY[:, yk[f"base|{c}|{it}"]] for it in inits], 0)
            mb, hb = paired_ci(base - pipe)
            d = out["items"].setdefault(f"{c}:{z}", {})
            d[en] = dict(truth=m, ci95=h, cls=int(np.sign(m) if abs(m) > h else 0), gain_vs_base=mb, ci95_base=hb,
                         pipe_bits=pipe.mean(), acc=np.mean([accY[:, yk[f"{c}:{z}|{it}"]].mean() for it in inits]),
                         init_spread=float(np.std([bitsY[:, yk[f"{c}:{z}|{it}"]].mean() for it in inits])),
                         gain_by_init=[float((bitsY[:, yk[f"base|{c}|{it}"]] - bitsY[:, yk[f"{c}:{z}|{it}"]]).mean())
                                       for it in inits])
        if en == "held":
            out["held_per_example"] = {
                "base|empty": np.mean([bitsY[:, yk[f"base|empty|{it}"]] for it in inits], 0),
                "empty:g2": np.mean([bitsY[:, yk[f"empty:g2|{it}"]] for it in inits], 0),
                "empty:nz05": np.mean([bitsY[:, yk[f"empty:nz05|{it}"]] for it in inits], 0)}
    out["cpu_secs"] = cpu_secs() - t0
    dump(f"truth_maze_s{seed}_r{rep}", out)


def job_pricerep(seed, rep):
    kind = "maze"
    t0 = cpu_secs()
    torch.manual_seed(seed_int("price", kind, seed, rep))
    its = items(kind)
    bank = price_bank(kind, seed)
    M, pP, sc = price_rep(kind, seed, rep, bank, its, (), SANITY_MAZE)
    dump(f"pricerep_maze_s{seed}_r{rep}", dict(seed=seed, rep=rep, N=bank.N, F=pP.per_block[0][0], blocks=pP.blocks,
                                               members=M, secs=sc, cpu_secs=cpu_secs() - t0))


def job_pricemerge(seed):
    """P8 job_price aggregation over the KREP replicate files (all must exist)."""
    kind = "maze"
    reps = []
    for r in range(KREP):
        d = json.load(open(os.path.join(RES, f"pricerep_maze_s{seed}_r{r}.json")))
        reps.append({k: dict(v, per_block=np.asarray(v["per_block"])) for k, v in d["members"].items()})
        N, F, blocks = d["N"], d["F"], d["blocks"]
    its = items(kind)
    M = avg_members(reps)
    out = {"kind": kind, "seed": seed, "N": N, "F": F, "blocks": blocks, "krep": KREP, "members": M}
    prices = {}
    for c, z in its + [("empty", s) for s in SANITY_MAZE]:
        p = price_of(M, c, z, N)
        per = [price_of(Mr, c, z, N) for Mr in reps]
        for k in ("R1", "R4", "R5", "L_Z"):
            p[f"{k}_reps"] = [q[k] for q in per]
            p[f"{k}_sd"] = float(np.std(p[f"{k}_reps"], ddof=1)) if KREP > 1 else None
        prices[f"{c}:{z}"] = p
    out["prices"] = prices
    dump(f"price_maze_s{seed}", out)


def job_r7brep(seed, r):
    """One replicate of P8 job_r7b (maze): own 80/20 split, own init. Test-loss difference scaled by N."""
    kind = "maze"
    t0 = cpu_secs()
    X = xs(kind)
    ctx = ctx_of(kind)
    its = items(kind)
    hats = [f"hat:{c}:{z}" for c, z in its]
    bank = price_bank(kind, seed, hats)
    N = bank.N
    rs = rep_seed(seed, r)
    g = torch.Generator().manual_seed(seed_int("r7b", kind, seed) if r == 0 else seed_int("r7b", kind, seed, r))
    perm = torch.randperm(N, generator=g)
    ntr = int(0.8 * N)
    tr_all, te = perm[:ntr], perm[ntr:].to(DEV)
    nv = max(1, ntr // 10)
    va, tr = tr_all[:nv], tr_all[nv:]
    zmem = [(X + ctx[c], z) for c, z in its]
    cin = max(len(X) + len(S) + 1 for S in ctx.values())
    mZ = new_model(kind, len(zmem), cin, [rs] * len(zmem))
    bZ = member_builder(bank, [m[0] for m in zmem], [m[1] for m in zmem], cin)
    iz = fit(mZ, bZ, tr, va, g, CFG_FIT)
    allr = torch.arange(N, device=DEV)
    pr = predict(mZ, bZ, allr)
    for j, h in enumerate(hats):
        bank.set(h, allr, pr[:, j, 0])
    ymem = [(f"Y|{c}", X + S) for c, S in ctx.items()] + [(f"{c}:{z}", X + ctx[c] + [f"hat:{c}:{z}"]) for c, z in its]
    mY = new_model(kind, len(ymem), cin, [rs] * len(ymem))
    bY = member_builder(bank, [m[1] for m in ymem], ["Y"] * len(ymem), cin)
    iy = fit(mY, bY, tr, va, g, CFG_FIT)
    te_loss = (eval_loss(mY, bY, te) / LN2).mean(0).cpu().numpy()
    k = {m[0]: i for i, m in enumerate(ymem)}
    res = {f"{c}:{z}": [float((te_loss[k[f"Y|{c}"]] - te_loss[k[f"{c}:{z}"]]) * N), float(te_loss[k[f"{c}:{z}"]])]
           for c, z in its}
    work = (iz["stop"].sum() * mZ.macs() + iy["stop"].sum() * mY.macs()) * CFG_FIT["batch"] * bank.cells
    dump(f"r7brep_maze_s{seed}_r{r}", dict(seed=seed, rep=r, N=N, res=res, work=work, cpu_secs=cpu_secs() - t0))


def job_r7bmerge(seed):
    reps = [json.load(open(os.path.join(RES, f"r7brep_maze_s{seed}_r{r}.json")))["res"] for r in range(KREP)]
    prices = {}
    for it in reps[0]:
        v = [rp[it][0] for rp in reps]
        prices[it] = dict(R7b=float(np.mean(v)), R7b_reps=v, te_bits_per_ex=float(np.mean([rp[it][1] for rp in reps])))
    dump(f"r7b_maze_s{seed}", dict(kind="maze", seed=seed, krep=KREP, prices=prices))


JOBS3 = {"b2gate": job_b2gate, "truthrep": job_truthrep, "pricerep": job_pricerep, "pricemerge": job_pricemerge,
         "r7brep": job_r7brep, "r7bmerge": job_r7bmerge}

if __name__ == "__main__":
    name, args = sys.argv[1], [int(a) for a in sys.argv[2:]]
    t = time.time()
    JOBS3[name](*args)
    print(f"done {name} {args} {time.time() - t:.0f}s", flush=True)
