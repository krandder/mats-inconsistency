"""P8 jobs. python jobs.py JOB SEED [extra]. Each job writes results/<job>_s<seed>.json.
Codelengths are prequential bits over the train stream (mazes N=1024, 11x11; parity N=4096); see p8lib.Preq."""
import itertools
import sys

from p8lib import *

torch.set_num_threads(int(os.environ.get("P8_THREADS", "1")))

N_TR = 64 if SMOKE else 1024
N_HELD = 64 if SMOKE else 1024
N_15 = 32 if SMOKE else 512
N_PAR = 128 if SMOKE else 4096
XM = ["wall", "start", "inside", "decoy"]
CTX_M = {"empty": [], "g2g4": ["g2", "g4"]}
CTX_P = {"empty": [], "p1234": ["p1", "p2", "p3", "p4"]}
CFG_FIT = dict(lr=3e-3, batch=32, max_steps=6000, patience=600, eval_every=40, hid=16)
if SMOKE:
    CFG_FIT.update(max_steps=30, patience=20, eval_every=10)


def items(kind):
    ctxs, pool = (CTX_M, POOL_MAZE) if kind == "maze" else (CTX_P, POOL_PAR)
    return [(c, z) for c in ctxs for z in pool if z not in ctxs[c]]


def xs(kind):
    return XM if kind == "maze" else X_PAR


def ctx_of(kind):
    return CTX_M if kind == "maze" else CTX_P


def cells_of(bank):
    return bank.cells


def new_model(kind, G, cin, seeds=None, layers=2, hid=None):
    if kind == "maze":
        return GConv(G, cin, hid or CFG["hid"], layers, 1, seeds)
    return GMLP(G, cin, hid or 128, layers, 1, seeds)


def price_bank(kind, seed, hats=()):
    if kind == "maze":
        return Bank(maze_data(N_TR, N_MAZE, seed_int("maze-price", seed)), "maze", hats)
    return Bank(parity_data(N_PAR, seed, "price"), "par", hats)


def truth_banks(kind, seed, hats=()):
    if kind == "maze":
        tr = Bank(maze_data(N_TR, N_MAZE, seed_int("maze-truth", seed)), "maze", hats)
        he = Bank(maze_data(N_HELD, N_MAZE, seed_int("maze-held", seed)), "maze", hats)
        f15 = Bank(maze_data(N_15, 15, seed_int("maze-15", seed)), "maze", hats)
        return tr, [he, f15]
    tr = Bank(parity_data(N_PAR, seed, "truth"), "par", hats)
    he = Bank(parity_data(N_PAR, seed, "held"), "par", hats)
    return tr, [he]


# ============================================================ (a) pricing: R1, R2, R4, R5, R7a in one paired run
KREP = int(os.environ.get("P8_KREP", "1" if SMOKE else "12"))  # replicates per price (Phase 0 gate, see RESULTS.md)
KREP_T = int(os.environ.get("P8_KREP_TOWER", "1" if SMOKE else "4"))  # replicates per tower step (greedy/analysis)


def rep_tag(r):
    return "P" if r == 0 else f"P{r}"


def rep_seed(seed, r):
    return seed + 1000 * r


def avg_members(reps):
    """reps: list of {key: dict(bits, per_block, lbm, work, stop)} -> mean bits/per_block/lbm, summed work."""
    out = {}
    for k in reps[0]:
        v = [r[k] for r in reps]
        out[k] = dict(bits=float(np.mean([x["bits"] for x in v])), bits_reps=[x["bits"] for x in v],
                      per_block=np.mean([x["per_block"] for x in v], 0), lbm=float(np.mean([x["lbm"] for x in v])),
                      work=float(np.sum([x["work"] for x in v])), stop=[x["stop"] for x in v])
    return out


def price_rep(kind, seed, r, bank, its, hats, sanity):
    """One paired replicate (own init and data order): P group (Y|ctx, Z|ctx:z, YZ|ctx:z) + R7a group."""
    X = xs(kind)
    ctx = ctx_of(kind)
    mem = []  # (key, inputs, target)
    for c, S in ctx.items():
        mem.append((f"Y|{c}", X + S, "Y"))
    for c, z in its + [("empty", s) for s in sanity]:
        S = ctx[c]
        mem.append((f"Z|{c}:{z}", X + S, z))
        mem.append((f"YZ|{c}:{z}", X + S + [z], "Y"))
    cin = max(len(m[1]) for m in mem)
    G = len(mem)
    rs = rep_seed(seed, r)
    mP = new_model(kind, G, cin, [rs] * G)
    bP = member_builder(bank, [m[1] for m in mem], [m[2] for m in mem], cin)
    rmem = [(f"R7a|{c}:{z}", X + ctx[c] + [f"hat:{c}:{z}"], "Y") for c, z in its]
    mR = new_model(kind, len(rmem), cin, [rs] * len(rmem))
    bR = member_builder(bank, [m[1] for m in rmem], [m[2] for m in rmem], cin)
    N = bank.N
    pP = Preq(mP, bP, N, seed, rep_tag(r), bank.cells)
    pR = Preq(mR, bR, N, seed, rep_tag(r), bank.cells)  # same data order as the P group (paired)
    key = {m[0]: i for i, m in enumerate(mem)}
    zi = torch.tensor([key[f"Z|{c}:{z}"] for c, z in its], device=DEV)
    for i in range(pP.n_stages):
        pP.train(i)
        pP.code(i)
        a, b = pP.ab(i)
        rows = torch.arange(a, b, device=DEV)
        pr = predict(mP, bP, rows)[:, zi, 0]
        for j, h in enumerate(hats):
            bank.set(h, rows, pr[:, j])
        pR.train(i)
        pR.code(i)
    rP, rR = pP.result(), pR.result()
    M = {}
    for kk, (res, keys) in (("P", (rP, key)), ("R", (rR, {m[0]: i for i, m in enumerate(rmem)}))):
        for k, i in keys.items():
            M[k] = dict(bits=res["bits"][i], per_block=res["per_block"][i], lbm=res["last_block_mean"][i],
                        work=res["work"][i], stop=res["stop_steps"][i])
    return M, pP, rP["secs"]


def price_of(M, c, z, N):
    LY, LZ, LYZ = M[f"Y|{c}"]["bits"], M[f"Z|{c}:{z}"]["bits"], M[f"YZ|{c}:{z}"]["bits"]
    lY, lZ, lYZ = (M[f"Y|{c}"]["lbm"] * N, M[f"Z|{c}:{z}"]["lbm"] * N, M[f"YZ|{c}:{z}"]["lbm"] * N)
    p = dict(L_Y=LY, L_Z=LZ, L_YZ=LYZ, R1=LY - LYZ, R2=-LZ, R4=LY - LYZ - LZ, R5=lY - lYZ - lZ, R3=lY - lYZ,
             work=M[f"Y|{c}"]["work"] + M[f"Z|{c}:{z}"]["work"] + M[f"YZ|{c}:{z}"]["work"])
    p["EDL"] = p["R4"] - p["R5"]
    p["curve_R4"] = (np.array(M[f"Y|{c}"]["per_block"]) - np.array(M[f"YZ|{c}:{z}"]["per_block"])
                     - np.array(M[f"Z|{c}:{z}"]["per_block"]))
    if f"R7a|{c}:{z}" in M:
        p["R7a"] = LY - M[f"R7a|{c}:{z}"]["bits"]
        p["work_R7a"] = M[f"Y|{c}"]["work"] + M[f"Z|{c}:{z}"]["work"] + M[f"R7a|{c}:{z}"]["work"]
    return p


def job_price(kind, seed):
    """R1-R5, EDL, R7a for every item: mean over KREP paired replicates (each: own init + own data order)."""
    t0 = cpu_secs()
    torch.manual_seed(seed_int("price", kind, seed))
    its = items(kind)
    hats = [f"hat:{c}:{z}" for c, z in its]
    bank = price_bank(kind, seed, hats)
    sanity = SANITY_MAZE if kind == "maze" else SANITY_PAR
    N = bank.N
    reps, secs = [], 0.0
    for r in range(KREP):
        M, pP, sc = price_rep(kind, seed, r, bank, its, hats, sanity)
        reps.append(M)
        secs += sc
    M = avg_members(reps)
    out = {"kind": kind, "seed": seed, "N": N, "F": pP.per_block[0][0], "blocks": pP.blocks, "krep": KREP,
           "members": M}
    prices = {}
    for c, z in its + [("empty", s) for s in sanity]:
        p = price_of(M, c, z, N)
        per = [price_of(Mr, c, z, N) for Mr in reps]
        for k in ("R1", "R4", "R5", "R7a"):
            if k in p:
                p[f"{k}_reps"] = [q[k] for q in per]
                p[f"{k}_sd"] = float(np.std(p[f"{k}_reps"], ddof=1)) if KREP > 1 else None
        prices[f"{c}:{z}"] = p
    out["prices"] = prices
    out["secs"] = secs
    out["cpu_secs"] = cpu_secs() - t0
    dump(f"price_{kind}_s{seed}", out)


# ============================================================ capable reference L_cap(Y|X)
def job_capref(kind, seed):
    t0 = cpu_secs()
    bank = price_bank(kind, seed)
    X = xs(kind)
    b1 = member_builder(bank, [X], ["Y"], len(X))
    res = {}
    if kind == "maze":
        arms = [("conv2_h16", lambda: GConv(1, 4, 16, 2, 1, [seed])), ("conv4_h16", lambda: GConv(1, 4, 16, 4, 1, [seed])),
                ("conv8_h16", lambda: GConv(1, 4, 16, 8, 1, [seed])), ("conv8_h32", lambda: GConv(1, 4, 32, 8, 1, [seed])),
                ("dtr_w16_i10", lambda: DTR(4, 16, 10, 0.5, seed))]
    else:
        arms = [("mlp2_h128", lambda: GMLP(1, 32, 128, 2, 1, [seed])), ("mlp6_h128", lambda: GMLP(1, 32, 128, 6, 1, [seed])),
                ("mlp2_h512", lambda: GMLP(1, 32, 512, 2, 1, [seed]))]
    for name, mk in arms:
        torch.manual_seed(seed_int("capref", name, seed))
        r = prequential(mk(), b1, bank.N, seed, "P", bank.cells)
        res[name] = dict(bits=r["bits"][0], per_block=r["per_block"][0], secs=r["secs"], work=r["work"][0])
    best = min(v["bits"] for v in res.values())
    dump(f"capref_{kind}_s{seed}", dict(kind=kind, seed=seed, arms=res, L_cap=best + 2, cpu_secs=cpu_secs() - t0))


# ============================================================ R7b wrapper validation (80/20)
def job_r7b(kind, seed):
    """R7b: 80/20 wrapper validation; mean over KREP replicates (own split, own init)."""
    t0 = cpu_secs()
    X = xs(kind)
    ctx = ctx_of(kind)
    its = items(kind)
    hats = [f"hat:{c}:{z}" for c, z in its]
    bank = price_bank(kind, seed, hats)
    N = bank.N
    reps, work = [], 0.0
    for r in range(KREP):
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
        reps.append({f"{c}:{z}": ((te_loss[k[f"Y|{c}"]] - te_loss[k[f"{c}:{z}"]]) * N, te_loss[k[f"{c}:{z}"]])
                     for c, z in its})
        work += (iz["stop"].sum() * mZ.macs() + iy["stop"].sum() * mY.macs()) * CFG_FIT["batch"] * bank.cells
    prices = {}
    for c, z in its:
        v = [rp[f"{c}:{z}"][0] for rp in reps]
        prices[f"{c}:{z}"] = dict(R7b=float(np.mean(v)), R7b_reps=v,
                                  te_bits_per_ex=float(np.mean([rp[f"{c}:{z}"][1] for rp in reps])))
    dump(f"r7b_{kind}_s{seed}", dict(kind=kind, seed=seed, krep=KREP, prices=prices, work_total=work,
                                     cpu_secs=cpu_secs() - t0))


# ============================================================ (c) R6 masked-reveal surrogate (+ R6b, T5 on parity)
def r6_builders(bank, kind, X, cands, p=0.25, void=False, seed=0):
    """Masked-reveal market. Train: each candidate revealed w.p. p. void=True (R6b, parity): the revealed value is a
    random proposal z' ~ Bern(1/2); the example counts only if every revealed proposal matches the truth.
    Scoring (st['mask'] set): true values revealed (model-based price) unless st['zp'] gives literal proposals."""
    xi = torch.tensor([bank.idx[c] for c in X], device=DEV)
    ci = torch.tensor([bank.idx[c] for c in cands], device=DEV)
    yi = bank.idx["Y"]
    K = len(cands)
    gfix = torch.Generator(device=DEV).manual_seed(seed_int("r6fix", seed))
    fixed = torch.rand((bank.N, K), generator=gfix, device=DEV) < p
    zfix = (torch.rand((bank.N, K), generator=gfix, device=DEV) < 0.5).float()
    st = {"mask": None, "zp": None}

    def feed(bi, bk, mode):
        bk = bk or bank
        C = bk.t[bi][:, ci]
        match = None
        if st["mask"] is not None:  # scoring under a fixed reveal set
            r = st["mask"].expand(len(bi), K)
            if st["zp"] is not None:
                zval = st["zp"][bi].float()
                match = ((zval == C) | ~r).all(1).float()
            else:
                zval = C
        else:
            r = (torch.rand((len(bi), K), device=DEV) < p) if mode == "train" else fixed[bi]
            zval = C
            if void:
                zval = (torch.rand((len(bi), K), device=DEV) < 0.5).float() if mode == "train" else zfix[bi]
                match = ((zval == C) | ~r).all(1).float()
        if kind == "maze":
            rf = r.float()[:, :, None, None]
            val, fl = zval * rf, rf.expand_as(zval)
        else:
            rf = r.float()
            val, fl = (2 * zval - 1) * rf, rf
        return torch.cat([bk.inp[bi][:, xi], val, fl], 1), C, r, match

    def build_y(bi, bk=None):
        xin, C, r, match = feed(bi, bk, getattr(build_y, "mode", "eval"))
        tgt = (bk or bank).t[bi][:, [yi]].unsqueeze(1)
        return xin, tgt, (None if match is None else match.view(-1, 1, 1))

    def build_z(bi, bk=None):
        xin, C, r, match = feed(bi, bk, getattr(build_z, "mode", "eval"))
        w = (~r).float()
        if match is not None:
            w = w * match[:, None]
        return xin, C.unsqueeze(1), w.unsqueeze(1)

    return build_y, build_z, st


def r6_score(model, build, st, rows, mask_vec, zp=None, zchan=None):
    """bits per example (len(rows),) for Y (zchan None) or per-candidate (len(rows),K) for Z under a fixed mask."""
    st["mask"] = mask_vec.view(1, -1)
    st["zp"] = zp
    xin, tgt, w = build(rows)
    lg = model.logits(xin)
    l = F.binary_cross_entropy_with_logits(lg, tgt.expand_as(lg), reduction="none")
    l = l.reshape(*l.shape[:3], -1).sum(-1)[:, 0] / LN2  # (B, out)
    st["mask"] = None
    st["zp"] = None
    return l


def job_r6(kind, seed, void=False):
    t0 = cpu_secs()
    torch.manual_seed(seed_int("r6", kind, seed, void))
    X = xs(kind)
    ctx = ctx_of(kind)
    its = items(kind)
    cands = POOL_MAZE if kind == "maze" else POOL_PAR
    K = len(cands)
    ci = {c: j for j, c in enumerate(cands)}
    bank = price_bank(kind, seed)
    N = bank.N
    by, bz, st = r6_builders(bank, kind, X, cands, void=void, seed=seed)
    cin = len(X) + 2 * K
    if kind == "maze":
        mY, mZ = GConv(1, cin, 32, 2, 1, [seed]), GConv(1, cin, 32, 2, K, [seed])
    else:
        mY, mZ = GMLP(1, cin, 256, 2, 1, [seed]), GMLP(1, cin, 256, 2, K, [seed])
    pY = Preq(mY, by, N, seed, "R6Y", bank.cells)
    pZ = Preq(mZ, bz, N, seed, "R6Z", bank.cells, uniform=pY.blocks[0] * bank.cells)
    ymasks = {}
    for c, S in ctx.items():
        ymasks[c] = S
        for z in cands:
            if z not in S:
                ymasks[f"{c}+{z}"] = S + [z]
    zmasks = {c: S for c, S in ctx.items()}

    def mvec(S):
        v = torch.zeros(K, dtype=torch.bool, device=DEV)
        for s in S:
            v[ci[s]] = True
        return v

    blkY = {k: [] for k in ymasks}
    blkZ = {k: [] for k in zmasks}
    lit = {k: [] for k in ymasks}  # R6b literal (void) scoring
    nmatch = {k: [] for k in ymasks}
    for i in range(pY.n_stages):
        pY.train(i)
        pZ.train(i)
        a, b = pY.ab(i)
        rows = torch.arange(a, b, device=DEV)
        with torch.no_grad():
            for k, S in ymasks.items():
                l = r6_score(mY, by, st, rows, mvec(S))[:, 0]
                blkY[k].append(float(l.sum()))
                if void:
                    gz = torch.Generator(device=DEV).manual_seed(seed_int("lit", seed, i, k))
                    zp = torch.rand((N, K), generator=gz, device=DEV) < 0.5
                    C = bank.t[rows][:, [bank.idx[c] for c in cands]]
                    m = mvec(S)
                    match = ((zp[rows].float() == C) | ~m).all(1).float()
                    lv = r6_score(mY, by, st, rows, m, zp=zp)[:, 0]
                    lit[k].append(float((lv * match).sum() * 2 ** len(S)))
                    nmatch[k].append(float(match.sum()))
            for k, S in zmasks.items():
                l = r6_score(mZ, bz, st, rows, mvec(S))
                blkZ[k].append(l.sum(0).cpu().numpy())
        pY.code(i)
        pZ.code(i)
    rY, rZ = pY.result(), pZ.result()
    F0 = pY.per_block[0][0]
    cum_work = rY["work_cum"][:, 0] + rZ["work_cum"][:, 0]
    prices, curves = {}, {}
    for c, z in its:
        S = ctx[c]
        yS = np.cumsum(blkY[c])
        yZ = np.cumsum(blkY[f"{c}+{z}"])
        lz = np.cumsum([bz_[ci[z]] for bz_ in blkZ[c]]) + F0
        cur = yS - yZ - lz
        prices[f"{c}:{z}"] = cur[-1]
        curves[f"{c}:{z}"] = cur
    out = dict(kind=kind, seed=seed, void=void, prices=prices, curves=curves, cum_work=cum_work, blocks=pY.blocks,
               last_block_Y={k: v[-1] / (pY.blocks[-1] - pY.blocks[-2]) for k, v in blkY.items()},
               last_block_Z={k: (v[-1] / (pY.blocks[-1] - pY.blocks[-2])) for k, v in blkZ.items()},
               cpu_secs=cpu_secs() - t0, secs=rY["secs"])
    if void:
        lp = {}
        for c, z in its:
            yS = np.array(lit[c])
            yZ = np.array(lit[f"{c}+{z}"])
            lz = np.array([bz_[ci[z]] for bz_ in blkZ[c]])
            lp[f"{c}:{z}"] = float((yS - yZ - lz).sum() - F0)
        out["prices_literal"] = lp
        out["n_matched"] = {k: float(np.sum(v)) for k, v in nmatch.items()}
        out["n_scored"] = float(N - pY.blocks[0])
    if kind == "par" and not void:
        out["t5"] = t5_search(mY, mZ, by, bz, st, bank, cands, seed)
        out["t5"]["r6_work"] = cum_work[-1]
    dump(f"r6{'b' if void else ''}_{kind}_s{seed}", out)


# ------------------------------------------------------------ (d) T5 backward search on parity
def t5_search(mY, mZ, by, bz, st, bank, cands, seed, depth=3, beam=3, top=3):
    N = bank.N
    K = len(cands)
    a = blocks_for(N)[-2]
    rows = torch.arange(a, N, device=DEV)
    sets = [frozenset()] + [frozenset([j]) for j in range(K)] + [frozenset(p) for p in itertools.combinations(range(K), 2)]
    sY, sZ = {}, {j: {} for j in range(K)}
    with torch.no_grad():
        for P in sets:
            v = torch.zeros(K, dtype=torch.bool, device=DEV)
            for j in P:
                v[j] = True
            sY[P] = float(r6_score(mY, by, st, rows, v)[:, 0].sum())
            lz = r6_score(mZ, bz, st, rows, v).sum(0).cpu().numpy()
            for j in range(K):
                if j not in P:
                    sZ[j][P] = float(lz[j])

    def score(node, P):
        return (sY if node == "Y" else sZ[node])[P] + 1.0 * len(P)

    def desc(par, node):  # nodes that have `node` as an ancestor (incl. Y chain)
        out, stack = set(), [node]
        while stack:
            n = stack.pop()
            for ch, ps in par.items():
                if n in ps and ch not in out:
                    out.add(ch)
                    stack.append(ch)
        return out

    def props(par, node):
        ds = desc(par, node)
        opts = [(score(node, P), P) for P in (sY if node == "Y" else sZ[node]) if not (P & ds)]
        return sorted(opts, key=lambda t: t[0])[:top]

    def acyclic(par):
        seen, onst = set(), set()

        def dfs(n):
            if n in onst:
                return False
            if n in seen:
                return True
            onst.add(n)
            ok = all(dfs(q) for q in par.get(n, ()))
            onst.discard(n)
            seen.add(n)
            return ok
        return all(dfs(n) for n in list(par))

    def dag_score(par):
        tot = score("Y", par["Y"])
        nodes = set().union(*par.values())
        for n in nodes:
            tot += score(n, par.get(n, frozenset()))
        return tot

    beam_dags = [({"Y": P}, [j for j in P]) for _, P in props({}, "Y")]
    trace = [[sorted(cands[j] for j in d[0]["Y"]) for d in beam_dags]]
    for lev in range(2, depth + 1):
        cand_dags = []
        for par, front in beam_dags:
            if not front:
                cand_dags.append((par, []))
                continue
            options = []
            for n in front:
                options.append([(n, P) for _, P in props(par, n)])
            for combo in itertools.product(*options):
                p2 = dict(par)
                newf = []
                for n, P in combo:
                    p2[n] = P
                if not acyclic(p2):
                    continue
                for n, P in combo:
                    for j in P:
                        if j not in p2 and j not in newf:
                            newf.append(j)
                cand_dags.append((p2, newf))
        uniq = {}
        for par, fr in cand_dags:
            k = tuple(sorted((str(n), tuple(sorted(P))) for n, P in par.items()))
            uniq[k] = (par, fr)
        beam_dags = sorted(uniq.values(), key=lambda t: dag_score(t[0]))[:beam]
        trace.append([{str(cands[n]) if n != "Y" else "Y": sorted(cands[j] for j in P) for n, P in d[0].items()}
                      for d in beam_dags])

    def named(par):
        full = dict(par)
        for n in set().union(*par.values()):
            full.setdefault(n, frozenset())
        return {("Y" if n == "Y" else cands[n]): sorted(cands[j] for j in P) for n, P in full.items()}

    finals = [named(d[0]) for d in beam_dags]
    truth = {"Y": ["q1", "q2"], "q1": ["p1", "p2"], "q2": ["p3", "p4"], "p1": [], "p2": [], "p3": [], "p4": []}
    ex = exact_dags(bank, finals + [truth, {"Y": []}], seed)
    best = int(np.argmin([e["L_S"] for e in ex["dags"][:len(finals)]]))
    lv1 = sorted(finals[best]["Y"])
    rec = lambda d: (sorted(d.get("Y", [])) == ["q1", "q2"] and sorted(d.get("q1", [])) == ["p1", "p2"]
                     and sorted(d.get("q2", [])) == ["p3", "p4"])
    return dict(trace=trace, finals=finals, exact=ex, best=best, best_dag=finals[best], recovered=rec(finals[best]),
                level1=lv1, r6_rank_Y=sorted(((sY[P], sorted(cands[j] for j in P)) for P in sY), key=lambda t: t[0])[:10])


def exact_dags(bank, dags, seed):
    """Exact prequential L_S = sum over nodes of L(node | X, parents) for each DAG (one grouped run)."""
    X = X_PAR
    nodes = {}
    for d in dags:
        for n, P in d.items():
            nodes[(n, tuple(sorted(P)))] = None
    keys = list(nodes)
    cin = len(X) + max(len(P) for _, P in keys)
    m = GMLP(len(keys), cin, 128, 2, 1, [seed] * len(keys))
    b = member_builder(bank, [X + list(P) for _, P in keys], [n for n, _ in keys], cin)
    r = prequential(m, b, bank.N, seed, "P", bank.cells)
    bits = {k: r["bits"][i] for i, k in enumerate(keys)}
    out = []
    for d in dags:
        out.append(dict(dag=d, L_S=sum(bits[(n, tuple(sorted(P)))] for n, P in d.items())))
    return dict(dags=out, work=r["work"].sum(), node_bits={f"{n}|{','.join(P)}": v for (n, P), v in bits.items()})


def job_bf(seed):
    """BF: exact L(Y|X, pair) for all 120 pairs of the 16 parity items (pool + copyY + random)."""
    t0 = cpu_secs()
    bank = price_bank("par", seed)
    its = POOL_PAR + SANITY_PAR
    pairs = list(itertools.combinations(its, 2))
    mem = [X_PAR] + [X_PAR + list(p) for p in pairs]
    cin = len(X_PAR) + 2
    m = GMLP(len(mem), cin, 128, 2, 1, [seed] * len(mem))
    b = member_builder(bank, mem, ["Y"] * len(mem), cin)
    r = prequential(m, b, bank.N, seed, "P", bank.cells)
    res = {",".join(p): r["bits"][i + 1] for i, p in enumerate(pairs)}
    order = sorted(res, key=res.get)
    nos = [k for k in order if not any(s in k.split(",") for s in SANITY_PAR)]
    dump(f"bf_par_s{seed}", dict(seed=seed, L_Y=r["bits"][0], pairs=res, best=order[0], best_no_sanity=nos[0],
                                 rank_q1q2=order.index("q1,q2"), work=r["work"].sum(), cpu_secs=cpu_secs() - t0))


# ============================================================ deployed truth (cross-fitted concept-bottleneck pipelines)
def fold_of(N, seed, tag):
    perm = np.random.default_rng(seed_int("fold", seed, tag)).permutation(N)
    f = np.zeros(N, np.int64)
    f[perm[N // 2:]] = 1
    return torch.tensor(f, device=DEV)


def ensure_hats(banks, names):
    for n in names:
        for bk in banks:
            if n not in bk.idx:
                bk.add(n, torch.full((bk.N, *bk.sp), 0.5))


def crossfit(kind, bank, evals, specs, seed, tag, cfg=None):
    """specs: [(hat, inputs, target)]. 2 fold members per spec; out-of-fold predictions -> bank[hat];
    evals[hat] = mean of the two fold members. Returns training MACs."""
    cfg = cfg or CFG_FIT
    ensure_hats([bank] + evals, [s[0] for s in specs])
    fold = fold_of(bank.N, seed, tag)
    ins, tg, W = [], [], []
    for h, inp, t in specs:
        for f in (0, 1):
            ins.append(inp)
            tg.append(t)
            W.append((fold == f).float())
    G = len(ins)
    cin = max(len(l) for l in ins)
    m = new_model(kind, G, cin, [seed] * G)
    b = member_builder(bank, ins, tg, cin, torch.stack(W))
    g = torch.Generator().manual_seed(seed_int(seed, tag, "fit"))
    perm = torch.randperm(bank.N, generator=g)
    nv = max(2, bank.N // 10)
    info = fit(m, b, perm[nv:], perm[:nv], g, cfg)
    allr = torch.arange(bank.N, device=DEV)
    pr = predict(m, b, allr)
    fb = (fold == 1).view(-1, *([1] * len(bank.sp)))
    for s, (h, _, _) in enumerate(specs):
        bank.set(h, allr, torch.where(fb, pr[:, 2 * s, 0], pr[:, 2 * s + 1, 0]))
    for ev in evals:
        er = torch.arange(ev.N, device=DEV)
        pe = predict(m, b, er, bk=ev)
        for s, (h, _, _) in enumerate(specs):
            ev.set(h, er, 0.5 * (pe[:, 2 * s, 0] + pe[:, 2 * s + 1, 0]))
    return float((info["stop"] * cfg["batch"] * bank.cells * m.macs()).sum())


def fit_eval(model, build, bank, evals, seed, tag, cfg=None):
    """Fit on all train rows (10% val), return per-eval-bank per-example bits (N_e, G) and exact acc (N_e, G)."""
    cfg = cfg or CFG_FIT
    g = torch.Generator().manual_seed(seed_int(seed, tag, "fit"))
    perm = torch.randperm(bank.N, generator=g)
    nv = max(2, bank.N // 10)
    info = fit(model, build, perm[nv:], perm[:nv], g, cfg)
    res = []
    for ev in evals:
        er = torch.arange(ev.N, device=DEV)
        bits = eval_loss(model, build, er, bk=ev) / LN2
        acc = exact_acc(predict(model, build, er, bk=ev), ev.t[er][:, ev.idx["Y"]])
        res.append((bits.cpu().numpy(), acc.float().cpu().numpy()))
    work = float((info["stop"] * cfg["batch"] * bank.cells * model.macs()).sum())
    return res, work, info


def paired_ci(d):
    d = np.asarray(d, float)
    m = float(d.mean())
    h = 1.96 * float(d.std(ddof=1)) / math.sqrt(len(d))
    return m, h


def job_truth(kind, seed):
    t0 = cpu_secs()
    torch.manual_seed(seed_int("truth", kind, seed))
    X = xs(kind)
    ctx = ctx_of(kind)
    its = items(kind)
    bank, evals = truth_banks(kind, seed)
    work = 0.0
    if kind == "maze":
        # Shat chain: g2 | X ; g4 | X, g2hat. Zhat | X, Shat for each item.
        work += crossfit(kind, bank, evals, [("ts:g2", X, "g2")] + [(f"tz:empty:{z}", X, z) for c, z in its if c == "empty"],
                         seed, "A")
        work += crossfit(kind, bank, evals, [("ts:g4", X + ["ts:g2"], "g4")], seed, "B")
        Shat = {"empty": [], "g2g4": ["ts:g2", "ts:g4"]}
    else:
        work += crossfit(kind, bank, evals, [(f"tz:empty:{z}", X, z) for c, z in its if c == "empty"], seed, "A")
        Shat = {"empty": [], "p1234": [f"tz:empty:{p}" for p in ("p1", "p2", "p3", "p4")]}
    c2 = [c for c in ctx if c != "empty"][0]
    work += crossfit(kind, bank, evals, [(f"tz:{c2}:{z}", X + Shat[c2], z) for c, z in its if c == c2], seed, "C")
    inits = (0, 1)
    ymem = []
    for c in ctx:
        for it in inits:
            ymem.append((f"base|{c}|{it}", X + Shat[c], it))
    for c, z in its:
        for it in inits:
            ymem.append((f"{c}:{z}|{it}", X + Shat[c] + [f"tz:{c}:{z}"], it))
    cin = max(len(m[1]) for m in ymem)
    mY = new_model(kind, len(ymem), cin, [seed * 10 + m[2] for m in ymem])
    bY = member_builder(bank, [m[1] for m in ymem], ["Y"] * len(ymem), cin)
    resY, wY, _ = fit_eval(mY, bY, bank, evals, seed, "Y")
    cmem = [(c, it) for c in ctx for it in inits]
    cb = len(X) + max(len(S) for S in Shat.values())
    mC = GStack(len(cmem), cb, 1, "conv" if kind == "maze" else "mlp", init_seeds=[seed * 10 + it for _, it in cmem])
    bC = member_builder(bank, [X + Shat[c] for c, _ in cmem], ["Y"] * len(cmem), cb)
    resC, wC, _ = fit_eval(mC, bC, bank, evals, seed, "C")
    work += wY + wC
    yk = {m[0]: i for i, m in enumerate(ymem)}
    ck = {m: i for i, m in enumerate(cmem)}
    names = ["held", "f15"] if kind == "maze" else ["held"]
    out = dict(kind=kind, seed=seed, items={}, controls={}, work=work)
    for e, en in enumerate(names):
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
                         init_spread=float(np.std([bitsY[:, yk[f"{c}:{z}|{it}"]].mean() for it in inits])))
    out["cpu_secs"] = cpu_secs() - t0
    dump(f"truth_{kind}_s{seed}", out)


# ============================================================ Phase 0: noise floor, schedule sensitivity, parity calibration
def job_noise(kind, seed, order):
    t0 = cpu_secs()
    bank = price_bank(kind, seed)
    X = xs(kind)
    if kind == "maze":
        types = [("L(Y|X)", X, "Y"), ("L(g2|X)", X, "g2"), ("L(Y|X,g2)", X + ["g2"], "Y"), ("L(g6|X)", X, "g6")]
    else:
        types = [("L(Y|X)", X, "Y"), ("L(p1|X)", X, "p1"), ("L(q1|X,p1,p2)", X + ["p1", "p2"], "q1"),
                 ("L(Y|X,q1,q2)", X + ["q1", "q2"], "Y")]
    inits = range(3 if SMOKE else 8)
    mem = [(t, it) for t in types for it in inits]
    cin = max(len(t[1]) for t in types)
    m = new_model(kind, len(mem), cin, [1000 * seed + it for _, it in mem])
    b = member_builder(bank, [t[1] for t, _ in mem], [t[2] for t, _ in mem], cin)
    r = prequential(m, b, bank.N, seed + 777 * order, f"noise{order}", bank.cells)
    res = {t[0]: [r["bits"][i] for i, (tt, _) in enumerate(mem) if tt[0] == t[0]] for t in types}
    dump(f"noise_{kind}_s{seed}_o{order}", dict(kind=kind, seed=seed, order=order, bits=res, cpu_secs=cpu_secs() - t0))


def job_sched(seed):
    """R4 for 10 maze items under fixed 250 steps per block (no early stopping); compare ranks with price_maze."""
    t0 = cpu_secs()
    bank = price_bank("maze", seed)
    sel = ["g1", "g2", "g3", "g4", "g6", "m4", "m8", "nz05", "d4", "reach"]
    mem = [("Y|empty", XM, "Y")] + [(f"Z|{z}", XM, z) for z in sel] + [(f"YZ|{z}", XM + [z], "Y") for z in sel]
    m = new_model("maze", len(mem), 5, [seed] * len(mem))
    b = member_builder(bank, [x[1] for x in mem], [x[2] for x in mem], 5)
    r = prequential(m, b, bank.N, seed, "P", bank.cells, fixed_steps=20 if SMOKE else 250)
    k = {x[0]: i for i, x in enumerate(mem)}
    R4 = {z: r["bits"][0] - r["bits"][k[f"YZ|{z}"]] - r["bits"][k[f"Z|{z}"]] for z in sel}
    dump(f"sched_maze_s{seed}", dict(seed=seed, R4_fixed250=R4, L_Y=r["bits"][0], cpu_secs=cpu_secs() - t0))


def job_pcalib(seed):
    """k-parity learnability at N (Phase 0). Chooses the smallest input width where 2-parity is learnable and
    4-parity is not; writes it to results/par_nbits.txt (read by p8lib at import)."""
    t0 = cpu_secs()
    out = {}
    for nb in (16, 32, 64):
        bank = Bank(parity_data(N_PAR, seed, "calib", nb), "par")
        X = [f"x{i}" for i in range(nb)]
        tg = [("k1", "bit"), ("k2", "p1"), ("k3", "k3"), ("k4", "q1"), ("k6", "k6"), ("k8", "Y")]
        m = GMLP(len(tg), nb, 128, 2, 1, [seed] * len(tg))
        b = member_builder(bank, [X] * len(tg), [t[1] for t in tg], nb)
        r = prequential(m, b, bank.N, seed, "P", 1)
        out[nb] = {t[0]: dict(bits=r["bits"][i], last_block_bits_per_ex=r["last_block_mean"][i],
                              learnable=bool(r["last_block_mean"][i] < 0.2)) for i, t in enumerate(tg)}
    ok = [nb for nb in out if out[nb]["k2"]["learnable"] and not out[nb]["k4"]["learnable"]]
    chosen = min(ok) if ok else 32
    if seed == 0 and not SMOKE:
        with open(os.path.join(RES, "par_nbits.txt"), "w") as f:
            f.write(str(chosen))
    dump(f"pcalib_s{seed}", dict(seed=seed, res=out, chosen_nbits=chosen, candidates_ok=ok, cpu_secs=cpu_secs() - t0))


def job_u6(seed):
    """U6: start channel masked from X; start (startZ) should be listed first by R4."""
    t0 = cpu_secs()
    bank = price_bank("maze", seed)
    Xp = ["wall", "inside", "decoy"]
    pool = POOL_MAZE + ["startZ"]
    mem = [("Y|", Xp, "Y")] + [(f"Z|{z}", Xp, z) for z in pool] + [(f"YZ|{z}", Xp + [z], "Y") for z in pool]
    m = new_model("maze", len(mem), 4, [seed] * len(mem))
    b = member_builder(bank, [x[1] for x in mem], [x[2] for x in mem], 4)
    r = prequential(m, b, bank.N, seed, "P", bank.cells)
    k = {x[0]: i for i, x in enumerate(mem)}
    R4 = {z: r["bits"][0] - r["bits"][k[f"YZ|{z}"]] - r["bits"][k[f"Z|{z}"]] for z in pool}
    dump(f"u6_maze_s{seed}", dict(seed=seed, R4=R4, first=max(R4, key=R4.get), L_Y_walls=r["bits"][0],
                                  L_start=r["bits"][k["Z|startZ"]], L_Y_walls_start=r["bits"][k["YZ|startZ"]],
                                  cpu_secs=cpu_secs() - t0))


# ============================================================ (b) towers: greedy on R4 / R7a, analysis, deployment
def greedy_level(bank, S, cands, seed, tag, want_r7a=True, kind="maze"):
    """Exact R4 (and R7a) for each candidate given the listed set S (teacher-forced): mean over KREP_T paired
    replicates (each with its own init and data order)."""
    X = xs(kind)
    hats = [f"hat:{tag}:{z}" for z in cands]
    ensure_hats([bank], hats)
    mem = [("Y", X + S, "Y")] + [(f"Z|{z}", X + S, z) for z in cands] + [(f"YZ|{z}", X + S + [z], "Y") for z in cands]
    cin = len(X) + len(S) + 1
    n = len(cands)
    bits, rbits, work, work_r = [], [], 0.0, 0.0
    for r in range(KREP_T):
        rs = rep_seed(seed, r)
        mP = new_model(kind, len(mem), cin, [rs] * len(mem))
        bP = member_builder(bank, [m[1] for m in mem], [m[2] for m in mem], cin)
        pP = Preq(mP, bP, bank.N, seed, rep_tag(r), bank.cells)
        if want_r7a:
            rmem = [X + S + [h] for h in hats]
            mR = new_model(kind, len(rmem), cin, [rs] * len(rmem))
            bR = member_builder(bank, rmem, ["Y"] * len(rmem), cin)
            pR = Preq(mR, bR, bank.N, seed, rep_tag(r), bank.cells)
        zi = torch.tensor([1 + j for j in range(n)], device=DEV)
        for i in range(pP.n_stages):
            pP.train(i)
            pP.code(i)
            if want_r7a:
                a, b = pP.ab(i)
                rows = torch.arange(a, b, device=DEV)
                pr = predict(mP, bP, rows)[:, zi, 0]
                for j, h in enumerate(hats):
                    bank.set(h, rows, pr[:, j])
                pR.train(i)
                pR.code(i)
        rr = pP.result()
        bits.append(np.asarray(rr["bits"]))
        work += float(np.sum(rr["work"]))
        if want_r7a:
            q = pR.result()
            rbits.append(np.asarray(q["bits"]))
            work_r += float(np.sum(q["work"]))
    B = np.mean(bits, 0)
    LY = B[0]
    out = dict(S=list(S), L_Y=LY, work=work, items={}, krep=KREP_T)
    for j, z in enumerate(cands):
        LZ, LYZ = B[1 + j], B[1 + n + j]
        r4 = [bb[0] - bb[1 + n + j] - bb[1 + j] for bb in bits]
        out["items"][z] = dict(L_Z=LZ, L_YZ=LYZ, R4=LY - LYZ - LZ, R1=LY - LYZ,
                               R4_sd=float(np.std(r4, ddof=1)) if KREP_T > 1 else None)
    if want_r7a:
        RB = np.mean(rbits, 0)
        out["work_r7a"] = work_r
        for j, z in enumerate(cands):
            out["items"][z]["R7a"] = LY - RB[j]
    return out


def greedy_tower(bank, pool, seed, tag, rule, depth=4, top=6, first=None):
    """Forward greedy: level 1 exact on all items, later levels exact on the `top` highest prior prices."""
    S, levels, prior = [], [], {}
    work = 0.0
    for lev in range(depth):
        rem = [z for z in pool if z not in S]
        if lev == 0 and first is not None:
            res = first
        else:
            cands = rem if lev == 0 else sorted(rem, key=lambda z: -prior.get(z, -1e18))[:top]
            res = greedy_level(bank, S, cands, seed, f"{tag}{lev}", want_r7a=(rule == "R7a"))
        work += res["work"] + res.get("work_r7a", 0.0)
        for z, v in res["items"].items():
            prior[z] = v[rule]
        best = max(res["items"], key=lambda z: res["items"][z][rule])
        levels.append(dict(S=list(S), prices={z: v[rule] for z, v in res["items"].items()}, R4={z: v["R4"] for z, v in res["items"].items()},
                           L_Y=res["L_Y"], best=best, best_price=res["items"][best][rule],
                           L_Z_best=res["items"][best]["L_Z"]))
        if res["items"][best][rule] <= 0:
            break
        S.append(best)
    return dict(listed=S, levels=levels, work=work)


def tower_analysis(bank, S, seed, kind="maze"):
    """L_S, used-by edges (L(W_j|X,W_<j minus p)) and direct/total/indirect pay for each listed W (paired run)."""
    X = xs(kind)
    need = {}

    def add(t, ins):
        need[(t, tuple(ins))] = None

    for j, W in enumerate(S):
        add(W, S[:j])
        for p in S[:j]:
            add(W, [q for q in S[:j] if q != p])
    add("Y", S)
    for W in S:
        rest = [q for q in S if q != W]
        add("Y", rest)
        for j, V in enumerate(S):
            if V != W:
                add(V, [q for q in S[:j] if q != W])
    keys = list(need)
    cin = len(X) + max(len(k[1]) for k in keys)
    b = member_builder(bank, [X + list(k[1]) for k in keys], [k[0] for k in keys], cin)
    bits, work = [], 0.0
    for rep in range(KREP_T):
        m = new_model(kind, len(keys), cin, [rep_seed(seed, rep)] * len(keys))
        r = prequential(m, b, bank.N, seed, rep_tag(rep), bank.cells)
        bits.append(np.asarray(r["bits"]))
        work += float(np.sum(r["work"]))
    B = np.mean(bits, 0)
    L = {k: B[i] for i, k in enumerate(keys)}
    LS = sum(L[(W, tuple(S[:j]))] for j, W in enumerate(S)) + L[("Y", tuple(S))]
    edges, pay = [], {}
    for j, W in enumerate(S):
        for p in S[:j]:
            edges.append(dict(p=p, W=W, delta=L[(W, tuple(q for q in S[:j] if q != p))] - L[(W, tuple(S[:j]))]))
    for W in S:
        rest = [q for q in S if q != W]
        direct = L[("Y", tuple(rest))] - L[("Y", tuple(S))]
        LSm = sum(L[(V, tuple(q for q in S[:j] if q != W))] for j, V in enumerate(S) if V != W) + L[("Y", tuple(rest))]
        total = LSm - LS
        pay[W] = dict(direct=direct, total=total, indirect=total - direct)
    return dict(S=list(S), L_S=LS, edges=edges, pay=pay, work=work, krep=KREP_T,
                node_bits={f"{t}|{','.join(i)}": v for (t, i), v in L.items()})


def deploy_tower(bank, evals, S, seed, tag, kind="maze"):
    X = xs(kind)
    hats, work = [], 0.0
    for j, W in enumerate(S):
        h = f"dt:{tag}:{W}"
        work += crossfit(kind, bank, evals, [(h, X + hats, W)], seed, f"{tag}{j}")
        hats.append(h)
    cin = len(X) + len(hats)
    m = new_model(kind, 1, cin, [seed])
    b = member_builder(bank, [X + hats], ["Y"], cin)
    res, w, info = fit_eval(m, b, bank, evals, seed, f"{tag}Y")
    work += w
    return summarize_eval(res, work, kind)


def summarize_eval(res, work, kind, g=0):
    names = ["held", "f15"] if kind == "maze" else ["held"]
    out = dict(work=work)
    for e, en in enumerate(names):
        bits, acc = res[e]
        out[en] = dict(bits=bits[:, g].mean(), acc=acc[:, g].mean(), bits_per_ex=bits[:, g])
    return out


def c2_pool(teacher, P, seed):
    """16-item non-oracle pool (pre-registered): per teacher layer 2, 4, 6, every channel binarized at its median,
    channels with <5% or >95% ones dropped, the 4 highest-entropy kept; plus 4 binarized random conv features
    (radius 2 and 4). Returns names, info and apply(bank) that adds the channels to any bank."""
    bx = member_builder(P, [XM], ["Y"], 4)
    specs = []
    with torch.no_grad():
        xin, _, _ = bx(torch.arange(P.N, device=DEV))
        for L in (2, 4, 6):
            h = teacher.features(xin, L)  # (N, hid, n, n)
            vals = h.permute(1, 0, 2, 3).reshape(h.shape[1], -1)
            med = vals.median(1).values
            ones = (vals > med[:, None]).float().mean(1)
            ent = -(ones * torch.log2(ones.clamp_min(1e-9)) + (1 - ones) * torch.log2((1 - ones).clamp_min(1e-9)))
            ok = (ones >= 0.05) & (ones <= 0.95)
            ent = torch.where(ok, ent, torch.full_like(ent, -1))
            for c in torch.argsort(ent, descending=True)[:4].tolist():
                if ent[c] >= 0:
                    specs.append(dict(name=f"T{L}c{c}", src="teacher", layer=L, chan=c, median=float(med[c]),
                                      frac_ones=float(ones[c])))
        rnets = {}
        for r, (layers, rs) in enumerate([(2, 0), (2, 1), (4, 0), (4, 1)]):
            net = GConv(1, 4, 16, layers, 1, [seed_int("randfeat", seed, r)])
            v = net.logits(xin)[:, 0, 0]
            med = v.median()
            nm = f"R{layers}r{rs}"
            rnets[nm] = net
            specs.append(dict(name=nm, src="random", layer=layers, chan=0, median=float(med),
                              frac_ones=float((v > med).float().mean())))

    @torch.no_grad()
    def apply(bk):
        bb = member_builder(bk, [XM], ["Y"], 4)
        xi, _, _ = bb(torch.arange(bk.N, device=DEV))
        cache = {}
        for sp in specs:
            if sp["src"] == "teacher":
                if sp["layer"] not in cache:
                    cache[sp["layer"]] = teacher.features(xi, sp["layer"])
                v = cache[sp["layer"]][:, sp["chan"]]
            else:
                v = rnets[sp["name"]].logits(xi)[:, 0, 0]
            bk.add(sp["name"], (v > sp["median"]).float() * bk.t[:, bk.idx["inside"]])

    return [sp["name"] for sp in specs], specs, apply


def b2_configs():
    return [(h, lr) for h in (16, 32, 64, 128) for lr in (1e-3, 3e-3)]


def b3_configs():
    return [(w, it, al) for w in (16, 32) for it in (10, 20) for al in (0.0, 0.5)]


def run_b2(bank, evals, seed, cfgs=None):
    out = {}
    for h, lr in (cfgs or b2_configs()):
        c = dict(CFG_FIT, lr=lr, max_steps=40 if SMOKE else 8000, patience=20 if SMOKE else 800)
        m = GConv(1, 4, h, 8, 1, [seed])
        b = member_builder(bank, [XM], ["Y"], 4)
        res, w, info = fit_eval(m, b, bank, evals, seed, f"B2{h}{lr}", c)
        v = val_of(m, b, bank, seed, f"B2{h}{lr}")
        out[f"h{h}_lr{lr}"] = dict(summarize_eval(res, w, "maze"), val=v, steps=info["steps"], params=m.n_params())
    return out


def val_of(m, b, bank, seed, tag):
    g = torch.Generator().manual_seed(seed_int(seed, tag, "fit"))
    perm = torch.randperm(bank.N, generator=g)
    nv = max(2, bank.N // 10)
    return float(val_loss(m, b, perm[:nv].to(DEV)).mean())


def run_b3(bank, evals, seed, cfgs=None):
    out = {}
    for w_, it, al in (cfgs or b3_configs()):
        c = dict(CFG_FIT, lr=1e-3, max_steps=40 if SMOKE else 6000, patience=20 if SMOKE else 600)
        torch.manual_seed(seed_int("b3", seed, w_, it, al))
        m = DTR(4, w_, it, al, seed)
        b = member_builder(bank, [XM], ["Y"], 4)
        res, wk, info = fit_eval(m, b, bank, evals, seed, f"B3{w_}{it}{al}", c)
        v = val_of(m, b, bank, seed, f"B3{w_}{it}{al}")
        curve = {}
        for k in (it // 2, it, 2 * it, 4 * it):
            m.iters = k
            r2 = []
            for ev in evals:
                er = torch.arange(ev.N, device=DEV)
                acc = exact_acc(predict(m, b, er, bk=ev), ev.t[er][:, ev.idx["Y"]])
                r2.append(float(acc.float().mean()))
            curve[k] = r2
        m.iters = it
        out[f"w{w_}_it{it}_a{al}"] = dict(summarize_eval(res, wk, "maze"), val=v, steps=info["steps"], params=m.n_params(),
                                          iters_curve=curve)
    return out


def best_of(d):
    ks = [k for k in d if not k.startswith("_")]
    return min(ks, key=lambda k: d[k]["val"])


def strip(d):
    """drop per-example arrays for JSON (keep means)."""
    if isinstance(d, dict):
        return {k: strip(v) for k, v in d.items() if k not in ("bits_per_ex", "_model")}
    return d


def train_teacher(P, seed, h, lr):
    teacher = GConv(1, 4, h, 8, 1, [seed + 1])
    bt = member_builder(P, [XM], ["Y"], 4)
    g = torch.Generator().manual_seed(seed_int(seed, "teacher"))
    perm = torch.randperm(P.N, generator=g)
    fit(teacher, bt, perm[P.N // 10:], perm[:P.N // 10], g,
        dict(CFG_FIT, lr=lr, max_steps=40 if SMOKE else 8000, patience=20 if SMOKE else 800))
    return teacher


def job_c2(seed):
    """Programme test at fixed depth: T1-C2 (greedy R4 over a non-oracle pool) vs B1-B5, T7-C2, T0-C2.
    Then the data-scaling point (N=4096 deployment split) for T1, T7, best B2, best B3 and B5."""
    t0 = cpu_secs()
    torch.manual_seed(seed_int("c2", seed))
    P = price_bank("maze", seed)
    T, evals = truth_banks("maze", seed)
    out = dict(seed=seed)
    b2 = run_b2(T, evals, seed)  # 8 configs on the deployment split; best by val = the teacher config
    kb2 = best_of(b2)
    h, lr = int(kb2.split("_")[0][1:]), float(kb2.split("_lr")[1])
    teacher = train_teacher(P, seed, h, lr)  # teacher trained on the price stream (no deployment data)
    pool, pool_info, apply = c2_pool(teacher, P, seed)
    for bk in [P, T] + evals:
        apply(bk)
    out["pool"] = pool_info
    lev1 = greedy_level(P, [], pool, seed, "c2L0", want_r7a=True)
    t1 = greedy_tower(P, pool, seed, "c2T1_", "R4", first=lev1)
    t7 = greedy_tower(P, pool, seed, "c2T7_", "R7a", first=lev1)
    rng = np.random.default_rng(seed_int("t0", seed))
    t0l = [pool[i] for i in rng.permutation(len(pool))[:max(1, len(t1["listed"]))]]
    out.update(T1=t1, T7=t7, T0_listed=t0l, level1=lev1)
    if t1["listed"]:
        out["T1_analysis"] = tower_analysis(P, t1["listed"], seed)
    dep = {}
    for name, S in (("T1", t1["listed"]), ("T7", t7["listed"]), ("T0", t0l), ("B4", ["g2", "g4", "g6"])):
        dep[name] = dict(deploy_tower(T, evals, S, seed, f"c2{name}"), S=S)
    m1 = GConv(1, 4, 16, 2, 1, [seed])
    res, w, _ = fit_eval(m1, member_builder(T, [XM], ["Y"], 4), T, evals, seed, "B1")
    dep["B1"] = summarize_eval(res, w, "maze")
    dep["B2"] = dict(b2[kb2], config=kb2)
    b3 = run_b3(T, evals, seed)
    kb3 = best_of(b3)
    dep["B3"] = dict(b3[kb3], config=kb3)
    K = max(1, len(t1["listed"]))
    mB5 = GStack(1, 4, K, "conv", init_seeds=[seed])
    res, w, _ = fit_eval(mB5, member_builder(T, [XM], ["Y"], 4), T, evals, seed, "B5")
    dep["B5"] = dict(summarize_eval(res, w, "maze"), K=K)
    pm = {k: dep[k]["held"]["bits_per_ex"] for k in dep}
    out["paired_vs_T1"] = {k: dict(zip(("bits_gain", "ci95"), paired_ci(pm[k] - pm["T1"]))) for k in pm if k != "T1"}
    out["deployed"] = strip(dep)
    out["b2_all"] = strip(b2)
    out["b3_all"] = strip(b3)
    out["cpu_secs"] = cpu_secs() - t0
    dump(f"c2_s{seed}", out)
    # data-scaling point: 4096 deployment mazes (same teacher, pool and listed towers)
    T4 = Bank(maze_data(256 if SMOKE else 4096, N_MAZE, seed_int("maze-truth4k", seed)), "maze")
    _, ev4 = truth_banks("maze", seed)
    for bk in [T4] + ev4:
        apply(bk)
    big = {}
    for name in ("T1", "T7"):
        S = out[name]["listed"]
        big[name] = dict(deploy_tower(T4, ev4, S, seed, f"big{name}"), S=S)
    big["B2"] = dict(list(run_b2(T4, ev4, seed, [(h, lr)]).values())[0], config=kb2)
    w_, it, al = int(kb3.split("_")[0][1:]), int(kb3.split("_it")[1].split("_")[0]), float(kb3.split("_a")[1])
    big["B3"] = dict(list(run_b3(T4, ev4, seed, [(w_, it, al)]).values())[0], config=kb3)
    mB5 = GStack(1, 4, K, "conv", init_seeds=[seed])
    res, w, _ = fit_eval(mB5, member_builder(T4, [XM], ["Y"], 4), T4, ev4, seed, "B5big")
    big["B5"] = dict(summarize_eval(res, w, "maze"), K=K)
    out["deployed_4k"] = strip(big)
    out["cpu_secs"] = cpu_secs() - t0
    dump(f"c2_s{seed}", out)


def job_toracle(seed):
    """T1-oracle: greedy R4 over the 17-item maze pool (depth <= 4), tower analysis, deployed vs B4; U7 ladder."""
    t0 = cpu_secs()
    torch.manual_seed(seed_int("toracle", seed))
    P = price_bank("maze", seed)
    T, evals = truth_banks("maze", seed)
    t1 = greedy_tower(P, POOL_MAZE, seed, "orT1_", "R4")
    out = dict(seed=seed, T1=t1)
    if t1["listed"]:
        out["T1_analysis"] = tower_analysis(P, t1["listed"], seed)
    out["ladder_analysis"] = tower_analysis(P, ["g2", "g4", "g6"], seed)
    dep = {}
    for name, S in (("T1oracle", t1["listed"]), ("B4", ["g2", "g4", "g6"])):
        dep[name] = dict(deploy_tower(T, evals, S, seed, f"or{name}"), S=S)
    out["deployed"] = strip(dep)
    out["cpu_secs"] = cpu_secs() - t0
    dump(f"toracle_s{seed}", out)


# ============================================================ parity towers: forward greedy (U5) is in price_par
JOBS = {
    "price_maze": lambda s: job_price("maze", s), "price_par": lambda s: job_price("par", s),
    "capref_maze": lambda s: job_capref("maze", s), "capref_par": lambda s: job_capref("par", s),
    "r7b_maze": lambda s: job_r7b("maze", s), "r7b_par": lambda s: job_r7b("par", s),
    "r6_maze": lambda s: job_r6("maze", s), "r6_par": lambda s: job_r6("par", s),
    "r6b_par": lambda s: job_r6("par", s, void=True),
    "truth_maze": lambda s: job_truth("maze", s), "truth_par": lambda s: job_truth("par", s),
    "noise_maze0": lambda s: job_noise("maze", s, 0), "noise_maze1": lambda s: job_noise("maze", s, 1),
    "noise_par0": lambda s: job_noise("par", s, 0), "noise_par1": lambda s: job_noise("par", s, 1),
    "noise_maze2": lambda s: job_noise("maze", s, 2), "noise_maze3": lambda s: job_noise("maze", s, 3),
    "noise_par2": lambda s: job_noise("par", s, 2), "noise_par3": lambda s: job_noise("par", s, 3),
    "sched_maze": job_sched, "pcalib": job_pcalib, "u6_maze": job_u6, "bf_par": job_bf,
    "c2": job_c2, "toracle": job_toracle,
}

if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2])
    t = time.time()
    JOBS[name](seed)
    print(f"done {name} {seed} {time.time() - t:.0f}s", flush=True)
