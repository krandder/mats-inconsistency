"""P19: rank candidate maze intermediates by a fast reader's prequential gain (FR) and by a first-order price
against the current reader (FO), next to P8's R4, R1 (importance) and R7b (wrapper validation).
Truth = deployed gain over base (P8/item3 job_truthrep).

python p19.py JOB SEED [REP]   (writes $P8_RES/<name>.json)
  truthrep SEED REP   deployed truth replicate (item3 job_truthrep on the P19 item set)
  pricerep SEED REP   one paired R1/R4 replicate (P8 price_rep, R7a off)
  r7brep SEED REP     one R7b 80/20 wrapper replicate
  frfo SEED           current readers, fast reader (K_FR stream orders), first-order prices, compute
  pilotlr SEED        pilot only: fast-reader codelength over the learning-rate grid (truth-free)

P8/item3 code (jobs.py, jobs3.py, p8lib.py) is used unchanged; this file patches the item set:
  key  = public random bit per cell, an extra reader input in every context (X = wall, start, inside, decoy, key)
  x2, x4, x6 = g_k XOR key          (XOR type: zero first-order value, full value to a reader that also reads key)
  xb4  = g4 XOR key2                (key2 is never an input: pure noise, matched twin of x4)
  pg4  = g4 of the next maze        (permuted: same marginals, unrelated)
  fg4  = 1 - g4                     (invertible relabelling of g4: every rule should price it like g4)
"""
import sys

import jobs
import p8lib
from p8lib import *

POOL19 = list(POOL_MAZE) + ["x2", "x4", "x6", "xb4", "pg4", "fg4"]
XM19 = ["wall", "start", "inside", "decoy", "key"]
_maze_data8 = p8lib.maze_data


def maze_data19(num, n, seed):
    ch = _maze_data8(num, n, seed)
    r = np.random.default_rng(seed_int("p19-key", seed))
    key = (r.random((num, n, n)) < 0.5).astype(np.float32)
    key2 = (r.random((num, n, n)) < 0.5).astype(np.float32)
    ch["key"] = key
    for k in (2, 4, 6):
        ch[f"x{k}"] = np.abs(ch[f"g{k}"] - key)
    ch["xb4"] = np.abs(ch["g4"] - key2)
    ch["pg4"] = np.roll(ch["g4"], 1, axis=0).copy()
    ch["fg4"] = (1.0 - ch["g4"]).astype(np.float32)
    return ch


jobs.maze_data = maze_data19
jobs.POOL_MAZE = POOL19
jobs.XM = XM19
import jobs3  # noqa: E402  (its functions resolve items/xs/banks through the patched jobs module)

from jobs import CFG_FIT, CTX_M, N_TR, fold_of, items, xs  # noqa: E402

N_F = N_TR                      # fast-reader stream: the whole price bank (same mazes R1/R4/R7b see)
CHUNK, REPLAY = (16, 16)        # code 16 new mazes, then one Adam step on them + 16 replayed past mazes
K_FR = int(os.environ.get("P19_KFR", "4"))
LR_FR = float(os.environ.get("P19_LR", "0.003"))
LR_GRID = [1e-3, 3e-3, 1e-2, 3e-2]
FO_RIDGE = 1.0
BUDGETS = (16, 32, 64)          # FR prefix budgets in steps (64 = whole stream at N_F = 1024)


def reader_bank(seed):
    return Bank(maze_data19(N_TR, N_MAZE, seed_int("p19-reader", seed)), "maze")


def train_readers(seed, RB):
    """The current reader per context: Y | X + S, trained to convergence (CFG_FIT) on a separate reader bank."""
    X = xs("maze")
    out = {}
    for c, S in CTX_M.items():
        m = GConv(1, len(X) + len(S), CFG["hid"], 2, 1, [seed_int("p19-reader", seed, c)])
        b = member_builder(RB, [X + S], ["Y"])
        g = torch.Generator().manual_seed(seed_int("p19-reader-fit", seed, c))
        perm = torch.randperm(RB.N, generator=g)
        nv = max(2, RB.N // 10)
        info = fit(m, b, perm[nv:], perm[:nv], g, CFG_FIT)
        stop = float(info["stop"][0])
        out[c] = dict(model=m, work=stop * CFG_FIT["batch"] * RB.cells * m.macs(), stop=stop)
    return out


def members_of(c):
    zs = [z for cc, z in items("maze") if cc == c]
    if c == "empty":
        zs = zs + list(SANITY_MAZE)
    return zs


def fr_run(reader, bank, c, zs, seed, rep, lr, steps=None):
    """Fast reader: G = 1 + len(zs) copies of the current reader, member 0 reads _zero (base), member j reads z_j
    through a new, zero-initialised input channel. Online prequential code with rehearsal on the price stream:
    code CHUNK new mazes, then one Adam step on them + REPLAY past mazes. Same order for all members (paired).
    Returns bits (T, G), per-member work (train + coding forward MACs)."""
    X = xs("maze")
    S = CTX_M[c]
    cin = len(X) + len(S) + 1
    G = len(zs) + 1
    m = GConv(G, cin, CFG["hid"], 2, 1, [0] * G)
    with torch.no_grad():
        m.W[0][:, :, :cin - 1] = reader.W[0]
        m.W[0][:, :, cin - 1] = 0
        for l in range(len(m.W)):
            if l > 0:
                m.W[l][:] = reader.W[l]
            m.b[l][:] = reader.b[l]
    b = member_builder(bank, [X + S + ["_zero"]] + [X + S + [z] for z in zs], ["Y"] * G, cin)
    T = min(N_F, bank.N) // CHUNK
    if steps:
        T = min(T, steps)
    g = torch.Generator().manual_seed(seed_int("p19-fr", seed, c, rep))
    order = torch.randperm(bank.N, generator=g)[:T * CHUNK].to(DEV)
    opt = torch.optim.Adam(m.params, lr=lr)
    bits = torch.zeros(T, G)
    for t in range(T):
        new = order[t * CHUNK:(t + 1) * CHUNK]
        bits[t] = (eval_loss(m, b, new).sum(0) / LN2).cpu()
        if t > 0 and REPLAY:
            old = order[torch.randint(t * CHUNK, (REPLAY,), generator=g).to(DEV)]
            bi = torch.cat([new, old])
        else:
            bi = new
        xin, tgt, _ = b(bi)
        loss = per_example_loss(m.logits(xin), tgt).mean((0, 2))
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        opt.step()
    ntrain = sum(CHUNK + (REPLAY if t > 0 else 0) for t in range(T))
    work = float((ntrain + T * CHUNK) * bank.cells * m.macs())
    return bits.numpy(), work


def _bce_bits(eta, y):
    return float(F.binary_cross_entropy_with_logits(eta, y, reduction="sum")) / LN2


def _newton(Phi, off, y, lam, iters=30):
    p = Phi.shape[1]
    beta = torch.zeros(p, dtype=torch.float64)
    I = torch.eye(p, dtype=torch.float64)
    obj = lambda bb: float(F.binary_cross_entropy_with_logits(off + Phi @ bb, y, reduction="sum")) + 0.5 * lam * float(bb @ bb)
    cur = obj(beta)
    for _ in range(iters):
        pr = torch.sigmoid(off + Phi @ beta)
        gr = Phi.T @ (pr - y) + lam * beta
        H = Phi.T @ (Phi * (pr * (1 - pr)).unsqueeze(1)) + lam * I
        step = torch.linalg.solve(H, gr)
        s = 1.0
        while s > 1e-4:
            nb = beta - s * step
            nv = obj(nb)
            if nv <= cur + 1e-12:
                break
            s *= 0.5
        if s <= 1e-4:
            break
        beta, delta, cur = nb, cur - nv, nv
        if delta < 1e-9:
            break
    return beta


def fo_prices(reader, bank, c, zs, seed):
    """First-order price against the frozen current reader (boosting / cascade-correlation / column generation):
    offset logistic regression logit = reader_logit + b + w . (3x3 neighbourhood of centred z), ridge FO_RIDGE, 2-fold
    cross-fitted on the price stream; price = held-fold bits of (reader + b) minus bits of (reader + b + w.z).
    FO1: the same with z's own cell only (1 weight)."""
    X = xs("maze")
    S = CTX_M[c]
    b = member_builder(bank, [X + S], ["Y"])
    allr = torch.arange(bank.N, device=DEV)
    with torch.no_grad():
        lg = torch.cat([reader.logits(b(allr[i:i + 256])[0])[:, 0, 0] for i in range(0, bank.N, 256)])
    off = lg.double().reshape(-1).cpu()
    y = bank.t[:, bank.idx["Y"]].double().reshape(-1).cpu()
    fold = fold_of(bank.N, seed, "p19-fo").cpu()
    fr = fold.view(-1, 1).expand(-1, bank.cells).reshape(-1)
    one = torch.ones(len(y), 1, dtype=torch.float64)

    def cross(Phi):
        tot = 0.0
        for f in (0, 1):
            trn, tst = fr != f, fr == f
            beta = _newton(Phi[trn], off[trn], y[trn], FO_RIDGE)
            tot += _bce_bits(off[tst] + Phi[tst] @ beta, y[tst])
        return tot

    base = cross(one)
    out = {"base_bits": base}
    n = bank.sp[0]
    for z in zs:
        zt = bank.t[:, bank.idx[z]].unsqueeze(1).double().cpu()
        zt = zt - zt.mean()  # centred, so the zero padding of the 3x3 window carries no border-position signal
        nb = F.unfold(zt, 3, padding=1).transpose(1, 2).reshape(-1, 9)  # (N*cells, 9)
        ctr = zt.reshape(-1, 1)
        out[z] = dict(FO=base - cross(torch.cat([one, nb], 1)), FO1=base - cross(torch.cat([one, ctr], 1)))
    rows = len(y) // 2
    out["work_per_item"] = float(2 * 30 * rows * 10 * 10 * 2)  # generous Newton MAC bound: folds x iters x rows x p^2 x (H + solve)
    return out


def job_frfo(seed):
    t0 = cpu_secs()
    torch.manual_seed(seed_int("p19-frfo", seed))
    RB = reader_bank(seed)
    readers = train_readers(seed, RB)
    bank = jobs.price_bank("maze", seed)
    out = dict(seed=seed, N_F=N_F, chunk=CHUNK, replay=REPLAY, lr=LR_FR, K_FR=K_FR, budgets=BUDGETS, readers={},
               items={}, work={})
    for c in CTX_M:
        rd = readers[c]["model"]
        zs = members_of(c)
        reps = []
        wm = 0.0
        for r in range(K_FR):
            bits, wm = fr_run(rd, bank, c, zs, seed, r, LR_FR)
            reps.append(bits)
        fo = fo_prices(rd, bank, c, zs, seed)
        n_rank = len([1 for cc, _ in items("maze") if cc == c])
        out["readers"][c] = dict(work=readers[c]["work"], stop=readers[c]["stop"],
                                 base_bits_stream=[float(bb[:, 0].sum()) for bb in reps], fo_base_bits=fo["base_bits"])
        out["work"][c] = dict(reader=readers[c]["work"], fr_member_rep=wm, n_rank=n_rank, n_members=len(zs) + 1,
                              fo_item=fo["work_per_item"])
        for j, z in enumerate(zs):
            d = {}
            for B in BUDGETS:
                v = [float(bb[:B, 0].sum() - bb[:B, j + 1].sum()) for bb in reps]
                d[f"FR_T{B}"] = float(np.mean(v))
                d[f"FR_T{B}_reps"] = v
            T = reps[0].shape[0]
            d["FR"] = d.get(f"FR_T{T}", float(np.mean([float(bb[:, 0].sum() - bb[:, j + 1].sum()) for bb in reps])))
            d["FR_reps"] = [float(bb[:, 0].sum() - bb[:, j + 1].sum()) for bb in reps]
            d["FO"], d["FO1"] = fo[z]["FO"], fo[z]["FO1"]
            out["items"][f"{c}:{z}"] = d
    out["cpu_secs"] = cpu_secs() - t0
    dump(f"frfo_maze_s{seed}", out)


def job_pilotlr(seed):
    """Truth-free choice of the FR learning rate: total FR codelength (all members, both contexts, 2 orders)."""
    t0 = cpu_secs()
    torch.manual_seed(seed_int("p19-pilot", seed))
    RB = reader_bank(seed)
    readers = train_readers(seed, RB)
    t1 = cpu_secs()
    bank = jobs.price_bank("maze", seed)
    out = dict(seed=seed, reader_cpu=t1 - t0, readers={c: dict(work=v["work"], stop=v["stop"]) for c, v in readers.items()},
               grid={})
    for lr in LR_GRID:
        res = {}
        tot = 0.0
        ts = cpu_secs()
        for c in CTX_M:
            zs = members_of(c)
            reps = [fr_run(readers[c]["model"], bank, c, zs, seed, r, lr)[0] for r in range(2)]
            tot += float(np.mean([bb.sum() for bb in reps]))
            for j, z in enumerate(zs):
                res[f"{c}:{z}"] = [float(bb[:, 0].sum() - bb[:, j + 1].sum()) for bb in reps]
            res[f"{c}:BASE_bits"] = [float(bb[:, 0].sum()) for bb in reps]
        out["grid"][str(lr)] = dict(total_bits=tot, prices=res, cpu=cpu_secs() - ts)
    fo = {}
    ts = cpu_secs()
    for c in CTX_M:
        f = fo_prices(readers[c]["model"], bank, c, members_of(c), seed)
        fo.update({f"{c}:{z}": v for z, v in f.items() if isinstance(v, dict)})
    out["fo"] = fo
    out["fo_cpu"] = cpu_secs() - ts
    out["cpu_secs"] = cpu_secs() - t0
    dump(f"pilotlr_maze_s{seed}", out)


JOBS19 = {"truthrep": jobs3.job_truthrep, "pricerep": jobs3.job_pricerep, "r7brep": jobs3.job_r7brep,
          "frfo": job_frfo, "pilotlr": job_pilotlr}

if __name__ == "__main__":
    name, args = sys.argv[1], [int(a) for a in sys.argv[2:]]
    t = time.time()
    JOBS19[name](*args)
    print(f"done {name} {args} {time.time() - t:.0f}s", flush=True)
