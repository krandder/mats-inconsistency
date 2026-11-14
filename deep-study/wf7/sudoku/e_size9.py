"""e_size9.py -- E6, unseen size. The size-generic network (trained on 4x4 and 6x6 grids only) applied to 9x9.
  [1] in-size sanity: route surprise on held-out 6x6 grids
  [2] 9x9: does M's rule transfer? mass on directly rule-breaking values at states of real 9x9 grids
  [3] 9x9: T (own-confidence order; no trust net was trained for this model) on valid grids and near-miss boards
  [4] 9x9: search S1 corrected against T, and one pass with no check, on puzzles with unique solutions
Usage: python -u e_size9.py NAME"""
import sys, time, math, json
import numpy as np, torch
import core, data
import oracle as O, eval_lib as E          # world generator (9x9 grids, puzzles) and scoring

name = sys.argv[1]; torch.set_num_threads(1); torch.manual_seed(0); T0 = time.time(); KM = 9
model, _ = core.load_model(f"ckpt/{name}.pt")


def log(*a):
    print(*a, flush=True)


def alpha_for(n, K):
    a = torch.zeros(n, KM, dtype=torch.bool); a[:, :K] = True
    return a


RES = {}
# ---- [1] in-size sanity at 6x6 (digits use slots 0..5)
d6 = np.load("data/k6_base.npz"); te = data.GridStream(d6["B"][d6["base_split"] == 2], 6, 9)
M6 = core.Net(model, 6, name); X6 = te.sample(500)
l6 = -core.rollout(M6, core.pol_entropy, torch.full_like(X6, -1), X=X6, alpha=alpha_for(500, 6))["logp"]
log(f"[1] 6x6 held-out grids, own-confidence (entropy) order: excess route surprise {l6.mean()-math.log(28200960):.4f} nats")
RES["k6_excess"] = float(l6.mean() - math.log(28200960))

# ---- 9x9 world (generator side)
rng = np.random.default_rng(909); K, N = 9, 81
G9 = np.array([O.random_grid_dfs(9, rng) for _ in range(500)]); assert O.is_valid_batch(G9, 9).all()
XV = torch.from_numpy(G9[:200]); XT = torch.from_numpy(G9[200:500])
M9 = core.Net(model, 9, name); A9 = lambda n: alpha_for(n, 9)

# ---- [2] rule transfer: mass on directly rule-breaking values
gg = torch.Generator().manual_seed(3); r = torch.floor(torch.rand(300, generator=gg) * (N - 1)).long()
rank = torch.rand(300, N, generator=gg).argsort(1).argsort(1)
S = torch.where(rank < r[:, None], XT, torch.full_like(XT, -1)); LP = M9.logp(S, alpha=A9(300))
loc = torch.from_numpy(O.local_candidates(S.numpy(), 9)); emp = S < 0
leak = (LP.exp() * ~loc).sum(-1)[emp]; nloc = loc.sum(-1)[emp]
forced = nloc == 1; top_ok = (LP.argmax(-1) == XT)[emp]
log(f"[2] 9x9 (never trained on): mass on directly rule-breaking values per atom: mean {leak.mean():.3e}, median {leak.median():.3e}, "
    f"99th percentile {leak.quantile(0.99):.3e}, max {leak.max():.3e} ({int(emp.sum())} atoms at 300 states of real grids)")
log(f"    locally forced cells: top value correct {top_ok[forced].float().mean():.4f} (n {int(forced.sum())}); surprise at the real value, all atoms: {(-LP[emp][torch.arange(int(emp.sum())), XT[emp]]).mean():.4f}")
RES["k9_leak_mean"] = float(leak.mean()); RES["k9_forced_top_ok"] = float(top_ok[forced].float().mean())

# ---- [3] T on complete 9x9 boards
pol = core.pol_entropy
lv = core.rollout(M9, pol, torch.full_like(XV, -1), X=XV, alpha=A9(len(XV)))["logp"].numpy().astype(np.float64)
lt = core.rollout(M9, pol, torch.full_like(XT, -1), X=XT, alpha=A9(len(XT)))["logp"].numpy().astype(np.float64)
tau = float(lv.min() - 2.0)
log(f"[3] 9x9 valid grids: log T mean {lt.mean():.2f} (ln of the number of 9x9 grids is {math.log(6.670903752021072936960e21):.2f}; the test grids come from a "
    f"randomised backtracking generator, not the uniform law); tau = {tau:.2f} from 200 real 9x9 grids; held-out valid grids rejected {np.mean(lt < tau):.4f}")
RES["k9_tau"] = tau; RES["k9_valid_mean_logT"] = float(lt.mean()); RES["k9_false_reject"] = float(np.mean(lt < tau))
for kind in ("cell", "swap"):
    Xn = E.near_miss(XT.numpy(), rng, kind, 9); assert not O.is_valid_batch(Xn, 9).any()
    ln = core.rollout(M9, pol, torch.full((len(Xn), N), -1, dtype=torch.long), X=torch.from_numpy(Xn), alpha=A9(len(Xn)))["logp"].numpy()
    log(f"    near-miss invalid boards ({kind}): accepted {np.mean(ln >= tau):.4f}; max log T {ln.max():.2f}; median {np.median(ln):.2f}")
    RES[f"k9_near_{kind}_fa"] = float(np.mean(ln >= tau)); RES[f"k9_near_{kind}_max"] = float(ln.max())

# ---- [4] search on 9x9 puzzles
def make9(sol, target):
    s = sol.copy()
    for i in rng.permutation(N):
        if (s >= 0).sum() <= target:
            break
        v = s[i]; s[i] = -1
        if O.count_solutions_dfs(s, 9, cap=2) != 1:
            s[i] = v
    return s


def checker(X):
    out = core.rollout(M9, pol, torch.full_like(X, -1), X=X, alpha=A9(len(X)), abort_below=tau)
    return out["logp"] >= tau, out["logp"], out["used"]


class WithAlpha:                                    # the proposal needs the alphabet too
    def __init__(self, net):
        self.net = net
    def logp(self, s, flag=None, alpha=None):
        return self.net.logp(s, flag, A9(len(s)))


for target, npz in ((45, 40), (36, 40), (30, 40)):
    P, Sx = [], []
    for g in G9[200:200 + npz]:
        P.append(make9(g, target)); Sx.append(g)
    puz = torch.from_numpy(np.array(P)); sol = torch.from_numpy(np.array(Sx))
    k0 = core.rollout(M9, core.pol_margin, puz, greedy=True, alpha=A9(len(puz)))["s"]
    gen = torch.Generator().manual_seed(target)
    r = core.search_first_accept(WithAlpha(M9), core.pol_margin, puz, None, checker, 64, gen, rounds=(1, 7, 24, 32))
    solved = r["acc_idx"] >= 0; correct = (r["answer"] == sol).all(1)
    fl = M9.flops_per_call; flops = (r["prop_steps"] + r["chk_steps"]).double() * fl
    res = dict(n=len(puz), clues=float((puz >= 0).sum(1).float().mean()), onepass=float((k0 == sol).all(1).float().mean()),
               accepted=float(solved.float().mean()), wrong=int((solved & ~correct).sum()),
               cands=float((r["acc_idx"][solved] + 1).float().mean()) if solved.any() else float("nan"),
               flops=float(flops[solved].mean()) if solved.any() else float("nan"))
    RES[f"k9_puz_{target}"] = res
    log(f"[4] 9x9 puzzles with {res['clues']:.1f} clues (n {res['n']}): one pass, no check: accuracy {res['onepass']:.3f}; "
        f"search corrected against T (cap 64): accepted {res['accepted']:.3f}, wrong answers {res['wrong']}, candidates to accept {res['cands']:.2f}, "
        f"FLOPs {res['flops']:.3e} ({time.time()-T0:.0f}s)")
json.dump(RES, open(f"out/e_size9_{name}.json", "w"), indent=1)
log(f"total time {time.time()-T0:.0f}s")
