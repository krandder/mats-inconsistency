"""c2_sudoku.py -- checker's own tests of the Sudoku tree (uses the author's sk.py / spol.py as the code under test;
everything else is mine: puzzle generator, exact solver, exact-integer recomputation of the books, adversary).
Part A: containment / nestedness / closed sets on FRESH puzzles I generate, for honest, bad and adversarial policies.
Part B: is the closing cost C a function of the ORDER rule only? (same order, scrambled value probabilities)
Part C: what the registered order rule does at perfect distillation (value head = the solution, one-hot).
Usage: c2_sudoku.py N_UNIQUE N_MULTI"""
import json, sys, time
from fractions import Fraction
import numpy as np
import torch
sys.dont_write_bytecode = True
sys.path.insert(0, '/home/agent/workspaces/mats/work/deep-study/wf13/loop')
import sk, spol          # noqa: E402

ROOT = '/home/agent/workspaces/mats/work/deep-study/'
OUT = ROOT + 'wf13/loop/check/out/'
K, N = 6, 36
r_, c_ = np.arange(N) // K, np.arange(N) % K
b_ = (r_ // 2) * 2 + c_ // 3
PEER = [[j for j in range(N) if j != i and (r_[i] == r_[j] or c_[i] == c_[j] or b_[i] == b_[j])] for i in range(N)]


def solve_all(puz, cap=5000, rnd=None):
    b = [int(v) for v in puz]
    out = []

    def rec():
        if len(out) >= cap:
            return
        best, bo = -1, None
        for i in range(N):
            if b[i] < 0:
                used = {b[p] for p in PEER[i]}
                o = [v for v in range(K) if v not in used]
                if bo is None or len(o) < len(bo):
                    best, bo = i, o
        if best < 0:
            out.append(list(b))
            return
        if rnd is not None:
            rnd.shuffle(bo)
        for v in bo:
            b[best] = v
            rec()
            if len(out) >= cap:
                break
        b[best] = -1
    rec()
    return np.array(out, dtype=np.int64).reshape(-1, N)


def gen(rnd, unique):
    full = solve_all(np.full(N, -1), cap=1, rnd=rnd)[0]
    p = full.copy()
    order = rnd.permutation(N)
    if unique:                                   # remove clues while the solution stays unique (minimal puzzle)
        for i in order:
            keep = p[i]
            p[i] = -1
            if len(solve_all(p, cap=2)) != 1:
                p[i] = keep
        return p
    nk = int(rnd.integers(7, 10))                # few clues -> several solutions; keep those with 2..400
    p[order[nk:]] = -1
    return p


class Anti:
    """Adversary that KNOWS the solutions: puts its value probability on values that occur in no solution where it
    can, and opens the cell with the most candidates first."""
    calls_per_state = 0

    def __init__(self, sols):
        self.in_sol = np.stack([(sols == v).any(0) for v in range(K)], -1)       # [36, 6]

    def __call__(self, b, c, rng):
        q = np.where(self.in_sol[None], 1e-6, 1.0) * c
        q = q / np.maximum(q.sum(-1, keepdims=True), 1e-300)
        j = np.where(b < 0, c.sum(-1) + 1e-3 * rng.random(b.shape), -1.0).argmax(1)
        return q, j


class Scramble:
    """Same ORDER as the wrapped policy, value probabilities replaced by random ones."""
    def __init__(self, pol):
        self.pol, self.calls_per_state = pol, pol.calls_per_state

    def __call__(self, b, c, rng):
        q, j = self.pol(b, c, rng)
        return np.random.default_rng(int(j.sum()) + len(j)).dirichlet(np.ones(K), size=q.shape[:2]), j


class Oracle:
    """Perfect value head (one-hot on the unique solution) under the registered order rule 'most confident cell'."""
    calls_per_state = 0

    def __init__(self, sol, order):
        self.p = (sol[:, None] == np.arange(K)[None]).astype(float)
        self.order = order

    def __call__(self, b, c, rng):
        p = np.repeat(self.p[None], len(b), 0)
        emp = b < 0
        if self.order == 'conf':
            j = np.where(emp, p.max(-1) + 1e-6 * rng.random(b.shape), -1.0).argmax(1)
        else:
            j = np.where(emp, c.sum(-1) + 1e-3 * rng.random(b.shape), 99.0).argmin(1)
        return p, j


def exact_books(T):
    """The interval recomputed with integers from the tree's own frontier (my formula, no floats)."""
    st = T.status[:T.N]
    sol = T.board[:T.N][st == 2]
    ZA = np.array([[(sol[:, j] == v).sum() for v in range(K)] for j in range(N)], dtype=object)
    Z = len(sol)
    UA = np.zeros((N, K), dtype=object)
    UN = np.zeros((N, K), dtype=object)
    Uu = np.zeros((N, K), dtype=object)
    for i in np.flatnonzero(st == 0):
        b, c = T.board[i], T.cand[i]
        U = 1
        for j in range(N):
            if b[j] < 0:
                U *= int(c[j].sum())
        for j in range(N):
            for v in range(K):
                if b[j] == v:
                    UA[j, v] += U
                elif b[j] < 0 and c[j, v]:
                    Uu[j, v] += U
                else:
                    UN[j, v] += U
    return ZA, Z, UA, UN, Uu


def main():
    nu, nm = int(sys.argv[1]), int(sys.argv[2])
    rnd = np.random.default_rng(987654321)
    torch.set_num_threads(2)
    rules = sk.Rules(np.load(ROOT + 'wf13/loop/out/sudoku_rules.npz')['allowed'])
    true = np.ones((N, K, N, K), dtype=bool)
    for i in range(N):
        for u in range(K):
            true[i, u, i, :] = False
            true[i, u, i, u] = True
            for j in PEER[i]:
                true[i, u, j, u] = False
    diff = int((true != rules.allowed).sum())
    offdiag = int(sum((true[i, :, j, :] != rules.allowed[i, :, j, :]).sum() for i in range(N) for j in range(N) if i != j))
    print('rule table read from M vs my true rules: %d entries differ (%d of them with i != j)' % (diff, offdiag), flush=True)
    puzzles = []
    while sum(1 for p in puzzles if p[2] == 'unique') < nu:
        p = gen(rnd, True)
        puzzles.append((p, solve_all(p), 'unique'))
    while sum(1 for p in puzzles if p[2] == 'multi') < nm:
        p = gen(rnd, False)
        s = solve_all(p, cap=401)
        if 2 <= len(s) <= 400:
            puzzles.append((p, s, 'multi'))
    tr = np.load(ROOT + 'wf7/sudoku/data/k6_train_min.npz')
    seen = {tuple(s) for s in tr['sol'].tolist()}
    print('fresh puzzles: %d unique (mean clues %.1f), %d multi (mean %.1f solutions); solution grids also in the training file: %d'
          % (nu, np.mean([(p[0] >= 0).sum() for p in puzzles if p[2] == 'unique']), nm,
             np.mean([len(p[1]) for p in puzzles if p[2] == 'multi']),
             sum(tuple(p[1][0]) in seen for p in puzzles if p[2] == 'unique')), flush=True)
    nets = {}
    for s in (1, 2):
        sd = torch.load(ROOT + 'wf13/loop/ckpt/s_s%d.pt' % s, map_location='cpu')
        for k in sd:
            n = spol.SNet(48, 3)
            n.load_state_dict(sd[k])
            n.eval()
            nets[(s, k)] = n
    marks = [16, 32, 64, 128, 256, 512, 1024]
    res = {}
    # ---------------- Part A
    armsA = [('hand_mrv', lambda s: spol.Hand('mrv'), 0.5, 0.02), ('random', lambda s: spol.Hand('random'), 0.5, 0.02),
             ('net5_s1', lambda s: spol.NetPolicy(nets[(1, '5')]), 0.5, 0.02), ('net0_s2', lambda s: spol.NetPolicy(nets[(2, '0')]), 0.5, 0.02),
             ('avoid_noguard_s2', lambda s: spol.NetPolicy(nets[(2, '5')], 'avoid'), 0.0, 0.0),
             ('perm_s1', lambda s: spol.NetPolicy(nets[(1, '5')], 'perm'), 0.5, 0.02),
             ('anti_noguard', lambda s: Anti(s), 0.0, 0.0), ('anti_share', lambda s: Anti(s), 0.5, 0.0)]
    for an, mk_pol, rho, eps in armsA:
        t0 = time.time()
        st = dict(intervals=0, out_tol=0, out_strict=0, float_lo_above_exact=0, float_hi_below_exact=0, exact_out=0,
                  exact_checked=0, not_nested=0, closed=0, closed_same_set=0, wrong_solution=0, C=[])
        for pi, (p, sols, kind) in enumerate(puzzles):
            rng = np.random.default_rng(555 * (pi + 1))
            pol = mk_pol(sols)
            exm = np.stack([(sols == v).mean(0) for v in range(K)], -1)
            emp = p < 0
            T = sk.Tree(rules, p, 6 * marks[-1] + 64)
            prev = None
            for mk in marks:
                sk.grow(T, pol, mk, rng, rho=rho, eps=eps)
                bk = sk.books(T)
                lo, hi = bk['lo'], bk['hi']
                st['intervals'] += int(emp.sum()) * K
                st['out_tol'] += int(((exm < lo - 1e-9) | (exm > hi + 1e-9))[emp].sum())
                st['out_strict'] += int(((exm < lo) | (exm > hi))[emp].sum())
                if prev is not None:
                    st['not_nested'] += int(((lo < prev[0] - 1e-12) | (hi > prev[1] + 1e-12))[emp].sum())
                prev = (lo, hi)
                if mk in (64, 1024) and pi % 4 == 0:                     # exact-integer recomputation on a subset
                    ZA, Z, UA, UN, Uu = exact_books(T)
                    for j in np.flatnonzero(emp):
                        for v in range(K):
                            dl, dh = Z + UN[j, v] + Uu[j, v], ZA[j, v] + UA[j, v] + Uu[j, v] + (Z - ZA[j, v])
                            elo = Fraction(int(ZA[j, v]), int(dl)) if dl > 0 else Fraction(0)
                            ehi = Fraction(int(ZA[j, v] + UA[j, v] + Uu[j, v]), int(dh)) if dh > 0 else Fraction(1)
                            tru = Fraction(int((sols[:, j] == v).sum()), len(sols))
                            st['exact_checked'] += 1
                            st['exact_out'] += int(not (elo <= tru <= ehi))
                            st['float_lo_above_exact'] += int(Fraction(float(lo[j, v])) > elo and float(lo[j, v]) - float(elo) > 1e-12)
                            st['float_hi_below_exact'] += int(Fraction(float(hi[j, v])) < ehi and float(ehi) - float(hi[j, v]) > 1e-12)
            got = {tuple(s) for s in bk['sol'].tolist()}
            tru = {tuple(s) for s in sols.tolist()}
            st['wrong_solution'] += len(got - tru)
            if bk['closed']:
                st['closed'] += 1
                st['closed_same_set'] += int(got == tru)
                if kind == 'unique':
                    st['C'].append(T.opened)
        st['C_mean_unique_closed'] = float(np.mean(st['C'])) if st['C'] else None
        st.pop('C')
        res['A_' + an] = st
        print('A %-18s rho=%.1f eps=%.2f %s | %.0fs' % (an, rho, eps, json.dumps(st), time.time() - t0), flush=True)
        json.dump(res, open(OUT + 'c2_sudoku.json', 'w'))
    # ---------------- Parts B and C on the unique fresh puzzles (cap 2048)
    d = np.load(ROOT + 'wf7/sudoku/data/k6_test_min.npz')
    hi_ = np.where(d['rating'] == 2)[0]
    hi_ = np.concatenate([hi_[250:300], hi_[16:66]])          # hard test puzzles the author's runs never scored
    uni = [(d['puz'][i], solve_all(d['puz'][i])) for i in hi_]
    assert all(len(s) == 1 for _, s in uni)
    print('parts B, C, D: %d hard wf7 test puzzles outside the registered (150:250) and smoke (0:16) ranges' % len(uni), flush=True)

    def cost(mk_pol, tag):
        Cs = []
        for pi, (p, sols) in enumerate(uni):
            rng = np.random.default_rng(777 * (pi + 1))
            T = sk.Tree(rules, p, 6 * 2048 + 64)
            sk.grow(T, mk_pol(sols), 2048, rng, rho=0.5, eps=0.02)
            Cs.append(T.opened if T.n_open() == 0 else 2048)
        res[tag] = dict(C_mean=float(np.mean(Cs)), C_median=float(np.median(Cs)), per=Cs)
        print('%-44s C mean %.1f median %.1f' % (tag, np.mean(Cs), np.median(Cs)), flush=True)
        json.dump(res, open(OUT + 'c2_sudoku.json', 'w'))
        return np.array(Cs)
    for s in (1, 2):
        a = cost(lambda sol: spol.NetPolicy(nets[(s, '5')]), 'B_net5_s%d_own_values' % s)
        b = cost(lambda sol: Scramble(spol.NetPolicy(nets[(s, '5')])), 'B_net5_s%d_same_order_random_values' % s)
        print('   per-puzzle C: correlation %.4f, mean abs difference %.2f, identical on %d of %d' % (np.corrcoef(a, b)[0, 1], np.abs(a - b).mean(), (a == b).sum(), len(a)), flush=True)
    cost(lambda sol: spol.Hand('mrv'), 'C_hand_mrv')
    cost(lambda sol: spol.Hand('random'), 'C_hand_random_order')
    cost(lambda sol: Oracle(sol[0], 'conf'), 'C_perfect_value_head_registered_order')
    cost(lambda sol: Oracle(sol[0], 'mrv'), 'C_perfect_value_head_mrv_order')
    # ---------------- Part D: C by round on these fresh hard puzzles, with how often the chosen cell is a
    # fewest-candidates cell and how saturated the confidence is
    class Rec:
        def __init__(self, pol):
            self.pol, self.calls_per_state, self.agree, self.n, self.conf = pol, pol.calls_per_state, 0, 0, []

        def __call__(self, b, c, rng):
            q, j = self.pol(b, c, rng)
            emp = b < 0
            cnt = np.where(emp, c.sum(-1), 99)
            self.agree += int((cnt[np.arange(len(b)), j] == cnt.min(1)).sum())
            self.n += len(b)
            self.conf.append(float(np.median(q.max(-1)[emp])))
            return q, j
    for s in (1, 2):
        for k in ['0', '1', '2', '3', '4', '5', 'oneshot']:
            rec = Rec(spol.NetPolicy(nets[(s, k)]))
            cost(lambda sol: rec, 'D_seed%d_round_%s' % (s, k))
            res['D_seed%d_round_%s' % (s, k)].update(mrv_agree=rec.agree / rec.n, median_conf=float(np.mean(rec.conf)))
            print('      chosen cell is a fewest-candidates cell: %.3f ; median confidence over empty cells: %.3f' % (rec.agree / rec.n, np.mean(rec.conf)), flush=True)
    json.dump(res, open(OUT + 'c2_sudoku.json', 'w'))
    print('done', flush=True)


if __name__ == '__main__':
    main()
