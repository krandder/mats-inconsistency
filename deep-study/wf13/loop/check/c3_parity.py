"""c3_parity.py -- checker's own tests in the parity world, on FRESH instances (my seed), with my own enumeration.
Code under test: the author's pw.py / ptree.py / ppol.py and checkpoints. Mine: instances, exact answers, adversary,
all counting. Usage: c3_parity.py WORLD N_INST
Part A: containment of the exact answer in the certain interval, bounds on the total, nestedness, audit misses.
Part B: the importance form under the avoiding policy against what it converges to BY CONSTRUCTION, T(. | avoided = 0).
Part C: rounds 0, 1, 5 and the one-shot control: E_IS(64), audit width / error at 256 openings; specification alone;
        one pass of M on the same fresh instances."""
import json, sys, time
import numpy as np
import torch
sys.dont_write_bytecode = True
ROOT = '/home/agent/workspaces/mats/work/deep-study/'
sys.path.insert(0, ROOT + 'wf13/loop')
import pdata, pw, ptree, ppol, ploop          # noqa: E402
from pw import UN                              # noqa: E402

OUT = ROOT + 'wf13/loop/check/out/'
ws, n_inst = int(sys.argv[1]), int(sys.argv[2])
tab = pdata.load_tab(ws)
n, m = tab['loga'].shape[0], tab['checks'].shape[0]
W = pdata.world(ws, tab)
assert (W.checks == tab['checks']).all()
# fresh real check vectors: my seed; drop any seen in M's training or in the loop's training rounds
_, xm = W.sample(50000, ws * 1000 + 1)
seen = set(pdata.pack(xm).tolist())
for r in range(5):
    seen |= set(pdata.pack(pdata.fresh_x(ws, tab, 128, 50 + r)).tolist())
xt, _ = pdata.test_x(ws, tab, 60)
seen |= set(pdata.pack(xt).tolist())
_, xs = W.sample(4 * n_inst, ws * 1000 + 941)
X = np.array([x for x, k in zip(xs, pdata.pack(xs).tolist()) if k not in seen][:n_inst])
print('world %d: %d fresh instances (none in M training, loop training or the author test set)' % (ws, len(X)), flush=True)

# ---- my enumeration of the score (2^n assignments)
z = np.arange(2 ** n, dtype=np.int64)
bits = ((z[:, None] >> np.arange(n)[None]) & 1).astype(np.float64)
cfg = np.zeros((2 ** n, m), dtype=np.int64)
for t in range(3):
    cfg += (((z[:, None] >> tab['checks'][None, :, t]) & 1) << t)


def exact(x, cond0=()):
    lr = bits @ (tab['loga'][:, 1] - tab['loga'][:, 0]) + tab['loga'][:, 0].sum()
    for c in range(m):
        lr = lr + tab['logf'][c, :, int(x[c])][cfg[:, c]]
    mx = lr.max()
    w = np.exp(lr - mx)
    out = [(w @ bits) / w.sum(), mx + np.log(w.sum())]
    if len(cond0):
        w2 = w * (bits[:, list(cond0)].sum(1) == 0)
        out.append((w2 @ bits) / w2.sum())
        out.append(float(w2.sum() / w.sum()))
    return out


sd = torch.load(ROOT + 'wf13/loop/ckpt/p_w%d_s1.pt' % ws, map_location='cpu')
nets = {}
for k in sd:
    g = ppol.GNN(32, 3)
    g.load_state_dict(sd[k])
    g.eval()
    nets[str(k)] = g
torch.set_num_threads(2)
AV = list(range(5))


class Anti:
    """Adversary that knows the exact posterior: proposes the unlikely value with probability 0.999, random order."""
    calls_per_state = 0

    def __init__(self, pstar):
        self.q = np.where(pstar > 0.5, 0.001, 0.999)

    def __call__(self, inst, e, rng):
        q = np.where(e['op'], self.q[None], e['p1'])
        return q, ppol.pick_random(e['op'], rng)


marks = [16, 64, 256, 1024]
EX = [exact(x, AV) for x in X]
insts = [pw.Inst(tab, x) for x in X]
res = dict(world=ws, n_inst=len(X))
armsA = [('round5', lambda i: ppol.NetPolicy(nets['5'], order='greedy1'), 0.5, 0.02),
         ('round0', lambda i: ppol.NetPolicy(nets['0'], order='greedy1'), 0.5, 0.02),
         ('round1', lambda i: ppol.NetPolicy(nets['1'], order='greedy1'), 0.5, 0.02),
         ('oneshot', lambda i: ppol.NetPolicy(nets['oneshot'], order='greedy1'), 0.5, 0.02),
         ('spec_alone', lambda i: ppol.Hand('greedy1', 'books'), 1.0, 0.02),
         ('random', lambda i: ppol.Hand('random', 'uniform'), 0.5, 0.02),
         ('flip', lambda i: ppol.NetPolicy(nets['5'], 'flip', order='greedy1'), 0.5, 0.02),
         ('avoid_noguard', lambda i: ppol.NetPolicy(nets['5'], 'avoid', AV, order='greedy1'), 0.0, 0.0),
         ('net5_conforder', lambda i: ppol.NetPolicy(nets['5'], order='conf'), 0.5, 0.02),
         ('anti_noguard', lambda i: Anti(EX[i][0]), 0.0, 0.0), ('anti_share', lambda i: Anti(EX[i][0]), 0.5, 0.0)]
for an, mk_pol, rho, eps in armsA:
    t0 = time.time()
    st = dict(intervals=0, out_cert=0, out_cert_strict=0, Z_bounds=0, out_Z=0, not_nested=0, aud_miss=0)
    wa, wc, ea, fd = {mk: [] for mk in marks}, {mk: [] for mk in marks}, {mk: [] for mk in marks}, {mk: [] for mk in marks}
    for i, inst in enumerate(insts):
        ps, lz = EX[i][0], EX[i][1]
        rng = np.random.default_rng(4242 * (i + 1) + ws)
        T = ploop.new_tree(inst, 2 * max(marks) + 8)
        pol = mk_pol(i)
        prev = None
        for mk in marks:
            ptree.grow(T, pol, mk, rng, rho=rho, eps=eps)
            bk = ptree.books(T)
            au = ptree.audit(T, bk, 4096, rng)
            st['intervals'] += n
            st['out_cert'] += int(((ps < bk['lo'] - 1e-9) | (ps > bk['hi'] + 1e-9)).sum())
            st['out_cert_strict'] += int(((ps < bk['lo']) | (ps > bk['hi'])).sum())
            st['Z_bounds'] += 1
            st['out_Z'] += int(not (bk['logZ_lo'] - 1e-7 <= lz <= bk['logZ_hi'] + 1e-7))
            st['aud_miss'] += int(((ps < au['lo'] - 1e-9) | (ps > au['hi'] + 1e-9)).sum())
            if prev is not None:
                st['not_nested'] += int(((bk['lo'] < prev[0] - 1e-12) | (bk['hi'] > prev[1] + 1e-12)).sum())
            prev = (bk['lo'], bk['hi'])
            apt = np.clip(au.get('pt', bk['pt']), au['lo'], au['hi'])
            wa[mk].append(float((au['hi'] - au['lo']).mean()))
            wc[mk].append(float((bk['hi'] - bk['lo']).mean()))
            ea[mk].append(float(np.abs(apt - ps).mean()))
            fd[mk].append(float(np.exp(np.log(max(bk['Z'], 1e-300)) + T.logU[0] - lz)))
    st.update(w_aud={mk: float(np.mean(v)) for mk, v in wa.items()}, w_cert={mk: float(np.mean(v)) for mk, v in wc.items()},
              err_aud={mk: float(np.mean(v)) for mk, v in ea.items()}, found={mk: float(np.mean(v)) for mk, v in fd.items()},
              per_inst_w_aud256=wa[256])
    res['A_' + an] = st
    print('A %-15s rho=%.1f eps=%.2f | intervals %d outside %d (strict %d) | Z bounds %d outside %d | not nested %d | audit misses %d | '
          'w_aud@256 %.3f @1024 %.3f | w_cert@256 %.3f | err_aud@256 %.4f | found@256 %.2f | %.0fs'
          % (an, rho, eps, st['intervals'], st['out_cert'], st['out_cert_strict'], st['Z_bounds'], st['out_Z'], st['not_nested'], st['aud_miss'],
             st['w_aud'][256], st['w_aud'][1024], st['w_cert'][256], st['err_aud'][256], st['found'][256], time.time() - t0), flush=True)
    json.dump(res, open(OUT + 'c3_parity_w%d.json' % ws, 'w'))

# ---- Part B and the importance form of Part C
def is_err(pol, K, eps, i, ref):
    rng = np.random.default_rng(999 * (i + 1) + ws)
    logw, bel, calls = ptree.run_is(insts[i], pol, K, rng, eps)
    w = np.exp(logw - logw.max())
    b = (w[:, None] * bel).sum(0) / w.sum()
    return [float(np.abs(b - r).mean()) for r in ref], float(w.sum() ** 2 / (w * w).sum())


t0 = time.time()
rows = []
for i in range(len(X)):
    ps, _, pc, mass = EX[i]
    e, ess = is_err(ppol.NetPolicy(nets['5'], 'avoid', AV, order='conf'), 256, 0.0, i, [ps, pc])
    rows.append([e[0], e[1], float(np.abs(pc - ps).mean()), mass, ess])
R = np.array(rows).mean(0)
res['B_avoid_noguard_IS256'] = dict(err_vs_T=R[0], err_vs_T_given_avoided_zero=R[1], by_construction_gap=R[2],
                                    T_mass_of_supported_part=R[3], ess=R[4])
print('B avoid, no guard, importance form, 256 rollouts: error against T %.3f | against T(. | the five avoided = 0) %.3f | '
      'distance between those two laws (fixed before any rollout) %.3f | T-mass of the part it can reach %.3f | ESS %.1f | %.0fs'
      % (R[0], R[1], R[2], R[3], R[4], time.time() - t0), flush=True)
for k in ['0', '1', '2', '3', '4', '5', 'oneshot']:
    v = [is_err(ppol.NetPolicy(nets[k], order='conf'), 64, 0.02, i, [EX[i][0]])[0][0] for i in range(len(X))]
    res['C_IS64_round_' + k] = dict(err=float(np.mean(v)), se=float(np.std(v) / np.sqrt(len(v))), per=v)
    print('C importance form, 64 rollouts, round %-7s: error %.4f (s.e. %.4f)' % (k, np.mean(v), np.std(v) / np.sqrt(len(v))), flush=True)
v = [is_err(ppol.Hand('conf', 'books'), 64, 0.02, i, [EX[i][0]])[0][0] for i in range(len(X))]
print('C importance form, 64 rollouts, hand (books numbers, confident first): error %.4f' % np.mean(v), flush=True)
res['C_IS64_hand'] = float(np.mean(v))
# one pass of the rollout policy and of M on the same fresh instances
for k in ['0', '5']:
    pol = ppol.NetPolicy(nets[k], order='conf')
    v = [float(np.abs(ploop.one_pass(insts[i], pol) - EX[i][0]).mean()) for i in range(len(X))]
    res['C_onepass_policy_round_' + k] = float(np.mean(v))
    print('C one pass of the rollout policy, round %s: error %.4f' % (k, np.mean(v)), flush=True)
try:
    sys.path.insert(0, ROOT + 'wf7/probworld')
    import model as MM, arms as A      # noqa: E402
    net = MM.build('M', n + m)
    net.load_state_dict(torch.load(ROOT + 'wf7/probworld/ckpt/M_w%d_M.pt' % ws, map_location='cpu')['state'])
    net.eval()
    A.N_H = n
    b = A.one_pass(net, torch.tensor(X)).double().numpy()
    pst = np.stack([e[0] for e in EX])
    res['C_onepass_M'] = float(np.abs(b - pst).mean())
    print('C one pass of M on the fresh instances: error %.4f' % res['C_onepass_M'], flush=True)
except Exception as ex:                # noqa: BLE001
    print('one pass of M not available:', repr(ex)[:200], flush=True)
res['C_prior'] = float(np.mean([np.abs(np.exp(tab['loga'][:, 1]) - e[0]).mean() for e in EX]))
print('C the score prior numbers: error %.4f' % res['C_prior'], flush=True)
json.dump(res, open(OUT + 'c3_parity_w%d.json' % ws, 'w'))
print('done', flush=True)
