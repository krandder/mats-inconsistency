"""c8_mask.py -- checker's re-run of the POST HOC 'confmask' order on fresh hard puzzles (author's checkpoints
ckpt/s_s{1,2}_mask.pt), plus: is the untrained network under that rule simply the fewest-values-left rule?"""
import sys
import numpy as np, torch
sys.dont_write_bytecode = True
ROOT = '/home/agent/workspaces/mats/work/deep-study/'
sys.path.insert(0, ROOT + 'wf13/loop')
import sk, spol
torch.set_num_threads(2)
rules = sk.Rules(np.load(ROOT + 'wf13/loop/out/sudoku_rules.npz')['allowed'])
d = np.load(ROOT + 'wf7/sudoku/data/k6_test_min.npz')
hi_ = np.where(d['rating'] == 2)[0]
hard = d['puz'][np.concatenate([hi_[250:300], hi_[16:66]])]
class Rec:
    def __init__(self, pol):
        self.pol, self.calls_per_state, self.agree, self.n = pol, pol.calls_per_state, 0, 0
    def __call__(self, b, c, rng):
        q, j = self.pol(b, c, rng)
        cnt = np.where(b < 0, c.sum(-1), 99)
        self.agree += int((cnt[np.arange(len(b)), j] == cnt.min(1)).sum()); self.n += len(b)
        return q, j
for s in (1, 2):
    sd = torch.load(ROOT + 'wf13/loop/ckpt/s_s%d_mask.pt' % s, map_location='cpu')
    row = []
    for k in ['0', '1', '2', '3', '4', '5', 'oneshot']:
        net = spol.SNet(48, 3); net.load_state_dict(sd[k]); net.eval()
        rec = Rec(spol.NetPolicy(net, order='confmask'))
        Cs = []
        for i, p in enumerate(hard):
            rng = np.random.default_rng(321 * (i + 1) + s)
            T = sk.Tree(rules, p, 6 * 2048 + 64)
            sk.grow(T, rec, 2048, rng, rho=0.5, eps=0.02)
            Cs.append(T.opened if T.n_open() == 0 else 2048)
        row.append('%s: C %.0f, fewest-values cell chosen %.2f' % (k, np.mean(Cs), rec.agree / rec.n))
    print('seed %d confmask | ' % s + ' | '.join(row), flush=True)
