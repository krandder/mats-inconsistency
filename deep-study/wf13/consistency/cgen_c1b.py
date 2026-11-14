"""cgen_c1b.py WORLD CKPT OUTDIR SHARD NTRIPLES (POST HOC, not in PREREG) : more unlabelled triples from M's own samples,
with 8 trajectories per context (the registered set c1 has 4), so that two DISJOINT trajectory draws exist at K = 4.
Used only by the post hoc noise-corrected pair term (ctrain_ph.py, arms idb and ib). No exact inference."""
import os
import sys

import numpy as np
import torch

sys.dont_write_bytecode = True
import cgen as G  # noqa: E402

if __name__ == '__main__':
    ws, ckpt, outdir, k, ntr = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    torch.set_num_threads(int(os.environ.get('NTHREADS', 2)))
    dev = torch.device('cpu')
    m = G.MM.build('M', G.N_VAR)
    m.load_state_dict(torch.load(ckpt, map_location=dev)['state'])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    loga = torch.load(outdir + '/score.pt')['loga']
    gen = torch.Generator()
    gen.manual_seed(ws * 100000 + 555001 + k)
    rng = np.random.default_rng(ws * 1000 + 555 + k)
    t = G.triple_set(m, loga, G.m_samples(m, loga, ntr, gen, dev), rng, 8, gen, dev)
    torch.save(t, '%s/c1b_%d.pt' % (outdir, k))
    print('c1b shard', k, 'triples', ntr, 'calls', G.CALLS['n'], flush=True)
