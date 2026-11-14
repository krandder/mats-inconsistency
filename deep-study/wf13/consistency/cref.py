"""cref.py WORLD CKPT SIZE FILE KREF : recompute the score's reference value log R-hat(x) of a c2 / t3 file with KREF
searcher trajectories (cgen.score_ref) and store it as `logRhat`; the earlier value is kept as `logRhat_old`.
Search side only: no exact inference."""
import os
import sys

import torch

sys.dont_write_bytecode = True
import cgen as G  # noqa: E402

if __name__ == '__main__':
    ws, ckpt, size, path, kref = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    torch.set_num_threads(int(os.environ.get('NTHREADS', 1)))
    dev = torch.device('cpu')
    ck = torch.load(ckpt, map_location=dev)
    m = G.MM.build(size, G.N_VAR)
    m.load_state_dict(ck['state'])
    m.eval()
    loga = torch.load(os.path.dirname(path) + '/score.pt')['loga']
    d = torch.load(path)
    gen = torch.Generator()
    gen.manual_seed(ws * 100000 + 777 + sum(ord(c) for c in os.path.basename(path)))
    new = G.score_ref(m, loga, d['xvec'].long(), kref, gen)
    d['logRhat_old'], d['logRhat'] = d['logRhat'], new
    torch.save(d, path)
    print(os.path.basename(path), 'Kref', kref, 'mean |new - old| %.3f' % float((new - d['logRhat_old']).abs().mean()),
          'calls', G.CALLS['n'], flush=True)
