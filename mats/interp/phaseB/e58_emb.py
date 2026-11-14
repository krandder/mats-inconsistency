"""E58 designed input embeddings (PREREG-B amendment 66): a frozen input embedding whose LayerNorm cap scale c is set by design.
Reference: the input embedding (inp.weight W, 32 x 36; inp.bias b) of E53's unif net of seed 0 (ctrl format; trained; c 1.39,
A's coefficient of variation 0.01, |rho| about 0). For a target cap scale c_t, kappa = c_ref / c_t and, in the sep format (e56_fmt:
summary readings in column 36):
  column 32 (o, now only the labelled tokens' y + r) -> kappa W[:, 32];
  column 33 (y)                                      -> W[:, 33] + (1 - kappa) W[:, 32]   (so y's total weight is unchanged);
  column 36 (summary readings)                       -> W[:, 32]                          (the readings enter as in the reference);
  every other column and the bias unchanged.
A labelled token's embedding is then a_i + kappa r_i u (a_i unchanged), so c = A / (kappa U) = c_ref / kappa = c_t exactly, the
summary and query tokens' embeddings equal the reference's, and only the error direction's length changes.
usage: python e58_emb.py OUTDIR C1 C2 ... -> OUTDIR/emb_c<C>.npz (W 32 x 37, b) and a printed check of c by lngeom.py
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
import e56_fmt as F  # noqa: E402
import lngeom as G  # noqa: E402

REF = 'runs/e53/e53_unif_s0/model.pt'


def main():
    out = sys.argv[1]; targets = [float(x) for x in sys.argv[2:]]
    os.makedirs(out, exist_ok=True)
    sd = torch.load(REF, map_location='cpu'); W = sd['inp.weight'].numpy().astype(np.float64); b = sd['inp.bias'].numpy().astype(np.float64)
    z = np.load('results/e27_data.npz')
    Xc, _ = F.make_X(z['o'][:500, 0], z['y'][:500], z['semb'][:500], z['qemb'][:500], 'ctrl')
    Xs, _ = F.make_X(z['o'][:500, 0], z['y'][:500], z['semb'][:500], z['qemb'][:500], 'sep')
    c_ref = G.geom_wb(W, b, Xc)['c_median']
    for ct in targets:
        k = c_ref / ct
        W2 = np.zeros((W.shape[0], 37)); W2[:, :36] = W
        W2[:, 32] = k * W[:, 32]; W2[:, 33] = W[:, 33] + (1 - k) * W[:, 32]; W2[:, 36] = W[:, 32]
        g = G.geom_wb(W2, b, Xs)
        np.savez(os.path.join(out, 'emb_c%g.npz' % ct), W=W2.astype(np.float32), b=b.astype(np.float32), kappa=k, c_ref=c_ref, c=g['c_median'])
        print('target c %.2f: kappa %.3f, lngeom c %.3f, A cv %.3f, |rho| %.3f' % (ct, k, g['c_median'], g['A_cv'], g['rho_abs_median']))


if __name__ == '__main__':
    main()
