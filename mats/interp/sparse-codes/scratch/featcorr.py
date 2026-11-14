"""One-line check of the coordinator's analytic argument (retarget 1), on saved SAEs: how dense are the couplings among
SAE features, and what does dropping them (an independent-feature product) do to the next layer's per-neuron variances?
For each saved SAE (arc, seed 65, layers 8/16/24): rho_rms over pairs of alive features; the predicted relative error
sqrt(2) rho_rms (for v_i ~ equal); and the measured median |Var_diag(z'_k) / Var_full(z'_k) - 1| with z' = W' (D f).
Same for the neuron basis. usage: featcorr.py RUNDIR N [layers]"""
import glob, json, math, os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import nets, sae
torch.set_num_threads(int(os.environ.get('NT', '2')))
rundir, n = sys.argv[1], int(sys.argv[2])
layers = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [8, 16, 24]
net = nets.SNet('arc', n, 65, 0.01)
g = torch.Generator().manual_seed(4321)
N = 1 << 15
X = net.sample(N, g, layers)
out = []


def coupling_stats(Fz, Dmat, Wn):
    """Fz (N, F) codes; Dmat (n, F) decoder (original units); Wn (n, n) next-layer weights."""
    alive = Fz.abs().sum(0) > 0
    Fz = Fz[:, alive].double(); D = Dmat[:, alive].double()
    C = torch.cov(Fz.T)
    sd = C.diagonal().clamp_min(1e-300).sqrt()
    R = C / (sd[:, None] * sd[None, :])
    off = ~torch.eye(R.shape[0], dtype=torch.bool)
    rho_rms = float((R[off] ** 2).mean().sqrt())
    # on/off phi among features firing in 1-99% of samples (comparable with the gate-level median |phi| 0.05-0.08)
    on = (Fz > 0).double(); pr = on.mean(0); mid = (pr > 0.01) & (pr < 0.99)
    phi_med = phi_rms = None
    if int(mid.sum()) >= 2:
        Pm = torch.corrcoef(on[:, mid].T); offm = ~torch.eye(Pm.shape[0], dtype=torch.bool)
        phi_med = float(Pm[offm].abs().median()); phi_rms = float((Pm[offm] ** 2).mean().sqrt())
    # energy double counting: sum of feature variances vs the variance of the decoded state
    energy_ratio = float((C.diagonal() * (D * D).sum(0)).sum() / torch.trace(D @ C @ D.T))
    A = Wn.double() @ D                                   # next-layer loadings of each feature
    vfull = ((A @ C) * A).sum(1)
    vdiag = (A * A * C.diagonal()[None]).sum(1)
    rel = (vdiag / vfull - 1).abs()
    return dict(alive=int(alive.sum()), rho_rms=rho_rms, pred_rel=math.sqrt(2) * rho_rms, phi_absmed=phi_med,
                phi_rms=phi_rms, n_mid=int(mid.sum()), energy_ratio=energy_ratio,
                var_relerr_med=float(rel.median()), var_relerr_p90=float(rel.quantile(0.9)))


for l in layers:
    A = X[l]
    Wn = net.W[l].double()                                # W_{l+1}
    st = coupling_stats(A - A.mean(0), torch.eye(n), Wn)
    st.update(layer=l, basis='neurons'); out.append(st)
    print(st, flush=True)
    for p in sorted(glob.glob(os.path.join(rundir, f'sae_L{l}_*F*.pt'))):
        ck = torch.load(p)
        F = ck['F']; origin = ck.get('origin', 'mean')
        model = sae.BatchedTopK(n, F, ck['ks'], torch.zeros(n), 'cpu', origin=origin)
        model.load_state_dict({k: v.float() for k, v in ck['state'].items()})
        with torch.no_grad():
            f, _, _, _ = model.encode(A / ck['c'])
        for si, k in enumerate(ck['ks']):
            st = coupling_stats(f[si] * ck['c'], model.W_dec[si].detach(), Wn)
            st.update(layer=l, basis=f'SAE_{origin}_F{F}_k{k}'); out.append(st)
            print(st, flush=True)
json.dump(out, open(os.path.join(HERE, '..', 'results', f'featcorr_n{n}.json'), 'w'), indent=1)
