"""E68 (PREREG-B amendment 76, planned; CPU): a reduced pooling model (RPM) of block 1's pooling heads, read by constrained Bayes.
For a source's four labelled errors r_j, LayerNorm with cap scale c gives each token the norm factor A / n_j = (1 + r_j^2 / c^2)^(-1/2)
(rho = 0). A pooling head h with mean norm-scaled own logit L_h and mean signed coefficient K_h (E63's measures) weights the four own
tokens by w_j = softmax_j((L_h + K_h r_j) A / n_j) (in the mag format the signed feature is |r_j|), and the model keeps, per head, the
even pooled feature E_h = sum_j w_j A / n_j and the signed one O_h = sum_j w_j r_j A / n_j (|r_j| in mag). A uniform pool is L = K = 0.
A constrained-Bayes reader sees, for each source, only T = (E_h, O_h) over the net's pooling heads, plus every query reading: p(v | T)
from a classifier (histogram gradient boosting, 3 levels, uniform prior) trained on simulated sources, then e53_cb's combination
over the 27 variance combinations and its effective weights; rom on E12's pairs as E12.
usage (module): rpm_rom(heads, c, fmt) with heads = [(L, K), ...]; python e68_rpm.py check  (pipeline checks: SS and the uniform
capped feature against e53_cb / amendment 63's table)
"""
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
import world as W  # noqa: E402
import e53_cb as CB  # noqa: E402

QL = W.QL
NSIM = 200_000


def rpm_stat(r, heads, c, fmt):
    """r (..., 4) errors -> T (..., 2 * len(heads))."""
    g = 1.0 / np.sqrt(1.0 + r ** 2 / c ** 2)                    # A / n_j
    s = np.abs(r) if fmt == 'mag' else r
    out = []
    for L, K in heads:
        z = (L + K * s) * g
        z = z - z.max(-1, keepdims=True); w = np.exp(z); w /= w.sum(-1, keepdims=True)
        out.append((w * g).sum(-1)); out.append((w * s * g).sum(-1))
    return np.stack(out, -1)


def fit_reader(statfn, seed=6801):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for j, v in enumerate(W.VLEV):
        r = rng.standard_normal((NSIM, QL)) * np.sqrt(v)
        Xs.append(statfn(r)); ys.append(np.full(NSIM, j))
    X = np.concatenate(Xs); y = np.concatenate(ys)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=0)
    clf.fit(X, y)

    def ll(r):
        sh = r.shape[:-1]; T = statfn(r.reshape(-1, QL))
        p = clf.predict_proba(T.reshape(len(T), -1))
        return np.log(np.maximum(p, 1e-12)).reshape(sh + (W.NV,))
    return ll


def rom_of(ll, z):
    src, rg = z['src'], z['range']; rows = np.arange(len(src))
    ws = [CB.cb_weights(z['d%d_o' % k].astype(np.float64), z['d%d_y' % k].astype(np.float64), ll)[rows, src] for k in (0, 1)]
    Wb = z['WB0'][rows, src]
    return {rr: float((ws[1] - ws[0])[rg == rr].sum() / Wb[rg == rr].sum()) for rr in ('large', 'middle')}


def rpm_rom(heads, c, fmt, z=None):
    z = z if z is not None else np.load('/home/agent/workspaces/mats/work/cm-interp-B/results/e12_pairs.npz')
    return rom_of(fit_reader(lambda r: rpm_stat(r, heads, c, fmt)), z)


def check():
    z = np.load('/home/agent/workspaces/mats/work/cm-interp-B/results/e12_pairs.npz')
    print('SS through the classifier (exact Bayes: 0, 0):', rom_of(fit_reader(lambda r: (r ** 2).sum(-1, keepdims=True)), z), flush=True)
    print('uniform pool at c = 1.4, E only (amendment 63 table: 0.165 / 0.102):',
          rom_of(fit_reader(lambda r: rpm_stat(r, [(0.0, 0.0)], 1.4, 'ctrl')[..., :1]), z), flush=True)
    print('uniform pool at c = 1.4, E and O:', rpm_rom([(0.0, 0.0)], 1.4, 'ctrl', z), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['check']:
        check()
