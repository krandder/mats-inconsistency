"""E1: TabPFN v2 off the shelf (ModelVersion.V2, default n_estimators), LOCO in context.

The joint likelihood of y is a chain of w single-target classifiers: target y_j, features B (+ b) + y_<j. b is NaN
before listing (TabPFN handles missing values natively). Prequential in 4-row blocks inside rows H..H+k-1: block i
predicts rows H+4i..H+4i+3 from context rows 0..H+4i-1. Returns per-row gain log R(y|with b) - log R(y|without b), nats.
A target that is constant in the context gets the KT estimate (n1 + .5)/(n + 1) in both copies (gain 0 there).
"""
import time
import numpy as np

_CLF = None
CALLS = {'n': 0, 's': 0.0}


def clf():
    global _CLF
    if _CLF is None:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        _CLF = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device='cuda')
    return _CLF


def _chain_logp(Xc, Yc, Xq, Yq):
    """sum over j of log p(y_j = observed | X, y_<j) for query rows -> (nq,)"""
    w = Yc.shape[1]
    out = np.zeros(len(Xq))
    for j in range(w):
        fc = np.concatenate([Xc, Yc[:, :j]], 1).astype(np.float32)
        fq = np.concatenate([Xq, Yq[:, :j]], 1).astype(np.float32)
        yc = Yc[:, j].astype(int)
        if yc.min() == yc.max():
            p1 = np.full(len(fq), (yc.sum() + .5) / (len(yc) + 1))
        else:
            c = clf()
            t0 = time.time()
            c.fit(fc, yc)
            pr = c.predict_proba(fq)
            CALLS['n'] += 1; CALLS['s'] += time.time() - t0
            p1 = pr[:, list(c.classes_).index(1)]
        p1 = np.clip(p1, 1e-6, 1 - 1e-6)
        out += np.where(Yq[:, j] == 1, np.log(p1), np.log1p(-p1))
    return out


def e1_gain(V, Y, H, k=16, block=4):
    """V (T, m+1) int with -1 = absent, Y (T, w) bits -> (k,) per-row gain (nats)."""
    Vf = V.astype(np.float32)
    Vf[V < 0] = np.nan
    m = V.shape[1] - 1
    g = np.zeros(k)
    for s in range(0, k, block):
        c1, q0, q1 = H + s, H + s, H + min(s + block, k)
        lw = _chain_logp(Vf[:c1], Y[:c1], Vf[q0:q1], Y[q0:q1])
        lo = _chain_logp(Vf[:c1, :m], Y[:c1], Vf[q0:q1, :m], Y[q0:q1])
        g[s:s + (q1 - q0)] = lw - lo
    return g
