"""129 mutually unbiased bases of R^256 from the Z4-linear Kerdock code (Calderbank, Cameron, Kantor, Seidel 1997).

GR(4,7) = Z4[xi]/(h(xi)), h = Hensel lift of the primitive binary polynomial x^7 + x + 1 (Graeffe: h(x^2) = -(e(x)^2 - o(x)^2)),
so h(y) = y^7 + 2y^4 + y + 3 over Z4. Teichmueller set T = {0, 1, xi, ..., xi^126}. For lambda0 in T, the codewords
c(x) = Tr((lambda0 + 2 lambda1) x) + eps (x in T, lambda1 in T, eps in Z4) form an RM(1,8)-coset after the Gray map
(0->00, 1->01, 2->11, 3->10); the 256 vectors (-1)^bits / 16 with eps in {0, 1} and lambda1 in T give one orthonormal basis,
and bases for different lambda0 are mutually unbiased (|<u, v>| = 1/16). With the identity basis: 129 bases.
build() returns a (129, 256, 256) float64 array (basis k, vector j, coordinate) and checks the properties.
"""
import numpy as np

R = 7


def _mul(a, b, h_tail):
    """Multiply two elements (length-R Z4 coefficient arrays) mod (h, 4). h_tail: xi^7 = sum h_tail[i] xi^i."""
    prod = np.zeros(2 * R - 1, dtype=np.int64)
    for i in range(R):
        if a[i]:
            prod[i:i + R] += a[i] * b
    for d in range(2 * R - 2, R - 1, -1):               # reduce xi^d = xi^(d-7) * xi^7
        c = prod[d] % 4
        if c:
            prod[d - R:d] += c * h_tail
        prod[d] = 0
    return prod[:R] % 4


def build():
    # xi^7 = -(2 xi^4 + xi + 3) = 2 xi^4 + 3 xi + 1 (mod 4)
    h_tail = np.array([1, 3, 0, 0, 2, 0, 0], dtype=np.int64)
    one = np.zeros(R, dtype=np.int64); one[0] = 1
    xi = np.zeros(R, dtype=np.int64); xi[1] = 1
    pw = [one]
    for _ in range(126):
        pw.append(_mul(pw[-1], xi, h_tail))
    assert np.array_equal(_mul(pw[-1], xi, h_tail), one), "xi does not have order 127"
    T = [np.zeros(R, dtype=np.int64)] + pw                 # 128 elements
    # trace: Tr(a) = trace of multiplication-by-a matrix in basis 1..xi^6 (mod 4)
    def trace(a):
        tr = 0
        for i in range(R):
            e = np.zeros(R, dtype=np.int64); e[i] = 1
            tr += _mul(a, e, h_tail)[i]
        return tr % 4
    # Tr(lambda x) for lambda, x in T: precompute products
    nT = len(T)
    TrTT = np.zeros((nT, nT), dtype=np.int64)
    for i in range(nT):
        for j in range(i, nT):
            TrTT[i, j] = TrTT[j, i] = trace(_mul(T[i], T[j], h_tail))
    gray = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    bases = [np.eye(256)]
    for l0 in range(nT):
        vecs = []
        for l1 in range(nT):
            for eps in (0, 1):
                # Tr((l0 + 2 l1) x) = Tr(l0 x) + 2 Tr(l1 x)
                c = (TrTT[l0] + 2 * TrTT[l1] + eps) % 4          # length 128, values in Z4
                bits = gray[c].reshape(-1)                        # length 256
                vecs.append((-1.0) ** bits / 16.0)
        bases.append(np.array(vecs))
    B = np.array(bases)                                           # (129, 256, 256)
    return B


def check(B):
    k = B.shape[0]
    worst_orth = max(np.abs(B[i] @ B[i].T - np.eye(256)).max() for i in range(k))
    worst_mub = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            G = np.abs(B[i] @ B[j].T)
            worst_mub = max(worst_mub, np.abs(G - 1 / 16).max())
    return worst_orth, worst_mub


if __name__ == '__main__':
    import time
    t = time.time(); B = build(); print('built', B.shape, f'{time.time() - t:.1f}s')
    o, m = check(B); print('max |B B^T - I| =', o, '   max ||<u,v>| - 1/16| across bases =', m)
    np.save(__file__.replace('kerdock.py', 'kerdock129.npy'), B.astype(np.float32))
