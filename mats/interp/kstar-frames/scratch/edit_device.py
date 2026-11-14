"""One-off source edit (2026-10-31 ~00:20Z): device support (KF_DEVICE) for the GPU run, and a batched dual transport.
Run once from the lane directory: python3 scratch/edit_device.py"""
import os, shutil

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
shutil.copy('lib/kf.py', 'notes/kf_cpu_v1.py.bak')


def sub(path, old, new, count=1):
    s = open(path).read()
    assert s.count(old) >= 1, (path, old[:60])
    s = s.replace(old, new) if count == 0 else s.replace(old, new, count)
    open(path, 'w').write(s)


p = 'lib/kf.py'
sub(p, "DT = torch.float64\nSIGMA = 1e-3",
    "DT = torch.float64\nDEV = torch.device(os.environ.get('KF_DEVICE', 'cpu'))   # 'cuda' on the GPU VM\nSIGMA = 1e-3")
sub(p, """    net32 = nd.Net('plain', seed, n, 32, sigma)
    if L == 32:
        return net32
    return nd.Net('plain', seed, n, L, sigma, r=net32.r[:L].tolist())
""", """    net32 = nd.Net('plain', seed, n, 32, sigma)
    net = net32 if L == 32 else nd.Net('plain', seed, n, L, sigma, r=net32.r[:L].tolist())
    net.W = [w.to(DEV) for w in net.W]
    net.r, net.s = net.r.to(DEV), net.s.to(DEV)
    return net
""")
sub(p, "    return torch.Generator().manual_seed(int(seed))",
    "    return torch.Generator(device=DEV).manual_seed(int(seed))")
sub(p, "    return torch.randn(P, n, generator=g, dtype=DT)",
    "    return torch.randn(P, n, generator=g, dtype=DT, device=DEV)")
sub(p, "                e = torch.randn(a.shape, generator=g, dtype=DT)",
    "                e = torch.randn(a.shape, generator=g, dtype=DT, device=DEV)")
sub(p, "    A = torch.randn(P, n, k, generator=g, dtype=DT)",
    "    A = torch.randn(P, n, k, generator=g, dtype=DT, device=DEV)")
s = open(p).read()
old_dual = s[s.index("def dual_forward("):s.index("def backward_one(")]
new_dual = '''def dual_forward_loop(net, F, D, c_from, c_to):
    """Dual (inverse-transpose) transport, one path at a time (numpy): G <- orth(pinv(B_l)^T G), B_l = D_l W_l D_{l-1}
    restricted to the active rows (layer l) and columns (layer l-1), for l = c_from+1..c_to. If m_l < k the frame has
    rank m_l (its other columns are zero)."""
    dev = F.device
    G = F.detach().cpu().clone()
    P, n, k = G.shape
    for l in range(c_from + 1, c_to + 1):
        Wl = net.W[l - 1].cpu().numpy()
        newG = np.zeros((P, n, k))
        Dl = D[l - 1].cpu().numpy() > 0.5
        Dp = D[l - 2].cpu().numpy() > 0.5
        Gn = G.numpy()
        for p in range(P):
            ra, ca = np.where(Dl[p])[0], np.where(Dp[p])[0]
            Bs = Wl[np.ix_(ra, ca)]                   # m_l x m_{l-1}
            g = np.linalg.pinv(Bs).T @ Gn[p][ca]      # (m_l x m_{l-1}) @ (m_{l-1} x k)
            q, _ = np.linalg.qr(g)                    # (m_l, min(m_l, k)); if m_l < k the frame is rank m_l
            newG[p][np.ix_(ra, np.arange(q.shape[1]))] = q
        G = torch.from_numpy(newG)
    return G.to(dev)


def dual_forward(net, F, D, c_from, c_to):
    """Batched dual transport (same subspaces as dual_forward_loop): the pinv of the masked n x n layer map is the
    embedded pinv of its active block. Paths whose active count drops below k on the way use the loop."""
    k = F.shape[-1]
    small = torch.zeros(F.shape[0], dtype=torch.bool, device=F.device)
    for l in range(c_from, c_to + 1):
        small |= D[l - 1].sum(-1) < k
    G = F
    for l in range(c_from + 1, c_to + 1):
        B = D[l - 1][:, :, None] * net.W[l - 1][None] * D[l - 2][:, None, :]
        G = orth(torch.linalg.pinv(B).transpose(-1, -2) @ G)
    if bool(small.any()):
        idx = torch.where(small)[0]
        G = G.clone()
        G[idx] = dual_forward_loop(net, F[idx], [None if d is None else d[idx] for d in D], c_from, c_to)
    return G


'''
open(p, 'w').write(s.replace(old_dual, new_dual))

for p in ['scratch/r1_layers.py', 'scratch/r2_big.py', 'scratch/r3_pairs.py', 'scratch/r5_states.py']:
    s = open(p).read()
    s = s.replace('.cpu().numpy()', '.numpy()').replace('.numpy()', '.cpu().numpy()')
    open(p, 'w').write(s)
p = 'scratch/r1_layers.py'
sub(p, "Macc = torch.zeros(A.batch, n, n, dtype=kf.DT)", "Macc = torch.zeros(A.batch, n, n, dtype=kf.DT, device=kf.DEV)")
sub(p, "torch.randn(h_c.shape, generator=gnoise, dtype=kf.DT)",
    "torch.randn(h_c.shape, generator=gnoise, dtype=kf.DT, device=kf.DEV)")
p = 'scratch/r2_big.py'
sub(p, "cnt = [torch.zeros(n, dtype=kf.DT) for _ in range(L)]",
    "cnt = [torch.zeros(n, dtype=kf.DT, device=kf.DEV) for _ in range(L)]")
sub(p, "Mt = {c: torch.zeros(n, n, dtype=kf.DT) for c in MAIN}",
    "Mt = {c: torch.zeros(n, n, dtype=kf.DT, device=kf.DEV) for c in MAIN}")
p = 'scratch/r5_states.py'
sub(p, "Z = {c: torch.zeros(NS, n, dtype=kf.DT) for c in layers}",
    "Z = {c: torch.zeros(NS, n, dtype=kf.DT, device=kf.DEV) for c in layers}")
sub(p, "d2grid = torch.tensor((2 * np.sin(GRID / 2)) ** 2, dtype=kf.DT)",
    "d2grid = torch.tensor((2 * np.sin(GRID / 2)) ** 2, dtype=kf.DT, device=kf.DEV)")
sub(p, "counts = torch.zeros(NS, len(GRID), dtype=torch.int64)",
    "counts = torch.zeros(NS, len(GRID), dtype=torch.int64, device=kf.DEV)")
sub(p, "d2[torch.arange(Ub.shape[0]), torch.arange(i0, i0 + Ub.shape[0])] = 1e9",
    "d2[torch.arange(Ub.shape[0], device=kf.DEV), torch.arange(i0, i0 + Ub.shape[0], device=kf.DEV)] = 1e9")
sub(p, "h = torch.zeros(Ub.shape[0], len(GRID) + 1, dtype=torch.int64)",
    "h = torch.zeros(Ub.shape[0], len(GRID) + 1, dtype=torch.int64, device=kf.DEV)")
p = 'lib/cover.py'
sub(p, "Gram = Gram + ridge * torch.eye(k, dtype=torch.float64)",
    "Gram = Gram + ridge * torch.eye(k, dtype=torch.float64, device=Gram.device)")
sub(p, "/ top[i0:i0 + a, None]).float()", "/ top[i0:i0 + a, None]).float().cpu()")
sub(p, "/ k).float()", "/ k).float().cpu()")
print('edits done')
