"""E53 plumbing (disclosed in amendment 61; no registered quantity; nets at step 0 or 3,000-step smoke runs only).
  (1) the smoke runs' step-0 weights equal the baselines' step 0: uni and unif against E37 ctrl s0 (every parameter); unif_noln
      (mag, sq) against E50's no-LayerNorm mag s0 and sq s0 (every parameter; blocks.0.ln1 absent in both);
  (2) block_fwd without fixed rows equals pfn2's Block.forward exactly;
  (3) in block 1 the summary rows are exactly 1/4 on their own source's labelled tokens and 0 elsewhere (uni, unif, unif_noln);
      for unif and unif_noln the query row puts 0 on every labelled token in both blocks;
  (4) factorisation at step 0: for unif and unif_noln, zeroing the labelled tokens' states after block 1 leaves the output
      unchanged (for uni it does not); for unif_noln in the sq format the outputs on E12's two pair versions are identical
      (a source's labelled errors enter only through SS); in the mag format they differ;
  (5) synthetic per-net results under /tmp/e53_synth/results/ with planted values for e53_summ.py (run it there).
usage: python e53_plumb.py
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_uni as PU  # noqa: E402
import e48_fmt as F  # noqa: E402
import world as W  # noqa: E402

SM = '/tmp/e53_smoke'
HERE = '/home/agent/workspaces/mats/work/cm-interp-B'
META = dict(depth=2, width=32, heads=2, mixk=4, ds=16)


def check_init():
    pairs = (('e53_uni_s0', 'runs/e37/e37_ctrl_s0'), ('e53_unif_s0', 'runs/e37/e37_ctrl_s0'),
             ('e53_unifnoln_mag_s0', 'runs/e50/e50_mag_s0'), ('e53_unifnoln_sq_s0', 'runs/e50/e50_sq_s0'))
    for sm, base in pairs:
        a = torch.load(os.path.join(SM, sm, 'ckpt/step_000000.pt'), map_location='cpu')
        b = torch.load(os.path.join(HERE, base, 'ckpt/step_000000.pt'), map_location='cpu')
        ok = set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
        print('(1)', sm, 'step 0 == %s step 0 (every parameter):' % base, ok, '| n params', len(a))


def model(variant, seed=0):
    torch.manual_seed(seed)
    return PU.build_any(dict(META, variant=variant), P2, 36).eval()


def data(n=64, seed=3, fmt='ctrl'):
    rng = np.random.default_rng(seed); d = W.sample_np(n, rng)
    X, v = F.make_X(d['o'], d['y'], rng.standard_normal((n, 3, 16)) / 4, rng.standard_normal((n, 5, 16)) / 4, fmt)
    return torch.from_numpy(X), torch.from_numpy(v)


def check_block():
    torch.manual_seed(0); b = P2.PFN(36, 2, 32, 2, mixk=4).eval()
    X, v = data()
    with torch.no_grad():
        x = b.inp(X)
        allow = torch.ones(64, 16, 16, dtype=torch.bool); allow[:, :, 15] = False; allow[:, 15, 15] = True
        y1, a1 = b.blocks[0](x, allow, True); y2, a2 = PU.block_fwd(b.blocks[0], x, allow, None, True)
    print('(2) block_fwd without fixed rows == pfn2 Block.forward:', torch.equal(y1, y2) and torch.equal(a1, a2))


def check_rows():
    X, v = data()
    for var in PU.VARIANTS:
        m = model(var)
        with torch.no_grad():
            out = m(X, v, want=True)
        att = out[-1]
        ok = True
        for s in range(3):
            row = att[0][:, :, PU.SUMM[s]]
            want = torch.zeros(16); want[s * 5:s * 5 + 4] = 0.25
            ok &= bool(torch.allclose(row, want.expand_as(row)))
        q0 = float(att[0][:, :, 15, PU.LAB].abs().max()); q1 = float(att[1][:, :, 15, PU.LAB].abs().max())
        print('(3)', var, 'summary rows exactly uniform on own labelled tokens:', ok, '| max query weight on labelled tokens, block 1 / 2: %.3g / %.3g' % (q0, q1))


def fwd_zero_lab(m, X, v):
    """the variant's forward with the labelled tokens' states set to 0 after block 1."""
    B, T, _ = X.shape; qi = T - 1
    ctx = v.clone(); ctx[:, qi] = False
    allow = ctx[:, None, :].expand(B, T, T).clone(); allow[:, qi, qi] = True
    if m.variant in ('unif', 'unif_noln'):
        allow[:, qi, PU.LAB] = False
    x = m.inp(X)
    x, _ = PU.block_fwd(m.blocks[0], x, allow, m._fixed)
    x = x.clone(); x[:, PU.LAB] = 0
    x, _ = PU.block_fwd(m.blocks[1], x, allow, None)
    return m.out(m.ln(x[:, qi]))


def check_factor():
    X, v = data()
    for var in PU.VARIANTS:
        m = model(var)
        with torch.no_grad():
            y0 = torch.cat(m(X, v), 1); y1 = fwd_zero_lab(m, X, v)
            K = 4; y1 = torch.cat([y1[:, K:2 * K], y1[:, 2 * K:].clamp(-6, 3), y1[:, :K]], 1)
        print('(4)', var, 'max |output change| when the labelled states after block 1 are zeroed: %.3g' % float((y0 - y1).abs().max()))
    z = np.load(os.path.join(HERE, 'results/e12_pairs.npz'))
    for fmt in ('sq', 'mag'):
        m = model('unif_noln'); outs = []
        for k in (0, 1):
            ps, pq = F.ids_from_X(z['X%d' % k][:500])
            Xk, vk = F.make_X(z['d%d_o' % k][:500].astype(np.float64), z['d%d_y' % k][:500].astype(np.float64), ps, pq, fmt)
            with torch.no_grad():
                outs.append(torch.cat(m(torch.from_numpy(Xk), torch.from_numpy(vk)), 1))
        print('(4) unif_noln %s: max |output difference| between the E12 pair versions (500 rows): %.3g' % (fmt, float((outs[0] - outs[1]).abs().max())))


def synth():
    R = '/tmp/e53_synth/results'
    rng = np.random.default_rng(2)
    plant = dict(uni=('e53', 'e53_uni_s%d', 0.010, 0.03, 0.04, 0.10), unif=('e53', 'e53_unif_s%d', 0.009, 0.10, 0.02, 0.05),
                 unifnoln_mag=('e53', 'e53_unifnoln_mag_s%d', 0.008, 0.13, 0.20, None), unifnoln_sq=('e53', 'e53_unifnoln_sq_s%d', 0.004, 0.0, 0.0, None),
                 ctrl=('e37', 'e37_ctrl_s%d', 0.0127, 0.133, 0.04, 0.20), mag=('e37', 'e37_mag_s%d', 0.0055, 0.073, -0.03, None),
                 mag_noln=('e50', 'e50_mag_s%d', 0.0177, 0.036, 0.12, None), sq_noln=('e50', 'e50_sq_s%d', 0.0174, 0.124, 0.05, None))
    for s in range(12):
        e = 1 + 0.01 * rng.uniform(-1, 1)
        for a, (sub, nm, k, rl, rm, cs) in plant.items():
            if a.startswith('unifnoln') and s >= 8:
                continue
            os.makedirs(os.path.join(R, sub), exist_ok=True)
            kw = dict(gap=(0.14 if (a == 'unif' and s == 5) else k * e), rom_large=rl * e, rom_middle=rm * e, lab_last=0.2, xq=0.01)
            if cs is not None:
                kw.update(sign_pc=cs * 0.01, sign_kl=0.01)
            np.savez(os.path.join(R, sub, nm % s + '.npz'), **kw)
    print('(5) synthetic results written: planted rom uni/ctrl 0.03/0.133 (P208 holds), unif/ctrl 0.10/0.133 with seed 5 stalled '
          '(P209 fails), unifnoln_mag 0.13 / 0.20 (P210 holds), KL unifnoln_mag - unifnoln_sq = 0.004 (P211 holds for KCB_MABS 0.0051)')


if __name__ == '__main__':
    check_block(); check_rows(); check_factor(); synth()
    if os.path.isdir(SM):
        check_init()
