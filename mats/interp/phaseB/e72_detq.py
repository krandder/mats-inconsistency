"""E72 addendum 1 (PREREG-B amendment 80, addendum 1): the like-for-like departure. The CB readers of e72_run.py hold the pooled
statistic fixed when they differentiate the predictive mean by a source's reading o*_s; the net's registered rom does not, because the
reading also enters the source's summary row as its block-1 query, which sets the own-pool attention. For every net, rom on E12's
pairs (autograd weights, as e15_run.eff_w, through the net's own forward and masks) twice:
  base   as registered (recomputed);
  detq   the same forward with the summary rows' (tokens 4, 9, 14) block-1 queries detached from the autograd graph: the forward and
         every prediction are unchanged, and the weight is the derivative at a fixed own-pool attention (for the uniform-pool variant,
         whose summary rows have fixed weights, detq equals base: a check).
Implementation: pfn2.Block.forward and pfn_uni.block_fwd (used by the own, nocons and uni variants) are wrapped so that a block marked
_detq detaches those rows of q; only model.blocks[0] is marked.
usage: python e72_detq.py OUTDIR TAG=RUNDIR:EVALFILE [...] -> OUTDIR/<tag>.json
"""
import json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_uni as PU  # noqa: E402
import e63_run as E63  # noqa: E402
import e15_run as R15  # noqa: E402
import pfn_b1hw as BH  # noqa: E402

BH.install()
import pfn_l2fix  # noqa: E402

pfn_l2fix.install()                      # E70 b1h4: block 2's two heads of dimension 16 (loader fix, found in E72)
torch.set_num_threads(1)
SUMM = [4, 9, 14]


def _block_forward(self, x, allow, want_attn=False):
    B, T, d = x.shape
    q, k, v = self.qkv(self.ln1(x)).split(self.h * self.dh, -1)
    sp = lambda t: t.view(B, T, self.h, self.dh).transpose(1, 2)
    q, k, v = sp(q), sp(k), sp(v)
    if getattr(self, '_detq', False):
        q = q.clone(); q[:, :, SUMM] = q[:, :, SUMM].detach()
    s = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
    s = s.masked_fill(~allow[:, None], float('-inf'))
    a = s.softmax(-1)
    x = x + self.o((a @ v).transpose(1, 2).reshape(B, T, self.h * self.dh))
    if not self.attn_only:
        x = x + self.mlp(self.ln2(x))
    return x, (a if want_attn else None)


def _block_fwd(blk, x, allow, fixed=None, want=False):
    B, Tn, d = x.shape
    q, k, v = blk.qkv(blk.ln1(x)).split(blk.h * blk.dh, -1)
    sp = lambda t: t.view(B, Tn, blk.h, blk.dh).transpose(1, 2)
    q, k, v = sp(q), sp(k), sp(v)
    if getattr(blk, '_detq', False):
        q = q.clone(); q[:, :, SUMM] = q[:, :, SUMM].detach()
    s = q @ k.transpose(-1, -2) / math.sqrt(blk.dh)
    s = s.masked_fill(~allow[:, None], float('-inf'))
    a = s.softmax(-1)
    if fixed is not None:
        rows, A = fixed
        a = torch.where(rows.to(a.device)[None, None, :, None], A.to(a.device, a.dtype)[None, None], a)
    x = x + blk.o((a @ v).transpose(1, 2).reshape(B, Tn, blk.h * blk.dh))
    if not blk.attn_only:
        x = x + blk.mlp(blk.ln2(x))
    return x, (a if want else None)


P2.Block.forward = _block_forward
PU.block_fwd = _block_fwd


def rom(model, arm, z):
    src, rg = z['src'], z['range']; Wb = z['WB0'][np.arange(len(src)), src]; ws = []
    for k in (0, 1):
        ps, pq = E63.F.ids_from_X(z['X%d' % k])
        Xk, vk = E63.F.make_X(z['d%d_o' % k].astype(np.float64), z['d%d_y' % k].astype(np.float64), ps, pq, arm)
        w, _ = R15.eff_w(model, Xk, vk, 32); ws.append(w[np.arange(len(src)), src])
    return {rr: float((ws[1] - ws[0])[rg == rr].sum() / Wb[rg == rr].sum()) for rr in ('large', 'middle')}


def main():
    outdir = sys.argv[1]; os.makedirs(outdir, exist_ok=True)
    z = np.load('results/e12_pairs.npz')
    for spec in sys.argv[2:]:
        tag, rest = spec.split('=', 1); run, evf = rest.split(':', 1); outp = os.path.join(outdir, tag + '.json')
        if os.path.exists(outp):
            continue
        t0 = time.time(); model, arm, meta = E63.load(run, None); evz = np.load(evf, allow_pickle=True)
        model.requires_grad_(False)
        o = dict(tag=tag, net_rom_large=float(evz['rom_large']), net_rom_middle=float(evz['rom_middle']), variant=meta.get('variant'))
        o['base'] = rom(model, arm, z)
        model.blocks[0]._detq = True
        o['detq'] = rom(model, arm, z)
        model.blocks[0]._detq = False
        o['seconds'] = time.time() - t0
        json.dump(o, open(outp + '.tmp', 'w'), indent=1); os.replace(outp + '.tmp', outp)
        print(tag, json.dumps(o), flush=True)


if __name__ == '__main__':
    main()
