"""E66 (PREREG-B amendment 74; CPU, existing nets): does the opposite-sign pair of an A net's two pooling heads cancel the sign
dependence? An evaluation-time surgery in block 1: for the summary rows (tokens 4, 9, 14) of a chosen head, the odd (signed-error) term
is removed from the logits of the labelled keys, exactly: logit_j -> logit_j - q.W_K(gamma * u~) r_j / (sigma_j sqrt(dh)), with u~
the centred o column of the input embedding, r_j = o_j - y_j and sigma_j block 1's LayerNorm standard deviation of token j (with eps).
The values, the other rows and heads and block 2 are unchanged; the scores' norm term and every other part stay.
Nets: E37's ctrl nets and E30's fix nets with at least one pooling head (E63's pooling heads, results/e63/{ctrl,fix}.json), and E60's
own nets (both heads pool; the own mask is kept). Conditions: base; each pooling head zeroed alone; all pooling heads zeroed.
Per condition (E27's data, first NB = 3,000 bases): KL of the base variant; the sign-flip family (flip_ind, 9 samples) as
e30_eval.family_cost: the KL reduction of geometric averaging (sign_pc) and the family's mean KL (sign_kl); for base and all-zeroed also
rom_large and rom_middle on the E12 pairs (e37_eval.rom with the surgery).
usage: python e66_run.py COHORT (ctrl | fix | own) -> results/e66/<cohort>.json
"""
import json, math, os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_own as PO  # noqa: E402

P2.build = lambda meta: PO.build_any(meta, P2, P2.fin_of(meta['ds']))
import e26_run as RU  # noqa: E402
import e27_data as E7  # noqa: E402
import e30_eval as E30  # noqa: E402
import e37_eval as EV  # noqa: E402
import e48_fmt as F  # noqa: E402
import pfn_uni as PU  # noqa: E402

E30.F = F; EV.F = F
torch.set_num_threads(1)
S, Q, QL, T = 3, 5, 4, 16
SUMM = [s * Q + QL for s in range(S)]
LAB = [s * Q + q for s in range(S) for q in range(QL)]
NB = int(os.environ.get('E66_NB', 3000)); OUT = os.environ.get('E66_OUT', 'results/e66')


class Surg(torch.nn.Module):
    """The net's forward (baseline or pfn_own's 'own'), with the odd term removed from block 1's summary-row logits in heads ZERO."""
    def __init__(self, m, zero, own):
        super().__init__(); self.m = m; self.zero = list(zero); self.own = own

    def forward(self, X, valid, want=False):
        m = self.m; B = X.shape[0]; qi = T - 1
        ctx = valid.clone(); ctx[:, qi] = False
        allow = ctx[:, None, :].expand(B, T, T).clone(); allow[:, qi, qi] = True
        eye = torch.eye(T, dtype=torch.bool)[None]; allow = allow | (eye & ~valid[:, :, None])
        allow0 = allow.clone()
        if self.own:
            for s in range(S):
                allow0[:, SUMM[s], :] = False; allow0[:, SUMM[s], s * Q:s * Q + QL] = True
        x = m.inp(X); blk = m.blocks[0]; ln = blk.ln1
        h = ln(x)
        qq, kk, vv = blk.qkv(h).split(blk.h * blk.dh, -1)
        sp = lambda t: t.view(B, T, blk.h, blk.dh).transpose(1, 2)
        qq, kk, vv = sp(qq), sp(kk), sp(vv)
        sc = qq @ kk.transpose(-1, -2) / math.sqrt(blk.dh)
        if self.zero:
            Wt = m.inp.weight; u = Wt[:, 32] - Wt[:, 32].mean()
            Wk = blk.qkv.weight[blk.h * blk.dh:2 * blk.h * blk.dh]
            ku = (Wk @ (ln.weight * u)).view(blk.h, blk.dh)                                   # W_K (gamma u~), per head
            xc = x - x.mean(-1, keepdim=True); sig = torch.sqrt((xc ** 2).mean(-1) + ln.eps)    # (B, T)
            r = X[:, :, 32] - X[:, :, 33]                                                      # o - y (labelled tokens: the error)
            odd = (r / sig)[:, LAB]                                                            # (B, 12)
            sc = sc.clone()
            for hh in self.zero:
                for s in range(S):
                    coef = (qq[:, hh, SUMM[s], :] * ku[hh][None]).sum(-1) / math.sqrt(blk.dh)   # (B,)
                    sc[:, hh, SUMM[s], LAB] = sc[:, hh, SUMM[s], LAB] - coef[:, None] * odd
        sc = sc.masked_fill(~allow0[:, None], float('-inf'))
        a = sc.softmax(-1)
        x = x + blk.o((a @ vv).transpose(1, 2).reshape(B, T, blk.h * blk.dh))
        x = x + blk.mlp(blk.ln2(x))
        for blk2 in m.blocks[1:]:
            x, _ = PU.block_fwd(blk2, x, allow, None, False)
        y = m.out(m.ln(x[:, qi])); K = m.mixk
        return (y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K])


def nets(cohort):
    out = []
    if cohort in ('ctrl', 'fix'):
        e63 = json.load(open('results/e63/%s.json' % cohort))
        for tag, o in sorted(e63.items()):
            if o['n_pool'] >= 1:
                out.append((tag, ('runs/e37/' if cohort == 'ctrl' else 'runs/e30/') + tag, o['pool_heads'], o['type']))
    else:
        for s in range(12):
            out.append(('e60_own_s%d' % s, 'runs/e60/e60_own_s%d' % s, [0, 1], 'A'))
    return out


def main():
    cohort = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
    z7 = np.load('results/e27_data.npz'); pz = np.load('results/e12_pairs.npz')
    g = z7['bmean'][:NB, None] + RU.GRID[None]
    lpB = RU.ED.mix_logp_points(z7['bmeans'][:NB], z7['bvars'], z7['bwts'][:NB], g)
    res = {}
    for tag, run, pool, typ in nets(cohort):
        model, arm, meta = E30.load(run); own = meta.get('variant') == 'own'
        conds = [('base', [])] + [('zero%d' % hh, [hh]) for hh in pool] + ([('zeroall', list(pool))] if len(pool) > 1 else [])
        o = dict(type=typ, pool_heads=pool, arm=arm)
        for name, zero in conds:
            net = Surg(model, zero, own); net.eval()
            X, v = F.make_X(z7['o'][:NB, 0], z7['y'][:NB], z7['semb'][:NB], z7['qemb'][:NB], arm)
            kl, _, _ = RU.set_stats([RU.heads(net, 'clean', X, v)], lpB, g)
            pc, sk, mass = E30.family_cost(net, arm, z7, E7.fam_idx('flip_ind'), NB)
            d = dict(kl=float(kl.mean()), sign_pc=pc, sign_kl=sk, mass=mass)
            if name in ('base', 'zeroall') or (len(pool) == 1 and name != 'base'):
                d.update(EV.rom(net, arm, pz))
            o[name] = d
            print(cohort, tag, name, json.dumps({k: round(x, 5) for k, x in d.items()}), flush=True)
        res[tag] = o
    json.dump(res, open(os.path.join(OUT, '%s.json' % cohort), 'w'), indent=1)


if __name__ == '__main__':
    main()
