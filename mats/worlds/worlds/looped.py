"""Weight-tied looped variant of harness.SetPFN, plus a thin CLI that runs harness.py with it (no harness edits).

LoopedSetPFN: input embedding e; a block of `loop_block` harness Blocks applied `loop_iters` times with shared
weights; input injection (x <- x + e before every iteration after the first; Yang et al. 2023 style) unless
--loop-no-inject. Same attention masking and heads as SetPFN, so a looped net with loop_iters = 1 is exactly a
SetPFN of depth loop_block. Rounds (layer applications) = loop_block * loop_iters.

usage (all harness.py arguments pass through):
  python looped.py train --loop-block 2 --loop-iters 3 --world world3_chains ... --out runs/X
  python looped.py eval  --run runs/X --evalset ES --loop-eval-iters 6 --out runs/X/eval_it6
Without --loop-iters this is plain harness.py (untied SetPFN of --depth). --amp-bf16 wraps either model in bf16
autocast (recorded in meta.json, so eval rebuilds the same wrapper).
"""
import sys
import torch
import torch.nn as nn
import harness as H


class LoopedSetPFN(nn.Module):
    def __init__(self, fin, loop_block, loop_iters, d, h, kind='cat', mix=1, n_classes=2, mlp_ratio=4, inject=True):
        super().__init__()
        self.kind, self.mix = kind, mix
        self.loop_iters, self.inject = loop_iters, inject
        self.inp = nn.Linear(fin, d)
        self.blocks = nn.ModuleList([H.Block(d, h, mlp_ratio) for _ in range(loop_block)])
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, 3 * mix if kind == 'real' else n_classes)

    def forward(self, X, mask, want_q=False, want_h=False, iters=None):
        ctx = mask.clone(); ctx[:, -1] = False
        empty = ~ctx.any(1)
        ctx[empty, -1] = True
        km = ctx[:, None, None, :]
        e = self.inp(X); x = e; qas, hs = [], [x]
        for t in range(iters or self.loop_iters):
            if t > 0 and self.inject:
                x = x + e
            for blk in self.blocks:
                x, qa = blk(x, km, want_q); qas.append(qa); hs.append(x)
        if want_h:
            return hs
        o = self.out(self.ln(x[:, -1]))
        if self.kind == 'real':
            lg, mu, ls = o.split(self.mix, -1)
            o = (lg.log_softmax(-1), mu, ls.clamp(-7, 4))
        return (o, qas) if want_q else o


class AmpWrap(nn.Module):
    """bf16 autocast around any harness-compatible model (CUDA only; no-op on CPU). Outputs are cast to float32."""

    def __init__(self, m):
        super().__init__()
        self.m = m
        self.kind, self.mix = m.kind, m.mix

    def forward(self, X, mask, *a, **k):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=X.is_cuda):
            o = self.m(X, mask, *a, **k)
        return _f32(o)


def _f32(o):
    if torch.is_tensor(o):
        return o.float()
    if isinstance(o, (list, tuple)):
        return type(o)(_f32(t) for t in o)
    return o


_orig_build = H.build_model
EVAL_ITERS = None


def build_model(world, a):
    if a.get('loop_iters'):
        iters = EVAL_ITERS or a['loop_iters']
        m = LoopedSetPFN(world.feat_dim, a['loop_block'], iters, a['width'], a['heads'], world.target_kind,
                         a.get('mix', 1), getattr(world, 'n_classes', 2), a.get('mlp_ratio', 4),
                         not a.get('loop_no_inject', False))
    else:
        m = _orig_build(world, a)
    return AmpWrap(m) if a.get('amp_bf16') else m


def main(argv):
    global EVAL_ITERS
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--loop-block', dest='loop_block', type=int, default=2)
    ap.add_argument('--loop-iters', dest='loop_iters', type=int, default=0)
    ap.add_argument('--loop-no-inject', dest='loop_no_inject', action='store_true')
    ap.add_argument('--loop-eval-iters', dest='loop_eval_iters', type=int, default=0)
    ap.add_argument('--amp-bf16', dest='amp_bf16', action='store_true')
    mine, rest = ap.parse_known_args(argv)
    EVAL_ITERS = mine.loop_eval_iters or None
    torch.backends.cuda.matmul.allow_tf32 = True     # all W3 arms (untied and looped) run through this wrapper
    torch.backends.cudnn.allow_tf32 = True
    H.build_model = build_model
    orig_train = H.train

    def train(a):                     # extra fields land in meta.json['args'], so load_run rebuilds the same model
        a.loop_block, a.loop_iters, a.loop_no_inject = mine.loop_block, mine.loop_iters, mine.loop_no_inject
        a.amp_bf16 = mine.amp_bf16
        if mine.loop_iters:
            a.depth = mine.loop_block * mine.loop_iters      # recorded as rounds; not used to build
        orig_train(a)
    H.train = train
    sys.argv = [sys.argv[0]] + rest
    H.main()


if __name__ == '__main__':
    main(sys.argv[1:])
