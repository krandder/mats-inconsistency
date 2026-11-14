"""E49 per-net evaluation (PREREG-B amendment 57): the unchanged evaluators of E37 (world 1: e37_eval.py; test KL, rom and type,
attention masses, xq, sign-flip cost) and E40 (world 2: e40_eval.py; test KL by level, sign-flip and negation costs), with the
model builders patched so that nets whose meta.json has variant 'pre' are built as pfn_pre.py's variant.
usage: python e49_eval.py w1 RUNDIR OUTDIR I0 I1    |    python e49_eval.py t2 RUNDIR OUTDIR I0 I1
"""
import sys

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_pre as PP  # noqa: E402
import train_t2 as TT  # noqa: E402

P2.build = lambda meta: PP.build_any(meta, P2, P2.fin_of(meta['ds']))
TT.build = lambda meta: PP.build_any(meta, P2, meta['D'] + 2)

if __name__ == '__main__':
    world = sys.argv.pop(1)
    if world == 'w1':
        import e37_eval as EV
    else:
        import e40_eval as EV
    EV.main()
