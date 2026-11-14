"""E50 per-net evaluation (PREREG-B amendment 58): e48_eval_w1.py's measures (e37_eval.py's: test KL in the net's format; rom and
type; block-2 labelled attention, xq; the sign-flip cost for formats with visible signs), with the model builder patched so that
nets whose meta.json has variant 'noln1' (or 'pre') are built as that variant.
usage: python e50_eval.py RUNDIR OUTDIR I0 I1
"""
import sys

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_noln as PN  # noqa: E402

P2.build = lambda meta: PN.build_any(meta, P2, P2.fin_of(meta['ds']))

if __name__ == '__main__':
    import e48_eval_w1 as EV
    EV.main()
