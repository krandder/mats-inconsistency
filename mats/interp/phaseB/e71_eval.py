"""E71 per-net evaluation (PREREG-B amendment 79): e48_eval_w1.py's measures, unchanged (test KL, rom_large, rom_middle, type,
block-1 and block-2 attention masses, xq, the flip_ind sign-flip cost), with pfn_b1hw's builder installed so that nets whose
meta.json has b1_heads load with their block-1 head count and dimension.
usage: python e71_eval.py RUNDIR OUTDIR I0 I1  -> OUTDIR/<tag>.npz
"""
import sys

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e48_eval_w1 as EW  # noqa: E402
import pfn_b1hw as BH  # noqa: E402

BH.install()

if __name__ == '__main__':
    EW.main()
