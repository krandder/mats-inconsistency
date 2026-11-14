"""E48 world-1 training (PREREG-B amendment 56): train_sw.py's recipe, schedule, checkpoints and --pair_init unchanged, with the
format module e48_fmt.py (e30_fmt.py's arms plus sq and sgnsq) in place of e30_fmt.py. Every e30_fmt arm builds byte-identical
tokens through e48_fmt, so a ctrl run of this script equals train_sw.py's.
usage: python train_lin.py --f0 sq --f1 sq --seed S --out DIR [train_sw.py options, e.g. --pair_init --ckpt sparse]
"""
import sys

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e48_fmt  # noqa: E402
import train_sw as TS  # noqa: E402

TS.F = e48_fmt

if __name__ == '__main__':
    TS.main()
