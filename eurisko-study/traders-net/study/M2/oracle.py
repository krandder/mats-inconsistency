"""M2 deliverable entry point: the exact held-out ablation oracle.

The implementation lives in w1/oracle.py (worker 1); this wrapper runs it with its own __file__
so its relative paths (wealth-net checkpoints, w1/results) stay correct.
Usage (same flags as w1/oracle.py):
  PYTHONPATH=../../../wealth-net/.pylib python3 oracle.py SEED [--steps final,20000,30000,8000] [--no-coalitions]
Outputs: w1/results/oracle_{units,coalitions,checks}_s{S}_step{K}.*; merge with w1/summarize.py.
"""
import os, runpy, sys

if __name__ == "__main__":
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "w1", "oracle.py"), run_name="__main__")
