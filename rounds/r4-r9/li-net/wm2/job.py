"""One wm2 job: python3 job.py --task t1|t2 (sn|wm|fc) ARGS...
Sets SMALLNET_DATA = $WM2_DATA/data-<task> before importing round-1 code, then runs
  sn: round-1 smallnet.py unchanged (baselines k0, rbtd8), wm: world model, fc: forecaster on a WM (wm2.py)."""
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # 7 jobs share one L4 (OOM seen 2026-10-11)

argv = sys.argv[1:]
assert argv[0] == "--task", "usage: job.py --task t1|t2 MODE ARGS"
task = argv[1]
assert task in ("t1", "t2")
os.environ["SMALLNET_DATA"] = os.path.join(os.environ.get("WM2_DATA", "/tmp/wm2"), f"data-{task}")
os.environ["WM2_TASK"] = task
mode, rest = argv[2], argv[3:]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SMALLNET = os.environ.get("WM2_SMALLNET", os.path.join(os.path.dirname(HERE), "smallnet"))
sys.path.insert(1, SMALLNET)
if mode == "sn":
    import smallnet
    sys.argv = ["smallnet.py"] + rest
    smallnet.main()
else:
    import wm2
    wm2.main([mode] + rest)
