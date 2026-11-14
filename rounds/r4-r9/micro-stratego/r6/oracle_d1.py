#!/usr/bin/env python3
"""R6-E3: exact Bayes posterior P*_full (and P*_c) for the 500 R5-E2 D1 two-flank test positions (r5 d1_pos.npz), for
SCORING ONLY (E3-D1 accuracy against P*_full). Same code path as r5/generate6.post_job. CPU, 6 workers.
   OMP_NUM_THREADS=1 python3 r6/oracle_d1.py"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r5"))
import generate6 as GEN  # noqa: E402

DATA = os.environ.get("R5E2_DATA", "/tmp/r5e2/data")

if __name__ == "__main__":
    t0 = time.time()
    d = np.load(os.path.join(DATA, "d1_pos.npz"))
    path = os.path.join(DATA, "games_test.npz")
    with Pool(6) as pool:
        res = pool.map(GEN.post_job, [(path, int(g), int(t)) for g, t in zip(d["game"], d["t"])], chunksize=4)
    np.savez_compressed(os.path.join(DATA, "oracle_d1_test.npz"), game=d["game"], t=d["t"],
                        pstar_full=np.stack([r[0] for r in res]), pstar_c=np.stack([r[1] for r in res]))
    print("oracle_d1_test", len(res), f"{time.time() - t0:.0f}s")
