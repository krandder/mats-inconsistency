"""Lock check: lmsr/meanbase_lmsr books_v2 under v1 (live attention) vs cp (attention detached) on
wealth-net final checkpoints, held-out tail. Prior/means from the train region (prequential state).
Writes results/cp_vs_v1_lmsr.csv."""
import csv, json, os, sys
import numpy as np, torch
import torch.nn.functional as F
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
WN = "/home/agent/workspaces/eurisko-study/wealth-net"
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, WN)
torch.set_num_threads(1)
from streams import make_questions, N_FAM
from model import Net
from books_v2 import BooksV2

rows = []
for seed in (1, 2, 3):
    meta = json.load(open(f"{WN}/results/train-s{seed}.json"))
    Q, W, T = meta["Q"], meta["W"], meta["T"]
    bits, fam, _ = make_questions(Q, T, seed)
    m = Net(Q, W, meta["d"], meta["H"], meta["L"]); m.load_state_dict(torch.load(f"{WN}/results/train-s{seed}.pt", map_location="cpu"))
    bt = torch.tensor(bits, dtype=torch.float32); wins = bt.unfold(1, W, 1); n_pos = T - W; split = int(n_pos * 0.8)
    def blk(idx): return wins[:, idx].permute(1, 0, 2).contiguous(), bt[:, idx + W].T.contiguous()
    res = {}
    for fa in (False, True):
        bk = BooksV2(m, fam, N_FAM)
        g = torch.Generator().manual_seed(seed)
        for _ in range(8):  # prequential state from 1024 train examples
            w_, y_ = blk(torch.randint(0, split, (128,), generator=g)); bk.observe(w_, y_, torch.ones_like(y_))
        diags = []
        for st in range(split, n_pos - 128, 256)[:8]:
            w_, y_ = blk(torch.arange(st, st + 128))
            acc = bk.account(w_, y_, freeze_attn=fa); bk.add(acc); diags.append(acc["diag"]["cons_resid_rel_lmsr"])
        res[fa] = (bk, float(np.mean(diags)))
    for book in ("lmsr", "meanbase_lmsr", "tangent"):
        for gran in ("neuron", "head", "out"):
            a = res[False][0].W[book][gran].sum(-1).reshape(-1).numpy(); b = res[True][0].W[book][gran].sum(-1).reshape(-1).numpy()
            rows.append(dict(seed=seed, book=book, gran=gran, spearman_v1_cp=float(spearmanr(a, b)[0]),
                             cons_resid_rel_lmsr_v1=res[False][1], cons_resid_rel_lmsr_cp=res[True][1]))
    print(seed, [r for r in rows if r["seed"] == seed and r["book"] == "lmsr"], flush=True)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results/cp_vs_v1_lmsr.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
