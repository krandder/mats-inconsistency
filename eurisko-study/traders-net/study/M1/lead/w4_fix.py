"""Re-aggregate M1/w4 per-unit books without the double 1/B (w4/leak.py books() returns
per-unit sums divided by B twice; with a short last chunk that overweights it). Calls w4's own
books() per chunk and multiplies the per-unit arrays by B_chunk, then takes the example-weighted
mean. Same eval positions as w4. Writes lead/results/w4fix-s{seed}-step{step}.json with
{rule: {head, neuron}} plus the Spearman/sign agreement with the original w4 arrays."""
import json, os, sys
import numpy as np
import torch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "w4"))
import leak  # noqa: E402
from model import Net  # noqa: E402
from streams import make_questions  # noqa: E402


def main(seed, threads=2):
    torch.set_num_threads(threads)
    Q, W, T = 64, 32, 24000
    bits, fam, desc = make_questions(Q, T, seed)
    bt = torch.tensor(bits, dtype=torch.float32); wins = bt.unfold(1, W, 1)
    n_pos = T - W; split = int(n_pos * 0.8)
    ev = torch.arange(split, n_pos, max(1, (n_pos - split) // 2048))
    ew = wins[:, ev].permute(1, 0, 2).contiguous(); ey = bt[:, ev + W].T.contiguous()
    n = ew.shape[0]
    for step in (8000, 16000, 24000, 30000):
        fn = os.path.join(HERE, "results", f"w4fix-s{seed}-step{step}.json")
        if os.path.exists(fn):
            continue
        m = Net(Q, W, 64, 4, 2)
        m.load_state_dict(torch.load(os.path.join(leak.WN, f"train-s{seed}-step{step}.pt"), map_location="cpu"))
        orig = json.load(open(os.path.join(HERE, "..", "w4", "results", f"s{seed}-step{step}.json")))["rules"]
        out = {"seed": seed, "step": step, "n_eval": int(n), "rules": {}, "agreement_with_w4": {}}
        for rn, rule in leak.RULES.items():
            h = 0.0; nu = 0.0
            for c in range(0, n, 512):
                b = leak.books(m, ew[c:c + 512], ey[c:c + 512], rule)
                B = ew[c:c + 512].shape[0]
                h = h + np.asarray(b["head"]) * B * (B / n); nu = nu + np.asarray(b["neuron"]) * B * (B / n)
            out["rules"][rn] = {"head": h.tolist(), "neuron": nu.tolist()}
            oh, on = np.ravel(orig[rn]["head"]), np.ravel(orig[rn]["neuron"])
            out["agreement_with_w4"][rn] = dict(
                sp_head=float(spearmanr(h.ravel(), oh).correlation), sp_neuron=float(spearmanr(nu.ravel(), on).correlation),
                sign_agree_head=float((np.sign(h.ravel()) == np.sign(oh)).mean()), sign_agree_neuron=float((np.sign(nu.ravel()) == np.sign(on)).mean()))
        json.dump(out, open(fn, "w"))
        print(seed, step, json.dumps(out["agreement_with_w4"]), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
