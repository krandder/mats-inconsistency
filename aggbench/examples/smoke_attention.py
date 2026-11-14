"""SMOKE ONLY: a tiny attention aggregator, run end to end to show the interfaces work.

Not a result. Trains a few hundred steps on a small train set and writes a SMOKE-track result
JSON (default /tmp/aggbench-smoke/result.json). Needs torch:

    /home/agent/.venvs/trader-cat/bin/python3 -m aggbench.examples.smoke_attention --env E2
"""
import argparse

import numpy as np
import torch

from aggbench import make_env
from aggbench.baselines import all_baselines
from aggbench.controls import run_with_controls
from aggbench.metrics import summarize
from aggbench.results import make_result, split_record, write_result
from aggbench.verify import oracle_check


class Net(torch.nn.Module):
    def __init__(self, in_dim, width=32):
        super().__init__()
        self.embed = torch.nn.Linear(in_dim, width)
        self.attn = torch.nn.MultiheadAttention(width, num_heads=2, batch_first=True)
        self.out = torch.nn.Linear(width, 1)

    def forward(self, tokens):                    # tokens: (n, agents, in_dim)
        z = self.embed(tokens)
        z = z + self.attn(z, z, z, need_weights=False)[0]
        return self.out(z).mean(1)[:, 0]          # permutation-invariant pooling


class AttentionAggregator:
    """Tokens = [report log-odds, key]; output = prior log-odds + network output."""

    def __init__(self, seed, key_dim, steps=300):
        torch.manual_seed(seed)
        self.module = Net(1 + key_dim)
        self.steps = steps

    def _tokens(self, view):
        x = np.concatenate([view.reports["logit"][..., None] / 4, view.keys], axis=-1)
        return torch.tensor(x, dtype=torch.float32)

    def fit(self, view, truth):
        x, y = self._tokens(view), torch.tensor(truth, dtype=torch.float32)
        opt = torch.optim.Adam(self.module.parameters(), lr=3e-3)
        for _ in range(self.steps):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(self.module(x), y).backward()
            opt.step()

    def predict(self, view):
        with torch.no_grad():
            out = self.module(self._tokens(view)).double().numpy()
        return {"logit": view.public["prior_logit"] + out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="E2", choices=["E2", "E3-sybil", "E3-biased", "E3-collude"])
    ap.add_argument("--visibility", default="full", choices=["none", "noisy", "full"])
    ap.add_argument("--out", default="/tmp/aggbench-smoke")
    args = ap.parse_args()

    env = make_env(args.env)
    tr, dv = env.sample("train", 2000), env.sample("dev", 1000)
    key_dim = tr.keys[args.visibility].shape[-1]
    arms = run_with_controls(lambda s: AttentionAggregator(s, key_dim), tr.view(args.visibility), tr.truth,
                             dv.view(args.visibility), seed=0)
    forecasts, _ = all_baselines(tr, dv, args.visibility)
    forecasts.update({f"attn_smoke:{k}": f for k, f in arms.items()})
    methods = {k: summarize(f, dv, n_boot=500) for k, f in forecasts.items()}
    for k, m in methods.items():
        r = m["regret"]
        print(f"{k:42s} regret {r['mean']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
    result = make_result(
        run_id=f"smoke-attention-{args.env}", track="SMOKE", smoke=True, env=env, visibility=args.visibility,
        splits={"train": split_record(tr, "train"), "eval": split_record(dv, "dev")},
        oracle_check=oracle_check(env), methods=methods, learned_arms=["attn_smoke:trained"],
        controls={"attn_smoke:trained": {"random_network": "attn_smoke:control:random_network",
                                         "untrained_init": "attn_smoke:control:untrained_init"}})
    print("wrote", write_result(args.out, result))


if __name__ == "__main__":
    main()
