"""E1 full_context_transformer scaling cells, fresh regime (never-reused episodes), unaligned inputs.

Each cell trains FullContextTransformer (aggbench defaults except width / layers / steps) on a FreshStream and
reports regret to Bayes on dev episodes 0..1999 (the same episodes as scripts/e1_floor_none.py, so per-episode
differences to the floor are paired) and on 2048 train episodes the model saw (the last 4 minibatches of the
stream): train/dev gap = dev regret - seen-train regret. Writes <out>/cell-<vis>-w<W>l<L>-s<steps>.json and the
per-episode dev regret (.npy) as each cell finishes.

Seeds: the train split holds 10M episodes and FreshStream's input statistics use train episodes from 9M, so a
fresh run is capped at 9M / 512 = 17.5k steps. Training episode j uses train-split seed j for j < 9M and seed
100_000_000 + (j - 9M) beyond (outside every split: train 0-10M, dev 10-20M, test 20-30M, smoke 90-100M), so no
episode is reused and none is a dev/test episode.

    python3 scripts/e1_fct_scaling.py --visibility none --cells 128x4:3000,128x4:12000 --samplers 7 --out DIR
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggbench import make_env  # noqa: E402
from aggbench.metrics import per_episode, summarize_values  # noqa: E402
from aggbench.envs import VISIBILITY, Batch  # noqa: E402
from aggbench.transformer import (DEFAULTS, FRESH_STATS_START, FreshStream, FullContextTransformer,  # noqa: E402
                                  _torch, full_context)

EXTRA_SEED_BASE = 100_000_000


def train_seeds(start, n):
    j = start + np.arange(n, dtype=np.int64)
    return np.where(j < FRESH_STATS_START, j, EXTRA_SEED_BASE + j - FRESH_STATS_START)


def sample_seeds(env, seeds):
    """env.sample for arbitrary seeds (same grouping as aggbench.envs._Env.sample)."""
    from dataclasses import asdict
    eps = [env.episode(int(s)) for s in seeds]
    grouped = {}
    for key in eps[0]:
        arr = np.stack([e[key] for e in eps])
        group, _, field = key.partition(".")
        if field:
            grouped.setdefault(group, {})[field] = arr
        else:
            grouped[group] = arr
    keys = {v: grouped["key"][v] for v in VISIBILITY}
    return Batch(env.name, asdict(env.config), env.kind, np.asarray(seeds), grouped["truth"], grouped["obs"],
                 grouped["local"], grouped["report"], keys, grouped["bayes"], grouped["oracle"], env.public())


_W = {}


def _init(visibility):
    _W.update(env=make_env("E1"), vis=visibility)


def _batch(job):
    b = sample_seeds(_W["env"], train_seeds(*job))
    return full_context(b, _W["vis"], "unaligned").tokens, np.asarray(b.truth, np.float32)


class LongFreshStream(FreshStream):
    """FreshStream with the seed map above (identical to FreshStream while steps * batch <= 9M)."""

    def __iter__(self):
        import multiprocessing
        from collections import deque
        jobs = [(s * self.batch, self.batch) for s in range(self.steps)]
        with multiprocessing.get_context("fork").Pool(self.samplers, initializer=_init,
                                                      initargs=(self.visibility,)) as pool:
            pending = deque()
            for k, j in enumerate(jobs):
                if k and k in getattr(self, "checkpoints", ()):
                    self.checkpoint(k)
                pending.append(pool.apply_async(_batch, (j,)))
                if len(pending) >= 2 * self.samplers:
                    yield pending.popleft().get()
            while pending:
                yield pending.popleft().get()


def dev_regret_now(model, cdv, dv):
    """Dev regret of the model mid-training (eval mode, back to train mode after); for salvage on timeouts."""
    from aggbench.transformer import _gauss_from_head
    torch = _torch()
    m = model.module
    dev = next(m.parameters()).device
    m.eval()
    with torch.no_grad():
        h = torch.cat([m(torch.tensor(cdv.tokens[s:s + 2048], device=dev)).double().cpu()
                       for s in range(0, len(cdv.tokens), 2048)])
    m.train()
    mean, L = _gauss_from_head(torch, h, model.d)
    return per_episode({"mean": mean.numpy(), "prec": (L @ L.transpose(1, 2)).numpy()}, dv)["regret"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visibility", required=True)
    ap.add_argument("--cells", required=True, help="comma list of WIDTHxLAYERS:STEPS")
    ap.add_argument("--samplers", type=int, default=7)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env("E1")
    dv = env.sample("dev", a.n_eval)
    cdv = full_context(dv, a.visibility, "unaligned")
    for cell in a.cells.split(","):
        arch, steps = cell.split(":")
        width, layers = map(int, arch.split("x"))
        steps = int(steps)
        cfg = dict(DEFAULTS, width=width, layers=layers, steps=steps)
        t0 = time.time()
        stream = LongFreshStream(env, a.visibility, "unaligned", cfg["batch"], steps, samplers=a.samplers)
        in_dim = stream.stats_tokens.shape[-1]
        model = FullContextTransformer(0, in_dim, dv.kind, d=dv.public["d"], device=device, **cfg)
        tag = f"{a.visibility}-w{width}l{layers}-s{steps}"
        stream.checkpoints = set(range(25000, steps, 25000))
        stream.checkpoint = lambda k, model=model, tag=tag: print(
            f"CKPT {tag} step {k} dev {dev_regret_now(model, cdv, dv).mean():.4f} (mid-schedule)", flush=True)
        model.fit(stream, None)
        seen = sample_seeds(env, train_seeds((steps - 4) * cfg["batch"], 4 * cfg["batch"]))
        reg_dev = per_episode(model.predict(cdv), dv)["regret"]
        reg_seen = per_episode(model.predict(full_context(seen, a.visibility, "unaligned")), seen)["regret"]
        np.save(out / f"cell-{tag}-dev-regret.npy", reg_dev)
        rec = {"env": "E1", "visibility": a.visibility, "alignment": "unaligned", "regime": "fresh",
               "device": device, "config": cfg, "seed_map": "train j<9M, else 1e8+j-9M", "n_train_episodes": steps * cfg["batch"],
               "dev_regret": summarize_values(reg_dev), "seen_train_regret": summarize_values(reg_seen),
               "train_dev_gap": float(reg_dev.mean() - reg_seen.mean()),
               "wall_seconds": time.time() - t0, "loss_log": model.log}
        (out / f"cell-{tag}.json").write_text(json.dumps(rec, indent=2) + "\n")
        print(f"CELL {tag} dev {reg_dev.mean():.4f} seen {reg_seen.mean():.4f} {rec['wall_seconds']:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
