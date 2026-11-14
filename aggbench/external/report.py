"""Default-config regret-to-Bayes tables of the literature methods (aggbench/external) on E1, E2 and E3.

    PY=/home/agent/.venvs/aggext/bin/python      # torch + tabpfn (versions in aggbench/README.md)
    $PY -m aggbench.external.report --cell E2 noisy-unaligned --part closed --out /tmp/aggext-runs  # one cell
    $PY -m aggbench.external.report --all --part learned --out /tmp/aggext-runs --jobs 2    # every cell
    $PY -m aggbench.external.report --collect /tmp/aggext-runs --write                   # README + JSON

Settings (cells):
  scalar            scalar-report baselines: reports only (keys none). Closed-form rules, a scalar Set Transformer
                    and TabPFN on the reports. All five environments.
  {none,noisy}-unaligned   RICH inputs, the headline: every learned method reads batch.agent_inputs(visibility,
                    alignment="unaligned") (each agent's raw observation, key and own report); the PIF row uses the
                    unaligned noisy keys with the reports.
  {none,noisy}-aligned     the same with alignment="aligned": a giveaway control (shared slot positions reveal the
                    overlap), not a result.
Rich cells cover E1, E2, E3-sybil and E3-collude (E3-biased is excluded: its raw observations are honest).
Learned methods train on train-split episodes 0..n_train-1 with full_context_transformer's budget (3000 AdamW steps
of 512, OneCycle lr 2e-3; TabPFN: up to 10000 of them as context on a GPU; rich TabPFN runs in the unaligned cells
only) and every method is scored on dev episodes 0..n_eval-1 (2000). Each cell part writes a result JSON
(aggbench.result/v1) with controls and the locality check of every message-passing arm. Run from a clean commit.
Split as run on 2026-10-21: `--part closed` on General (numpy), `--part learned` on spot L4 GPUs
(/home/agent/workspaces/mats/work/run_on_l4.sh); --collect merges the parts of each cell.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ENVS = ("E1", "E2", "E3-sybil", "E3-biased", "E3-collude")
RICH_ENVS = ("E1", "E2", "E3-sybil", "E3-collude")
RICH = ("none-unaligned", "noisy-unaligned", "none-aligned", "noisy-aligned")
SETTINGS = ("scalar", *RICH)
REFS = ("extremized_log_pool", "oracle_clearing")
SCALAR_COLS = (*REFS, "pif_oracle", "winkler_gls", "covariance_intersection", "inverse_covariance_intersection",
               "set_transformer", "set_transformer_pool", "tabpfn")
RICH_LEARNED = ("tabpfn", "deepsets", "set_transformer", "tarmac_r1", "tarmac_r3")
POOL = tuple(f"{a}_pool" for a in RICH_LEARNED[1:])
RICH_COLS = (*REFS, "pif_oracle", "full_context_transformer", *RICH_LEARNED)
POOL_COLS = ("full_context_transformer_pool", *POOL)
HERE = Path(__file__).resolve().parent
README = HERE.parent / "README.md"
BEGIN, END = "<!-- EXTERNAL:BEGIN -->", "<!-- EXTERNAL:END -->"
APP_BEGIN, APP_END = "<!-- EXTERNAL-MEMORISED:BEGIN -->", "<!-- EXTERNAL-MEMORISED:END -->"


def cells():
    return [(e, "scalar") for e in ENVS] + [(e, s) for e in RICH_ENVS for s in RICH]


def run_cell(env_name, setting, n_train=20000, n_eval=2000, steps=3000, seed=0, n_boot=2000, skip=(),
             allow_dirty=False, part="all"):
    """part: "closed" (closed-form rules and references; numpy, run on General), "learned" (learned arms with
    controls; run on a GPU), or "all". Parts of one cell are merged by --collect."""
    import torch
    torch.set_num_threads(int(os.environ.get("AGGEXT_THREADS", "2")))
    from .. import make_env
    from ..baselines import all_baselines
    from ..controls import run_with_controls
    from ..metrics import summarize
    from ..results import make_result, split_record
    from ..verify import oracle_check
    from . import closed_form as cf
    from .learned import locality, maker, rich_inputs
    from .tabpfn_adapter import config as tabpfn_config, run_tabpfn_with_controls

    env = make_env(env_name)
    tr, dv = env.sample("train", n_train), env.sample("dev", n_eval)
    f, alpha = all_baselines(tr, dv, "none")
    forecasts = {k: f[k] for k in ("bayes", *REFS)}
    extra = {"setting": setting, "part": part, "extremize_alpha": alpha, "seed": seed, "n_train": n_train,
             "learned": {"steps": steps, "batch": 512, "optimizer": "AdamW lr 1e-3 wd 0.01", "schedule": "OneCycle, "
                         "5% warm-up", "grad_clip": 1.0, "device": "cuda" if torch.cuda.is_available() else "cpu",
                         "regimes": {"<arch>": "fresh: new simulator episodes every step (headline)",
                                     "<arch>_pool": "20k train pool, early stopping on 10% of train"}},
             "tabpfn": tabpfn_config(kind=env.kind)}
    closed = part in ("all", "closed")
    if setting == "scalar":
        xtr, xdv = tr.view("none"), dv.view("none")
    if setting == "scalar" and closed:
        forecasts["pif_oracle"] = cf.pif_oracle(xdv, cf.fit_pif_compound(xtr))
        if dv.kind == "gaussian":
            forecasts["winkler_gls"] = cf.winkler_gls(xdv, cf.fit_winkler(xtr, tr.truth))
            ci = cf.covariance_intersection(xdv)
            forecasts["covariance_intersection"] = {"mean": ci["mean"], "prec": ci["prec"]}
            forecasts["inverse_covariance_intersection"] = cf.inverse_covariance_intersection(xdv)
    if setting == "scalar":
        archs = ("set_transformer", "tabpfn")
        visibility = "none"
    else:
        visibility, alignment = setting.split("-")
        xtr, xdv = rich_inputs(tr, visibility, alignment), rich_inputs(dv, visibility, alignment)
        if visibility == "noisy" and closed:
            # provenance keys only (what the aggregator is told), not the agents' own sensor models in xdv.keys
            prov = dv.agent_inputs("noisy", alignment=alignment)[1]
            forecasts["pif_oracle"] = cf.pif_oracle(cf.with_keys(dv.view("noisy"), prov))
        archs = RICH_LEARNED if alignment == "unaligned" else RICH_LEARNED[1:]
        extra["alignment"] = alignment
    learned_arms, controls, loc = [], {}, {}
    learned = part in ("all", "learned")
    stream = None

    def record(name, arms, ctl, t0):
        forecasts[name] = arms.pop("trained")
        forecasts.update({f"{name}:{k}": v for k, v in arms.items()})
        learned_arms.append(name)
        controls[name] = ctl
        extra.setdefault("seconds", {})[name] = round(time.time() - t0, 1)
        print(f"{env_name} {setting} {name} done in {time.time() - t0:.0f}s", flush=True)

    for arch in (archs if learned else ()):
        if arch in skip:
            continue
        t0 = time.time()
        if arch == "tabpfn":  # in-context: the train pool is its context (same in both regimes)
            record(arch, run_tabpfn_with_controls(xtr, tr.truth, xdv, seed=seed),
                   {"random_network": f"{arch}:control:random_network",
                    "untrained_init": f"{arch}:control:shuffled_labels"}, t0)
            continue
        # secondary regime: the 20k train pool, early stopping on 10% of train
        mk, made = maker(arch, xtr, steps=steps), []
        record(f"{arch}_pool", run_with_controls(lambda s: made.append(mk(s)) or made[-1], xtr, tr.truth, xdv,
                                                 seed=seed),
               {"random_network": f"{arch}_pool:control:random_network",
                "untrained_init": f"{arch}_pool:control:untrained_init"}, t0)
        extra.setdefault("pool_early_stopping", {})[arch] = made[0].log[-1]   # the trained arm is made first
        if setting != "scalar" and arch != "set_transformer":
            probe = mk(seed)  # architecture-level locality check: initial weights, train standardization, 0 steps
            probe.hp["steps"] = 0
            loc[arch] = locality(probe.fit(xtr, tr.truth), xdv)
        # headline regime: fresh simulator episodes every step (materialized once per cell, shared by the arms)
        t0 = time.time()
        if stream is None:
            stream = fresh_stream(env, setting, steps)
            extra["fresh"] = {"episodes": stream.n_episodes, "first_seed": stream.start,
                              "generate_seconds": round(time.time() - t0, 1)}
        record(arch, run_with_controls(maker(arch, stream), stream, None, xdv, seed=seed),
               {"random_network": f"{arch}:control:random_network",
                "untrained_init": f"{arch}:control:untrained_init"}, t0)
    if loc:
        extra["locality_check"] = loc
    methods = {k: summarize(v, dv, n_boot=n_boot) for k, v in forecasts.items()}
    return make_result(
        run_id=f"external-default-{env_name}-{setting}-{part}", track="REFERENCE", env=env, visibility=visibility,
        splits={"train": split_record(tr, "train"), "eval": split_record(dv, "dev")},
        oracle_check=oracle_check(env), methods=methods, learned_arms=learned_arms, controls=controls,
        allow_dirty=allow_dirty, extra=extra)


def fresh_stream(env, setting, steps, batch=512):
    """The fresh-regime minibatches of a cell, generated once (in parallel) and shared by every architecture, so all
    arms see the same new episodes; the stream itself is aggbench.transformer.FreshStream (rich) or
    learned.ScalarFreshStream (scalar baseline)."""
    from .learned import ScalarFreshStream
    from ..transformer import FreshStream
    if setting == "scalar":
        base = ScalarFreshStream(env, batch, steps)
    else:
        vis, al = setting.split("-")
        base = FreshStream(env, vis, al, batch, steps, samplers=max(1, (os.cpu_count() or 4) - 2))
    return CachedStream(base)


def CachedStream(base):
    """A FreshStream whose batches are materialized once and replayed identically to each architecture."""
    cache = list(iter(base))
    cls = type("CachedStream", (type(base),), {"__iter__": lambda self: iter(self._cache)})
    obj = cls.__new__(cls)
    obj.__dict__.update(base.__dict__)
    obj._cache = cache
    return obj


def _cell_path(out, env_name, setting, part="all"):
    return Path(out) / f"{env_name}__{setting}__{part}.json"


def merge(parts):
    """One result from the closed and learned parts of a cell (same harness commit required)."""
    base = json.loads(json.dumps(parts[0]))
    for r in parts[1:]:
        # parts may come from different commits (each part's harness is kept in extra.parts and named in the
        # table note); the README says what changed between them
        base["methods"].update(r["methods"])
        base["learned_arms"] += [a for a in r["learned_arms"] if a not in base["learned_arms"]]
        base["controls"].update(r["controls"])
        for k, v in r["extra"].items():
            base["extra"].setdefault(k, v)
        base["extra"].setdefault("parts", {})[r["extra"]["part"]] = {"harness": r["harness"],
                                                                     "created_utc": r["created_utc"]}
    for r in parts[:1]:
        base["extra"].setdefault("parts", {})[r["extra"]["part"]] = {"harness": r["harness"],
                                                                     "created_utc": r["created_utc"]}
    base["extra"]["part"] = "+".join(sorted(p["extra"]["part"] for p in parts))
    base["run_id"] = base["run_id"].rsplit("-", 1)[0]
    return base


def fmt(s):
    return f"{s['mean']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]"


def _fct(r, regime="fresh"):
    """full_context_transformer (aggbench/transformer.py, main's reference) for the same env and setting, from
    aggbench/baselines_transformer_fresh.json (regime fresh) or baselines_transformer.json (regime pool) when they
    have been generated."""
    from ..report import FCT_CACHE, FRESH_CACHE, _fct_cells
    if r["extra"]["setting"] == "scalar":
        return None
    vis, al = r["extra"]["setting"].split("-")
    c = _fct_cells(None, FRESH_CACHE if regime == "fresh" else FCT_CACHE)[0].get((r["env"]["name"], al, vis))
    return c["methods"]["full_context_transformer"]["regret"] if c else None


def _table(results, cols, learned):
    head = ["env", "setting", *cols]
    lines = ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    ctl = ["| env | setting | " + " | ".join(learned) + " |", "|---" * (2 + len(learned)) + "|"]
    for r in results:
        m = r["methods"]
        cells_ = [r["env"]["name"], r["extra"]["setting"]]
        for c in cols:
            if c.startswith("full_context_transformer"):
                s = _fct(r, "pool" if c.endswith("_pool") else "fresh")
            else:
                s = m[c]["regret"] if c in m else None
            cells_.append(fmt(s) if s else "n/a")
        lines.append("| " + " | ".join(cells_) + " |")
        cc = [r["env"]["name"], r["extra"]["setting"]]
        for a in learned:
            c = r["controls"].get(a)
            cc.append(f"{m[c['random_network']]['regret']['mean']:.3f} / {m[c['untrained_init']]['regret']['mean']:.3f}"
                      if c else "n/a")
        ctl.append("| " + " | ".join(cc) + " |")
    return "\n".join(lines), "\n".join(ctl)


def _commits(results):
    """Which harness commit produced which parts (closed-form on General, learned on GPU)."""
    by = {}
    for r in results:
        for part, info in (r["extra"].get("parts") or {r["extra"]["part"]: {"harness": r["harness"]}}).items():
            h = info["harness"]
            by.setdefault(part, set()).add(h["git_commit"][:10] + (" (dirty)" if h["git_dirty"] else ""))
    return "harness commits: " + ", ".join(f"{p} parts `{'`, `'.join(sorted(c))}`" for p, c in sorted(by.items()))


def table(results):
    scalar = [r for r in results if r["extra"]["setting"] == "scalar"]
    rich = sorted((r for r in results if r["extra"]["setting"] != "scalar"),
                  key=lambda r: (RICH_ENVS.index(r["env"]["name"]), RICH.index(r["extra"]["setting"])))
    t1, c1 = _table(rich, RICH_COLS, RICH_LEARNED)
    t3, c3 = _table(rich, POOL_COLS, POOL)
    t2, c2 = _table(scalar, SCALAR_COLS, ("set_transformer", "set_transformer_pool", "tabpfn"))
    r0 = results[0]
    note = (f"Mean regret to the global Bayes posterior (nats per episode), 95% bootstrap CIs over "
            f"{r0['splits']['eval']['n']} dev episodes. Learned arms use the optimizer of full_context_transformer "
            f"(AdamW lr 1e-3, weight decay 0.01, OneCycle, {r0['extra']['learned']['steps']} steps of 512), on an L4 "
            f"GPU. Headline regime: fresh simulator episodes every step, never reused (the same episodes for every "
            f"architecture of a cell). Pool regime: the {r0['splits']['train']['n']}-episode train pool with early "
            f"stopping on 10% of it. TabPFN is in-context (no training): v2 weights, the first 10000 train episodes as "
            f"context, package-default ensemble (E1 regressors: 1 estimator, a cost cap). `full_context_transformer` "
            f"is main's reference (aggbench/transformer.py) in the same regime, copied from "
            f"aggbench/baselines_transformer_fresh.json / baselines_transformer.json when generated. n/a: not defined "
            f"for that env/setting (closed-form Gaussian rules are E1 only; PIF needs keys or the scalar fit; rich "
            f"TabPFN runs only unaligned). Generated by `python -m aggbench.external.report --collect DIR --write`; "
            + _commits(results) + "; full numbers, controls, early-stopping steps and locality checks in "
            f"`aggbench/external/results_default.json`.")
    ctl_note = ("Controls: regret of the random-network / untrained-init control (TabPFN: random network / "
                "shuffled-label context).")
    return (f"**Rich inputs, fresh regime (headline)** (`aggbench.transformer.full_context` tokens: each agent's raw "
            f"observation, own sensor model, provenance key and report; `*-unaligned` is the headline, `*-aligned` a "
            f"giveaway control):\n\n{t1}\n\n{ctl_note}\n\n{c1}\n\n"
            f"**Rich inputs, pool regime (secondary)** (20k train pool, early stopping on 10% of train):\n\n{t3}"
            f"\n\n{ctl_note}\n\n{c3}\n\n"
            f"**Scalar-report baselines** (reports only, keys none; `set_transformer` fresh, `_pool` pool):\n\n{t2}"
            f"\n\n{ctl_note}\n\n{c2}\n\n{note}")


def appendix(results):
    """The superseded equal-budget run (fixed 20k pool, 3000 steps of 512, no early stopping): memorised."""
    rich = sorted((r for r in results if r["extra"]["setting"] != "scalar" and r["learned_arms"]),
                  key=lambda r: (RICH_ENVS.index(r["env"]["name"]), RICH.index(r["extra"]["setting"])))
    t, c = _table(rich, (*REFS, *RICH_LEARNED), RICH_LEARNED)
    return (f"{t}\n\nControls (random network / untrained init; TabPFN: random network / shuffled labels):\n\n{c}"
            f"\n\nSame tokens as the main table except the own sensor model (pre-20ee6a0 full key), 20k train pool "
            f"reused for 3000 AdamW steps of 512 (lr 2e-3, wd 1e-4), no early stopping, 2000 dev episodes; "
            + _commits(results) + ".")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", nargs=2, metavar=("ENV", "SETTING"))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--collect")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--appendix", action="store_true", help="with --collect --write: write the superseded run")
    ap.add_argument("--out", default="/tmp/aggext-runs")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--part", default="all", choices=["all", "closed", "learned"])
    ap.add_argument("--envs", nargs="*", help="--all: only these environments")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--allow-dirty", action="store_true", help="smoke only: accept uncommitted harness changes")
    args = ap.parse_args()
    if args.cell:
        env_name, setting = args.cell
        res = run_cell(env_name, setting, steps=args.steps, n_eval=args.n_eval, skip=args.skip,
                       allow_dirty=args.allow_dirty, part=args.part)
        p = _cell_path(args.out, env_name, setting, args.part)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=2) + "\n")
        print("wrote", p)
    elif args.all:
        procs, logs = [], Path(args.out)
        logs.mkdir(parents=True, exist_ok=True)
        for e, s in cells():
            if _cell_path(args.out, e, s, args.part).exists() or (args.envs and e not in args.envs):
                continue
            while sum(p.poll() is None for p in procs) >= args.jobs:
                time.sleep(5)
            cmd = [sys.executable, "-m", "aggbench.external.report", "--cell", e, s, "--out", args.out,
                   "--steps", str(args.steps), "--n-eval", str(args.n_eval), "--part", args.part]
            if args.skip:
                cmd += ["--skip", *args.skip]
            procs.append(subprocess.Popen(cmd, stdout=open(logs / f"{e}__{s}__{args.part}.log", "w"),
                                          stderr=subprocess.STDOUT))
        for p in procs:
            p.wait()
    elif args.collect:
        from ..results import validate
        results = []
        for e, s in cells():
            parts = []
            for part in ("all", "closed", "learned"):
                for p in Path(args.collect).rglob(_cell_path("", e, s, part).name):
                    r = json.loads(p.read_text())
                    validate(r)
                    parts.append(r)
            if parts:
                results.append(merge(parts))
        text = appendix(results) if args.appendix else table(results)
        print(text)
        if args.write:
            name, begin, end = (("results_memorised.json", APP_BEGIN, APP_END) if args.appendix
                                else ("results_default.json", BEGIN, END))
            (HERE / name).write_text(json.dumps({"results": results}, indent=2) + "\n")
            readme = README.read_text()
            head, rest = readme.split(begin)
            _, tail = rest.split(end)
            README.write_text(f"{head}{begin}\n{text}\n{end}{tail}")


if __name__ == "__main__":
    main()
