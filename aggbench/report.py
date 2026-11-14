"""Compute the default-config baseline table of every environment (aggbench.zoo.list_envs: E1-E4 and the
structure zoo, held-out structures included) and write it into README.md and baselines_default.json.

    python3 -m aggbench.report            # print the table
    python3 -m aggbench.report --write    # also rewrite the README block and the JSON
    python3 -m aggbench.report --only Z1 Z2 --n 500   # a subset (no --write)
    python3 -m aggbench.report --transformer --scaling --write  # also retrain full_context_transformer and its
                                                                # data-scaling curve (torch; hours on a busy CPU)

The full_context_transformer columns come from aggbench/baselines_transformer.json (written by --transformer,
reused otherwise): unaligned agent inputs with keys at none and at noisy, trained on 20k train episodes.

Train split episodes 0..n-1 fit the extremizing factor; dev split episodes 0..n-1 are scored.
Run it from a clean checkout of a commit (results.validate rejects a dirty harness unless --allow-dirty).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import make_env
from .baselines import all_baselines
from .envs import RolloutConfig, rollout_trace_posterior_logit, union_identified
from .metrics import summarize
from .results import make_result, split_record
from .verify import oracle_check
from .zoo import list_envs

HERE = Path(__file__).resolve().parent
BEGIN, END = "<!-- BASELINES:BEGIN -->", "<!-- BASELINES:END -->"
COLUMNS = ("prior", "floor", "floor_none", "oracle_clearing", "linear_pool", "log_pool", "sum_logodds", "extremized_log_pool")
FCT_SETTINGS = (("unaligned", "none"), ("unaligned", "noisy"))
FCT_CACHE = HERE / "baselines_transformer.json"              # regime "pool" (secondary row)
FRESH_CACHE = HERE / "baselines_transformer_fresh.json"      # regime "fresh" (headline)
SCALING_ENVS = ("E2", "Z8", "Z1", "Z6")
SCALING_SIZES = ((20000, 3000), (80000, 6000), (320000, 12000))   # (train episodes, steps)
SCALING_SETTING = ("unaligned", "none")
SCALING_CACHE = HERE / "baselines_transformer_scaling.json"
SCALING_BEGIN, SCALING_END = "<!-- SCALING:BEGIN -->", "<!-- SCALING:END -->"


def prior_forecast(batch):
    """The public prior alone: how much the agents' information is worth (regret of ignoring them)."""
    n = len(batch)
    if batch.kind == "bernoulli":
        return {"logit": np.full(n, float(batch.public["prior_logit"]))}
    d = batch.public["d"]
    return {"mean": np.zeros((n, d)), "prec": np.broadcast_to(np.eye(d), (n, d, d)).copy()}


def achievable_floor(batch):
    """Exact posterior given everything the agents can send at the headline setting (unaligned inputs, own
    models, provenance none): Bayes, except E4, where it is the posterior given every rollout's whole trace."""
    if "W" in batch.oracle:
        c, l0 = RolloutConfig(**batch.config), batch.public["prior_logit"]
        return {"logit": np.array([rollout_trace_posterior_logit(batch.obs[e], batch.oracle["order"][e],
                                                                 batch.oracle["mu"][e], c, l0)
                                   for e in range(len(batch))])}
    return {k: v.copy() for k, v in batch.bayes.items()}


def floor_without_provenance(env, batch):
    """The floor at provenance `none`: the exact posterior given the agents' observations and own models with
    source identities marginalized (zoo.none_floor_logit). Where own models or content identify the sources
    (E2-E4 by exact values, the non-blind zoo) it equals `floor`; None (unknown, not computed) for E1 (shared noise
    factors) and Z8 (piece overlap)."""
    from .zoo import ZOO, none_floor_logit
    if batch.env in ZOO:
        f = none_floor_logit(env, batch)
        return None if f is None else {"logit": f}
    if batch.kind == "gaussian":
        return None
    return achievable_floor(batch)


def run(n=2000, n_boot=2000, only=None, allow_dirty=False):
    results = []
    for info in list_envs():
        name = info["name"]
        if only and name not in only:
            continue
        env = make_env(name, allow_heldout=True)
        tr, dv = env.sample("train", n), env.sample("dev", n)
        forecasts, alpha = all_baselines(tr, dv)
        forecasts["prior"] = prior_forecast(dv)
        forecasts["floor"] = achievable_floor(dv)
        fn = floor_without_provenance(env, dv)
        if fn is not None:
            forecasts["floor_none"] = fn
        extra = {"extremize_alpha": alpha, "family": info["family"], "heldout": info["heldout"]}
        if "W" in dv.oracle:  # E4: union of the rollouts' conditioning sets covers the whole context
            extra["rollouts_cover_context_fraction"] = float(dv.oracle["cov"].any(1).all(1).mean())
        elif "report_logit" in dv.oracle:
            extra["reports_lossless_fraction"] = float(np.mean(
                np.abs(dv.oracle["report_logit"] - dv.bayes["logit"]) < 1e-9))
        elif "cov" in dv.oracle:
            extra["union_identified_fraction"] = float(np.mean([union_identified(c) for c in dv.oracle["cov"]]))
        results.append(make_result(
            run_id=f"baselines-default-{name}", track="REFERENCE", env=env, visibility="none",
            splits={"train": split_record(tr, "train"), "eval": split_record(dv, "dev")},
            oracle_check=oracle_check(env), allow_dirty=allow_dirty,
            methods={k: summarize(f, dv, n_boot=n_boot) for k, f in forecasts.items()}, extra=extra))
    return results


def _fct_job(args):
    import sys
    import time
    name, alignment, visibility, n_train, threads, allow_dirty, steps, *rest = args
    regime = rest[0] if rest else "pool"
    from .transformer import evaluate_full_context_transformer
    t0 = time.time()
    env = make_env(name, allow_heldout=True)
    rec = evaluate_full_context_transformer(env, n_train=n_train, visibility=visibility, alignment=alignment,
                                            threads=threads, steps=steps, regime=regime)
    print(f"[fct] {name} {alignment}/{visibility} {regime} n_train={rec['config']['n_train']} steps={steps}: regret "
          f"{rec['methods']['full_context_transformer']['regret']['mean']:.4f} in {time.time() - t0:.0f}s",
          file=sys.stderr, flush=True)
    tag = f"n{n_train}" if regime == "pool" else "fresh"
    result = make_result(
        run_id=f"full-context-transformer-{name}-{alignment}-{visibility}-{tag}", track="REFERENCE", env=env,
        visibility=visibility, splits=rec["splits"], oracle_check=oracle_check(env), methods=rec["methods"],
        learned_arms=rec["learned_arms"], controls=rec["controls"], allow_dirty=allow_dirty,
        extra={"alignment": alignment, "regime": regime, "training": rec["config"], "loss_log": rec["log"]})
    return result


def _job_file(job_dir, job):
    name, alignment, visibility, n_train, _, _, steps, *rest = job
    fresh = "-fresh" if rest and rest[0] == "fresh" else ""
    return Path(job_dir) / f"{name}-{alignment}-{visibility}-n{n_train}-s{steps}{fresh}.json"


def _fct_job_saved(args):
    job, job_dir = args
    result = _fct_job(job)
    if job_dir:
        _job_file(job_dir, job).write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_jobs(jobs, workers=4, job_dir=None):
    """Run _fct_job over jobs in parallel processes, longest first; results in job order. With job_dir, every
    finished job is saved there at once and a rerun skips jobs already saved (resume)."""
    from concurrent.futures import ProcessPoolExecutor
    done = {}
    if job_dir:
        Path(job_dir).mkdir(parents=True, exist_ok=True)
        for j, job in enumerate(jobs):
            if _job_file(job_dir, job).exists():
                done[j] = json.loads(_job_file(job_dir, job).read_text())
    first = {}                                   # identical jobs (e.g. the 20k scaling point) run once
    for j, job in enumerate(jobs):
        first.setdefault(job, j)
    cost = lambda j: -jobs[j][6] * (512 if jobs[j][7:] == ("fresh",) else 0.01 * jobs[j][3])
    todo = sorted({first[jobs[j]] for j in range(len(jobs)) if j not in done}, key=cost)
    from concurrent.futures import as_completed
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_fct_job_saved, (jobs[j], job_dir)): j for j in todo}
        for f in as_completed(futs):             # a failing job stops the run at once, not at the end
            done[futs[f]] = f.result()
    return [done[j] if j in done else done[first[jobs[j]]] for j in range(len(jobs))]


def transformer_jobs(names, n_train=20000, steps=3000, threads=2, allow_dirty=False, regime="pool"):
    if regime == "fresh":
        return [(n, a, v, n_train, threads, allow_dirty, steps, "fresh") for n in names for a, v in FCT_SETTINGS]
    return [(n, a, v, n_train, threads, allow_dirty, steps) for n in names for a, v in FCT_SETTINGS]


def scaling_jobs(threads=2, allow_dirty=False):
    a, v = SCALING_SETTING
    return [(n, a, v, size, threads, allow_dirty, steps) for n in SCALING_ENVS for size, steps in SCALING_SIZES]


def scaling_table(scaling, baselines):
    """Data-scaling curve of full_context_transformer at the headline setting, against the floor and the
    best report baseline."""
    base = {r["env"]["name"]: r for r in baselines}
    sizes = sorted({r["extra"]["training"]["n_train"] for r in scaling})
    cell = {(r["env"]["name"], r["extra"]["training"]["n_train"]): r for r in scaling}
    steps = {r["extra"]["training"]["n_train"]: r["extra"]["training"]["steps"] for r in scaling}
    head = ["env", *[f"{n // 1000}k train, {steps[n]} steps" for n in sizes], "floor (provenance none)", "prior",
            "extremized_log_pool"]
    lines = ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    for name in dict.fromkeys(r["env"]["name"] for r in scaling):
        cells = [name]
        for n in sizes:
            s = cell[(name, n)]["methods"]["full_context_transformer"]["regret"]
            cells.append(f"{s['mean']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]")
        for m in ("floor_none", "prior", "extremized_log_pool"):
            s = base[name]["methods"][m]["regret"] if name in base and m in base[name]["methods"] else None
            cells.append(f"{s['mean']:.3f}" if s else "unknown")
        lines.append("| " + " | ".join(cells) + " |")
    h = scaling[0]["harness"]
    a, v = SCALING_SETTING
    lines += ["", f"`full_context_transformer` regret to Bayes (nats, 95% bootstrap CI, 2000 dev episodes) versus "
              f"training episodes, {a} inputs with own models, provenance keys `{v}`; steps grow with the data. "
              f"Generated by `python3 -m aggbench.report --scaling --write` at harness commit "
              f"`{h['git_commit'][:10]}`{' (dirty)' if h['git_dirty'] else ''}, {scaling[0]['created_utc']}; "
              f"full numbers and controls in `aggbench/baselines_transformer_scaling.json`."]
    return "\n".join(lines)


def _fct_cells(fct_results=None, cache_file=None):
    """{(env, alignment, visibility): result} from fct_results or else the cache ({} if neither)."""
    cache_file = cache_file or FCT_CACHE
    if fct_results is None:
        if not cache_file.exists():
            return {}, None
        fct_results = json.loads(cache_file.read_text())["results"]
    cache = fct_results
    cells = {(r["env"]["name"], r["extra"]["alignment"], r["visibility"]): r for r in cache}
    return cells, cache[0]["harness"] if cache else None


def _ceiling_note(extra):
    if "union_identified_fraction" in extra:
        return f"union identified {extra['union_identified_fraction']:.1%}"
    if "reports_lossless_fraction" in extra:
        return f"reports lossless {extra['reports_lossless_fraction']:.1%}"
    if "rollouts_cover_context_fraction" in extra:
        return "reports ceiling (rollouts)"
    return "reports lossless (always)"


def table(results, fct_results=None, fresh_results=None):
    fct, fct_harness = _fct_cells(fct_results)
    fresh, fresh_harness = _fct_cells(fresh_results, FRESH_CACHE)
    fct_cols = ([f"transformer fresh ({a}, provenance {v})" for a, v in FCT_SETTINGS] if fresh else []) + \
               ([f"transformer pool 20k ({a}, provenance {v})" for a, v in FCT_SETTINGS] if fct else [])
    head = ["env", "family", *COLUMNS, *fct_cols, "Bayes log loss", "alpha", "reports ceiling"]
    lines = ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    for r in results:
        cells = [r["env"]["name"] + (" (held out)" if r["extra"]["heldout"] else ""), r["extra"]["family"]]
        for m in COLUMNS:
            if m not in r["methods"]:
                cells.append("unknown")
                continue
            s = r["methods"][m]["regret"]
            cells.append(f"{s['mean']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]")
        for src, a, v in [(fresh, a, v) for a, v in (FCT_SETTINGS if fresh else ())] + \
                [(fct, a, v) for a, v in (FCT_SETTINGS if fct else ())]:
            c = src.get((r["env"]["name"], a, v))
            s = c["methods"]["full_context_transformer"]["regret"] if c else None
            cells.append(f"{s['mean']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]" if s else "n/a")
        cells += [f"{r['methods']['bayes']['logloss']['mean']:.3f}", f"{r['extra']['extremize_alpha']:.2f}",
                  _ceiling_note(r["extra"])]
        lines.append("| " + " | ".join(cells) + " |")
    h = results[0]["harness"]
    lines.append("")
    lines.append(f"Mean regret to the global Bayes posterior (nats per episode) with 95% bootstrap CIs over "
                 f"{results[0]['splits']['eval']['n']} dev episodes; `prior` = the public prior alone (what the "
                 f"agents' information is worth). Generated by `python3 -m aggbench.report --write` "
                 f"at harness commit `{h['git_commit'][:10]}`{' (dirty)' if h['git_dirty'] else ''}, "
                 f"{results[0]['created_utc']}; full numbers in `aggbench/baselines_default.json`; brute-force "
                 f"oracle checks: " + ", ".join(
                     f"{r['env']['name']} {r['oracle_check']['max_abs_diff']:.0e}" for r in results) + ".")
    if fresh:
        first = next(iter(fresh.values()))
        t = first["extra"]["training"]
        lines.append("")
        lines.append(f"`transformer fresh` (headline regime): `full_context_transformer` trained PFN-style on "
                     f"{t['steps']} steps x {t['batch']} FRESH simulator episodes (no reuse, slot permutations "
                     f"resampled every minibatch), lr {t['lr']}, weight decay {t['weight_decay']}, dropout "
                     f"{t['dropout']}; harness commit `{fresh_harness['git_commit'][:10]}`"
                     f"{' (dirty)' if fresh_harness['git_dirty'] else ''}, {first['created_utc']}; controls in "
                     f"`aggbench/baselines_transformer_fresh.json`. `transformer pool 20k` (secondary): a fixed pool "
                     f"of 20k train episodes with early stopping on 10% of them.")
    if fct:
        first = next(iter(fct.values()))
        t = first["extra"]["training"]
        lines.append("")
        lines.append(f"`full_context_transformer` (`aggbench/transformer.py`): {t['layers']}-layer pre-norm transformer "
                     f"encoder, width {t['width']}, {t['heads']} heads, CLS readout, one token per agent = [raw "
                     f"observation, present mask, own sensor model, provenance key, report] from "
                     f"`agent_inputs(alignment=..., own_model=True)`; trained from "
                     f"outcomes on {t['n_train']} train episodes, {t['steps']} AdamW steps of {t['batch']}, OneCycle lr "
                     f"{t['lr']}; evaluated on {first['splits']['eval']['n']} dev episodes with the harness controls "
                     f"(random-network and untrained-init regrets are in `aggbench/baselines_transformer.json`). "
                     f"Harness commit `{fct_harness['git_commit'][:10]}`"
                     f"{' (dirty)' if fct_harness['git_dirty'] else ''}, {first['created_utc']}. E3-biased: the "
                     f"biased agent's raw observation is honest, so the transformer sees through the bias there.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--only", nargs="*", help="environment names (default: all)")
    ap.add_argument("--allow-dirty", action="store_true", help="accept uncommitted harness changes")
    ap.add_argument("--transformer", action="store_true", help="retrain full_context_transformer (needs torch)")
    ap.add_argument("--scaling", action="store_true", help="retrain its data-scaling curve (needs torch)")
    ap.add_argument("--fresh", action="store_true", help="retrain it in the fresh regime (headline; needs torch)")
    ap.add_argument("--n-train-transformer", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    ap.add_argument("--job-dir", help="save each finished transformer job here; a rerun resumes from it")
    args = ap.parse_args()
    if args.write and args.only:
        ap.error("--write needs the full set of environments")
    results = run(args.n, only=args.only, allow_dirty=args.allow_dirty)
    jobs = []
    if args.transformer:
        jobs += transformer_jobs([r["env"]["name"] for r in results], n_train=args.n_train_transformer,
                                 threads=args.threads, allow_dirty=args.allow_dirty)
    n_main = len(jobs)
    if args.scaling:
        jobs += scaling_jobs(threads=args.threads, allow_dirty=args.allow_dirty)
    n_scaling = len(jobs)
    if args.fresh:
        jobs += transformer_jobs([r["env"]["name"] for r in results], threads=args.threads,
                                 allow_dirty=args.allow_dirty, regime="fresh")
    done = run_jobs(jobs, args.workers, args.job_dir) if jobs else []
    fct, scaling, fresh = (done[:n_main] or None), done[n_main:n_scaling], (done[n_scaling:] or None)
    if args.write and fresh:
        FRESH_CACHE.write_text(json.dumps({"results": fresh}, indent=2) + "\n")
    if args.write and fct:
        FCT_CACHE.write_text(json.dumps({"results": fct}, indent=2) + "\n")
    if args.write and scaling:
        SCALING_CACHE.write_text(json.dumps({"results": scaling}, indent=2) + "\n")
    text = table(results, fct, fresh)
    print(text)
    if not scaling and SCALING_CACHE.exists():
        scaling = json.loads(SCALING_CACHE.read_text())["results"]
    stext = scaling_table(scaling, results) if scaling else None
    if stext:
        print(stext)
    if args.write:
        (HERE / "baselines_default.json").write_text(json.dumps({"results": results}, indent=2) + "\n")
        readme = (HERE / "README.md").read_text()
        head, rest = readme.split(BEGIN)
        _, tail = rest.split(END)
        readme = f"{head}{BEGIN}\n{text}\n{END}{tail}"
        if stext and SCALING_BEGIN in readme:
            head, rest = readme.split(SCALING_BEGIN)
            _, tail = rest.split(SCALING_END)
            readme = f"{head}{SCALING_BEGIN}\n{stext}\n{SCALING_END}{tail}"
        (HERE / "README.md").write_text(readme)


if __name__ == "__main__":
    main()
