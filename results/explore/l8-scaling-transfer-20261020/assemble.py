"""Build harness-validated result JSONs and README tables from the forecasts that l8.py saved.

Runs locally (numpy only) at a clean harness commit, so harness.git_commit is genuine (the GPU VM gets
the repo without .git). It regenerates every dev batch from its seeds, checks that the saved seeds and
Bayes logits match the regenerated batch exactly, and then scores every saved forecast with
aggbench.metrics. Usage: python3 assemble.py RUN_DIR
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../..")))
from aggbench import make_env  # noqa: E402
from aggbench.metrics import paired_difference, per_episode, summarize, summarize_values  # noqa: E402
from aggbench.results import harness_info, make_result, split_record, write_result  # noqa: E402
from aggbench.verify import oracle_check  # noqa: E402

RUN_ID = "l8-scaling-transfer-20261020"
MAIN_ARMS = ("attn1", "attnloop", "attnloop_T8", "clearloop", "clearloop_T64")
CURVE_ARMS = ("attn1", "attnloop", "clearloop")


def fmt(s, nd=3):
    return f"{s['mean']:.{nd}f} [{s['ci_low']:.{nd}f}, {s['ci_high']:.{nd}f}]"


def main(run_dir):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    local_h = harness_info()
    assert not local_h["git_dirty"], "harness must be clean"
    same_harness = meta["harness"]["sha256"] == local_h["sha256"]
    assert same_harness or os.environ.get("L8_TEST_ONLY"), "VM harness sha256 differs from local harness"
    n_eval = meta["n_eval"]
    seeds = sorted({int(sp[2]) for sp in meta["specs"] if sp[6] is None})
    out_json = os.path.join(os.environ.get("L8_OUT", HERE), "json")
    summary = {"run_id": RUN_ID, "harness": local_h, "vm_harness_sha256_matches": same_harness,
               "device": meta["device"], "steps": meta["steps"], "n_eval": n_eval, "seeds": seeds,
               "param_counts": meta["param_counts"], "alpha_train_cfg": meta["alpha_train_cfg"],
               "tables": {}}
    for vis in ("full", "noisy"):
        for cname, ov in meta["conditions"].items():
            path = os.path.join(run_dir, f"preds_{vis}_{cname}.npz")
            if not os.path.exists(path):
                continue
            z = np.load(path)
            env = make_env("E2", **{k: tuple(v) if isinstance(v, list) else v for k, v in ov.items()})
            dv = env.sample("dev", n_eval)
            assert np.array_equal(z["seeds"], dv.seeds)
            bayes_diff = float(np.abs(z["bayes"] - dv.bayes["logit"]).max())
            assert bayes_diff < 1e-9, bayes_diff
            fc = {k: {"logit": z[k]} for k in z.files if k not in ("seeds", "bayes", "truth")}
            methods = {k.replace("baseline:", ""): summarize(f, dv) for k, f in fc.items()}
            learned = sorted(k[:-len(":trained")] for k in fc if k.endswith(":trained"))
            controls = {f"{a}:trained": {"random_network": f"{a}:control:random_network",
                                         "untrained_init": f"{a}:control:untrained_init"} for a in learned}
            paired = {}
            for a in learned:
                f = fc[f"{a}:trained"]
                paired[a] = {"minus_extremized_log_pool": paired_difference(f, fc["baseline:extremized_log_pool"], dv),
                             "minus_extremized_log_pool@train_alpha":
                                 paired_difference(f, fc["baseline:extremized_log_pool@train_alpha"], dv),
                             "minus_oracle_clearing": paired_difference(f, fc["baseline:oracle_clearing"], dv)}
            # seed-averaged per-episode regret for the main arms (CI over episodes; seed range shown separately)
            seed_avg = {}
            orc = per_episode(fc["baseline:oracle_clearing"], dv)["regret"]
            for arm in MAIN_ARMS:
                keys = [f"{arm}:s{s}" for s in seeds if f"{arm}:s{s}:trained" in fc]
                if not keys:
                    continue
                regs = np.array([per_episode(fc[f"{k}:trained"], dv)["regret"] for k in keys])
                rn = np.array([per_episode(fc[f"{k}:control:random_network"], dv)["regret"] for k in keys])
                ui = np.array([per_episode(fc[f"{k}:control:untrained_init"], dv)["regret"] for k in keys])
                seed_avg[arm] = {"regret_bayes": summarize_values(regs.mean(0)),
                                 "regret_oracle_clearing": summarize_values(regs.mean(0) - orc),
                                 "seed_means": regs.mean(1).tolist(),
                                 "random_network_regret": summarize_values(rn.mean(0)),
                                 "untrained_init_regret": summarize_values(ui.mean(0))}
            splits = {"eval": split_record(dv, "dev")}
            for rec in meta["train"][f"{vis}:n{max(int(k.split(':n')[1]) for k in meta['train'] if k.startswith(vis))}"]:
                splits[f"train_N{rec['n_agents']}"] = {k: rec[k] for k in ("split", "first_seed", "n")}
            splits["alpha_target_fit"] = {"split": "train", "first_seed": 8_000_000, "n": 2000}
            res = make_result(
                run_id=f"{RUN_ID}:{vis}:{cname}", track="EXPLORATORY", env=env, visibility=vis, splits=splits,
                oracle_check=oracle_check(env), methods=methods,
                learned_arms=[f"{a}:trained" for a in learned], controls=controls,
                extra={"condition": cname, "train_config": {"env": "E2", "n_agents": meta["train_ns"],
                                                            **meta["train_cfg"]},
                       "labels": "truth (train split); no distillation", "steps": meta["steps"],
                       "paired_differences": paired, "seed_averaged": seed_avg,
                       "alpha_train_cfg": meta["alpha_train_cfg"][vis]["used"],
                       "alpha_target_cfg": meta["alpha_target_cfg"][f"{vis}:{cname}"],
                       "saved_bayes_max_abs_diff": bayes_diff, "vm_device": meta["device"],
                       "vm_harness_sha256_matches": same_harness,
                       "fit_memoized_note": "one trained model per (arm, seed, width, data); run_with_controls "
                                            "re-run per condition with fit() memoized"})
            write_result(out_json, res, name=f"{vis}_{cname}.json")
            row = {m: methods[m]["regret"] for m in ("oracle_clearing", "extremized_log_pool",
                                                     "extremized_log_pool@train_alpha", "log_pool")}
            row.update({f"{a}:{k}": v[k] for a, v in seed_avg.items()
                        for k in ("regret_bayes", "regret_oracle_clearing", "random_network_regret",
                                  "untrained_init_regret")})
            row.update({f"{a}:seed_means": v["seed_means"] for a, v in seed_avg.items()})
            row["curve"] = {a: {"regret_bayes": methods[f"{a}:trained"]["regret"],
                                "random_network": methods[f"{a}:control:random_network"]["regret"],
                                "regret_oracle_clearing": paired[a]["minus_oracle_clearing"]}
                            for a in learned if ":w" in a}
            summary["tables"][f"{vis}:{cname}"] = row
            print(vis, cname, "ok", flush=True)
    json.dump(summary, open(os.path.join(os.environ.get("L8_OUT", HERE), "summary.json"), "w"), indent=1)
    write_tables(summary, meta)


def write_tables(summary, meta):
    T = summary["tables"]
    lines = []
    for vis in ("full", "noisy"):
        if f"{vis}:N8" not in T:
            continue
        lines += [f"\n### Regret to Bayes, keys `{vis}` (nats/episode, mean over {len(summary['seeds'])} training seeds, "
                  "95% bootstrap CI over episodes)\n",
                  "| test condition | oracle_clearing | extremized (alpha fit at target) | extremized (train alpha) | "
                  "attn1 | attnloop | attnloop T8 | clearloop | clearloop T64 | lowest random-net control (any arm) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for c in meta["conditions"]:
            r = T.get(f"{vis}:{c}")
            if not r:
                continue
            cells = [fmt(r["oracle_clearing"]), fmt(r["extremized_log_pool"]), fmt(r["extremized_log_pool@train_alpha"])]
            cells += [fmt(r[f"{a}:regret_bayes"]) if f"{a}:regret_bayes" in r else "-" for a in MAIN_ARMS]
            ctl = min(r[f"{a}:random_network_regret"]["mean"] for a in MAIN_ARMS if f"{a}:random_network_regret" in r)
            lines.append(f"| {c} | " + " | ".join(cells) + f" | {ctl:.3f} |")
        lines += [f"\n### Regret to oracle_clearing, keys `{vis}` (paired, seed-averaged; < 0 means better than the "
                  "reports-level ceiling)\n",
                  "| test condition | " + " | ".join(MAIN_ARMS) + " | clearloop regret to Bayes, min-max over seeds |",
                  "|---|" + "---|" * (len(MAIN_ARMS) + 1)]
        for c in meta["conditions"]:
            r = T.get(f"{vis}:{c}")
            if not r:
                continue
            cells = [fmt(r[f"{a}:regret_oracle_clearing"]) if f"{a}:regret_oracle_clearing" in r else "-"
                     for a in MAIN_ARMS]
            sm = r.get("clearloop:seed_means", [np.nan])
            lines.append(f"| {c} | " + " | ".join(cells) + f" | {min(sm):.3f}-{max(sm):.3f} |")
        lines += [f"\n### Learning curves, keys `{vis}`, seed 0 (regret to Bayes; same 6000 steps for every model)\n",
                  "| arm | width | train episodes | params | N8 (in distribution) | N64 (zero-shot) | N64 random-net control |",
                  "|---|---|---|---|---|---|---|"]
        for arm in CURVE_ARMS:
            rows = []
            for key in T[f"{vis}:N8"]["curve"]:
                if key.startswith(arm + ":w"):
                    w, n = key.split(":")[1:3]
                    rows.append((int(w[1:]), int(n[1:]), key))
            main_key = f"{arm}:s0"
            main_n = max(int(sp[4]) for sp in meta["specs"] if sp[6] is None)
            rows.append((32, main_n, None))
            for w, n, key in sorted(rows, key=lambda t: (t[0], t[1])):
                if key is None:
                    a8 = T[f"{vis}:N8"][f"{arm}:seed_means"][0]
                    a64 = T[f"{vis}:N64"][f"{arm}:seed_means"][0]
                    lines.append(f"| {arm} | {w} | {n} | {summary['param_counts'][f'{arm}:w{w}']} | {a8:.3f} (main, s0) | "
                                 f"{a64:.3f} (main, s0) | - |")
                    continue
                c8, c64 = T[f"{vis}:N8"]["curve"][key], T[f"{vis}:N64"]["curve"][key]
                lines.append(f"| {arm} | {w} | {n} | {summary['param_counts'][f'{arm}:w{w}']} | "
                             f"{fmt(c8['regret_bayes'])} | {fmt(c64['regret_bayes'])} | {c64['random_network']['mean']:.3f} |")
    open(os.path.join(os.environ.get("L8_OUT", HERE), "tables.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
