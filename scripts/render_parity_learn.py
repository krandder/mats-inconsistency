"""Render the parity-learnability tables of aggbench/README.md from aggbench/compression_results/parity_learn*.json.

    python3 scripts/render_parity_learn.py --merge OUT.json RUN_DIR... --commit SHA   # merge L4 run JSONs, add harness
    python3 scripts/render_parity_learn.py --write                                   # rewrite the README block

Rows group runs by (experiment, b, channel, decoder, decoder keys, input norm, key warm-up) over seeds. gap = mean dev
KL minus the REFERENCE: the uncompressed reader trained on the same stages (same curriculum and data) when one
exists, else the best uncompressed reader on the same final spec. "escapes" = seeds whose dev KL is below 0.9 x the
prior's. Every number comes from the JSON; nothing is typed by hand.
"""
import argparse
import glob
import json
import statistics as st
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "aggbench" / "compression_results"
BEGIN, END = "<!-- PARITY-LEARN:BEGIN -->", "<!-- PARITY-LEARN:END -->"


def expected_sha256(commit):
    """sha256 of the aggbench harness sources at `commit` (aggbench.results.harness_info's algorithm), from git."""
    import hashlib
    import io
    import tarfile
    data = subprocess.run(["git", "-C", str(ROOT), "archive", commit, "aggbench"], capture_output=True,
                          check=True).stdout
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        for m in t.getmembers():
            p = Path(m.name)
            if m.isfile() and p.suffix in (".py", ".json") and "__pycache__" not in p.parts \
                    and not p.name.startswith("baselines_"):
                files[str(p.relative_to("aggbench"))] = t.extractfile(m).read()
    h = hashlib.sha256()
    for rel in sorted(files, key=lambda r: Path(r)):
        h.update(rel.encode() + b"\0" + files[rel] + b"\0")
    return h.hexdigest()


def merge(out, run_dirs, commit):
    """Merge L4 run JSONs. Cleanliness is CHECKED, not assumed: each run's harness sha256 (computed on the VM over the
    synced aggbench sources) must equal the sha256 of `commit`'s aggbench in git; git_dirty records the result."""
    full = subprocess.run(["git", "-C", str(ROOT), "rev-parse", commit], capture_output=True, text=True,
                          check=True).stdout.strip()
    expect = expected_sha256(full)
    results, runs = [], []
    for d in run_dirs:
        for f in glob.glob(f"{d}/**/parity_learn.json", recursive=True):
            j = json.load(open(f))
            rid = Path(f).parent.name
            sha = (j.get("harness") or {}).get("sha256")
            runs.append({"run": rid, "sha256": sha, "matches_commit": sha == expect})
            for r in j["results"]:
                r["gpu_run"] = rid
                results.append(r)
    dirty = not all(r["matches_commit"] for r in runs)
    json.dump({"schema": "aggbench.parity-learn/v1",
               "harness": {"git_commit": full, "expected_sha256": expect, "git_dirty": dirty,
                           "source": "work/run_on_l4.sh trains from a checkout of this commit; git_dirty is false "
                                     "only if every run's harness sha256 equals this commit's"},
               "runs": runs, "results": results}, open(out, "w"), indent=0)
    print(f"merged {len(results)} results from {len(runs)} runs; dirty={dirty}")


def key(r):
    return (r["name"], r["b"], r["channel"], r["decoder"], r["dec_keys"], r.get("in_norm", False),
            r.get("key_warmup", 0), r.get("key_scale", 1.0), r.get("pretrain", 0), r.get("pretrain_mode"),
            r.get("power_norm", False))


SPEC_DEFAULTS = {"readers_alpha": 0.0}       # fields added later; older JSON lacks them


def _norm(spec):
    return {**SPEC_DEFAULTS, **spec}


def spec_key(r):
    return json.dumps(_norm(r["stages"][-1][0]), sort_keys=True)


def stages_key(r):
    """The training data an arm saw: its stages plus any pretraining (pretraining adds steps and episodes)."""
    return json.dumps([[[_norm(s), n] for s, n in r["stages"]], r.get("pretrain", 0), r.get("pretrain_mode")],
                      sort_keys=True)


def _load(files):
    rs, seen = [], set()
    for f in files:
        for r in json.load(open(f))["results"]:
            k = key(r) + (r["seed"], stages_key(r))
            if k not in seen:
                seen.add(k)
                rs.append(r)
    groups = defaultdict(list)
    for r in rs:
        groups[key(r) + (stages_key(r),)].append(r)
    return groups


def table(files, ref_files=None):
    """Rows from `files`; references (uncompressed readers) from `ref_files` (default: every parity_learn JSON)."""
    groups = _load(files)
    refs = _load(ref_files if ref_files is not None else sorted(RES.glob("parity_learn*.json")))
    unc = [g for g in refs.values() if g[0]["channel"] == "none"]
    by_stages = {}
    for g in unc:
        m, s = st.mean(x["regret"]["mean"] for x in g), stages_key(g[0])
        by_stages[s] = m if s not in by_stages else min(by_stages[s], m)
    # step-matched same-decoder anchor (gate on batch 9): the uncompressed transformer on the same spec whose total
    # training steps (stages + pretraining) equal the arm's; used as the reference when it exists
    tr_steps = {}
    for g in unc:
        if g[0]["decoder"] != "transformer":
            continue
        n = sum(n_ for _, n_ in g[0]["stages"]) + g[0].get("pretrain", 0)
        key2 = (spec_key(g[0]), n)
        m = st.mean(x["regret"]["mean"] for x in g)
        if key2 not in tr_steps or m < tr_steps[key2][0]:
            tr_steps[key2] = (m, g[0]["name"])
    best_spec = defaultdict(lambda: None)
    for g in unc:
        m = st.mean(x["regret"]["mean"] for x in g)
        s = spec_key(g[0])
        best_spec[s] = m if best_spec[s] is None else min(best_spec[s], m)
    lines = ["| experiment | b | channel / decoder | keys (warm-up) | supervision / curriculum | dev KL, mean (per seed) "
             "| escapes | reference | gap | best uncompressed reader | symbol entropy (bits) | after sign fine-tune |"
             " prior KL |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2], kv[0][4]))
    for k, g in order:
        g.sort(key=lambda r: r["seed"])
        kl = [r["regret"]["mean"] for r in g]
        prior = g[0]["prior_regret"]
        esc = sum(x < 0.9 * prior for x in kl)
        sk, stk = spec_key(g[0]), stages_key(g[0])
        n_tot = sum(n_ for _, n_ in g[0]["stages"]) + g[0].get("pretrain", 0)
        ref = by_stages.get(stk) if g[0]["channel"] != "none" else None
        ref_name = "same stages" if ref is not None else None
        if ref is None and (sk, n_tot) in tr_steps:
            ref, nm = tr_steps[(sk, n_tot)]
            ref_name = f"transformer, {n_tot} steps (`{nm}`, {ref:.4f})"
        if ref is None:
            ref = best_spec[sk]
            ref_name = "best uncompressed reader on spec" if ref is not None else "n/a"
        gap = f"{st.mean(kl) - ref:+.3f}" if ref is not None and g[0]["channel"] != "none" else "n/a"
        ent = [r["symbol_entropy_bits"] for r in g if "symbol_entropy_bits" in r]
        ft = [r["regret_after_sign_finetune"]["mean"] for r in g if "regret_after_sign_finetune" in r]
        tag = g[0].get("curriculum_tag")
        if not tag and len({st_[0]["k"] for st_ in g[0]["stages"]}) > 1:
            tag = "structure-informed (k curriculum)"        # older runs did not record the tag
        if not tag and len({st_[0].get("bit_prob", 0.5) for st_ in g[0]["stages"]}) > 1:
            tag = "biased-bit curriculum"
        sup = g[0]["supervision"] + (f"; {tag}" if tag else "")
        if g[0]["dec_keys"]:
            sup += "; STRUCTURE-KNOWN (decoder reads every agent's key outside b)"
        if g[0].get("pretrain"):
            q = g[0].get("pretrain_query") or "held-out agent's own key (training only)"
            sup += f"; pretraining {g[0]['pretrain']} steps ({q})"
        if g[0].get("queried"):
            sup += "; QUERIED encoder (reads earlier agents' b-bit messages, unbounded, read-only)"
        keys = ("yes" + (f" ({g[0]['key_warmup']})" if g[0].get("key_warmup") else "")) if g[0]["dec_keys"] else "no"
        b = "inf" if g[0]["b"] is None else g[0]["b"]
        lines.append(
            f"| {k[0]} | {b} | {'fsq' if g[0].get('fsq') else g[0]['channel']}{' + power' if g[0].get('power_norm') else ''}"
            f" / {g[0]['decoder']}{' (in_norm)' if g[0].get('in_norm') else ''} | "
            f"{keys} | {sup} | {st.mean(kl):.3f} ({', '.join(f'{x:.3f}' for x in kl)}) | {esc}/{len(kl)} | "
            f"{ref_name if g[0]['channel'] != 'none' else 'n/a'} | {gap} | "
            f"{'n/a' if best_spec[sk] is None else format(best_spec[sk], '.4f')} | {', '.join(f'{abs(x):.2f}' for x in ent) if ent else 'n/a'} | "
            f"{', '.join(f'{x:.3f}' for x in ft) if ft else 'n/a'} | {prior:.3f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", nargs="+", default=None, help="OUT.json RUN_DIR ...")
    ap.add_argument("--commit", default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge(a.merge[0], a.merge[1:], a.commit)
        return
    blocks = []
    for f in sorted(RES.glob("parity_learn*.json")):
        j = json.load(open(f))
        h = j.get("harness") or {}
        blocks.append(f"`{f.name}` (harness `{str(h.get('git_commit'))[:10]}`, {len(j['results'])} runs):\n\n"
                      + table([f]))
    text = "\n\n".join(blocks)
    if not a.write:
        print(text)
        return
    readme = ROOT / "aggbench" / "README.md"
    s = readme.read_text()
    i, j = s.index(BEGIN), s.index(END)
    readme.write_text(s[:i + len(BEGIN)] + "\n" + text + "\n" + s[j:])


if __name__ == "__main__":
    main()
