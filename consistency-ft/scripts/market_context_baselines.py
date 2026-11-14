#!/usr/bin/env python3
"""Market-price baselines and prompt lengths for the market-context dataset.

Baselines on validation and test, all twelve forms and the observed form,
log loss and Brier, split into single-question forms (P, NOT P, Q, NOT Q) and
AND/OR forms:

* market_indep: price at t for P and Q (clipped to [0.01, 0.99]); NOT = 1 - p;
  AND/OR of the signed literals under independence (a*b, a+b-a*b).
* market_frechet_clip: the independence value clipped into the Fréchet bounds
  (AND in [max(0,a+b-1), min(a,b)], OR in [max(a,b), min(1,a+b)]).  With
  independent inputs this never binds; it is reported as a check.
* market_frechet_mid: the midpoint of the Fréchet interval (no independence).
* market_raw: independence without the [0.01, 0.99] clip (eps 1e-7).
* base_rate_all12: per-form YES rate on the train split (all twelve labels),
  the table that scored 0.570 all-12 log loss on the old test split.
* boolean_prior: 0.5 literal, 0.25 AND, 0.75 OR.

Subsets: ``both_open`` (both markets traded at t; a pure market baseline) and
``all`` (a proposition with no market at t falls back to the train YES rate of
its kind, binary or option).  Prompt lengths use the Qwen3 tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.supervised_pairs import (FORM_ORDER, FORM_PRIOR, LABEL_FILES, load_split, probability_metrics,  # noqa: E402
                                       render_prompt)
from training import verbalized_baseline as vb  # noqa: E402

SINGLE = {"P", "NOT P", "Q", "NOT Q"}
MODES = ("none", "asof_text", "prices", "comments", "prices+comments")


def signed(p: float, sign: int) -> float:
    return p if sign > 0 else 1 - p


def form_probs(a: float, b: float, clip: float | None) -> dict[str, dict[str, float]]:
    """Probabilities for every form from literal prices a = p(P), b = p(Q)."""
    if clip is not None:
        a, b = min(1 - clip, max(clip, a)), min(1 - clip, max(clip, b))
    out = {"indep": {}, "frechet_clip": {}, "frechet_mid": {}}
    for form in FORM_ORDER:
        op, sa, sb = vb._parts(form)
        if op is None:
            value = signed(a, sa) if sa else signed(b, sb)
            for k in out:
                out[k][form] = value
            continue
        x, y = signed(a, sa), signed(b, sb)
        if op == "AND":
            indep, lo, hi = x * y, max(0.0, x + y - 1), min(x, y)
        else:
            indep, lo, hi = x + y - x * y, max(x, y), min(1.0, x + y)
        out["indep"][form] = indep
        out["frechet_clip"][form] = min(hi, max(lo, indep))
        out["frechet_mid"][form] = (lo + hi) / 2
    return out


def train_tables(root: Path) -> tuple[dict[str, float], dict[str, float]]:
    full = load_split(root, "train", labels_file=LABEL_FILES["full"])
    per_form: dict[str, list[int]] = {f: [] for f in FORM_ORDER}
    kind: dict[str, list[int]] = {"binary": [], "option": []}
    for e in full:
        for f, y in e.labeled_forms.items():
            per_form[f].append(y)
        for form, pid in (("P", e.pair["p_id"]), ("Q", e.pair["q_id"])):
            kind["option" if ":" in pid else "binary"].append(e.labeled_forms[form])
    return ({f: statistics.fmean(v) for f, v in per_form.items()},
            {k: statistics.fmean(v) for k, v in kind.items()})


def literal_price(prop: dict, kind_rate: dict[str, float]) -> tuple[float, bool]:
    ctx = prop["context"]
    if ctx["open_at_t"]:
        return ctx["price"]["p_t"], True
    return kind_rate["option" if ctx["answer_id"] else "binary"], False


def score_split(root: Path, split: str, table: dict[str, float], kind_rate: dict[str, float]) -> dict:
    full = load_split(root, split, labels_file=LABEL_FILES["full"])
    rows = []  # (subset flags, form, observed, label, {method: p})
    for e in full:
        a, open_a = literal_price(e.propositions[e.pair["p_id"]], kind_rate)
        b, open_b = literal_price(e.propositions[e.pair["q_id"]], kind_rate)
        clipped = form_probs(a, b, 0.01)
        raw = form_probs(a, b, None)["indep"]
        for form, y in e.labeled_forms.items():
            preds = {"market_indep": clipped["indep"][form], "market_frechet_clip": clipped["frechet_clip"][form],
                     "market_frechet_mid": clipped["frechet_mid"][form], "market_raw": raw[form],
                     "base_rate_all12": table[form], "boolean_prior": FORM_PRIOR[form]}
            rows.append((open_a and open_b, form, form == e.observed_form, y, preds, e.pair.get("stratum")))
    out = {}
    strata = sorted({r[5] for r in rows if r[5]})
    for subset in ("both_open", "all", *(f"stratum:{s}" for s in strata)):
        if subset.startswith("stratum:"):
            chosen = [r for r in rows if r[5] == subset.split(":", 1)[1]]
        else:
            chosen = [r for r in rows if subset == "all" or r[0]]
        out[subset] = {"pairs": len(chosen) // 12}
        for method in chosen[0][4]:
            res = {}
            for scope in ("all12", "observed"):
                for group in ("all", "single", "and_or"):
                    sel = [r for r in chosen if (scope == "all12" or r[2])
                           and (group == "all" or (r[1] in SINGLE) == (group == "single"))]
                    if sel:
                        m = probability_metrics([r[4][method] for r in sel], [r[3] for r in sel])
                        res[f"{scope}/{group}"] = {"n": len(sel), "log_loss": round(m["log_loss"], 4),
                                                   "brier": round(m["brier"], 4)}
            out[subset][method] = res
    return out


def pct(values: list[int]) -> dict:
    values = sorted(values)
    q = lambda f: values[min(len(values) - 1, int(f * len(values)))]  # noqa: E731
    return {"n": len(values), "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": values[-1],
            "mean": round(statistics.fmean(values), 1)}


def prompt_lengths(root: Path, tokenizer_path: str) -> dict:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    out = {"tokenizer": tokenizer_path,
           "tokenizer_sha256": hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest(),
           "note": "supervised = render_prompt text only; v6_direct = system + user text plus chat-template "
                   "markers (about 20 tokens); counts are over every pair and all twelve forms"}
    for split in ("train", "validation", "test"):
        examples = load_split(root, split, labels_file=LABEL_FILES["full"])
        out[split] = {}
        for mode in MODES:
            sup = [render_prompt(e, f, mode) for e in examples for f in FORM_ORDER]
            chat = [f"<|im_start|>system\n{m[0]['content']}<|im_end|>\n<|im_start|>user\n{m[1]['content']}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                    for e in examples for f in FORM_ORDER for m in [vb.messages(e, f, "v6_direct", context=mode)]]
            out[split][mode] = {"supervised": pct([len(x.ids) for x in tok.encode_batch(sup)]),
                                "v6_direct_chat": pct([len(x.ids) for x in tok.encode_batch(chat)])}
        print(split, json.dumps({m: out[split][m]["supervised"] for m in MODES}), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/agent/data/jev-market-context-20260923")
    ap.add_argument("--old-data", default="/home/agent/data/jev-disjoint-v1-filtered-20260919")
    ap.add_argument("--tokenizer", default="/home/agent/data/qwen32b-baseline-20260923/tokenizer/tokenizer.json")
    a = ap.parse_args()
    root = Path(a.data)
    table, kind_rate = train_tables(root)
    old_table, _ = train_tables(Path(a.old_data))
    result = {"train_base_rate_all12": table, "train_literal_yes_rate_by_kind": kind_rate, "splits": {}}
    for split in ("validation", "test"):
        result["splits"][split] = score_split(root, split, table, kind_rate)
    # Reference: the old table on the old test (should reproduce 0.570 all-12).
    old = load_split(Path(a.old_data), "test", labels_file=LABEL_FILES["full"])
    ps = [(old_table[f], y) for e in old for f, y in e.labeled_forms.items()]
    result["old_test_old_table_all12_log_loss"] = round(probability_metrics([p for p, _ in ps], [y for _, y in ps])["log_loss"], 4)
    result["prompt_lengths"] = prompt_lengths(root, a.tokenizer)
    (root / "baselines.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["baselines_file"] = "baselines.json"
    manifest["files_sha256"]["baselines.json"] = hashlib.sha256((root / "baselines.json").read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for split, res in result["splits"].items():
        for subset, methods in res.items():
            print(split, subset, methods["pairs"], "pairs")
            for method, m in methods.items():
                if method == "pairs":
                    continue
                print(f"  {method:20s}", " ".join(f"{k}={v['log_loss']:.3f}/{v['brier']:.3f}" for k, v in m.items()))
    print("old test, old table all-12 log loss", result["old_test_old_table_all12_log_loss"])


if __name__ == "__main__":
    main()
