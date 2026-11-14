#!/usr/bin/env python3
"""Rebuild JEV one-pair-per-proposition data with both questions open at t.

Same pipeline as the September pairs (consistency-ft-followups-20260913:
build_jev_pairings.py then build_jev_disjoint.py, then the excluded-parent
filter), with one extra rule at candidate generation.

* Pool: the tournament-selected, label-time-safe propositions of
  /home/agent/data/jev-pairings-v1 (JEV 16 rounds >= 8 on either axis), with
  their frozen split and family.  Parents on the 178-parent exclusion list are
  removed before pairing, not after.  Every pool proposition was already
  screened by the JEV context audit (jev-context-audit-v1: current, earliest
  and comments), so no new JEV calls are needed.
* Candidate rule (new): for a candidate (i, j), t = 00:00 UTC of
  (min(closing_week_i, closing_week_j) - 7 days).  Both must be open at t:
  market and option created before t, neither resolved at or before t, and
  no resolution edit logged before t.
* Nominations: each proposition nominates its nearest valid partner by BGE
  cosine and one distinct uniform random valid partner (seed 20260919 + split
  index), as before; same parent and duplicate normalized titles excluded.
* One pair per proposition: networkx max-cardinality max-weight matching,
  weight = round(cosine * 1e6).
* Observed form: one of twelve per pair, random.Random(seed + 100 + split
  index) in sorted edge order; outcomes are never consulted.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import random
import re
import sqlite3
from pathlib import Path

import numpy as np

SEED = 20260919
SPLITS = ("train", "validation", "test")
DAY = 86_400_000
FORMS = [("P", "lit", 1, 0), ("NOT P", "lit", -1, 0), ("Q", "lit", 0, 1), ("NOT Q", "lit", 0, -1)]
for _x in (1, -1):
    for _y in (1, -1):
        for _op in ("AND", "OR"):
            FORMS.append((f"({'P' if _x == 1 else 'NOT P'}) {_op} ({'Q' if _y == 1 else 'NOT Q'})", _op, _x, _y))


def truth(form, p, q):
    _, op, x, y = form
    u = p if x == 1 else 1 - p
    v = q if y == 1 else 1 - q
    return int((u if x else v) if op == "lit" else (u and v) if op == "AND" else (u or v))


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(r: dict) -> str:
    return re.sub(r"\W+", " ", r["parent_question"] + " " + str(r.get("answer_text") or "")).strip().casefold()


def week_ms(week: str) -> int:
    return int(dt.datetime.fromisoformat(week).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def open_windows(index: Path, props: list[dict]) -> dict[str, tuple[int, int]]:
    """(available_from, end) per proposition: open at t iff available_from < t < end.

    end = earliest of market resolution, option resolution and the first
    contract edit that touches resolution fields (0 if unknown -> never open).
    """
    db = sqlite3.connect(index)
    contracts = {cid: (created, res) for cid, created, res in
                 db.execute("SELECT id, created_time, resolution_time FROM contracts")}
    answers = {aid: (created, res) for aid, created, res in
               db.execute("SELECT id, created_time, resolution_time FROM answers")}
    first_resolution_edit: dict[str, int] = {}
    for cid, created, keys in db.execute("SELECT contract_id, created_time, updated_keys FROM contract_edits"):
        if {"isResolved", "resolution", "resolutions"} & set(json.loads(keys or "[]")):
            first_resolution_edit[cid] = min(created, first_resolution_edit.get(cid, created))
    current_answers = {}
    for cid, data in db.execute("SELECT id, data FROM contracts"):
        for a in json.loads(data).get("answers") or []:
            current_answers[a["id"]] = (a.get("createdTime"), a.get("resolutionTime"))
    big = 1 << 62
    out = {}
    for r in props:
        cid = r["parent_id"]
        if cid not in contracts:
            out[r["id"]] = (big, 0)
            continue
        created, res = contracts[cid]
        ends = [res or big, first_resolution_edit.get(cid, big)]
        avail = created
        if ":" in r["id"]:
            aid = r["id"].split(":", 1)[1]
            a_created, a_res = answers.get(aid) or current_answers.get(aid) or (None, None)
            if a_created is None:
                out[r["id"]] = (big, 0)
                continue
            avail = max(avail, a_created)
            ends.append(a_res or big)
        out[r["id"]] = (avail, min(ends))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=Path("/home/agent/data/jev-pairings-v1"))
    ap.add_argument("--embeddings", type=Path, required=True, help="directory with ids.json and embeddings.npy")
    ap.add_argument("--index", type=Path, default=Path("/home/agent/data/manifold-context-20260923/index-pool.sqlite"))
    ap.add_argument("--excluded", type=Path,
                    default=Path("/home/agent/data/jev-disjoint-v1-filtered-20260919/excluded-parent-ids.json"))
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    import networkx as nx
    assert not a.output.exists(), "refusing to overwrite a frozen dataset"
    a.output.mkdir(parents=True)
    excluded = set(json.loads(a.excluded.read_text()))
    ids = json.loads((a.embeddings / "ids.json").read_text())
    emb = np.load(a.embeddings / "embeddings.npy").astype(np.float32)
    row_of = {pid: k for k, pid in enumerate(ids)}
    summary, family_sets = {}, {}
    for si, split in enumerate(SPLITS):
        src = a.pool / split
        labels = {r["id"]: r["label"] for r in read(src / "single-targets.jsonl")}
        pool_all = sorted(read(src / "propositions.jsonl"), key=lambda r: r["id"])
        rs = [r for r in pool_all if r["parent_id"] not in excluded]
        family_sets[split] = {r["family_id"] for r in rs}
        windows = open_windows(a.index, rs)
        n = len(rs)
        vec = emb[[row_of[r["id"]] for r in rs]]
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        parents = np.array([r["parent_id"] for r in rs])
        names = np.array([norm(r) for r in rs])
        weeks = np.array([week_ms(r["closing_week"]) for r in rs], dtype=np.int64)
        avail = np.array([windows[r["id"]][0] for r in rs], dtype=np.int64)
        end = np.array([windows[r["id"]][1] for r in rs], dtype=np.int64)
        rng = random.Random(SEED + si)
        edges: dict[tuple[int, int], list[dict]] = {}
        no_partner = 0
        for i in range(n):
            t = np.minimum(weeks[i], weeks) - 7 * DAY
            valid = (parents != parents[i]) & (names != names[i]) & (avail < t) & (end > t) & (avail[i] < t) & (end[i] > t)
            valid[i] = False
            if not valid.any():
                no_partner += 1
                continue
            sims = vec @ vec[i]
            sims[~valid] = -np.inf
            j = int(np.argmax(sims))
            eligible = np.flatnonzero(valid)
            eligible = eligible[eligible != j]
            nominated = [(j, "near")]
            if len(eligible):
                nominated.append((int(eligible[rng.randrange(len(eligible))]), "random"))
            for partner, kind in nominated:
                edges.setdefault(tuple(sorted((i, partner))), []).append({"anchor_id": rs[i]["id"], "kind": kind})
        graph = nx.Graph()
        for (i, j) in edges:
            graph.add_edge(i, j, weight=round(float(vec[i] @ vec[j]) * 1_000_000))
        matched = {tuple(sorted(e)) for e in nx.max_weight_matching(graph, maxcardinality=True, weight="weight")}
        mask_rng = random.Random(SEED + 100 + si)
        pairs, queries, one, full, audit = [], [], [], [], []
        for z, key in enumerate(sorted(matched)):
            i, j = key
            p, q = rs[i], rs[j]
            pair_id = f"{split}-open-{z:05d}"
            chosen = mask_rng.randrange(12)
            qids = [f"{pair_id}-q{k:02d}" for k in range(12)]
            t = min(weeks[i], weeks[j]) - 7 * DAY
            assert avail[i] < t < end[i] and avail[j] < t < end[j]
            pairs.append({"pair_id": pair_id, "p_id": p["id"], "q_id": q["id"], "p_parent": p["parent_id"],
                          "q_parent": q["parent_id"], "p_family": p["family_id"], "q_family": q["family_id"],
                          "cosine": float(vec[i] @ vec[j]), "nominations": edges[key], "p_degree": 1, "q_degree": 1,
                          "pair_weight": 1.0, "observed_query_id": qids[chosen], "observed_form": FORMS[chosen][0],
                          "hidden_query_ids": [v for k, v in enumerate(qids) if k != chosen]})
            yp, yq = int(labels[p["id"]]), int(labels[q["id"]])
            observed_label = truth(FORMS[chosen], yp, yq)
            worlds = [(u, v) for u in (0, 1) for v in (0, 1) if truth(FORMS[chosen], u, v) == observed_label]
            for k, f in enumerate(FORMS):
                queries.append({"query_id": qids[k], "pair_id": pair_id, "p_id": p["id"], "q_id": q["id"], "form": f[0],
                                "operator": f[1], "p_sign": f[2], "q_sign": f[3], "accuracy_observed": k == chosen})
                full.append({"query_id": qids[k], "label": truth(f, yp, yq)})
                if k == chosen:
                    one.append({"query_id": qids[k], "label": observed_label})
                else:
                    values = {truth(f, u, v) for u, v in worlds}
                    audit.append({"query_id": qids[k], "pair_id": pair_id, "entailed_by_observed_label": len(values) == 1,
                                  "remaining_atomic_worlds": len(worlds),
                                  "evaluation_stratum": "logically_determined" if len(values) == 1 else "underdetermined"})
        atoms = [x for pr in pairs for x in (pr["p_id"], pr["q_id"])]
        assert len(atoms) == len(set(atoms))
        keep = set(atoms)
        out = a.output / split
        out.mkdir()
        write(out / "pairs.jsonl", pairs)
        write(out / "queries.jsonl", queries)
        write(out / "propositions.jsonl", [r for r in rs if r["id"] in keep])
        write(out / "one-label-targets.jsonl", one)
        write(out / "full-label-targets.jsonl", full)
        write(out / "hidden-label-audit.jsonl", audit)
        write(out / "single-targets.jsonl", [{"id": x, "label": labels[x]} for x in sorted(keep)])
        summary[split] = {"pool": len(pool_all), "pool_after_excluded_parents": n,
                          "propositions_without_open_partner": no_partner, "candidate_edges": len(edges),
                          "pairs": len(pairs), "propositions": len(atoms),
                          "entailed_hidden": sum(r["entailed_by_observed_label"] for r in audit),
                          "underdetermined_hidden": sum(not r["entailed_by_observed_label"] for r in audit),
                          "median_cosine": float(np.median([pr["cosine"] for pr in pairs])) if pairs else None,
                          "observed_form_counts": dict(collections.Counter(pr["observed_form"] for pr in pairs))}
        print(split, json.dumps(summary[split]), flush=True)
    assert not family_sets["train"] & family_sets["validation"]
    assert not family_sets["train"] & family_sets["test"]
    assert not family_sets["validation"] & family_sets["test"]
    (a.output / "excluded-parent-ids.json").write_text(a.excluded.read_text())
    manifest = {
        "seed": SEED, "pool": str(a.pool), "pool_manifest_sha256": sha(a.pool / "manifest.json"),
        "embeddings": str(a.embeddings), "embeddings_sha256": sha(a.embeddings / "embeddings.npy"),
        "index": str(a.index), "excluded_parent_ids_sha256": sha(a.excluded), "excluded_parent_count": len(excluded),
        "open_rule": "candidate (i,j) valid only if both are open at t = min(closing_week) - 7 days, 00:00 UTC: "
                     "market and option created before t, not resolved at or before t, no resolution edit before t",
        "pair_policy": "nearest valid + one random valid nomination per proposition; same parent and duplicate "
                       "normalized titles excluded; max-cardinality max-weight matching (weight=round(cos*1e6))",
        "jev_audit": "all pool propositions were screened by jev-context-audit-v1 (current, earliest, comments); "
                     "the 178 excluded parents come from it; no new JEV calls",
        "families_disjoint_across_splits": True, "networkx_version": nx.__version__, "splits": summary,
        "files_sha256": {str(p.relative_to(a.output)): sha(p) for p in sorted(a.output.rglob("*")) if p.is_file()},
    }
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
