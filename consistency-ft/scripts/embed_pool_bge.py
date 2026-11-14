#!/usr/bin/env python3
"""BGE-small embeddings for the JEV pairing pool, recomputed on CPU.

The original embeddings (consistency-ft-followups-20260913
embed_binary_submarkets.py) live only on the Mac Studio.  This recomputes them
with the same model and revision (BAAI/bge-small-en-v1.5 @ 5c38ec7c),
the same text (``embedding_text`` from the relevance tournament's
propositions.jsonl) and normalize_embeddings=True, and checks the cosines of
the original jev-pairings-v1 pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

MODEL, REVISION = "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=Path("/home/agent/data/jev-pairings-v1"))
    ap.add_argument("--texts", type=Path, default=Path("/home/agent/data/relevance-swiss-jev-full/propositions.jsonl"))
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    text = {}
    for line in a.texts.read_text().splitlines():
        r = json.loads(line)
        text[r["id"]] = r["embedding_text"]
    ids = sorted(json.loads(line)["id"] for split in ("train", "validation", "test")
                 for line in (a.pool / split / "propositions.jsonl").read_text().splitlines() if line.strip())
    start = time.time()
    model = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
    vec = model.encode([text[i] for i in ids], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    vec = np.asarray(vec, dtype=np.float32)
    a.output.mkdir(parents=True, exist_ok=True)
    np.save(a.output / "embeddings.npy", vec)
    (a.output / "ids.json").write_text(json.dumps(ids))
    row = {pid: k for k, pid in enumerate(ids)}
    diffs = []
    for split in ("train", "validation", "test"):
        for line in (a.pool / split / "pairs.jsonl").read_text().splitlines():
            p = json.loads(line)
            diffs.append(abs(float(vec[row[p["p_id"]]] @ vec[row[p["q_id"]]]) - p["cosine"]))
    manifest = {"model": MODEL, "revision": REVISION, "device": "cpu", "normalized": True, "n": len(ids),
                "seconds": round(time.time() - start, 1),
                "embeddings_sha256": hashlib.sha256((a.output / "embeddings.npy").read_bytes()).hexdigest(),
                "check_vs_original_pair_cosines": {"pairs": len(diffs), "max_abs_diff": max(diffs),
                                                   "mean_abs_diff": sum(diffs) / len(diffs)}}
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
