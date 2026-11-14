#!/usr/bin/env python3
"""Download the explicitly pinned public Qwen archive into a new local directory.

No authentication, paid inference, model loading or model execution. Downloads
are streamed and checked against upstream SHA-256/size pins before publication.
Partial files are retained on failure; this bounded collector has no crash resume.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/research/qwen-checkpoint-artifacts-20260913.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True, help="New directory; existing paths are refused")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest["model_id"] != "Qwen/Qwen3.5-9B"
            or manifest["revision"] != "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
            or len(manifest["files"]) != 6):
        raise ValueError("unexpected candidate manifest")
    args.directory.mkdir(parents=True, exist_ok=False)
    total = sum(item["bytes"] for item in manifest["files"].values())
    if shutil.disk_usage(args.directory).free < total + 10 * 2**30:
        raise ValueError("insufficient free disk for archive plus 10GiB reserve")
    (args.directory / "download-manifest.json").write_bytes(manifest_bytes)
    started = time.monotonic()

    def download(item):
        name, expected = item
        prefix = "https://huggingface.co/Qwen/Qwen3.5-9B/resolve/" + manifest["revision"] + "/"
        if Path(name).name != name or not expected["url"].startswith(prefix + name):
            raise ValueError("file path/URL outside pinned public archive")
        partial = args.directory / (name + ".part")
        count, last_report = 0, 0
        digest = hashlib.sha256()
        request = urllib.request.Request(expected["url"], headers={"User-Agent": "consistency-ft-candidate-download"})
        with partial.open("xb") as stream:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 200:
                    raise ValueError("expected a complete public archive response")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != expected["bytes"]:
                    raise ValueError("upstream archive length differs from pinned value")
                while True:
                    block = response.read(min(4 * 2**20, expected["bytes"] - count + 1))
                    if not block:
                        break
                    count += len(block)
                    if count > expected["bytes"]:
                        raise ValueError("upstream archive exceeds pinned byte count")
                    digest.update(block)
                    stream.write(block)
                    if count - last_report >= 2**30:
                        print(json.dumps({"file": name, "received_bytes": count, "expected_bytes": expected["bytes"]}), flush=True)
                        last_report = count
            stream.flush()
            os.fsync(stream.fileno())
        if count != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
            raise ValueError("archive byte count/SHA-256 does not match upstream pin: " + name)
        os.chmod(partial, 0o444)
        os.link(partial, args.directory / name)  # Refuse an existing destination.
        partial.unlink()
        result = {"file": name, "bytes": count, "sha256": digest.hexdigest(), "verified": True}
        print(json.dumps(result), flush=True)
        return result

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            files = list(pool.map(download, sorted(manifest["files"].items())))
    except Exception as error:
        (args.directory / "download-failure.json").write_text(json.dumps({"error": str(error),
            "completed": False, "partial_files_retained": True}, indent=2) + "\n")
        raise
    report = {"completed": True, "model_id": manifest["model_id"], "revision": manifest["revision"],
              "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "files": files,
              "total_bytes": total, "wall_seconds": time.monotonic() - started,
              "finished_at": datetime.now(timezone.utc).isoformat(), "model_loaded": False,
              "provider_inference_requests": 0, "new_paid_resources_allocated": False}
    (args.directory / "download-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": True, "total_bytes": total, "wall_seconds": report["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
