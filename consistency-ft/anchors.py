"""Offline frozen base beliefs and the proposal's A2 Bernoulli-KL reward.

No inference or training is performed here. Snapshot hashes must be pinned by
the caller; a hash detects changes, not whether a source model is trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Mapping

from common import CHECK_FIELDS
from metrics import exploit


def canonical_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def proposition_id(text: str) -> str:
    return sha256(text.encode("utf-8"))


def probability(value, name="probability") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number in [0, 1]") from error
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return value


def _epsilon(value):
    if value is None:
        return None
    value = probability(value, "smoothing epsilon")
    if not 0 < value < 0.5 or 1 - value == 1:
        raise ValueError("epsilon must be in (0, .5) with 1-epsilon representably below 1")
    return value


def _smooth(p, epsilon):
    # An explicit mixture with a fair coin; never clipping an observation.
    if p == 1:
        return 1 - epsilon
    return math.fsum((epsilon, (1 - 2 * epsilon) * p))


def _log1p_remainder(t):
    """(1+t)log(1+t)-t near zero, without subtracting leading terms."""
    return math.fsum((-1) ** n * t ** n / (n * (n - 1)) for n in range(2, 12))


def bernoulli_kl(candidate, base, *, smoothing_epsilon=None) -> float:
    """KL(Bern(candidate) || Bern(base)), in nats, with exact endpoints.

    KL(p||0)=infinity for p>0; KL(p||1)=infinity for p<1. Equal endpoints
    have KL zero. Optional smoothing transforms BOTH distributions by
    T(p)=epsilon+(1-2*epsilon)*p and is a different, explicitly named treatment.
    """
    p, q = probability(candidate, "candidate"), probability(base, "base")
    epsilon = _epsilon(smoothing_epsilon)
    if epsilon is not None:
        p, q = _smooth(p, epsilon), _smooth(q, epsilon)
    if p == q:
        return 0.0
    if q in (0.0, 1.0):
        return math.inf
    if p == 0:
        return -math.log1p(-q)
    if p == 1:
        return -math.log(q)
    delta = p - q
    if abs(delta) <= 0.01 * min(q, 1 - q):
        # Algebraically identical Bregman form: nonnegative O(delta^2)
        # terms avoid cancellation for adjacent floating-point probabilities.
        return math.fsum((q * _log1p_remainder(delta / q),
                          (1 - q) * _log1p_remainder(-delta / (1 - q))))
    # Difference of logarithms avoids overflow in p/q near subnormal q.
    return math.fsum((p * (math.log(p) - math.log(q)),
                      (1 - p) * (math.log1p(-p) - math.log1p(-q))))


def mean_bernoulli_kl(candidate: Mapping, base: Mapping, *, smoothing_epsilon=None) -> float:
    """Uniform mean over the same nonempty set of roles; never skip missing roles."""
    if not candidate or set(candidate) != set(base):
        raise ValueError("candidate and base must have exactly the same nonempty roles")
    terms = [bernoulli_kl(candidate[k], base[k], smoothing_epsilon=smoothing_epsilon)
             for k in sorted(candidate)]
    return math.fsum(terms) / len(terms)


def _timestamp(value, field):
    if not isinstance(value, str):
        raise ValueError(f"missing {field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"invalid {field}") from error
    if parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def _jsonl(data):
    rows = [_json(line) for line in data.splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL records must be objects")
    return rows


def episode_index(path, split):
    """Bind exact split bytes and full prompt text; allow reuse across episodes."""
    if split not in {"train", "valid", "heldout"}:
        raise ValueError("split must be train, valid, or heldout")
    path = Path(path)
    raw = path.read_bytes()
    episodes, texts = {}, {}
    for row in _jsonl(raw):
        ident = row.get("id")
        if not isinstance(ident, str) or not ident or ident in episodes:
            raise ValueError("missing or duplicate episode id")
        fields = CHECK_FIELDS.get(row.get("check_type"))
        if fields is None or row.get("constraint") != row["check_type"]:
            raise ValueError(f"invalid check/constraint for {ident}")
        prompts = row.get("prompts", [])
        if (not isinstance(prompts, list) or any(not isinstance(p, dict) for p in prompts)
                or [p.get("role") for p in prompts] != list(fields)):
            raise ValueError(f"missing, duplicate, or misordered roles for {ident}")
        for prompt in prompts:
            text = prompt.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"missing full proposition text for {ident}")
            texts[proposition_id(text)] = text
        episodes[ident] = row
    if not episodes:
        raise ValueError("empty episode split")
    binding = {"split": split, "file": path.name, "sha256": sha256(raw),
               "episode_sha256": {k: sha256(canonical_bytes(v)) for k, v in sorted(episodes.items())},
               "proposition_ids": sorted(texts)}
    return binding, episodes, texts


def _bound_file(directory, filename, expected_hash):
    if not isinstance(filename, str) or not isinstance(expected_hash, str):
        raise ValueError("source file/hash missing")
    path = (directory / filename).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("source file escapes the manifest directory")
    raw = path.read_bytes()
    if sha256(raw) != expected_hash:
        raise ValueError(f"source hash mismatch: {filename}")
    return raw


def freeze_from_manifest(manifest_path, run_id, episodes_path, split, *,
                         allow_partial=False, allowed_ids=None, smoothing_epsilon=None,
                         created_at=None):
    """Build a snapshot from exact published receipts, never inferred legacy metadata.

    Missing/refused observations become coverage exclusions only when partial
    freezing is explicit. Malformed probabilities or provenance mismatches fail.
    ``allowed_ids`` restricts eligible source observations without changing the
    bound dataset or its denominator; it is recorded in the snapshot.
    A requested provider model identifier is not proof of an immutable weight
    revision or identity with a future trainable checkpoint.
    """
    manifest_path = Path(manifest_path)
    raw_manifest = manifest_path.read_bytes()
    manifest = _json(raw_manifest)
    matches = [r for r in manifest["runs"] if r["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError("source run id must match exactly one published run")
    run = matches[0]
    model, reasoning, protocol_id = (run.get(k) for k in ("requested_model", "reasoning", "protocol_id"))
    if not model or reasoning not in {"none", "low", "medium", "high", "xhigh", "max", "native-default"}:
        raise ValueError("source lacks an exact model/reasoning treatment; legacy fallback labels are insufficient")
    protocol = manifest.get("native_protocols", {}).get(protocol_id)
    if not protocol or not run.get("receipts_file"):
        raise ValueError("source lacks an exact protocol and full receipt provenance")
    if not isinstance(protocol.get("verbal_prompt"), str) or not isinstance(protocol.get("system_instruction"), str):
        raise ValueError("source protocol lacks exact templates")
    directory = manifest_path.parent
    raw_receipts = _bound_file(directory, run["receipts_file"], run.get("receipts_sha256"))
    raw_props = _bound_file(directory, manifest.get("propositions_file"), manifest.get("propositions_sha256"))
    catalog = {}
    for row in _jsonl(raw_props):
        ident, text = row.get("id"), row.get("text")
        if not isinstance(text, str) or ident != proposition_id(text) or ident in catalog:
            raise ValueError("duplicate or mismatched source proposition id/text")
        catalog[ident] = text
    receipts = {}
    for receipt in _jsonl(raw_receipts):
        if receipt.get("is_current") is not True:
            continue
        ident = receipt.get("id")
        if ident not in catalog or ident in receipts:
            raise ValueError("duplicate or unknown current receipt id")
        receipts[ident] = receipt
    binding, episodes, texts = episode_index(episodes_path, split)
    allowed = None if allowed_ids is None else list(allowed_ids)
    if allowed is not None and (len(allowed) != len(set(allowed)) or set(allowed) - set(catalog)):
        raise ValueError("allowed IDs must be unique published proposition IDs")
    allowed_set = None if allowed is None else set(allowed)
    epsilon = _epsilon(smoothing_epsilon)
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    frozen_time = _timestamp(created_at, "created_at")
    anchors, missing = {}, {}
    for ident, text in sorted(texts.items()):
        if allowed_set is not None and ident not in allowed_set:
            missing[ident] = "outside-allowed-source-ids"
            continue
        receipt = receipts.get(ident)
        if receipt is None:
            missing[ident] = "no-source-receipt"
            continue
        if catalog[ident] != text:
            raise ValueError(f"source text mismatch: {ident}")
        expected = {"requested_model": model, "reasoning": reasoning, "protocol_id": protocol_id,
                    "campaign": run.get("campaign"),
                    "prompt_sha256": proposition_id(protocol["verbal_prompt"].format(proposition=text)),
                    "system_instruction_sha256": proposition_id(protocol["system_instruction"])}
        if any(receipt.get(k) != value for k, value in expected.items()):
            raise ValueError(f"receipt treatment/prompt mismatch: {ident}")
        status = receipt.get("readout_status")
        if status in {"refusal", "abstention"}:
            if receipt.get("probability") is not None:
                raise ValueError(f"refusal/abstention carries a probability: {ident}")
            missing[ident] = status
            continue
        if (receipt.get("completed") is not True or receipt.get("exit_code") != 0
                or receipt.get("tools_used") is not False or receipt.get("error_kind") is not None):
            missing[ident] = "unsuccessful-source-receipt"
            continue
        if status != "probability":
            raise ValueError(f"unsupported or malformed source readout: {ident}")
        value = probability(receipt.get("probability"))
        response = receipt.get("response")
        if (not isinstance(response, str) or not re.fullmatch(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", response.strip())
                or float(response.strip()) != value):
            raise ValueError(f"probability disagrees with complete source response: {ident}")
        observed = _timestamp(receipt.get("finished_at"), "receipt finished_at")
        if observed > frozen_time:
            raise ValueError("snapshot creation precedes a source observation")
        anchors[ident] = {"text": text, "probability": value, "observed_at": receipt["finished_at"],
                          "published_receipt_sha256": sha256(canonical_bytes(receipt)),
                          "prompt_sha256": receipt["prompt_sha256"]}
    complete_episodes = [ident for ident, row in episodes.items()
                         if all(proposition_id(p["text"]) in anchors for p in row["prompts"])]
    coverage = {"target_propositions": len(texts), "anchored_propositions": len(anchors),
                "missing_propositions": missing, "target_episodes": len(episodes),
                "complete_episode_ids": sorted(complete_episodes),
                "missing_episode_ids": sorted(set(episodes) - set(complete_episodes))}
    if missing and not allow_partial:
        raise ValueError(f"partial anchor: {len(anchors)}/{len(texts)} propositions; explicit allow_partial required")
    payload = {"schema_version": 1, "created_at": created_at,
               "status": "partial" if missing else "complete-for-bound-split",
               "base": {"model": model, "reasoning": reasoning, "protocol_id": protocol_id,
                        "protocol_sha256": sha256(canonical_bytes(protocol))},
               "protocol": protocol, "dataset": binding,
               "treatment": {"mode": "exact" if epsilon is None else "symmetric-mixture",
                             "epsilon": epsilon, "unit": "nats", "aggregation": "mean-over-episode-roles"},
               "source": {"manifest_file": manifest_path.name, "manifest_sha256": sha256(raw_manifest),
                          "run_id": run_id, "run_metadata_sha256": sha256(canonical_bytes(run)),
                          "recorded_at": run.get("recorded_at"), "timestamp_source": run.get("timestamp_source"),
                          "receipts_file": run["receipts_file"], "receipts_sha256": sha256(raw_receipts),
                          "propositions_file": manifest["propositions_file"], "propositions_sha256": sha256(raw_props),
                          "allowed_proposition_ids": None if allowed is None else sorted(allowed)},
               "anchors": anchors, "coverage": coverage}
    return {"snapshot_sha256": sha256(canonical_bytes(payload)), "payload": payload}


def write_snapshot(path, snapshot):
    """Atomically create a read-only artifact, refusing existing files/symlinks."""
    path = Path(path)
    raw = canonical_bytes(snapshot) + b"\n"
    if snapshot.get("snapshot_sha256") != sha256(canonical_bytes(snapshot["payload"])):
        raise ValueError("snapshot content hash mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".anchor-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
        os.link(temporary, path)  # Atomic no-replace, including dangling symlinks.
    finally:
        os.unlink(temporary)


@dataclass(frozen=True)
class FrozenBeliefs:
    snapshot_sha256: str
    base: Mapping
    dataset: Mapping
    probabilities: Mapping
    episode_sha256: Mapping
    smoothing_epsilon: float | None
    status: str


def load_snapshot(path, *, expected_sha256, expected_base, episodes_path, split, allow_partial=False):
    """Verify a pinned snapshot and expected treatment against current split bytes."""
    envelope = _json(Path(path).read_bytes())
    payload = envelope["payload"]
    actual = sha256(canonical_bytes(payload))
    if envelope.get("snapshot_sha256") != actual or expected_sha256 != actual:
        raise ValueError("snapshot content/pinned hash mismatch")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported snapshot schema")
    if set(expected_base) != {"model", "reasoning", "protocol_id", "protocol_sha256"} or payload["base"] != expected_base:
        raise ValueError("expected model/reasoning/protocol binding mismatch")
    if sha256(canonical_bytes(payload["protocol"])) != payload["base"]["protocol_sha256"]:
        raise ValueError("protocol content hash mismatch")
    binding, episodes, texts = episode_index(episodes_path, split)
    if payload["dataset"] != binding:
        raise ValueError("dataset split/text/content hash mismatch")
    frozen_time = _timestamp(payload["created_at"], "created_at")
    treatment = payload["treatment"]
    epsilon = _epsilon(treatment.get("epsilon"))
    if treatment != {"mode": "exact" if epsilon is None else "symmetric-mixture", "epsilon": epsilon,
                     "unit": "nats", "aggregation": "mean-over-episode-roles"}:
        raise ValueError("unsupported KL treatment/normalization")
    probabilities = {}
    for ident, row in payload["anchors"].items():
        if ident not in texts or row["text"] != texts[ident] or proposition_id(row["text"]) != ident:
            raise ValueError("anchor id/text mismatch")
        if _timestamp(row.get("observed_at"), "observed_at") > frozen_time:
            raise ValueError("snapshot creation precedes observation")
        probabilities[ident] = probability(row["probability"])
    missing = set(texts) - set(probabilities)
    coverage = payload["coverage"]
    complete = sorted(ident for ident, row in episodes.items()
                      if all(proposition_id(p["text"]) in probabilities for p in row["prompts"]))
    if (set(coverage["missing_propositions"]) != missing or coverage["target_propositions"] != len(texts)
            or coverage["anchored_propositions"] != len(probabilities)
            or coverage["target_episodes"] != len(episodes) or coverage["complete_episode_ids"] != complete
            or coverage["missing_episode_ids"] != sorted(set(episodes) - set(complete))):
        raise ValueError("snapshot coverage does not match its anchors")
    status = "partial" if missing else "complete-for-bound-split"
    if payload["status"] != status or (missing and not allow_partial):
        raise ValueError("partial snapshot requires explicit allow_partial")
    return FrozenBeliefs(actual, MappingProxyType(dict(payload["base"])),
                         MappingProxyType({k: binding[k] for k in ("split", "file", "sha256")}),
                         MappingProxyType(probabilities), MappingProxyType(binding["episode_sha256"]),
                         epsilon, status)


def a2_reward(episode, candidate: Mapping, frozen: FrozenBeliefs, *, lambda_kl):
    """R=-Exploit(c,S)-lambda*mean_i KL(c_i||base_i), with strict coverage.

    Each episode role contributes once. No batch aggregation, silent skipping,
    malformed-output penalty, optimizer, or trainer policy is implied.
    """
    if isinstance(lambda_kl, bool) or not isinstance(lambda_kl, (int, float)):
        raise ValueError("lambda_kl must be finite and nonnegative")
    try:
        lambda_kl = float(lambda_kl)
    except OverflowError as error:
        raise ValueError("lambda_kl must be finite and nonnegative") from error
    if not math.isfinite(lambda_kl) or lambda_kl < 0:
        raise ValueError("lambda_kl must be finite and nonnegative")
    ident = episode.get("id")
    if frozen.episode_sha256.get(ident) != sha256(canonical_bytes(episode)):
        raise ValueError("episode absent from or changed since the frozen split")
    fields = CHECK_FIELDS[episode["check_type"]]
    if set(candidate) != set(fields):
        raise ValueError("candidate roles do not exactly match the episode")
    candidate = {k: probability(candidate[k], k) for k in fields}
    base = {}
    for prompt in episode["prompts"]:
        pid = proposition_id(prompt["text"])
        if pid not in frozen.probabilities:
            raise ValueError(f"missing frozen base belief: {pid}")
        base[prompt["role"]] = frozen.probabilities[pid]
    mean_kl = mean_bernoulli_kl(candidate, base, smoothing_epsilon=frozen.smoothing_epsilon)
    exploitability = exploit(episode["check_type"], candidate)
    # 0 * infinity is mathematically irrelevant when lambda=0; preserve the
    # infinite KL in diagnostics without manufacturing a NaN reward.
    penalty = 0.0 if lambda_kl == 0 else lambda_kl * mean_kl
    if math.isfinite(mean_kl) and not math.isfinite(penalty):
        raise OverflowError("finite KL penalty exceeds floating-point range")
    return {"reward": -exploitability - penalty, "exploitability": exploitability,
            "mean_kl": mean_kl, "lambda_kl": float(lambda_kl), "role_count": len(fields),
            "snapshot_sha256": frozen.snapshot_sha256, "smoothing_epsilon": frozen.smoothing_epsilon}
