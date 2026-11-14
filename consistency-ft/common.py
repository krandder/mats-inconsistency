"""Shared tuple, consistency, and Fireworks plumbing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import requests


MODEL = "accounts/fireworks/models/qwen3p5-9b"
BASE_URL = "https://api.fireworks.ai/inference/v1"
VERBAL_PROMPT = """Estimate the probability that the proposition below is true.
Return only a number from 0.0 to 1.0, with no commentary.

Proposition:
{proposition}"""
LOGPROB_PROMPT = """Decide whether the proposition below is true. Answer with exactly one word.

Proposition:
{proposition}

True or False:"""

CHECK_FIELDS = {
    "Negation": ("P", "not_P"),
    "Paraphrase": ("P", "para_P"),
    "Consequence": ("P", "cons_P"),
    "And": ("P", "Q", "P_and_Q"),
    "Or": ("P", "Q", "P_or_Q"),
    "AndOr": ("P", "Q", "P_and_Q", "P_or_Q"),
    "But": ("P", "Q_and_not_P", "P_or_Q"),
    "Cond": ("P", "Q_given_P", "P_and_Q"),
    "CondCond": ("P", "Q_given_P", "R_given_P_and_Q", "P_and_Q_and_R"),
    "ExpEvidence": ("P", "Q", "P_given_Q", "P_given_not_Q"),
}


def violation(check_type: str, p: dict[str, float]) -> float:
    """Basic algebraic boundary breach; deliberately not an arbitrage metric."""
    if check_type == "Negation":
        return abs(p["P"] + p["not_P"] - 1)
    if check_type == "Paraphrase":
        return abs(p["P"] - p["para_P"])
    if check_type == "Consequence":
        return max(0.0, p["P"] - p["cons_P"])
    if check_type == "And":
        return max(max(p["P"] + p["Q"] - 1, 0) - p["P_and_Q"],
                   p["P_and_Q"] - min(p["P"], p["Q"]), 0.0)
    if check_type == "Or":
        return max(max(p["P"], p["Q"]) - p["P_or_Q"],
                   p["P_or_Q"] - min(1, p["P"] + p["Q"]), 0.0)
    if check_type == "AndOr":
        return abs(p["P"] + p["Q"] - p["P_and_Q"] - p["P_or_Q"])
    if check_type == "But":
        return abs(p["P"] + p["Q_and_not_P"] - p["P_or_Q"])
    if check_type == "Cond":
        return abs(p["P"] * p["Q_given_P"] - p["P_and_Q"])
    if check_type == "CondCond":
        return abs(p["P"] * p["Q_given_P"] * p["R_given_P_and_Q"]
                   - p["P_and_Q_and_R"])
    if check_type == "ExpEvidence":
        return abs(p["P"] - p["P_given_Q"] * p["Q"]
                   - p["P_given_not_Q"] * (1 - p["Q"]))
    raise ValueError(f"unknown check type: {check_type}")


def proposition_text(question: dict) -> str:
    title = question["title"].strip()
    body = (question.get("body") or "").strip()
    return title if not body else f"{title}\n\nResolution criteria: {body}"


def load_tuples(directory: str | Path, limit: int | None = None) -> list[dict]:
    tuples = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        count = 0
        with path.open() as f:
            for line in f:
                if limit is not None and count >= limit:
                    break
                if line.strip():
                    tuples.append(json.loads(line))
                    count += 1
    if not tuples:
        raise SystemExit(f"No tuple JSONL records found in {directory}")
    return tuples


def _api_key() -> str | None:
    if os.environ.get("FIREWORKS_API_KEY"):
        return os.environ["FIREWORKS_API_KEY"]
    path = Path.home() / ".config/fireworks/env"
    if path.exists():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.removeprefix("export ").split("=", 1)
            if key.strip() == "FIREWORKS_API_KEY":
                return value.strip().strip("'\"")
    return None


def _number(text: str) -> float:
    match = re.fullmatch(r"\s*`*\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*`*\s*", text)
    if not match:
        match = re.search(r"(?<![\d.])(0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])", text)
    if not match:
        raise ValueError(f"could not parse a probability from {text!r}")
    return float(match.group(1))


class FireworksClient:
    def __init__(self, model: str = MODEL):
        self.model = model
        self.key = _api_key()
        if not self.key:
            raise SystemExit(
                "Missing Fireworks API key. Set FIREWORKS_API_KEY or add "
                "FIREWORKS_API_KEY=... to ~/.config/fireworks/env; use --mock offline."
            )

    def _chat(self, prompt: str, **options) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            # Qwen3.5 otherwise may spend the short completion on hidden reasoning.
            "reasoning_effort": "none",
            **options,
        }
        for attempt in range(5):
            try:
                response = requests.post(
                    f"{BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {self.key}"},
                    json=payload,
                    timeout=90,
                )
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
            if response.status_code == 400 and ("reasoning effort" in response.text or "reasoning_effort" in response.text) and payload.get("reasoning_effort") == "none":
                # ponytail: models that reject Qwen3.5's "none" get "low" — uncapped reasoning can loop to the token limit
                payload["reasoning_effort"] = "low"
                continue
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise RuntimeError(f"Fireworks API error {response.status_code}: {response.text[:500]}") from exc
            return response.json()
        raise AssertionError("unreachable")

    def verbal_probability(self, proposition: str) -> float:
        # ponytail: 2048 not 16 — reasoning models burn budget before emitting content
        data = self._chat(VERBAL_PROMPT.format(proposition=proposition), max_tokens=2048)
        return _number(data["choices"][0]["message"].get("content") or "")

    def true_false_probability(self, proposition: str) -> float:
        data = self._chat(
            LOGPROB_PROMPT.format(proposition=proposition),
            max_tokens=2048,
            logprobs=True,
            top_logprobs=5,
        )
        positions = data["choices"][0]["logprobs"]["content"]
        # ponytail: scan from the end — reasoning models mention true/false mid-analysis,
        # but the answer token is the last place both appear in the top logprobs
        for position in reversed(positions):
            candidates = position.get("top_logprobs", []) + [position]
            scores = {}
            for candidate in candidates:
                token = re.sub(r"[^a-z]", "", candidate["token"].lower())
                if token in ("true", "false"):
                    scores[token] = max(scores.get(token, -math.inf), candidate["logprob"])
            if scores.keys() >= {"true", "false"}:
                offset = max(scores.values())
                true, false = math.exp(scores["true"] - offset), math.exp(scores["false"] - offset)
                return true / (true + false)
        raise ValueError("Fireworks top_logprobs did not contain both True and False tokens")


class MockClient:
    """Deterministic fake readouts for offline end-to-end runs."""
    model = "mock/sha256"

    @staticmethod
    def _probability(kind: str, proposition: str) -> float:
        n = int.from_bytes(hashlib.sha256(f"{kind}\0{proposition}".encode()).digest()[:8], "big")
        return round(0.01 + (n / (2**64 - 1)) * 0.98, 6)

    def verbal_probability(self, proposition: str) -> float:
        return self._probability("verbal", proposition)

    def true_false_probability(self, proposition: str) -> float:
        return self._probability("logprob", proposition)


def make_client(mock: bool, model: str = MODEL):
    return MockClient() if mock else FireworksClient(model)
