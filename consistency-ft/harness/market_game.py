"""Market game: a frozen frontier model trades a combinatorial LMSR market.

The market is the bayes-market factored (junction-tree) LMSR over the 1,265
binary questions of Turtel et al. 2025's released test set. One session is
one model invocation that sees N questions with their current prices and
frozen evidence and answers with a block of trades. P passes cover every
question; the market is scored against the known resolutions after each
pass. Nothing is trained.

Library layout (one module, in reading order):
  dataset      mirror + load the test set, Table 1 verification
  engine       thin thread-safe wrapper over FactoredMarket (prices, links,
               KL accounting, budget, snapshots)
  grouping     TF-IDF similarity, sibling sets, per-pass session groups
  prompt       the frozen developer rule and the session prompt builder
  parser       tolerant extraction + validation of the trade block
  providers    MockProvider (deterministic, offline) and CodexProvider
  arena        the run loop (thread pool, logging, snapshots, resume)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = Path(os.environ.get("BAYES_ENGINE_DIR", "/home/agent/workspaces/bayes-engine"))
DATA_ROOT = Path(os.environ.get("FORECASTING_BENCHMARKS_DIR", "/home/agent/data/forecasting-benchmarks"))
HF_REPO = "LightningRodLabs/outcome-rl-test-dataset"
DATASET_DIR = DATA_ROOT / HF_REPO.replace("/", "--")
PARQUET_PATH = DATASET_DIR / "data" / "train-00000-of-00001.parquet"
QUESTION_CACHE = ROOT / "runs" / "market-game" / "questions-cache.jsonl"
USER_AGENT = "forecasting-benchmarks-mirror/1.0"

MODELS = {"astra": "gpt-6-astra", "luna": "gpt-5.6-luna", "sol": "gpt-6-sol"}
CODEX_EXECUTABLE = "/home/agent/.local/bin/codex-any"
CAMPAIGN = "market-game-20260922"

LIQUIDITY = 100.0
MAX_WIDTH = 8
TARGET_MIN, TARGET_MAX = 0.01, 0.99
MAX_TRADES_PER_QUESTION = 2
BUDGET_FACTOR = 0.3  # eps_total = BUDGET_FACTOR * N on passes after the first
TABLE1 = {"prediction_polymarket": 0.151, "prediction_o1": 0.202, "prediction_remax": 0.191}
TABLE1_TOLERANCE = 0.002


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def binary_kl(q: float, p: float) -> float:
    """KL(Bernoulli(q) || Bernoulli(p)) in nats; inf for degenerate p."""
    if not 0.0 < p < 1.0:
        return math.inf
    total = 0.0
    if q > 0.0:
        total += q * math.log(q / p)
    if q < 1.0:
        total += (1.0 - q) * math.log((1.0 - q) / (1.0 - p))
    return max(total, 0.0)


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    close_date: str
    resolution: int
    prompt: str
    today: str
    evidence: str
    prediction_polymarket: float | None = None
    prediction_o1: float | None = None
    prediction_remax: float | None = None
    volume: float | None = None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Question":
        fields = {k: record.get(k) for k in cls.__dataclass_fields__}
        fields["resolution"] = int(fields["resolution"])
        return cls(**fields)


def _http_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def ensure_dataset(root: Path = DATA_ROOT, repo: str = HF_REPO) -> Path:
    """Mirror the Hugging Face dataset into the archive; never overwrite.

    Follows the archive's conventions (download-additions-20260920.py):
    <owner>--<repo>/ folder, source-metadata.json with URL, fetch time, the
    revision and per-file sha256, .part files, existing files left alone.
    Returns the parquet path.
    """
    out = root / repo.replace("/", "--")
    parquet = out / "data" / "train-00000-of-00001.parquet"
    metadata_path = out / "source-metadata.json"
    if parquet.exists() and metadata_path.exists():
        return parquet
    out.mkdir(parents=True, exist_ok=True)
    info = _http_json("https://huggingface.co/api/datasets/" + repo)
    revision = info["sha"]
    files = _http_json(f"https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true&limit=1000")
    records = []
    for entry in files:
        if entry.get("type") != "file":
            continue
        target = out / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        url = (f"https://huggingface.co/datasets/{repo}/resolve/{revision}/"
               + urllib.parse.quote(entry["path"]) + "?download=true")
        if not target.exists():
            part = target.with_suffix(target.suffix + ".part")
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as handle:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
            if not target.exists():
                part.replace(target)
        digest = sha256_file(target)
        lfs = (entry.get("lfs") or {}).get("oid")
        records.append({"path": entry["path"], "url": url, "bytes": target.stat().st_size,
                        "expected_bytes": entry.get("size"), "sha256": digest, "lfs_oid": lfs,
                        "sha256_matches_lfs": (digest == lfs) if lfs else None})
    if not metadata_path.exists():
        metadata_path.write_text(json.dumps({
            "source": "huggingface", "url": "https://huggingface.co/datasets/" + repo,
            "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "revision": revision,
            "license": (info.get("cardData") or {}).get("license", "not specified"),
            "files": records, "api": info}, indent=2) + "\n")
    return parquet


_TODAY_RE = re.compile(r"^Today's date: (.+)$", re.M)


def split_turtel_prompt(prompt: str, question_text: str | None = None) -> dict[str, str]:
    """Split Turtel's frozen prompt into header / question / evidence / footer.

    The evidence block runs from "Question Background:" through "</News>"
    inclusive, so it keeps the background, the resolution criteria, the
    "Today's date" and "Question close date" lines and the retrieved articles.
    """
    header_end = prompt.find("\nQuestion:\n")
    header = prompt[:header_end].strip() if header_end >= 0 else ""
    start = prompt.find("Question Background:")
    if start < 0:
        marker = "\nQuestion:\n" + (question_text or "")
        pos = prompt.find(marker)
        start = pos + len(marker) if pos >= 0 else 0
    end = prompt.find("</News>")
    if end >= 0:
        end += len("</News>")
    else:
        footer_pos = prompt.find("Think about your answer")
        end = footer_pos if footer_pos >= 0 else len(prompt)
    evidence = prompt[start:end].strip()
    footer = prompt[end:].strip()
    today_match = _TODAY_RE.search(prompt)
    return {"header": header, "evidence": evidence, "footer": footer,
            "today": today_match.group(1).strip() if today_match else ""}


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def question_from_row(row: Mapping[str, Any]) -> Question:
    parts = split_turtel_prompt(row["prompt"], row.get("question_text"))
    return Question(
        question_id=str(row["question_id"]), text=str(row["question_text"]),
        close_date=str(row.get("question_close_date") or ""), resolution=int(row["resolution"]),
        prompt=str(row["prompt"]), today=parts["today"], evidence=parts["evidence"],
        prediction_polymarket=_clean_float(row.get("prediction_polymarket")),
        prediction_o1=_clean_float(row.get("prediction_o1")),
        prediction_remax=_clean_float(row.get("prediction_remax")),
        volume=_clean_float(row.get("volume")))


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq  # optional dependency; venv-cft has it
    return pq.read_table(str(path)).to_pylist()


def load_questions(path: Path = PARQUET_PATH, cache: Path | None = QUESTION_CACHE) -> list[Question]:
    """Load the test set from parquet (pyarrow), a JSONL export, or the cache.

    The cache is a JSONL export keyed by the parquet's sha256, written next
    to the runs so that the stdlib interpreter (no pyarrow) can run the game.
    """
    path = Path(path)
    if path.suffix in {".jsonl", ".json"}:
        return [Question.from_record(r) for r in _read_jsonl(path)]
    digest = sha256_file(path)
    if cache is not None and cache.exists():
        with cache.open() as handle:
            first = handle.readline()
        try:
            head = json.loads(first)
        except json.JSONDecodeError:
            head = {}
        if head.get("parquet_sha256") == digest:
            return [Question.from_record(r) for r in _read_jsonl(cache)[1:]]
    try:
        rows = _read_parquet_rows(path)
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is not importable and no matching question cache exists; run once with "
            "/home/agent/workspaces/venv-cft/bin/python (has pyarrow) to build "
            f"{cache}") from exc
    questions = [question_from_row(row) for row in rows]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("w") as handle:
            handle.write(json.dumps({"parquet_sha256": digest, "parquet": str(path), "count": len(questions)}) + "\n")
            for question in questions:
                handle.write(json.dumps(question.to_record()) + "\n")
    return questions


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def brier(pairs: Iterable[tuple[float, int]]) -> float:
    pairs = list(pairs)
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs) if pairs else math.nan


def log_loss(pairs: Iterable[tuple[float, int]], eps: float = 1e-6) -> float:
    pairs = list(pairs)
    total = 0.0
    for p, y in pairs:
        p = min(max(p, eps), 1.0 - eps)
        total -= math.log(p) if y == 1 else math.log(1.0 - p)
    return total / len(pairs) if pairs else math.nan


def verify_table1(questions: Sequence[Question]) -> dict[str, Any]:
    """Recompute Turtel's Table 1 Brier scores from the released file."""
    result: dict[str, Any] = {"rows": len(questions), "expected": dict(TABLE1),
                              "tolerance": TABLE1_TOLERANCE, "computed": {}, "ok": True,
                              "base_rate": sum(q.resolution for q in questions) / max(len(questions), 1)}
    for column, expected in TABLE1.items():
        pairs = [(getattr(q, column), q.resolution) for q in questions if getattr(q, column) is not None]
        score = brier(pairs)
        entry = {"brier": round(score, 6), "log_loss": round(log_loss(pairs), 6), "n": len(pairs),
                 "expected": expected, "within_tolerance": abs(score - expected) <= TABLE1_TOLERANCE}
        result["computed"][column] = entry
        result["ok"] = result["ok"] and entry["within_tolerance"] and len(pairs) == len(questions)
    return result


# ---------------------------------------------------------------------------
# engine wrapper
# ---------------------------------------------------------------------------


def load_engine(engine_dir: Path = ENGINE_DIR):
    """Import FactoredMarket and JointMarketError from the bayes-engine copy."""
    engine_dir = str(Path(engine_dir))
    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)
    from inference.factored_market import FactoredMarket  # noqa: WPS433 (deferred import)
    from inference.joint_market import JointMarketError
    return FactoredMarket, JointMarketError


@dataclass(frozen=True)
class Trade:
    variable: str
    target: float
    context: tuple[tuple[str, str], ...] = ()
    index: int = 0  # position in the model's block

    def as_dict(self) -> dict[str, Any]:
        return {"variable": self.variable, "target": self.target,
                "context": [{"variable": v, "outcome": o} for v, o in self.context], "index": self.index}


class Market:
    """Thread-safe wrapper: prices, link registry, KL accounting, snapshots.

    The engine is single-threaded state, so every read or write goes through
    `lock`; callers that build prompts take a consistent `view` under it and
    later `execute` under it again while provider calls run in parallel.
    """

    def __init__(self, engine: Any, question_ids: Sequence[str], liquidity: float, max_width: int,
                 links: Mapping[str, Iterable[str]] | None = None, version: int = 0) -> None:
        self.engine = engine
        self.question_ids = list(question_ids)
        self.liquidity = float(liquidity)
        self.max_width = int(max_width)
        self.version = int(version)
        self.lock = threading.RLock()
        # links[v] = set of other variables v has been conditionally traded with
        # (either as the traded variable or as the context).
        self.links: dict[str, set[str]] = {str(k): set(v) for k, v in (links or {}).items()}
        _, self._error = load_engine()

    @classmethod
    def fresh(cls, question_ids: Sequence[str], liquidity: float = LIQUIDITY, max_width: int = MAX_WIDTH,
              prior: float = 0.5) -> "Market":
        factored, _ = load_engine()
        nodes = [{"variable_id": qid, "outcomes": ["yes", "no"], "parents": [],
                  "rows": {frozenset(): {"yes": prior, "no": 1.0 - prior}}} for qid in question_ids]
        engine = factored.from_nodes(nodes, liquidity, max_width)
        return cls(engine, question_ids, liquidity, max_width)

    @classmethod
    def from_snapshot(cls, data: Mapping[str, Any]) -> "Market":
        factored, _ = load_engine()
        engine = factored.from_snapshot(data["engine"])
        return cls(engine, data["question_ids"], data["liquidity"], data["max_width"],
                   links=data.get("links", {}), version=data.get("version", 0))

    def snapshot(self, meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
        with self.lock:
            return {"format": "market-game-v1", "engine": self.engine.snapshot(),
                    "question_ids": list(self.question_ids), "liquidity": self.liquidity,
                    "max_width": self.max_width, "version": self.version,
                    "links": {k: sorted(v) for k, v in sorted(self.links.items()) if v},
                    "meta": dict(meta or {}), "saved_at": utcnow()}

    # -- reads ---------------------------------------------------------------

    def price(self, variable: str, context: Mapping[str, str] | None = None) -> float | None:
        with self.lock:
            marginal = self.engine.marginal(variable, evidence=dict(context) if context else None)
        return None if marginal is None else float(marginal["yes"])

    def prices(self) -> dict[str, float]:
        with self.lock:
            return {qid: float(self.engine.marginal(qid)["yes"]) for qid in self.question_ids}

    def view(self, variables: Sequence[str]) -> dict[str, Any]:
        """Prices and current conditionals for a session's variables."""
        with self.lock:
            entries: dict[str, Any] = {}
            for variable in variables:
                conditionals = []
                for other in sorted(self.links.get(variable, ())):
                    given_yes = self.price(variable, {other: "yes"})
                    given_no = self.price(variable, {other: "no"})
                    other_yes = self.price(other, {variable: "yes"})
                    other_no = self.price(other, {variable: "no"})
                    conditionals.append({"other": other, "p_given_other_yes": given_yes,
                                         "p_given_other_no": given_no, "p_other_given_yes": other_yes,
                                         "p_other_given_no": other_no})
                entries[variable] = {"price": self.price(variable), "conditionals": conditionals}
            return {"version": self.version, "variables": entries}

    def stats(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.engine.stats())

    # -- writes --------------------------------------------------------------

    def execute(self, trades: Sequence[Trade], budget: float | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Execute trades in order under the KL budget; returns (fills, rejected).

        KL per trade is KL(target || pre) of the traded slice (the marginal for
        a marginal trade, the conditional for a conditional trade). Once a trade
        would exceed the budget it and every later trade are rejected with
        reason "kl_budget". Engine refusals (width budget, degenerate prices)
        are recorded per trade and execution continues.
        """
        fills: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        limit = math.inf if budget is None else float(budget)
        used = 0.0
        budget_hit = False
        with self.lock:
            for trade in trades:
                context = dict(trade.context)
                if budget_hit:
                    rejected.append({"trade": trade.as_dict(), "reason": "kl_budget",
                                     "detail": "budget exhausted by an earlier trade"})
                    continue
                pre = self.price(trade.variable, context)
                if pre is None:
                    rejected.append({"trade": trade.as_dict(), "reason": "context_zero_probability",
                                     "detail": "context has zero probability"})
                    continue
                kl = binary_kl(trade.target, pre)
                if used + kl > limit + 1e-12:
                    budget_hit = True
                    rejected.append({"trade": trade.as_dict(), "reason": "kl_budget",
                                     "detail": f"kl {kl:.4f} would exceed remaining {limit - used:.4f}"})
                    continue
                pre_marginal = self.price(trade.variable)
                try:
                    fill = self.engine.trade_to_probability(trade.variable, "yes", trade.target,
                                                            context=context or None)
                except self._error as exc:
                    rejected.append({"trade": trade.as_dict(), "reason": "engine_error", "detail": str(exc)})
                    continue
                used += kl
                self.version += 1
                if context:
                    for other in context:
                        self.links.setdefault(trade.variable, set()).add(other)
                        self.links.setdefault(other, set()).add(trade.variable)
                post = self.price(trade.variable, context)
                fills.append({"trade": trade.as_dict(), "pre": pre, "post": post,
                              "pre_marginal": pre_marginal, "post_marginal": self.price(trade.variable),
                              "cost": fill["cost"], "shares": fill["shares"], "kl": kl,
                              "kl_cumulative": used, "version": self.version})
        return fills, rejected


# ---------------------------------------------------------------------------
# grouping: TF-IDF similarity, sibling sets, session groups
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['.][a-z0-9]+)*")
STOPWORDS = frozenset("""a an and are as at be by for from in is it its of on or that the this to will with
what which who whom whose does do did has have had than then there their they them if any""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def tfidf_vectors(texts: Sequence[str]) -> list[dict[str, float]]:
    """L2-normalised TF-IDF vectors (log tf, smoothed idf); pure Python."""
    docs = [tokenize(t) for t in texts]
    df: dict[str, int] = {}
    for tokens in docs:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    n = len(docs)
    idf = {term: math.log((1 + n) / (1 + count)) + 1.0 for term, count in df.items()}
    vectors = []
    for tokens in docs:
        counts: dict[str, int] = {}
        for term in tokens:
            counts[term] = counts.get(term, 0) + 1
        vector = {term: (1.0 + math.log(c)) * idf[term] for term, c in counts.items()}
        norm = math.sqrt(sum(w * w for w in vector.values())) or 1.0
        vectors.append({term: w / norm for term, w in vector.items()})
    return vectors


def top_neighbours(vectors: Sequence[Mapping[str, float]], k: int = 20, min_sim: float = 0.15) -> list[list[tuple[int, float]]]:
    """Per document, the k most cosine-similar others above min_sim (desc)."""
    postings: dict[str, list[int]] = {}
    for i, vector in enumerate(vectors):
        for term in vector:
            postings.setdefault(term, []).append(i)
    result = []
    for i, vector in enumerate(vectors):
        scores: dict[int, float] = {}
        for term, weight in vector.items():
            for j in postings[term]:
                if j != i:
                    scores[j] = scores.get(j, 0.0) + weight * vectors[j][term]
        ranked = sorted(((j, s) for j, s in scores.items() if s >= min_sim), key=lambda x: (-x[1], x[0]))
        result.append(ranked[:k])
    return result


_SLUG_RE = re.compile(r"polymarket\.com/(?:event|market)/([a-z0-9-]+)", re.I)
_VERSUS_RE = re.compile(r'^Will the following market resolve to "(.+?)"\?\s*(.+)$')
_QUOTED_RE = re.compile(r'"[^"]*"|“[^”]*”')
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|"
                      r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:\s*-\s*\d{1,2})?\b", re.I)
_NUMBER_RE = re.compile(r"[$€£]?\d[\d,]*(?:\.\d+)?[kmb%]?(?:\s*°\s*f)?", re.I)
_RANGE_RE = re.compile(r"(?:between\s+)?#(?:\s*(?:-|and|to)\s*#)?(?:\s*or\s+(?:higher|lower|more|less|fewer|above|below))?(?:\s*\+)?", re.I)
_IS_RANGE_RE = re.compile(r"\bbetween\b|#\s*(?:-|and|to)\s*#", re.I)  # "x or more" alone is a threshold


def polymarket_slug(*texts: str) -> str | None:
    for text in texts:
        match = _SLUG_RE.search(text or "")
        if match:
            return match.group(1).lower()
    return None


def mentions(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(dates, numbers) mentioned in a question text, lowercased."""
    dates = tuple(d.lower() for d in _DATE_RE.findall(text))
    numbers = tuple(n.lower() for n in _NUMBER_RE.findall(_DATE_RE.sub(" ", text)))
    return dates, numbers


def event_key(question: Question) -> tuple[str, str, bool] | None:
    """(kind, key, is_range) shared by sibling questions of one event, or None.

    kinds: "slug" (Polymarket event slug), "versus" (two-sided match market),
    "award" (quoted nominee for a category), "numeric" (same template with
    numbers replaced; the date mentions stay in the key). is_range flags a
    bin such as "between 42-43" or "54 or higher"; a numeric sibling set
    with a range member is treated as a partition ("bins"), otherwise as a
    threshold/count "series".
    """
    slug = polymarket_slug(question.question_id, question.text)
    if slug:
        return ("slug", slug, False)
    text = question.text.strip()
    match = _VERSUS_RE.match(text)
    if match:
        return ("versus", re.sub(r"\s+", " ", match.group(2)).strip().lower(), False)
    if _QUOTED_RE.search(text) and re.search(r"\bwin\b", text, re.I):
        return ("award", re.sub(r"\s+", " ", _QUOTED_RE.sub('""', text)).strip().lower(), False)
    dates = _DATE_RE.findall(text)
    protected = _DATE_RE.sub(" @date ", text)
    if not _NUMBER_RE.search(protected):
        return None
    numbered = _NUMBER_RE.sub(" # ", protected)
    numbered = re.sub(r"\s+", " ", numbered).strip()
    is_range = bool(_IS_RANGE_RE.search(numbered))
    collapsed = re.sub(r"\s+", " ", _RANGE_RE.sub("#", numbered)).strip().lower()
    key = collapsed + " |" + "|".join(d.lower() for d in dates)
    return ("numeric", key, is_range)


@dataclass
class SiblingSet:
    set_id: int
    kind: str
    members: list[str]
    exclusive: bool
    duplicate_texts: int = 0  # members whose text repeats another member's (same market listed twice)

    def as_dict(self) -> dict[str, Any]:
        return {"set_id": self.set_id, "kind": self.kind, "members": list(self.members),
                "exclusive": self.exclusive, "duplicate_texts": self.duplicate_texts}


def detect_siblings(questions: Sequence[Question], neighbours: Sequence[Sequence[tuple[int, float]]] | None = None,
                    sim_threshold: float = 0.8) -> list[SiblingSet]:
    """Union-find over event keys, then over near-identical texts.

    Structural keys (slug, versus, award, numeric template + dates) define
    the sets whose exclusivity is reported. The similarity rule only joins
    pairs that mention the same dates and numbers and are not structurally
    keyed to different events, so a wording-only sibling ("Will Canada win
    the ..." / "Will Sweden win the ...") is found without merging
    different dates or categories.
    """
    parent = list(range(len(questions)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    keys = [event_key(q) for q in questions]
    by_key: dict[tuple[str, str], list[int]] = {}
    for i, key in enumerate(keys):
        if key is not None:
            by_key.setdefault((key[0], key[1]), []).append(i)
    for members in by_key.values():
        for i in members[1:]:
            union(members[0], i)
    if neighbours is not None:
        marks = [mentions(q.text) for q in questions]
        for i, ranked in enumerate(neighbours):
            for j, sim in ranked:
                if sim < sim_threshold or marks[i] != marks[j]:
                    continue
                ki, kj = keys[i], keys[j]
                structural = {"slug", "versus", "award"}
                if ((ki is not None and ki[0] in structural) or (kj is not None and kj[0] in structural)) \
                        and (ki is None or kj is None or (ki[0], ki[1]) != (kj[0], kj[1])):
                    continue  # structurally keyed to different events
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(len(questions)):
        groups.setdefault(find(i), []).append(i)
    order = {"slug": 0, "versus": 1, "award": 2, "bins": 3, "series": 4, "similar": 5}
    sets = []
    for members in groups.values():
        if len(members) < 2:
            continue
        member_keys = [keys[i] for i in members if keys[i] is not None]
        kinds = {k[0] for k in member_keys}
        if "slug" in kinds:
            kind = "slug"
        elif "versus" in kinds:
            kind = "versus"
        elif "award" in kinds:
            kind = "award"
        elif "numeric" in kinds:
            kind = "bins" if any(k[2] for k in member_keys) else "series"
        else:
            kind = "similar"
        texts = [re.sub(r"\s+", " ", questions[i].text).strip().lower() for i in members]
        sets.append(SiblingSet(0, kind, [questions[i].question_id for i in members],
                               exclusive=kind in {"versus", "award", "bins"},
                               duplicate_texts=len(texts) - len(set(texts))))
    sets.sort(key=lambda s: (order[s.kind], s.members[0]))
    for n, sibling in enumerate(sets):
        sibling.set_id = n
    return sets


def group_sizes(count: int, n: int) -> list[int]:
    """As-equal-as-possible sizes n or n-1 for ceil(count/n) groups."""
    if count <= 0:
        return []
    groups = math.ceil(count / n)
    base, extra = divmod(count, groups)
    return [base + 1 if i < extra else base for i in range(groups)]


def build_groups(question_ids: Sequence[str], neighbours: Mapping[str, Sequence[tuple[str, float]]],
                 siblings: Mapping[str, Sequence[str]], n: int, rng: random.Random,
                 min_sim: float = 0.2) -> list[list[str]]:
    """Greedy similarity clustering into groups of about n, random fill.

    Seeds are visited in random order; each seed gathers its sibling-set
    members first, then its nearest unassigned neighbours above min_sim.
    Cores of size >= 2 become groups; everything else is dealt at random into
    the remaining slots, so a question's non-sibling neighbours change with
    the seed and the pass.
    """
    ids = list(question_ids)
    sizes = group_sizes(len(ids), n)
    if not sizes:
        return []
    order = list(ids)
    rng.shuffle(order)
    unassigned = set(ids)

    def bundle(qid: str, taken: Sequence[str]) -> list[str]:
        """qid plus its still-unassigned siblings (sibling sets travel together)."""
        members = [qid] + [m for m in sorted(siblings.get(qid, ())) if m in unassigned and m != qid and m not in taken]
        return members

    groups: list[list[str]] = []
    for seed in order:
        if seed not in unassigned or len(groups) >= len(sizes):
            continue
        cap = sizes[len(groups)]
        core = bundle(seed, [])[:cap]
        for other, sim in neighbours.get(seed, ()):
            if len(core) >= cap:
                break
            if sim < min_sim or other not in unassigned or other in core:
                continue
            extra = bundle(other, core)
            if len(core) + len(extra) <= cap:
                core.extend(extra)
        if len(core) < 2:
            continue
        for member in core:
            unassigned.discard(member)
        groups.append(core)
    while len(groups) < len(sizes):
        groups.append([])
    leftovers = sorted(unassigned)
    rng.shuffle(leftovers)
    while leftovers:
        qid = leftovers.pop()
        if qid not in unassigned:
            continue
        extra = bundle(qid, [])
        room = [(sizes[i] - len(g), i) for i, g in enumerate(groups) if sizes[i] - len(g) >= len(extra)]
        if room:
            index = max(room)[1]  # the group with the most room takes the whole bundle
            groups[index].extend(extra)
            unassigned.difference_update(extra)
        else:  # bundle does not fit anywhere whole: deal this question alone
            index = max((sizes[i] - len(g), i) for i, g in enumerate(groups))[1]
            groups[index].append(qid)
            unassigned.discard(qid)
    assert not unassigned and all(len(g) == sizes[i] for i, g in enumerate(groups)), "group sizes do not cover the set"
    return [g for g in groups if g]


class Grouper:
    """Precomputes similarity once; builds seeded groups per pass."""

    def __init__(self, questions: Sequence[Question], sibling_sets: Sequence[SiblingSet] | None = None,
                 k: int = 20, min_sim: float = 0.15) -> None:
        self.ids = [q.question_id for q in questions]
        vectors = tfidf_vectors([q.text for q in questions])
        ranked = top_neighbours(vectors, k=k, min_sim=min_sim)
        self.neighbours = {self.ids[i]: [(self.ids[j], s) for j, s in row] for i, row in enumerate(ranked)}
        self.sibling_sets = list(sibling_sets) if sibling_sets is not None else detect_siblings(questions, ranked)
        self.siblings: dict[str, list[str]] = {}
        for sibling in self.sibling_sets:
            for member in sibling.members:
                self.siblings[member] = [m for m in sibling.members if m != member]

    def groups_for_pass(self, pass_index: int, n: int, seed: int) -> list[list[str]]:
        rng = random.Random(f"{seed}:groups:{pass_index}")
        groups = build_groups(self.ids, self.neighbours, self.siblings, n, rng)
        rng.shuffle(groups)  # random order of groups within a pass
        return groups


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------

DEVELOPER_RULE = (
    "You are a careful forecaster trading in a combinatorial prediction market. "
    "Follow the date rule, the market rules and the output format in the user message exactly. "
    "Do not use tools, browse, read files, consult memory, or delegate. "
    "End your reply with exactly one fenced JSON block containing your trades."
)

DATE_RULE = (
    "Answer as of each question's stated date (the \"Today's date\" line inside its evidence). "
    "Do not use any knowledge from after that date, even if you have it; pretend you do not. "
    "Reason only from the evidence shown and from general knowledge available before that date."
)


def budget_for_pass(pass_index: int, n: int, factor: float = BUDGET_FACTOR, first_pass_uncapped: bool = True) -> float | None:
    """Session KL budget in nats: pass 1 uncapped, later passes factor * N."""
    if pass_index == 1 and first_pass_uncapped:
        return None
    return round(factor * n, 6)


def format_conditionals(entry: Mapping[str, Any], texts: Mapping[str, str]) -> str:
    conditionals = entry.get("conditionals") or []
    if not conditionals:
        return "none"
    lines = []
    for c in conditionals:
        other = c["other"]
        label = texts.get(other, other)
        if len(label) > 90:
            label = label[:87] + "..."
        short = texts.get("__alias__", {}).get(other, other) if isinstance(texts.get("__alias__"), dict) else other
        lines.append(f"  - with {short} (\"{label}\"): P(this=yes | other=yes) = {c['p_given_other_yes']:.3f}, "
                     f"P(this=yes | other=no) = {c['p_given_other_no']:.3f}; "
                     f"P(other=yes | this=yes) = {c['p_other_given_yes']:.3f}, "
                     f"P(other=yes | this=no) = {c['p_other_given_no']:.3f}")
    return "\n" + "\n".join(lines)


def build_prompt(questions: Sequence[Question], view: Mapping[str, Any], budget: float | None, *,
                 session_id: str, pass_index: int, liquidity: float = LIQUIDITY,
                 feedback: Sequence[str] | None = None) -> str:
    """The user prompt for one session. Deterministic in its inputs."""
    n = len(questions)
    texts = {q.question_id: q.text for q in questions}
    texts["__alias__"] = {q.question_id: f"q{k}" for k, q in enumerate(questions, start=1)}
    if budget is None:
        budget_rule = ("5. KL budget: this pass is uncapped. There is no limit on the total KL divergence you move "
                       "(KL(new || old) of the traded probability, summed over your trades, in nats); later passes "
                       "cap it. Trades are executed in the order given.")
    else:
        budget_rule = (f"5. KL budget: the total KL divergence you move in this session, summed over your trades "
                       f"(KL(new || old) of the traded probability, in nats), must stay within eps_total = "
                       f"{budget:.3f} nats. Trades are executed in the order given; once the budget is exhausted the "
                       "remaining trades are rejected.")
    lines = [
        f"COMBINATORIAL PREDICTION MARKET SESSION {session_id} (pass {pass_index}, {n} questions)",
        "",
        "DATE RULE",
        DATE_RULE,
        "",
        "MARKET RULES",
        "1. Every question is a binary variable with outcomes \"yes\" and \"no\". The price shown is the "
        "current market probability of \"yes\".",
        f"2. A trade sets a new market probability (\"target\") for one variable. The market maker is a "
        f"logarithmic market scoring rule (LMSR) with liquidity b = {liquidity:g}.",
        "3. A trade may carry a context: one or more other variables from this session fixed to \"yes\" or "
        "\"no\". Such a conditional trade sets P(A | B=yes) (or P(A | B=no)) instead of P(A); it creates a "
        "dependency between A and B in the market, and P(B) itself does not move.",
        "4. Payoff at resolution: b * (log p_new(w) - log p_old(w)), where w is the realised outcome of the "
        "traded variable and p_old, p_new are its probabilities before and after your trade (conditional "
        "probabilities for a conditional trade). You gain when you move the price towards the truth and "
        "lose when you move it away. A conditional trade is called off (payoff 0) if its context does not "
        "resolve as stated.",
        budget_rule,
        "6. Trades execute against the live market; prices may have moved slightly since this prompt was "
        "built.",
        "",
        "QUESTIONS",
    ]
    entries = view["variables"]
    for k, question in enumerate(questions, start=1):
        entry = entries[question.question_id]
        lines += [
            f"--- Question {k} of {n} ---",
            f"id: q{k}   (full id {question.question_id})",
            f"question: {question.text}",
            f"close date: {question.close_date}",
            f"current market probability of \"yes\": {entry['price']:.3f}",
            f"current conditionals involving this question: {format_conditionals(entry, texts)}",
            "evidence (frozen retrieval; the \"Today's date\" line is the date you must answer as of):",
            question.evidence,
            "",
        ]
    lines += [
        "REQUIRED OUTPUT",
        "First, for each question, a short rationale: the evidence you rely on (cite article numbers) and "
        "the key considerations. Then, at the very end of your reply, exactly one fenced JSON block of the "
        "form:",
        "```json",
        '{"trades": [{"variable": "q1", "target": 0.xx, "context": [{"variable": "q2", "outcome": "yes"}]}]}',
        "```",
        "Constraints:",
        f"- \"variable\" is the short id of a question above: q1 to q{n} (the full id is also accepted).",
        f"- \"target\" is a number within [{TARGET_MIN:.2f}, {TARGET_MAX:.2f}] (the new probability of \"yes\").",
        "- \"context\" is optional; when present it lists one or more other question ids from this session, "
        "each with an \"outcome\" of \"yes\" or \"no\". Do not repeat a variable inside a context and do not "
        "put the traded variable in its own context.",
        f"- At least one trade per question. Express \"no change\" by targeting the current price. At most "
        f"{MAX_TRADES_PER_QUESTION} trades per question (for example one marginal trade and one conditional "
        "trade).",
        "- Use conditional trades only where you believe one question's outcome should change the "
        "probability of another.",
        "- Trades that violate these constraints are rejected individually; a response without a valid "
        "JSON block is rejected as a whole.",
    ]
    if feedback:
        lines += ["", "PREVIOUS ATTEMPT REJECTED",
                  "Your previous response for this session was rejected for the reasons below. Respond "
                  "again, following the output format exactly."]
        lines += [f"- {item}" for item in feedback]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# parser and validation
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```[ \t]*(?:json|JSON)?[ \t]*\r?\n(.*?)```", re.S)
_TRADES_RE = re.compile(r"\{\s*\"trades\"")


def parse_trades(text: str) -> tuple[list[Any] | None, list[str]]:
    """Extract the trade list from a response; (None, errors) when absent.

    Tolerant: the last fenced json block that parses wins; otherwise the
    first `{"trades"` object in the text is decoded in place. Malformed
    JSON (trailing commas, comments) is rejected with the decoder's message.
    """
    errors: list[str] = []
    if not isinstance(text, str) or not text.strip():
        return None, ["empty response"]
    blocks = _FENCE_RE.findall(text)
    for number, block in reversed(list(enumerate(blocks, start=1))):
        try:
            data = json.loads(block)
        except json.JSONDecodeError as exc:
            errors.append(f"fenced block {number}: invalid JSON ({exc.msg} at line {exc.lineno} column {exc.colno})")
            continue
        if isinstance(data, dict) and isinstance(data.get("trades"), list):
            return data["trades"], errors
        errors.append(f"fenced block {number}: JSON has no \"trades\" list")
    decoder = json.JSONDecoder()
    for match in _TRADES_RE.finditer(text):
        try:
            data, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError as exc:
            errors.append(f"inline trades object at offset {match.start()}: invalid JSON ({exc.msg})")
            continue
        if isinstance(data, dict) and isinstance(data.get("trades"), list):
            return data["trades"], errors
    if not blocks and not errors:
        errors.append("no fenced json block and no {\"trades\": [...]} object found")
    return None, errors


@dataclass
class Validated:
    raw: Any
    trade: Trade | None
    error: str | None


def resolve_variable(name: Any, session_ids: Sequence[str], min_prefix: int = 12) -> str | None:
    """Exact id, else a unique prefix of at least min_prefix characters."""
    if not isinstance(name, str):
        return None
    name = name.strip()
    if name in session_ids:
        return name
    alias = name.lower()
    if len(alias) >= 2 and alias[0] == "q" and alias[1:].isdigit():
        k = int(alias[1:])
        if 1 <= k <= len(session_ids):
            return session_ids[k - 1]
    if len(name) >= min_prefix:
        matches = [qid for qid in session_ids if qid.startswith(name)]
        if len(matches) == 1:
            return matches[0]
    return None


def validate_trades(raw_trades: Sequence[Any], session_ids: Sequence[str],
                    max_per_question: int = MAX_TRADES_PER_QUESTION) -> list[Validated]:
    """Per-trade validation with machine-readable reasons (error strings)."""
    result: list[Validated] = []
    per_question: dict[str, int] = {}
    for index, raw in enumerate(raw_trades):
        if not isinstance(raw, dict):
            result.append(Validated(raw, None, "malformed: trade is not an object"))
            continue
        variable = resolve_variable(raw.get("variable"), session_ids)
        if variable is None:
            result.append(Validated(raw, None, f"unknown_variable: {raw.get('variable')!r} is not a question id in this session"))
            continue
        target = raw.get("target")
        if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(float(target)):
            result.append(Validated(raw, None, f"target_not_number: {target!r}"))
            continue
        target = float(target)
        if not TARGET_MIN <= target <= TARGET_MAX:
            result.append(Validated(raw, None, f"target_range: {target} is outside [{TARGET_MIN}, {TARGET_MAX}]"))
            continue
        context_raw = raw.get("context")
        if context_raw is None:
            context_raw = []
        if not isinstance(context_raw, list):
            result.append(Validated(raw, None, "context_malformed: context must be a list"))
            continue
        context: list[tuple[str, str]] = []
        error = None
        seen: set[str] = set()
        for item in context_raw:
            if not isinstance(item, dict):
                error = "context_malformed: context entry is not an object"
                break
            other = resolve_variable(item.get("variable"), session_ids)
            outcome = item.get("outcome")
            if other is None:
                error = f"context_unknown_variable: {item.get('variable')!r} is not a question id in this session"
                break
            if other == variable:
                error = "context_self: the traded variable cannot be in its own context"
                break
            if other in seen:
                error = f"context_duplicate: {other} appears twice in the context"
                break
            if outcome not in ("yes", "no"):
                error = f"context_bad_outcome: {outcome!r} is not \"yes\" or \"no\""
                break
            seen.add(other)
            context.append((other, outcome))
        if error:
            result.append(Validated(raw, None, error))
            continue
        count = per_question.get(variable, 0) + 1
        per_question[variable] = count
        if count > max_per_question:
            result.append(Validated(raw, None, f"too_many_trades: more than {max_per_question} trades for {variable}"))
            continue
        result.append(Validated(raw, Trade(variable, target, tuple(context), index), None))
    return result


def reason_code(error: str) -> str:
    return error.split(":", 1)[0].strip()


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------


@dataclass
class SessionRequest:
    session_id: str
    attempt: int
    pass_index: int
    group_index: int
    prompt: str
    developer_rule: str
    question_ids: list[str]
    prices: dict[str, float]
    budget: float | None


@dataclass
class ProviderResult:
    text: str
    ok: bool = True
    error: str | None = None
    thread_id: str | None = None
    usage: dict[str, Any] | None = None
    elapsed: float = 0.0
    exit_code: int | str | None = None
    stderr: str = ""
    command: list[str] | None = None
    events: dict[str, Any] = field(default_factory=dict)


class Provider:
    name = "base"
    model = "none"
    effort = "none"

    def complete(self, request: SessionRequest) -> ProviderResult:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "effort": self.effort}


class MockProvider(Provider):
    """Deterministic offline stand-in for the frontier model.

    Moves every question toward a noisy view of its resolution (flip_rate),
    sometimes adds a conditional trade between two session questions,
    sometimes emits an invalid trade (bad target, unknown id, self-context),
    occasionally proposes a big jump (exercises the KL budget) and, rarely,
    returns an unparseable block (exercises the retry path). Everything is
    seeded by (seed, session id, attempt) so runs are reproducible.
    """

    name = "mock"
    model = "mock"
    effort = "none"

    def __init__(self, resolutions: Mapping[str, int], seed: int = 0, flip_rate: float = 0.2, step: float = 0.5,
                 jitter: float = 0.05, conditional_rate: float = 0.15, invalid_rate: float = 0.1,
                 big_jump_rate: float = 0.05, garbage_rate: float = 0.02, latency: float = 0.0) -> None:
        self.resolutions = dict(resolutions)
        self.seed = seed
        self.flip_rate = flip_rate
        self.step = step
        self.jitter = jitter
        self.conditional_rate = conditional_rate
        self.invalid_rate = invalid_rate
        self.big_jump_rate = big_jump_rate
        self.garbage_rate = garbage_rate
        self.latency = latency

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "seed": self.seed, "flip_rate": self.flip_rate, "step": self.step,
                "jitter": self.jitter, "conditional_rate": self.conditional_rate, "invalid_rate": self.invalid_rate,
                "big_jump_rate": self.big_jump_rate, "garbage_rate": self.garbage_rate}

    def complete(self, request: SessionRequest) -> ProviderResult:
        started = time.monotonic()
        if self.latency:
            time.sleep(self.latency)
        rng = random.Random(f"{self.seed}:{request.session_id}:{request.attempt}")
        ids = list(request.question_ids)
        trades: list[dict[str, Any]] = []
        rationale = []
        for qid in ids:
            price = request.prices.get(qid, 0.5)
            truth = self.resolutions.get(qid, 0)
            belief = truth if rng.random() >= self.flip_rate else 1 - truth
            if rng.random() < self.big_jump_rate:
                target = 0.98 if belief else 0.02
            else:
                target = price + self.step * (belief - price) + rng.gauss(0.0, self.jitter)
            target = min(max(round(target, 3), TARGET_MIN), TARGET_MAX)
            rationale.append(f"{qid}: evidence points to {'yes' if belief else 'no'}; move {price:.3f} -> {target:.3f}.")
            trades.append({"variable": qid, "target": target})
            if len(ids) > 1 and rng.random() < self.conditional_rate:
                other = rng.choice([o for o in ids if o != qid])
                outcome = rng.choice(["yes", "no"])
                shift = 0.15 if belief else -0.15
                conditional = min(max(round(target + shift, 3), TARGET_MIN), TARGET_MAX)
                trades.append({"variable": qid, "target": conditional,
                               "context": [{"variable": other, "outcome": outcome}]})
        if rng.random() < self.invalid_rate and ids:
            kind = rng.choice(["target_range", "unknown_variable", "context_self", "context_duplicate"])
            victim = rng.choice(ids)
            if kind == "target_range":
                trades.append({"variable": victim, "target": 1.2})
            elif kind == "unknown_variable":
                trades.append({"variable": "polymarket_0xdeadbeef" + "0" * 56, "target": 0.4})
            elif kind == "context_self":
                trades.append({"variable": victim, "target": 0.4, "context": [{"variable": victim, "outcome": "yes"}]})
            else:
                other = rng.choice(ids)
                trades.append({"variable": victim, "target": 0.4,
                               "context": [{"variable": other, "outcome": "yes"}, {"variable": other, "outcome": "no"}]})
        block = json.dumps({"trades": trades})
        if request.attempt == 1 and rng.random() < self.garbage_rate:
            block = block[:-2] + ",]}"  # trailing comma: must be rejected cleanly
        text = "\n".join(rationale) + "\n\n```json\n" + block + "\n```\n"
        return ProviderResult(text=text, thread_id=f"mock-{sha256_text(request.prompt)[:12]}",
                              usage={"input_tokens": len(request.prompt) // 4, "output_tokens": len(text) // 4},
                              elapsed=time.monotonic() - started, exit_code=0)


def codex_command(provider: str, prompt: str, empty: Path, output: Path, effort: str, *,
                  rule: str = DEVELOPER_RULE, campaign: str = CAMPAIGN,
                  environ: Mapping[str, str] | None = None) -> tuple[list[str], dict[str, str], str]:
    """Exact copy of harness/run_native_checks.py native_command() for astra/luna.

    test_market_game.py asserts equality with the original whenever it is
    importable, so this copy cannot drift silently.
    """
    if provider not in MODELS:
        raise ValueError(f"unknown Codex provider: {provider}")
    env = dict(os.environ if environ is None else environ)
    for name in list(env):
        if name.startswith(("ANTHROPIC_", "OPENAI_", "XAI_", "ZAI_", "ZHIPU_", "CODEX_API_")):
            env.pop(name)
    env.pop("TMUX", None)
    actor = "consistency-" + campaign
    env.update(FLEET_ACTOR=actor, FUTARCHY_AGENT_NAME=actor, AI_AGENT_NAME=actor)
    native_runtime = env.get("CONSISTENCY_CODEX_RUNTIME")
    executable = CODEX_EXECUTABLE
    if native_runtime:
        executable = "/home/agent/.local/bin/codex"
        env["CODEX_HOME"] = str(Path(native_runtime).resolve())
        env["CODEX_ANY_KEY"] = "proxy-managed-client"
        env["CODEX_REFRESH_TOKEN_URL_OVERRIDE"] = "http://127.0.0.1:9/oauth/token"
    cmd = [executable, "exec", "--ephemeral", "--json",
           "--skip-git-repo-check", "--sandbox", "read-only", "-C", str(empty),
           "--model", MODELS[provider], "-c", "model_reasoning_effort=" + json.dumps(effort),
           "-c", 'service_tier="default"', "-c", "project_doc_max_bytes=0",
           "-c", 'web_search="disabled"', "-c", "features.multi_agent_v2=false",
           "-c", "features.shell_tool=false", "-c", "features.exec_tool=false",
           "-c", "developer_instructions=" + json.dumps(rule),
           "-o", str(output), "-"]
    return cmd, env, prompt


def json_events(stdout: str) -> list[dict[str, Any]]:
    events = []
    for line in stdout.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def decode_codex_events(stdout: str, final_path: Path | None) -> dict[str, Any]:
    """Final agent_message, thread id and usage from the Codex JSON stream.

    Mirrors harness/run_native_probes.py validate_output for astra/luna: the
    final text is the -o file when present, else the single agent_message;
    exactly one turn.completed and one thread.started are expected.
    """
    events = json_events(stdout)
    messages = [(i, e["item"].get("text", "")) for i, e in enumerate(events)
                if e.get("type") == "item.completed" and e.get("item", {}).get("type") == "agent_message"]
    completions = [i for i, e in enumerate(events) if e.get("type") == "turn.completed"]
    starts = [e.get("thread_id") for e in events if e.get("type") == "thread.started"]
    usage = next((e.get("usage") for e in reversed(events) if e.get("type") == "turn.completed"), None)
    final_text = final_path.read_text() if final_path is not None and final_path.exists() else None
    text = final_text if final_text is not None else (messages[-1][1] if messages else "")
    errors = []
    if len(completions) != 1:
        errors.append("native_completion_not_established")
    if len(starts) != 1 or not isinstance(starts[0], str) or not starts[0]:
        errors.append("native_session_missing")
    if not messages:
        errors.append("native_message_missing")
    elif final_text is not None and messages[-1][1].strip() != final_text.strip():
        errors.append("native_final_file_or_message_mismatch")
    if completions and messages and completions[-1] < messages[-1][0]:
        errors.append("native_completion_order")
    for event in events:
        if event.get("type") == "turn.failed":
            errors.append("native_turn_failed")
        if event.get("type") == "error":
            errors.append("native_error_event")
        kind = event.get("item", {}).get("type")
        if kind and kind not in {"agent_message", "reasoning"}:
            errors.append("tools_used:" + str(kind))
    return {"text": text if isinstance(text, str) else "", "thread_id": starts[0] if starts else None,
            "usage": usage, "errors": sorted(set(errors)), "event_count": len(events),
            "message_count": len(messages), "event_types": sorted({str(e.get("type")) for e in events})}


class CodexProvider(Provider):
    """GPT-6 Astra / GPT-5.6 Luna through the Codex CLI (subscription-native)."""

    name = "codex"

    def __init__(self, provider: str, effort: str = "medium", timeout: float = 1800.0, campaign: str = CAMPAIGN,
                 work_root: Path | None = None, rule: str = DEVELOPER_RULE) -> None:
        if provider not in MODELS:
            raise ValueError(f"unknown Codex provider: {provider}")
        self.name = provider
        self.provider = provider
        self.model = MODELS[provider]
        self.effort = effort
        self.timeout = timeout
        self.campaign = campaign
        self.rule = rule
        root = Path(work_root) if work_root else Path(os.environ.get("TMPDIR") or "/tmp") / "market-game"
        self.work_root = root.resolve()  # codex runs with cwd=empty dir, so -C/-o must be absolute

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "campaign": self.campaign, "timeout": self.timeout,
                "executable": CODEX_EXECUTABLE, "developer_rule_sha256": sha256_text(self.rule)}

    def build(self, prompt: str, session_dir: Path) -> tuple[list[str], dict[str, str], str, Path]:
        empty = session_dir / "empty"
        empty.mkdir(parents=True, exist_ok=True)
        output = session_dir / "final.txt"
        cmd, env, stdin = codex_command(self.provider, prompt, empty, output, self.effort,
                                        rule=self.rule, campaign=self.campaign)
        env["TMPDIR"] = str(session_dir)
        return cmd, env, stdin, output

    def dry_run(self, request: SessionRequest, session_dir: Path | None = None) -> dict[str, Any]:
        session_dir = session_dir or (self.work_root / "dry-run")
        cmd, env, stdin, output = self.build(request.prompt, session_dir)
        return {"command": cmd, "command_line": shlex.join(cmd), "stdin": stdin, "output_file": str(output),
                "env_overrides": {k: env[k] for k in ("FLEET_ACTOR", "FUTARCHY_AGENT_NAME", "AI_AGENT_NAME", "TMPDIR")}}

    def complete(self, request: SessionRequest) -> ProviderResult:
        started = time.monotonic()
        session_dir = self.work_root / request.session_id / f"attempt-{request.attempt}"
        session_dir.mkdir(parents=True, exist_ok=True)
        cmd, env, stdin, output = self.build(request.prompt, session_dir)
        output.unlink(missing_ok=True)
        stdout, stderr, exit_code = "", "", "not_started"
        try:
            process = subprocess.Popen(cmd, cwd=session_dir / "empty", env=env, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                       start_new_session=True)
            try:
                stdout, stderr = process.communicate(stdin, timeout=self.timeout)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
                exit_code = "timeout"
        except OSError as exc:
            stderr = type(exc).__name__ + ": " + str(exc)
            exit_code = "launch_error"
        (session_dir / "stdout.jsonl").write_text(stdout)
        (session_dir / "stderr.txt").write_text(stderr)
        decoded = decode_codex_events(stdout, output)
        errors = list(decoded["errors"])
        if exit_code != 0:
            errors.append(f"native_exit:{exit_code}")
        return ProviderResult(text=decoded["text"], ok=not errors, error="; ".join(errors) if errors else None,
                              thread_id=decoded["thread_id"], usage=decoded["usage"],
                              elapsed=time.monotonic() - started, exit_code=exit_code, stderr=stderr[-4000:],
                              command=cmd, events={k: decoded[k] for k in ("event_count", "message_count", "event_types")})


def make_provider(name: str, questions: Sequence[Question], effort: str = "medium", seed: int = 0,
                  timeout: float = 1800.0, mock_options: Mapping[str, Any] | None = None,
                  work_root: Path | None = None) -> Provider:
    if name == "mock":
        return MockProvider({q.question_id: q.resolution for q in questions}, seed=seed, **dict(mock_options or {}))
    if name in MODELS:
        return CodexProvider(name, effort=effort, timeout=timeout, work_root=work_root)
    raise ValueError(f"unknown provider: {name}")


# ---------------------------------------------------------------------------
# quota gate
# ---------------------------------------------------------------------------


class QuotaGate:
    """Runs --quota-check-cmd before sessions; pauses the pool on "PAUSE"."""

    def __init__(self, command: str | None, poll: float = 30.0, min_interval: float = 20.0,
                 log: Callable[[str], None] | None = None) -> None:
        self.command = command
        self.poll = poll
        self.min_interval = min_interval
        self.log = log or (lambda message: None)
        self._lock = threading.Lock()
        self._last_check = -math.inf
        self.pauses = 0

    def _paused(self) -> bool:
        try:
            completed = subprocess.run(self.command, shell=True, capture_output=True, text=True, timeout=120)
            return "PAUSE" in (completed.stdout or "")
        except (OSError, subprocess.SubprocessError) as exc:
            self.log(f"quota check failed ({exc}); continuing")
            return False

    def wait(self) -> None:
        if not self.command:
            return
        with self._lock:  # other workers block here while one worker is waiting out a pause
            if time.monotonic() - self._last_check < self.min_interval:
                return
            while self._paused():
                self.pauses += 1
                self.log(f"quota check printed PAUSE; sleeping {self.poll:g}s")
                time.sleep(self.poll)
            self._last_check = time.monotonic()


# ---------------------------------------------------------------------------
# arena
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    out: str
    provider: str = "mock"
    effort: str = "medium"
    n_per_session: int = 8
    passes: int = 3
    workers: int = 6
    seed: int = 0
    limit: int | None = None
    liquidity: float = LIQUIDITY
    max_width: int = MAX_WIDTH
    budget_factor: float = BUDGET_FACTOR
    first_pass_uncapped: bool = True
    snapshot_every: int = 50
    quota_check_cmd: str | None = None
    timeout: float = 1800.0
    data: str = str(PARQUET_PATH)
    mock_options: dict[str, Any] = field(default_factory=dict)

    def identity(self) -> dict[str, Any]:
        """Fields that must match for --resume (more passes may be added)."""
        return {"provider": self.provider, "effort": self.effort, "n_per_session": self.n_per_session,
                "seed": self.seed, "limit": self.limit, "liquidity": self.liquidity,
                "max_width": self.max_width, "budget_factor": self.budget_factor,
                "first_pass_uncapped": self.first_pass_uncapped, "mock_options": self.mock_options}


def session_key(pass_index: int, group_index: int) -> str:
    return f"p{pass_index}-g{group_index:03d}"


class Arena:
    """Runs P passes of N-question sessions against the market; logs JSONL."""

    def __init__(self, config: RunConfig, questions: Sequence[Question], provider: Provider,
                 grouper: Grouper | None = None, log: Callable[[str], None] | None = None) -> None:
        self.config = config
        self.questions = list(questions)
        self.by_id = {q.question_id: q for q in self.questions}
        self.provider = provider
        self.grouper = grouper or Grouper(self.questions)
        self.out = Path(config.out)
        self.snapshot_dir = self.out / "snapshots"
        self.log_path = self.out / "sessions.jsonl"
        self.say = log or (lambda message: print(message, flush=True))
        self.market: Market | None = None
        self.completed: set[str] = set()
        self.sessions_logged = 0
        self.stats = {"sessions": 0, "retries": 0, "failed": 0, "executed": 0, "rejected": 0, "proposed": 0}
        self.gate = QuotaGate(config.quota_check_cmd, log=self.say)
        self._stop = threading.Event()

    # -- persistence ---------------------------------------------------------

    def prepare(self, resume: bool = False) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(exist_ok=True)
        config_path = self.out / "config.json"
        if resume and config_path.exists():
            existing = json.loads(config_path.read_text())
            if existing.get("identity") != self.config.identity():
                raise ValueError("--resume: run configuration differs from config.json in the run directory")
        if not resume and self.log_path.exists() and self.log_path.stat().st_size > 0:
            raise ValueError(f"{self.log_path} exists; pass --resume or choose a new --out")
        config_path.write_text(json.dumps({"config": asdict(self.config), "identity": self.config.identity(),
                                           "provider": self.provider.describe(), "developer_rule": DEVELOPER_RULE,
                                           "developer_rule_sha256": sha256_text(DEVELOPER_RULE),
                                           "questions": len(self.questions), "created_at": utcnow()},
                                          indent=2, sort_keys=True) + "\n")
        questions_path = self.out / "questions.jsonl"
        if not questions_path.exists():
            set_of = {m: s.set_id for s in self.grouper.sibling_sets for m in s.members}
            with questions_path.open("w") as handle:
                for question in self.questions:
                    record = question.to_record()
                    record["sibling_set"] = set_of.get(question.question_id)
                    handle.write(json.dumps(record) + "\n")
            (self.out / "sibling-sets.json").write_text(json.dumps(
                [s.as_dict() for s in self.grouper.sibling_sets], indent=1) + "\n")
        if resume:
            self._reload()
        else:
            self.market = Market.fresh([q.question_id for q in self.questions], self.config.liquidity, self.config.max_width)

    def _reload(self) -> None:
        """Latest snapshot + replay of executed fills logged after it."""
        records = _read_jsonl(self.log_path) if self.log_path.exists() else []
        self.sessions_logged = len(records)
        for record in records:
            retry_failed = os.environ.get("MARKET_GAME_RETRY_FAILED") == "1"
            if record.get("status") == "completed" or (record.get("status") == "failed" and not retry_failed):
                self.completed.add(record["session_id"])
                self.stats["sessions"] += 1
                self.stats["failed"] += record["status"] == "failed"
                self.stats["executed"] += len(record.get("executed", []))
                self.stats["rejected"] += len(record.get("rejected", []))
                self.stats["proposed"] += record.get("proposed", 0)
            elif record.get("status") == "retry":
                self.stats["retries"] += 1
        latest, latest_lines = None, -1
        for path in self.snapshot_dir.glob("*.json"):
            data = json.loads(path.read_text())
            lines = data.get("meta", {}).get("log_lines", -1)
            if lines > latest_lines:
                latest, latest_lines = data, lines
        ids = [q.question_id for q in self.questions]
        if latest is None:
            self.market = Market.fresh(ids, self.config.liquidity, self.config.max_width)
            latest_lines = 0
        else:
            self.market = Market.from_snapshot(latest)
        replayed = 0
        for record in records[latest_lines:]:
            for fill in record.get("executed", []):
                spec = fill["trade"]
                trade = Trade(spec["variable"], spec["target"], tuple((c["variable"], c["outcome"]) for c in spec["context"]))
                pre = self.market.price(trade.variable, dict(trade.context))
                if pre is None or abs(pre - fill["pre"]) > 1e-6:
                    raise ValueError(f"--resume replay mismatch for {record['session_id']}: pre {pre} vs logged {fill['pre']}")
                fills, rejected = self.market.execute([trade], None)
                if rejected:
                    raise ValueError(f"--resume replay rejected a logged fill: {rejected[0]['reason']}")
                replayed += 1
        self.say(f"resume: {len(self.completed)} sessions done, snapshot at log line {latest_lines}, replayed {replayed} fills")

    def _append_log(self, record: Mapping[str, Any]) -> None:
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.sessions_logged += 1

    def _write_snapshot(self, name: str, pass_index: int) -> Path:
        assert self.market is not None
        with self.market.lock:
            data = self.market.snapshot({"pass": pass_index, "log_lines": self.sessions_logged,
                                         "sessions_completed": len(self.completed), "engine_stats": self.market.stats()})
        path = self.snapshot_dir / f"{name}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data))
        temporary.replace(path)
        return path

    # -- sessions ------------------------------------------------------------

    def budget(self, pass_index: int, n: int) -> float | None:
        return budget_for_pass(pass_index, n, self.config.budget_factor, self.config.first_pass_uncapped)

    def build_request(self, pass_index: int, group_index: int, ids: Sequence[str], attempt: int,
                      feedback: Sequence[str] | None = None) -> tuple[SessionRequest, dict[str, Any]]:
        assert self.market is not None
        sid = session_key(pass_index, group_index)
        questions = [self.by_id[qid] for qid in ids]
        budget = self.budget(pass_index, len(ids))
        with self.market.lock:
            view = self.market.view(ids)
        prompt = build_prompt(questions, view, budget, session_id=sid, pass_index=pass_index,
                              liquidity=self.config.liquidity, feedback=feedback)
        request = SessionRequest(sid, attempt, pass_index, group_index, prompt, DEVELOPER_RULE, list(ids),
                                 {qid: view["variables"][qid]["price"] for qid in ids}, budget)
        return request, view

    def run_session(self, pass_index: int, group_index: int, ids: Sequence[str]) -> dict[str, Any]:
        assert self.market is not None
        sid = session_key(pass_index, group_index)
        feedback: list[str] | None = None
        record: dict[str, Any] = {}
        for attempt in (1, 2):
            if self._stop.is_set():
                return {"session_id": sid, "status": "aborted"}
            self.gate.wait()
            timings: dict[str, float] = {}
            t0 = time.monotonic()
            request, view = self.build_request(pass_index, group_index, ids, attempt, feedback)
            timings["prompt_build"] = time.monotonic() - t0
            t1 = time.monotonic()
            try:
                result = self.provider.complete(request)
            except Exception as exc:  # provider crash: log and fail the attempt
                result = ProviderResult(text="", ok=False, error=f"provider_exception: {type(exc).__name__}: {exc}")
            timings["provider"] = time.monotonic() - t1
            raw_trades, parse_errors = parse_trades(result.text)
            validated = validate_trades(raw_trades or [], ids)
            valid = [v.trade for v in validated if v.trade is not None]
            invalid = [{"trade": v.raw, "reason": reason_code(v.error), "detail": v.error} for v in validated if v.error]
            traded = {t.variable for t in valid}
            missing = [qid for qid in ids if qid not in traded]
            record = {
                "type": "session", "session_id": sid, "attempt": attempt, "pass": pass_index, "group": group_index,
                "question_ids": list(ids), "n": len(ids), "prompt_sha256": sha256_text(request.prompt),
                "prompt_chars": len(request.prompt), "snapshot_version": view["version"],
                "provider": self.provider.name, "model": self.provider.model, "effort": self.provider.effort,
                "thread_id": result.thread_id, "provider_ok": result.ok, "provider_error": result.error,
                "exit_code": result.exit_code, "usage": result.usage, "raw_response": result.text,
                "parse_errors": parse_errors, "parsed_trades": raw_trades, "proposed": len(raw_trades or []),
                "invalid": invalid, "missing_questions": missing, "kl_budget": request.budget,
                "prices_at_prompt": request.prices, "started_at": utcnow(), "feedback": feedback,
            }
            if not valid:
                errors = parse_errors + [v.error for v in validated if v.error]
                if not raw_trades and not errors:
                    errors = ["no trades in the block"]
                if not result.ok and result.error:
                    errors = [result.error] + errors
                record.update(status="retry" if attempt == 1 else "failed", executed=[], rejected=invalid,
                              kl_used=0.0, errors=errors, timings=timings, finished_at=utcnow())
                with self.market.lock:
                    self._append_log(record)
                if attempt == 1:
                    with self.market.lock:
                        self.stats["retries"] += 1
                    feedback = errors[:12]
                    continue
                with self.market.lock:
                    self._finish(record)
                return record
            t2 = time.monotonic()
            with self.market.lock:
                fills, rejected = self.market.execute(valid, request.budget)
                timings["execute"] = time.monotonic() - t2
                timings["total"] = time.monotonic() - t0
                record.update(status="completed", executed=fills, rejected=invalid + rejected,
                              kl_used=fills[-1]["kl_cumulative"] if fills else 0.0, errors=[],
                              timings=timings, finished_at=utcnow(), version_after=self.market.version)
                self._append_log(record)
                self._finish(record)
            return record
        return record

    def _finish(self, record: Mapping[str, Any]) -> None:
        """Bookkeeping after a session's final attempt; caller holds the market lock."""
        self.completed.add(record["session_id"])
        self.stats["sessions"] += 1
        self.stats["failed"] += record["status"] == "failed"
        self.stats["executed"] += len(record.get("executed", []))
        self.stats["rejected"] += len(record.get("rejected", []))
        self.stats["proposed"] += record.get("proposed", 0)
        if self.config.snapshot_every and self.stats["sessions"] % self.config.snapshot_every == 0:
            self._write_snapshot("latest", record["pass"])

    # -- passes --------------------------------------------------------------

    def groups_for_pass(self, pass_index: int) -> list[list[str]]:
        return self.grouper.groups_for_pass(pass_index, self.config.n_per_session, self.config.seed)

    def run(self) -> dict[str, Any]:
        assert self.market is not None, "call prepare() first"
        started = time.monotonic()
        groups_path = self.out / "groups.json"
        all_groups = json.loads(groups_path.read_text()) if groups_path.exists() else {}
        for pass_index in range(1, self.config.passes + 1):
            groups = self.groups_for_pass(pass_index)
            all_groups[str(pass_index)] = groups
            groups_path.write_text(json.dumps(all_groups) + "\n")
            pending = [(g, ids) for g, ids in enumerate(groups) if session_key(pass_index, g) not in self.completed]
            self.say(f"pass {pass_index}: {len(groups)} sessions, {len(pending)} pending, "
                     f"budget {self.budget(pass_index, self.config.n_per_session)}")
            pass_started = time.monotonic()
            with ThreadPoolExecutor(max_workers=self.config.workers) as pool:
                futures = {pool.submit(self.run_session, pass_index, g, ids): g for g, ids in pending}
                done = 0
                for future in as_completed(futures):
                    record = future.result()
                    done += 1
                    if done % 25 == 0 or done == len(pending):
                        self.say(f"  pass {pass_index}: {done}/{len(pending)} sessions "
                                 f"({time.monotonic() - pass_started:.0f}s), last {record.get('session_id')} "
                                 f"{record.get('status')}")
            self._write_snapshot(f"pass-{pass_index}", pass_index)
            self.say(f"pass {pass_index} done in {time.monotonic() - pass_started:.1f}s; "
                     f"engine {self.market.stats()}")
        summary = {"config": asdict(self.config), "provider": self.provider.describe(), "stats": dict(self.stats),
                   "elapsed_seconds": round(time.monotonic() - started, 3), "sessions_logged": self.sessions_logged,
                   "engine_stats": self.market.stats(), "quota_pauses": self.gate.pauses, "finished_at": utcnow()}
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary


def select_questions(questions: Sequence[Question], limit: int | None, seed: int) -> list[Question]:
    """The full set, or a seeded random subset for smoke runs."""
    if limit is None or limit >= len(questions):
        return list(questions)
    rng = random.Random(f"{seed}:limit:{limit}")
    chosen = sorted(rng.sample(range(len(questions)), limit))
    return [questions[i] for i in chosen]
