"""Auditable supervised training data for Boolean proposition pairs.

The September pilot accidentally recovered the twelve expressions by sorting
query IDs and then applied relation penalties to positions 0--3.  This module
keeps the proposition text and builds a named form map for every pair.  Any
missing, duplicated, or mismatched form fails before a batch reaches a model.
It deliberately contains no reinforcement-learning code.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
import re
from pathlib import Path


FORM_ORDER = (
    "P", "NOT P", "Q", "NOT Q",
    "(P) AND (Q)", "(P) OR (Q)",
    "(P) AND (NOT Q)", "(P) OR (NOT Q)",
    "(NOT P) AND (Q)", "(NOT P) OR (Q)",
    "(NOT P) AND (NOT Q)", "(NOT P) OR (NOT Q)",
)
FORMS = frozenset(FORM_ORDER)


@dataclass(frozen=True)
class PairExample:
    pair: dict
    queries: dict[str, dict]
    propositions: dict[str, dict]
    labels: dict[str, int]

    @property
    def relation_map(self) -> dict[str, str]:
        """Map a semantic form name to its query ID, never by row position."""
        return {form: self.queries[form]["query_id"] for form in FORM_ORDER}

    @property
    def labeled_forms(self) -> dict[str, int]:
        """Forms with a supervised label in this example's label set, by name."""
        return {form: int(self.labels[self.queries[form]["query_id"]])
                for form in FORM_ORDER if self.queries[form]["query_id"] in self.labels}

    @property
    def observed_form(self) -> str:
        observed = self.pair["observed_query_id"]
        return next(form for form in FORM_ORDER if self.queries[form]["query_id"] == observed)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


MAX_CRITERIA_CHARS = 1500
"""Criteria text is cut at this many characters per proposition.

Validation-week markets carry criteria several times longer than train/test
(median ~2,750 vs ~520 characters).  A fixed cap keeps every prompt under the
token budget so the trailing ``Answer [YES/NO]:`` readout is never truncated
away, and applies identically to every split and arm.
"""


EDIT_PARAGRAPH = re.compile(r"^\s*(update\s+20\d\d|edit\s*[:\-(]|edit\b)", re.IGNORECASE)
INLINE_EDIT = re.compile(r"[Uu]pdate\s+20\d\d-\d\d-\d\d|\b(?:EDIT|Edit)(?:\s+[A-Z][a-z]{2,8}\.?\s+\d{1,2}(?:,\s*\d{4})?)?\s*:")


def strip_creator_edits(criteria: str) -> str:
    """Drop creator edits added after a market opened ("Update 2025-..", "Edit:").

    They can state or hint at the outcome; historical_text_verified is False for
    every row, so these are removed rather than trusted.
    """
    kept = []
    for para in re.split(r"\n\s*\n", criteria):
        if EDIT_PARAGRAPH.match(para):
            continue
        cut = INLINE_EDIT.search(para)
        if cut:
            para = para[:cut.start()].rstrip()
            if not re.sub(r"\[[a-z ]+\]", "", para).strip():
                continue
        kept.append(para)
    return "\n\n".join(kept).strip()


def proposition_text(row: dict, *, max_criteria_chars: int = MAX_CRITERIA_CHARS,
                     criteria_override: str | None = None) -> str:
    """Render one proposition as a self-contained yes/no question.

    For multiple-choice markets ``answer_text`` is the option name and defines
    the proposition ("Which lab releases an LLM in October?" / option "OpenAI").
    It is not a label. Before 2026-09-23 this function dropped it, leaving about
    79% of propositions undefined; runs before that date used undefined prompts.
    Creator edits are stripped from the criteria.  ``criteria_override`` (the
    as-of text from a market-context dataset) is used verbatim instead: it is
    the text as it stood at the forecast time, so its edits predate that time.
    """
    question = str(row.get("parent_question", "")).strip()
    option = str(row.get("answer_text") or "").strip()
    if criteria_override is not None:
        criteria = criteria_override.strip()
    else:
        criteria = strip_creator_edits(str(row.get("criteria_text", "")).strip())
    if not question and not criteria:
        raise ValueError(f"proposition {row.get('id')} has no text")
    if len(criteria) > max_criteria_chars:
        criteria = criteria[:max_criteria_chars].rstrip() + " [...]"
    lines = [f"Question: {question}"]
    if option:
        lines += [f"Option: {option}", "This proposition is true if the market resolves YES for this option."]
    else:
        lines.append("This proposition is true if the market resolves YES.")
    if row.get("closing_week"):
        lines.append(f"Closes in the week of {row['closing_week']}.")
    if criteria:
        lines.append(f"Resolution criteria:\n{criteria}")
    return "\n".join(lines)


def _expected_signs(form: str) -> tuple[int, int]:
    return {
        "P": (1, 0), "NOT P": (-1, 0), "Q": (0, 1), "NOT Q": (0, -1),
        "(P) AND (Q)": (1, 1), "(P) OR (Q)": (1, 1),
        "(P) AND (NOT Q)": (1, -1), "(P) OR (NOT Q)": (1, -1),
        "(NOT P) AND (Q)": (-1, 1), "(NOT P) OR (Q)": (-1, 1),
        "(NOT P) AND (NOT Q)": (-1, -1), "(NOT P) OR (NOT Q)": (-1, -1),
    }[form]


def validate_pair(pair: dict, query_rows: list[dict], labels: dict[str, int], propositions: dict[str, dict],
                  *, require_observed: bool = True) -> PairExample:
    if not isinstance(pair.get("pair_id"), str):
        raise ValueError("pair_id must be a string")
    rows = [row for row in query_rows if row.get("pair_id") == pair["pair_id"]]
    if len(rows) != len(FORM_ORDER):
        raise ValueError(f"{pair['pair_id']} has {len(rows)} queries, expected {len(FORM_ORDER)}")
    by_form: dict[str, dict] = {}
    for row in rows:
        form = row.get("form")
        if form not in FORMS or form in by_form:
            raise ValueError(f"{pair['pair_id']} has duplicate/unknown form {form!r}")
        if row.get("p_id") != pair.get("p_id") or row.get("q_id") != pair.get("q_id"):
            raise ValueError(f"{pair['pair_id']} query proposition IDs disagree with pair")
        if (row.get("p_sign"), row.get("q_sign")) != _expected_signs(form):
            raise ValueError(f"{pair['pair_id']} has incorrect signs for {form}")
        by_form[form] = row
    if set(by_form) != FORMS:
        raise ValueError(f"{pair['pair_id']} does not contain the complete Boolean form set")
    for ident in (pair.get("p_id"), pair.get("q_id")):
        if ident not in propositions:
            raise ValueError(f"{pair['pair_id']} references missing proposition {ident}")
        # Validate before the prompt is rendered; avoid materialising the full
        # (often long) prompt once per pair during archive validation.
        source = propositions[ident]
        if not str(source.get("parent_question", "")).strip() and not str(source.get("criteria_text", "")).strip():
            raise ValueError(f"proposition {ident} has no text")
    # One-label training files contain only the observed query; full-label files
    # all twelve; singles files label only P and Q.  The observed label is
    # required for one-label data; other label sets must label at least one form.
    observed = pair.get("observed_query_id")
    query_ids = [row["query_id"] for row in by_form.values()]
    if observed not in set(query_ids):
        raise ValueError(f"{pair['pair_id']} has an unknown observed query")
    if require_observed and observed not in labels:
        raise ValueError(f"missing observed label for {observed}")
    if not any(query_id in labels for query_id in query_ids):
        raise ValueError(f"{pair['pair_id']} has no labeled form")
    for query_id in query_ids:
        if query_id in labels and labels[query_id] not in (0, 1):
            raise ValueError(f"non-binary label for {query_id}")
    return PairExample(pair=pair, queries=by_form, propositions=propositions, labels=labels)


LABEL_FILES = {
    "one": "one-label-targets.jsonl",
    "full": "full-label-targets.jsonl",
    "single": "single-targets.jsonl",
}


def _query_labels(label_rows: list[dict], queries: list[dict], labels_file: str) -> dict[str, int]:
    """Key labels by query ID.

    ``single-targets.jsonl`` is keyed by proposition ID and labels only the
    literal forms ``P`` and ``Q``; negations and compounds stay unlabeled so a
    singles arm cannot see them.
    """
    if not label_rows:
        raise ValueError(f"{labels_file} is empty")
    if all("query_id" in row for row in label_rows):
        return {row["query_id"]: row["label"] for row in label_rows}
    if not all("id" in row for row in label_rows):
        raise ValueError(f"{labels_file} mixes query and proposition keys")
    by_proposition = {row["id"]: row["label"] for row in label_rows}
    labels: dict[str, int] = {}
    for row in queries:
        if row.get("form") == "P" and row.get("p_id") in by_proposition:
            labels[row["query_id"]] = by_proposition[row["p_id"]]
        elif row.get("form") == "Q" and row.get("q_id") in by_proposition:
            labels[row["query_id"]] = by_proposition[row["q_id"]]
    return labels


def load_split(root: str | Path, split: str, *, labels_file: str = "one-label-targets.jsonl") -> list[PairExample]:
    root = Path(root)
    directory = root / split
    propositions = {row["id"]: row for row in _jsonl(directory / "propositions.jsonl")}
    pairs = _jsonl(directory / "pairs.jsonl")
    queries = _jsonl(directory / "queries.jsonl")
    labels = _query_labels(_jsonl(directory / labels_file), queries, labels_file)
    if any(label not in (0, 1) for label in labels.values()):
        raise ValueError(f"non-binary label in {labels_file}")
    require_observed = labels_file == LABEL_FILES["one"]
    queries_by_pair: dict[str, list[dict]] = {}
    for row in queries:
        queries_by_pair.setdefault(row.get("pair_id"), []).append(row)
    result = [validate_pair(pair, queries_by_pair.get(pair.get("pair_id"), []), labels, propositions,
                            require_observed=require_observed) for pair in pairs]
    if len({example.pair["pair_id"] for example in result}) != len(result):
        raise ValueError(f"duplicate pair IDs in {split}")
    prompts_path = directory / "prompts.jsonl"
    if prompts_path.exists():
        # Precomputed prompts (e.g. with an evidence pack): one shared prefix per
        # pair and one suffix per form; render_prompt returns prefix + suffix.
        prompts = {row["pair_id"]: row for row in _jsonl(prompts_path)}
        for example in result:
            row = prompts.get(example.pair["pair_id"])
            if row is None or set(row["suffix"]) != FORMS:
                raise ValueError(f"prompts.jsonl lacks a complete prompt set for {example.pair['pair_id']}")
            example.pair["_prefix"] = row["prefix"]
            example.pair["_suffix"] = row["suffix"]
    return result


def prompt_parts(example: PairExample, form: str, context: str = "none") -> tuple[str, str]:
    """(shared prefix, form suffix); prefix + suffix == render_prompt(example, form)."""
    if form not in FORMS:
        raise ValueError(f"unknown form {form}")
    if "_prefix" in example.pair:
        return example.pair["_prefix"], example.pair["_suffix"][form]
    full = render_prompt(example, form, context)
    cut = full.index("Statement to assess: ")
    return full[:cut], full[cut:]


def _statement(form: str) -> str:
    word = {"P": "Question A resolves YES", "NOT P": "Question A resolves NO",
            "Q": "Question B resolves YES", "NOT Q": "Question B resolves NO"}
    if form in word:
        return word[form] + "."
    left, op, right = re.fullmatch(r"\((.+)\) (AND|OR) \((.+)\)", form).groups()
    r = word[right]
    lead = "Both of the following are true" if op == "AND" else "At least one of the following is true"
    return f"{lead}: (1) {word[left]}; (2) {r}."


def question_text(row: dict, context: str = "none") -> str:
    """One question block; with a market context mode, the as-of text plus market lines."""
    from training.market_context import asof_row, context_block
    shown, criteria = asof_row(row, context)
    text = proposition_text(shown, criteria_override=criteria)
    extra = context_block(row["context"], context) if context != "none" else ""
    return f"{text}\n{extra}" if extra else text


def render_prompt(example: PairExample, form: str, context: str = "none") -> str:
    """Render both propositions and the form as a plain-language statement.

    Both questions are shown for every form so the twelve prompts of a pair share
    a prefix; the as-of date is one week before the earlier closing week.
    ``context`` ("none", "asof_text", "prices", "comments", "prices+comments")
    adds as-of market lines under each question; it needs a market-context
    dataset (propositions with a ``context`` object).  "none" is the old prompt.
    """
    import datetime as _dt
    if form not in FORMS:
        raise ValueError(f"unknown form {form}")
    if "_prefix" in example.pair:
        return example.pair["_prefix"] + example.pair["_suffix"][form]
    rp = example.propositions[example.pair["p_id"]]
    rq = example.propositions[example.pair["q_id"]]
    weeks = [w for w in (rp.get("closing_week"), rq.get("closing_week")) if w]
    head = ""
    if weeks:
        as_of = (_dt.date.fromisoformat(min(weeks)) - _dt.timedelta(days=7)).isoformat()
        head = (f"Assume today is {as_of}. Use only information available up to that date; "
                "do not use knowledge of later events.\n\n")
    return (f"{head}Question A\n{question_text(rp, context)}\n\nQuestion B\n{question_text(rq, context)}\n\n"
            f"Statement to assess: {_statement(form)}\n"
            "Is this statement true? Answer [YES/NO]:")


def _relu(value):
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return torch.relu(value)
    except ImportError:  # pragma: no cover
        pass
    return max(0.0, float(value))


def _mean(values):
    if not values:
        raise ValueError("empty relation penalty")
    result = values[0]
    for value in values[1:]:
        result = result + value
    return result / len(values)


def relation_penalties(probabilities: dict[str, object]) -> dict[str, object]:
    """Return named Boolean constraints for probabilities in [0,1].

    Tensor inputs remain differentiable.  The four conjunction/disjunction
    pairs use Fréchet bounds; the four De Morgan complements and two literal
    complements are explicit.  This is a conservative supervised consistency
    loss, not a reward or RL objective.
    """
    missing = FORMS - set(probabilities)
    if missing:
        raise ValueError(f"missing probabilities for {sorted(missing)}")
    p, np_, q, nq = (probabilities[name] for name in ("P", "NOT P", "Q", "NOT Q"))
    terms = {
        "P_complement": (p + np_ - 1) ** 2,
        "Q_complement": (q + nq - 1) ** 2,
    }
    for conj, left, right in (
        ("(P) AND (Q)", p, q), ("(P) AND (NOT Q)", p, nq),
        ("(NOT P) AND (Q)", np_, q), ("(NOT P) AND (NOT Q)", np_, nq),
    ):
        disj = {
            "(P) AND (Q)": "(NOT P) OR (NOT Q)",
            "(P) AND (NOT Q)": "(NOT P) OR (Q)",
            "(NOT P) AND (Q)": "(P) OR (NOT Q)",
            "(NOT P) AND (NOT Q)": "(P) OR (Q)",
        }[conj]
        value = probabilities[conj]
        other = probabilities[disj]
        terms[f"{conj}_demorgan"] = (value + other - 1) ** 2
        terms[f"{conj}_lower"] = _relu(left + right - 1 - value) ** 2
        terms[f"{conj}_upper_left"] = _relu(value - left) ** 2
        terms[f"{conj}_upper_right"] = _relu(value - right) ** 2
    for disj, left, right in (
        ("(P) OR (Q)", p, q), ("(P) OR (NOT Q)", p, nq),
        ("(NOT P) OR (Q)", np_, q), ("(NOT P) OR (NOT Q)", np_, nq),
    ):
        value = probabilities[disj]
        terms[f"{disj}_lower_left"] = _relu(left - value) ** 2
        terms[f"{disj}_lower_right"] = _relu(right - value) ** 2
        terms[f"{disj}_upper"] = _relu(value - left - right) ** 2
    terms["total"] = _mean(list(terms.values()))
    return terms


FORM_PRIOR = {form: (0.5 if "AND" not in form and "OR" not in form else 0.25 if "AND" in form else 0.75)
              for form in FORM_ORDER}
"""Boolean-structure prior: 0.5 for a literal, 0.25 for an AND, 0.75 for an OR."""

PERMUTATION_SEED = 20260923


def form_permutation(seed: int = PERMUTATION_SEED) -> dict[str, str]:
    """A fixed random relabelling of the twelve form names (control arm).

    ``permutation[form]`` is the form whose probability fills the ``form`` slot
    when the Boolean constraints are evaluated.
    """
    shuffled = list(FORM_ORDER)
    random.Random(seed).shuffle(shuffled)
    return dict(zip(FORM_ORDER, shuffled))


def prior_pull_penalty(probabilities: dict[str, object]) -> object:
    """Mean over the twelve forms of (p_form - prior_form)^2 (shrinkage control).

    Pulls every form toward its Boolean-structure prior without any relation
    between forms; zero exactly at the prior.
    """
    missing = FORMS - set(probabilities)
    if missing:
        raise ValueError(f"missing probabilities for {sorted(missing)}")
    return _mean([(probabilities[form] - FORM_PRIOR[form]) ** 2 for form in FORM_ORDER])


def permuted_relation_penalties(probabilities: dict[str, object],
                                permutation: dict[str, str] | None = None) -> dict[str, object]:
    """``relation_penalties`` applied after a fixed permutation of form names.

    Same functional form and scale as the real penalty but logically
    meaningless: it asks, e.g., p(Q) + p((P) OR (NOT Q)) = 1.
    """
    permutation = permutation or form_permutation()
    missing = FORMS - set(probabilities)
    if missing:
        raise ValueError(f"missing probabilities for {sorted(missing)}")
    return relation_penalties({form: probabilities[permutation[form]] for form in FORM_ORDER})


def probability_metrics(probabilities: list[float], labels: list[int], *, bins: int = 10) -> dict[str, float]:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("probabilities and labels must be equally nonempty")
    eps = 1e-7
    log_loss = -sum(y * math.log(max(eps, min(1 - eps, p))) + (1 - y) * math.log(max(eps, min(1 - eps, 1 - p)))
                    for p, y in zip(probabilities, labels, strict=True)) / len(labels)
    brier = sum((p - y) ** 2 for p, y in zip(probabilities, labels, strict=True)) / len(labels)
    accuracy = sum((p >= 0.5) == bool(y) for p, y in zip(probabilities, labels, strict=True)) / len(labels)
    ece = 0.0
    for index in range(bins):
        rows = [(p, y) for p, y in zip(probabilities, labels, strict=True) if index / bins <= p < (index + 1) / bins or (index == bins - 1 and p == 1)]
        if rows:
            ece += len(rows) / len(labels) * abs(sum(p for p, _ in rows) / len(rows) - sum(y for _, y in rows) / len(rows))
    return {"log_loss": log_loss, "brier": brier, "accuracy": accuracy, "ece": ece}
