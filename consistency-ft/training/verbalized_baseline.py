"""Zero-shot verbalized-probability baseline for the twelve Boolean forms.

The model reads two questions (A = P, B = Q), is asked about one form, and must
end its reply with ``Probability: 0.xx``.  No training, no logits: the number
the model writes is the forecast.

Proposition text differs from ``supervised_pairs.proposition_text`` on purpose:

* ``answer_text`` is included.  For multiple-choice markets (proposition ID
  contains ``:``) it is the option name, and without it the proposition is
  undefined ("Which lab releases an LLM in October?" with no option).  The
  20 September 0.6B runs dropped it; that was a defect, not a leak guard.
* Creator edits appended after the market opened ("Update 2025-..", "Edit:",
  "EDIT") are removed paragraph by paragraph before the 1,500-character cap,
  because they can reveal the outcome.
* An as-of frame (closing week minus seven days) sets the tense.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from pathlib import Path

from training.supervised_pairs import FORM_ORDER, MAX_CRITERIA_CHARS, PairExample

EDIT_PARAGRAPH = re.compile(r"^\s*(update\s+20\d\d|edit\s*[:\-(]|edit\b)", re.IGNORECASE)
INLINE_EDIT = re.compile(r"[Uu]pdate\s+20\d\d-\d\d-\d\d|\b(?:EDIT|Edit)(?:\s+[A-Z][a-z]{2,8}\.?\s+\d{1,2}(?:,\s*\d{4})?)?\s*:")


def strip_creator_edits(criteria: str) -> tuple[str, int]:
    """Drop paragraphs that start with a creator edit marker; return text and count removed."""
    paragraphs = re.split(r"\n\s*\n", criteria)
    kept, removed = [], 0
    for p in paragraphs:
        if EDIT_PARAGRAPH.match(p):
            removed += 1
            continue
        # Edits embedded after a placeholder ("[image]Update 2025-..") or inline
        # ("... Edit Jun 16: ..."): cut from the marker to the paragraph end.
        cut = INLINE_EDIT.search(p)
        if cut:
            removed += 1
            p = p[:cut.start()].rstrip()
            if not re.sub(r"\[[a-z ]+\]", "", p).strip():  # nothing left but "[image]" etc.
                continue
        kept.append(p)
    return "\n\n".join(kept).strip(), removed


def as_of_date(closing_week: str) -> str:
    return (dt.date.fromisoformat(closing_week) - dt.timedelta(days=7)).isoformat()


def question_block(row: dict, *, max_criteria_chars: int = MAX_CRITERIA_CHARS, context: str = "none") -> str:
    """Render one proposition as a self-contained yes/no question.

    ``context`` other than "none" uses the as-of question and criteria text
    from the proposition's market context and appends the as-of market lines
    (price history and/or comments); see ``training.market_context``.
    """
    if context != "none":
        from training.market_context import asof_row, context_block
        shown, criteria = asof_row(row, context)
        text = _question_lines(shown, criteria.strip(), max_criteria_chars)
        extra = context_block(row["context"], context)
        return f"{text}\n{extra}" if extra else text
    criteria, _ = strip_creator_edits(str(row.get("criteria_text") or ""))
    return _question_lines(row, criteria, max_criteria_chars)


def _question_lines(row: dict, criteria: str, max_criteria_chars: int) -> str:
    question = str(row.get("parent_question") or "").strip()
    option = str(row.get("answer_text") or "").strip()
    if len(criteria) > max_criteria_chars:
        criteria = criteria[:max_criteria_chars].rstrip() + " [...]"
    lines = [f"Question: {question}"]
    if option:
        lines.append(f"Option: {option}")
        lines.append("This proposition is true if the market resolves YES for this option.")
    else:
        lines.append("This proposition is true if the market resolves YES.")
    lines.append(f"Closes in the week of {row['closing_week']}.")
    if criteria:
        lines.append(f"Resolution criteria:\n{criteria}")
    return "\n".join(lines)


# --- form statements -------------------------------------------------------

_SIGN_WORD = {1: "YES", -1: "NO"}


def _parts(form: str) -> tuple[str | None, int, int]:
    """(operator, sign of A, sign of B) for a named form; sign 0 = absent."""
    literal = {"P": (None, 1, 0), "NOT P": (None, -1, 0), "Q": (None, 0, 1), "NOT Q": (None, 0, -1)}
    if form in literal:
        return literal[form]
    left, op, right = re.fullmatch(r"\((NOT )?P\) (AND|OR) \((NOT )?Q\)", form).groups()
    return op, -1 if left else 1, -1 if right else 1


def symbolic_statement(form: str) -> str:
    text = form.replace("P", "A").replace("Q", "B")
    return f"{text}   (A means 'Question A resolves YES', B means 'Question B resolves YES')"


def plain_statement(form: str) -> str:
    op, a, b = _parts(form)
    if op is None:
        name, sign = ("A", a) if a else ("B", b)
        return f"Question {name} resolves {_SIGN_WORD[sign]}."
    one, two = f"Question A resolves {_SIGN_WORD[a]}", f"Question B resolves {_SIGN_WORD[b]}"
    if op == "AND":
        return f"Both of the following are true: (1) {one}; (2) {two}."
    return f"At least one of the following is true: (1) {one}; (2) {two}."


# --- v7_explicit (2026-09-29) ---------------------------------------------
# v6 put "This proposition is true if the market resolves YES." in every question
# block and stated A / not A as "Question A resolves YES." / "... resolves NO.", so
# the two prompts differed by one word at the end.  v7 describes each question
# neutrally, restates the proposition inside the statement in plain words, and
# spells the negation out in capitals.  Placeholder-only or empty criteria say so.

CRITERIA_NA = "Resolution criteria: not available (see the question)."
PLACEHOLDER = re.compile(r"\[(?:image|video|embed|link|tweet)\]", re.IGNORECASE)


def _clean_criteria(criteria: str) -> str:
    text = PLACEHOLDER.sub("", criteria).strip()
    return text


def question_block_v7(row: dict, name: str, *, max_criteria_chars: int = MAX_CRITERIA_CHARS) -> str:
    question = str(row.get("parent_question") or "").strip()
    option = str(row.get("answer_text") or "").strip()
    criteria, _ = strip_creator_edits(str(row.get("criteria_text") or ""))
    criteria = _clean_criteria(criteria)
    if len(criteria) > max_criteria_chars:
        criteria = criteria[:max_criteria_chars].rstrip() + " [...]"
    if option:
        lines = [f"Question {name} (multiple choice; each option either happens or does not): {question}",
                 f"Option under consideration: {option}"]
    else:
        lines = [f"Question {name} (yes/no): {question}"]
    lines.append(f"Closes in the week of {row['closing_week']}.")
    lines.append(f"Resolution criteria:\n{criteria}" if criteria else CRITERIA_NA)
    return "\n".join(lines)


def explicit_clause(row: dict, name: str, sign: int) -> str:
    """One literal in plain words; the negation is written out in capitals."""
    question = str(row.get("parent_question") or "").strip()
    option = str(row.get("answer_text") or "").strip()
    if option:
        if sign > 0:
            return f'the option "{option}" of Question {name} WILL happen (it resolves YES)'
        return f'the option "{option}" of Question {name} will NOT happen (it resolves NO)'
    if sign > 0:
        return f'the answer to Question {name} ("{question}") is YES'
    return f'the answer to Question {name} ("{question}") is NOT yes: it is NO'


def _truth_note(row: dict, sign: int = -1) -> str:
    """v7b/v7c: say when a literal statement is true and when it is false."""
    option = str(row.get("answer_text") or "").strip()
    if option:
        if sign > 0:
            return f' This statement is TRUE if "{option}" happens, and FALSE if "{option}" does NOT happen.'
        return f' This statement is TRUE if "{option}" does NOT happen, and FALSE if "{option}" happens.'
    if sign > 0:
        return " This statement is TRUE if the answer is YES, and FALSE if the answer is NO."
    return " This statement is TRUE if the answer is NO, and FALSE if the answer is YES."


def explicit_statement(form: str, p_row: dict, q_row: dict, *, negation_note: bool = False,
                       both_notes: bool = False) -> str:
    op, a, b = _parts(form)
    if op is None:
        row, name, sign = (p_row, "A", a) if a else (q_row, "B", b)
        c = explicit_clause(row, name, sign)
        text = c[0].upper() + c[1:] + "."
        if (negation_note or both_notes) and sign < 0:
            return "(a NEGATION) " + text + _truth_note(row, sign)
        if both_notes:
            return text + _truth_note(row, sign)
        return text
    one, two = explicit_clause(p_row, "A", a), explicit_clause(q_row, "B", b)
    if op == "AND":
        return f"BOTH of these are true: (1) {one}; AND (2) {two}."
    return f"AT LEAST ONE of these is true (possibly both): (1) {one}; OR (2) {two}."


# --- prompt variants -------------------------------------------------------

BASE_SYSTEM = "You are a careful forecaster. You estimate probabilities of future events."
SUPER_SYSTEM = ("You are an expert superforecaster with an excellent calibration record. You start from base rates, "
                "adjust for the specifics, avoid overconfidence, and never give 0 or 1.")
ANSWER_WITH_REASONING = ("Think briefly (at most five sentences), then end your reply with a final line of exactly this form:\n"
                         "Probability: 0.xx\n(a single number between 0.01 and 0.99, nothing after it).")
ANSWER_DIRECT = ("Reply with exactly one line of this form and nothing else:\nProbability: 0.xx\n"
                 "(a single number between 0.01 and 0.99).")
LOGIC_HINT = ("Work it out step by step: first estimate the probability that Question A resolves YES and the probability "
              "that Question B resolves YES. Then combine them for the statement: 'resolves NO' is 1 minus 'resolves YES'; "
              "a 'both' statement can be no more likely than either part; an 'at least one' statement is at least as likely "
              "as either part and at most their sum; if the two questions look unrelated, treat them as independent.")

VARIANTS = {
    # name: (system, statement style, show both questions for literal forms, extra instruction, answer instruction)
    "v1_plain_symbolic": (BASE_SYSTEM, "symbolic", False, None, ANSWER_WITH_REASONING),
    "v2_plain_language": (BASE_SYSTEM, "plain", False, None, ANSWER_WITH_REASONING),
    "v3_superforecaster_baserate": (SUPER_SYSTEM, "plain", False, "BASE_RATE", ANSWER_WITH_REASONING),
    "v4_logic_steps": (BASE_SYSTEM, "plain", True, LOGIC_HINT, ANSWER_WITH_REASONING),
    "v5_pair_context": (BASE_SYSTEM, "plain", True, None, ANSWER_WITH_REASONING),
    "v6_direct": (BASE_SYSTEM, "plain", False, None, ANSWER_DIRECT),
    "v7_explicit": (BASE_SYSTEM, "explicit", False, None, ANSWER_DIRECT),
    # validation iteration 1: v7 plus "(a NEGATION) ... TRUE if ... does NOT happen, FALSE if ..." on negated literals
    "v7b_explicit": (BASE_SYSTEM, "explicit_note", False, None, ANSWER_DIRECT),
    # validation iteration 2: the TRUE/FALSE sentence on positive literals too (same shape for A and not A)
    "v7c_explicit": (BASE_SYSTEM, "explicit_both", False, None, ANSWER_DIRECT),
}


def base_rate_text(rates: dict[str, float]) -> str:
    return (f"Base rate: in comparable past questions from this forecasting site, about {rates['binary']:.0%} of yes/no "
            f"questions and about {rates['option']:.0%} of individual multiple-choice options resolved YES. "
            "Start from that and adjust for the specifics.")


def train_base_rates(root: str | Path) -> dict[str, float]:
    """Literal-P YES rates on the TRAIN split only, by binary vs option propositions."""
    root = Path(root) / "train"
    queries = {r["query_id"]: r for r in map(json.loads, (root / "queries.jsonl").read_text().splitlines())}
    counts = {"binary": [0, 0], "option": [0, 0]}
    for row in map(json.loads, (root / "full-label-targets.jsonl").read_text().splitlines()):
        q = queries[row["query_id"]]
        if q["form"] == "P":
            kind = "option" if ":" in q["p_id"] else "binary"
            counts[kind][0] += row["label"]; counts[kind][1] += 1
    return {k: yes / n for k, (yes, n) in counts.items()}


def user_prompt(example: PairExample, form: str, variant: str, *, rates: dict[str, float] | None = None,
                context: str = "none") -> str:
    _, style, both, extra, answer = VARIANTS[variant]
    p_row = example.propositions[example.pair["p_id"]]
    q_row = example.propositions[example.pair["q_id"]]
    op, a, b = _parts(form)
    show_a, show_b = (True, True) if (both or op) else (bool(a), bool(b))
    # One "today" per pair (the earlier closing week) so all twelve forms share the same frame.
    today = as_of_date(min(p_row["closing_week"], q_row["closing_week"]))
    parts = [f"Assume today is {today}. Use only information available up to that date; do not use knowledge of later events."]
    if extra == "BASE_RATE":
        parts.append(base_rate_text(rates or {}))
    if style in ("explicit", "explicit_note", "explicit_both"):
        if context != "none":
            raise ValueError("v7 variants are question-only")
        if show_a:
            parts.append(question_block_v7(p_row, "A"))
        if show_b:
            parts.append(question_block_v7(q_row, "B"))
        statement = explicit_statement(form, p_row, q_row, negation_note=style == "explicit_note",
                                       both_notes=style == "explicit_both")
        parts.append(f"Statement: {statement}\nWhat is the probability that this statement is true?")
    else:
        if show_a:
            parts.append("Question A\n" + question_block(p_row, context=context))
        if show_b:
            parts.append("Question B\n" + question_block(q_row, context=context))
        if style == "symbolic":
            statement = symbolic_statement(form)
        else:
            statement = plain_statement(form)
        parts.append(f"Statement to assess: {statement}\nWhat is the probability that this statement is true?")
    if extra and extra != "BASE_RATE":
        parts.append(extra)
    parts.append(answer)
    return "\n\n".join(parts) + " /no_think"


def messages(example: PairExample, form: str, variant: str, *, rates=None, context: str = "none") -> list[dict]:
    return [{"role": "system", "content": VARIANTS[variant][0]},
            {"role": "user", "content": user_prompt(example, form, variant, rates=rates, context=context)}]


RETRY_REMINDER = ("Your reply did not end with a valid line of the form 'Probability: 0.xx'. "
                  "Reply with exactly one line, 'Probability: ' followed by a single number between 0.01 and 0.99, and nothing else.")


# --- parser ----------------------------------------------------------------

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
PROB_LINE = re.compile(
    r"probability\s*\**\s*(?:[:=]|is)\s*\**\s*(?:~|≈|about|approximately|approx\.?)?\s*"
    r"(?P<num>\d*\.?\d+)(?![\d.]*[A-Za-z_])(?P<frac>\s*/\s*\d+)?\s*(?P<pct>%|percent)?(?P<tail>\s*(?:-|–|to)\s*\d*\.?\d+\s*%?)?",
    re.IGNORECASE)


def parse_probability(text: str) -> tuple[float | None, str]:
    """Return (probability, reason).  The LAST 'Probability:' match wins.

    Rules: thinking blocks are removed; an unclosed <think> is unparseable;
    a range ("60-70%") is unparseable; a number > 1 without '%' is read as a
    percent only if it is <= 100; anything outside [0, 1] is unparseable.
    No default is ever returned.
    """
    if text is None:
        return None, "empty"
    body = THINK_BLOCK.sub("", text)
    if re.search(r"<think>", body, re.IGNORECASE):
        return None, "unclosed_think"
    matches = list(PROB_LINE.finditer(body))
    if not matches:
        return None, "no_probability_line"
    m = matches[-1]
    if m.group("tail"):
        return None, "range"
    if m.group("frac"):
        return None, "fraction"
    value = float(m.group("num"))
    if m.group("pct"):
        value /= 100
    elif value > 1:
        if value <= 100:
            value /= 100
        else:
            return None, "out_of_range"
    if not 0 <= value <= 1:
        return None, "out_of_range"
    return value, "ok"


# --- scoring ---------------------------------------------------------------

def clip(p: float, lo: float = 0.01) -> float:
    return min(1 - lo, max(lo, p))


def metrics(ps: list[float], ys: list[int], *, bins: int = 10) -> dict:
    n = len(ys)
    if not n:
        return {"n": 0}
    ll = -sum(y * math.log(p) + (1 - y) * math.log(1 - p) for p, y in zip(ps, ys)) / n
    brier = sum((p - y) ** 2 for p, y in zip(ps, ys)) / n
    acc = sum((p >= 0.5) == bool(y) for p, y in zip(ps, ys)) / n
    ece = 0.0; reliability = []
    for i in range(bins):
        rows = [(p, y) for p, y in zip(ps, ys) if i / bins <= p < (i + 1) / bins or (i == bins - 1 and p == 1)]
        if rows:
            mp = sum(p for p, _ in rows) / len(rows); my = sum(y for _, y in rows) / len(rows)
            ece += len(rows) / n * abs(mp - my)
            reliability.append({"bin": i, "n": len(rows), "mean_p": mp, "freq_yes": my})
    return {"n": n, "log_loss": ll, "brier": brier, "accuracy": acc, "ece": ece, "reliability": reliability}
