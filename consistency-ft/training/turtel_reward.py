"""Reward for the Turtel et al. 2025 replication: strict negative Brier plus reasoning guardrails.

Paper (arXiv 2505.17989 v4, sec. 3): R = -(p_hat - y)^2; the strict rule assigns the maximum loss 1.0 whenever
no probability in [0, 1] can be extracted; a scorer deducts for out-of-context non-English passages, gibberish
and a missing explanation inside the <think> block (lambda values are not published: the defaults below are
ours and are flagged as unverified in the runbook); the text handed to the scorer is hard-truncated at 16,000
characters. Evaluation uses the soft Brier: 0.25 for a missing forecast.

Parsing precedence (the prompt asks for "*<probability>*"): the last asterisk-wrapped number after the
reasoning block, then "Probability: 0.xx", then the last number in [0, 1] on the final non-empty line (that
fallback requires a decimal point or a percent sign, so a bare "1" or "0" in running text is not a forecast).
Percentages ("35%", "*35%*") are accepted and divided by 100. Stdlib only; unit tests in test_turtel_reward.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable, Optional, Sequence

MAX_SCORE_CHARS = 16_000
THINK_CLOSE = "</think>"

_ASTERISK_RE = re.compile(r"\*\s*(\d*\.?\d+)\s*(%?)\s*\*")
_LABEL_RE = re.compile(r"probability[^\d\n]{0,16}?(\d*\.?\d+)\s*(%?)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![\w.])(\d*\.\d+|\d+(?=\s*%))\s*(%?)")  # fallback needs a decimal point or a %


def _to_unit(number: str, percent: str) -> Optional[float]:
    try:
        value = float(number)
    except ValueError:
        return None
    if percent:
        value /= 100.0
    if 0.0 <= value <= 1.0:
        return value
    return None


def split_reasoning(text: str) -> tuple[str, str]:
    """(reasoning, answer_region). With a </think> tag the answer region is what follows it."""
    if THINK_CLOSE in text:
        head, _, tail = text.rpartition(THINK_CLOSE)
        head = head.replace("<think>", "", 1)
        return head, tail
    lines = text.rstrip().split("\n")
    if len(lines) > 1:
        return "\n".join(lines[:-1]), lines[-1]
    return "", text


def parse_probability(text: str, max_chars: int = MAX_SCORE_CHARS) -> Optional[float]:
    """Extract the forecast from a completion; None when nothing parseable (the strict rule then applies)."""
    if not text:
        return None
    text = text[:max_chars]
    _, answer = split_reasoning(text)
    # The answer region (after </think>, or the last line) is tried with every pattern before the full text,
    # so a number quoted inside the reasoning never overrides the final statement.
    for region in (answer, text):
        for pattern in (_ASTERISK_RE, _LABEL_RE):
            for number, percent in reversed(pattern.findall(region)):
                value = _to_unit(number, percent)
                if value is not None:
                    return value
        final_lines = [line for line in region.strip().split("\n") if line.strip()]
        if final_lines:
            for number, percent in reversed(_NUMBER_RE.findall(final_lines[-1])):
                value = _to_unit(number, percent)
                if value is not None:
                    return value
    return None


@dataclass(frozen=True)
class GuardrailConfig:
    lambda_lang: float = 0.5  # unverified: paper gives no numbers
    lambda_gib: float = 0.5
    lambda_miss: float = 0.25
    min_reasoning_words: int = 30
    max_score_chars: int = MAX_SCORE_CHARS


_LATIN_MAX = 0x024F  # Basic Latin .. Latin Extended-B; accented European letters stay "English-compatible"
_VOWELS = set("aeiouy")


def non_english_proportion(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) > _LATIN_MAX) / len(letters)


def gibberish_proportion(text: str) -> float:
    """Share of non-word tokens (long, vowel-less, or with 4+ repeated characters) or of repeated 8-word windows."""
    tokens = [t.strip(".,;:!?()[]{}\"'*_-") for t in text.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return 0.0
    non_word = 0
    for token in tokens:
        lower = token.lower()
        if len(lower) > 30:
            non_word += 1
        elif re.search(r"(.)\1{3,}", lower):
            non_word += 1
        elif len(lower) >= 5 and lower.isascii() and lower.isalpha() and not (_VOWELS & set(lower)):
            non_word += 1
    non_word_share = non_word / len(tokens)
    window = 8
    if len(tokens) >= 2 * window:
        windows = [" ".join(tokens[i:i + window]).lower() for i in range(0, len(tokens) - window + 1)]
        seen: set[str] = set()
        repeated = 0
        for w in windows:
            if w in seen:
                repeated += 1
            seen.add(w)
        repeat_share = repeated / len(windows)
    else:
        repeat_share = 0.0
    return max(non_word_share, repeat_share)


@dataclass(frozen=True)
class RewardBreakdown:
    probability: Optional[float]
    parsed: bool
    brier: float  # strict: 1.0 when unparsed
    non_english_proportion: float
    gibberish_proportion: float
    explains_answer: bool
    penalty: float
    reward: float
    reasoning_words: int
    scored_chars: int
    truncated_for_scoring: bool

    def to_dict(self) -> dict:
        return asdict(self)


def score_completion(text: str, outcome: int, config: GuardrailConfig = GuardrailConfig()) -> RewardBreakdown:
    text = text or ""
    truncated = len(text) > config.max_score_chars
    scored = text[:config.max_score_chars]
    probability = parse_probability(scored, config.max_score_chars)
    brier = (probability - float(outcome)) ** 2 if probability is not None else 1.0
    reasoning, _ = split_reasoning(scored)
    words = len(reasoning.split())
    explains = words >= config.min_reasoning_words
    lang = non_english_proportion(reasoning) if reasoning.strip() else 0.0
    gib = gibberish_proportion(reasoning) if reasoning.strip() else 0.0
    penalty = config.lambda_lang * lang + config.lambda_gib * gib + (0.0 if explains else config.lambda_miss)
    return RewardBreakdown(probability=probability, parsed=probability is not None, brier=brier,
                           non_english_proportion=lang, gibberish_proportion=gib, explains_answer=explains,
                           penalty=penalty, reward=-brier - penalty, reasoning_words=words,
                           scored_chars=len(scored), truncated_for_scoring=truncated)


def group_advantages(rewards: Sequence[float], group_size: int) -> list[float]:
    """Modified GRPO: advantage = reward - group mean, no division by the group standard deviation."""
    if len(rewards) % group_size:
        raise ValueError(f"{len(rewards)} rewards are not a multiple of group size {group_size}")
    out: list[float] = []
    for start in range(0, len(rewards), group_size):
        group = rewards[start:start + group_size]
        mean = sum(group) / group_size
        out.extend(r - mean for r in group)
    return out


def soft_brier(pairs: Iterable[tuple[Optional[float], int]]) -> float:
    """Turtel's evaluation rule: a missing forecast costs 0.25 (guessing 50%)."""
    pairs = list(pairs)
    if not pairs:
        return float("nan")
    return sum(0.25 if p is None else (p - y) ** 2 for p, y in pairs) / len(pairs)


def expected_calibration_error(pairs: Iterable[tuple[Optional[float], int]], bins: int = 10) -> float:
    pairs = [(p, y) for p, y in pairs if p is not None]
    if not pairs:
        return float("nan")
    totals = [[0, 0.0, 0.0] for _ in range(bins)]  # count, sum p, sum y
    for p, y in pairs:
        index = min(bins - 1, int(p * bins))
        totals[index][0] += 1
        totals[index][1] += p
        totals[index][2] += y
    return sum(abs(s_p / n - s_y / n) * n for n, s_p, s_y in totals if n) / len(pairs)
