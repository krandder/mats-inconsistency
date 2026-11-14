"""Turtel et al. 2025 (arXiv 2505.17989) forecasting prompt: exact template and parser.

The template was reverse-engineered from the 1,265 prompts in
``LightningRodLabs/outcome-rl-test-dataset`` (``prompt`` column) and is verified by
``verify_against_test_prompts``: every released prompt is parsed into its components
and re-rendered byte-for-byte (see ``scripts/build_turtel_prompts.py --verify``).

Stdlib only, so the builder can run under the system interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence
from urllib.parse import urlparse

HEADER = ("You are an expert superforecaster, familiar with Structured Analytic Techniques as well as "
          "Superforecasting by Philip Tetlock and related work. Predict the probability that the following "
          "question will be resolved as true/yes. You MUST give a probability estimate between 0 and 1 UNDER "
          "ALL CIRCUMSTANCES.")
FOOTER = ("Think about your answer and output your final prediction (a number between 0 and 1) with an asterisk "
          "at the beginning and end of the decimal (Ex: *<probability>*).")
NO_INFORMATION = "No information found"
UNKNOWN_DATE = "Unknown Date"


@dataclass(frozen=True)
class Article:
    title: str
    published: str  # Exa ISO string as returned ("2025-02-13T12:00:00.000Z") or "Unknown Date"
    source: str  # host name without a leading "www."
    summary: str

    def render(self, index: int) -> str:
        return f"[{index}] {self.title} (published on {self.published} by {self.source})\nSummary: {self.summary}"


def source_from_url(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def iso_to_prompt_datetime(value: str) -> str:
    """'2024-01-14T11:26:05Z' -> '2024-01-14 11:26:05+00:00' (the test set's "Today's date" form)."""
    value = value.strip()
    match = re.match(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(?:\.\d+)?(?:Z|\+00:00)?$", value)
    if not match:
        raise ValueError(f"unexpected datetime {value!r}")
    return f"{match.group(1)} {match.group(2)}+00:00"


def close_date_to_prompt_datetime(close_date: str) -> str:
    """'2025-02-19' -> '2025-02-19 00:00:00+00:00' (the test set renders the close *day* at midnight)."""
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", close_date):
        raise ValueError(f"expected YYYY-MM-DD close date, got {close_date!r}")
    return f"{close_date} 00:00:00+00:00"


def render_prompt(question: str, background: str, criteria: str, today: str, close: str,
                  articles: Sequence[Article]) -> str:
    lines = [HEADER, "", "Question:", question, "", "Question Background:", background, "",
             "Resolution Criteria:", criteria, "", f"Today's date: {today}", f"Question close date: {close}",
             "", "", "", "We have retrieved the following information for this question:", "<News>"]
    if articles:
        lines += ["---", "ARTICLES", "\n\n".join(a.render(i) for i, a in enumerate(articles, 1)), "----"]
    else:
        lines.append(NO_INFORMATION)
    lines += ["</News>", "", FOOTER]
    return "\n".join(lines)


def order_articles(articles: Iterable[Article]) -> list[Article]:
    """Newest first, unknown dates first: the order observed in all 1,232 test prompts with articles."""
    return sorted(articles, key=lambda a: (a.published != UNKNOWN_DATE, _neg_date_key(a.published)))


def _neg_date_key(published: str) -> str:
    if published == UNKNOWN_DATE:
        return ""
    # Lexicographic descending on the ISO string: invert each character.
    return "".join(chr(0x10FFFF - ord(c)) for c in published)


_PROMPT_RE = re.compile(
    r"^(?P<header>.*?)\n\nQuestion:\n(?P<question>.*?)\n\nQuestion Background:\n(?P<background>.*?)\n\n"
    r"Resolution Criteria:\n(?P<criteria>.*?)\n\nToday's date: (?P<today>[^\n]*)\nQuestion close date: "
    r"(?P<close>[^\n]*)\n\n\n\nWe have retrieved the following information for this question:\n<News>\n"
    r"(?P<news>.*)\n</News>\n\n(?P<footer>.*)$", re.S)
_ARTICLE_RE = re.compile(r"^\[(\d+)\] (?P<title>.*?) \(published on (?P<published>.*?) by (?P<source>[^\s)]*)\)\n"
                         r"Summary: (?P<summary>.*)$", re.S)


def parse_prompt(text: str) -> dict:
    """Split a Turtel prompt into its components; raises ValueError when the structure does not match."""
    match = _PROMPT_RE.match(text)
    if not match:
        raise ValueError("prompt does not match the Turtel template")
    parts = match.groupdict()
    news = parts.pop("news")
    articles: list[Article] = []
    if news == NO_INFORMATION:
        pass
    elif news.startswith("---\nARTICLES\n") and news.endswith("\n----"):
        body = news[len("---\nARTICLES\n"):-len("\n----")]
        for index, block in enumerate(re.split(r"\n\n(?=\[\d+\] )", body), 1):
            amatch = _ARTICLE_RE.match(block)
            if not amatch or int(amatch.group(1)) != index:
                raise ValueError(f"article block {index} does not match")
            articles.append(Article(amatch.group("title"), amatch.group("published"), amatch.group("source"),
                                    amatch.group("summary")))
    else:
        raise ValueError("news block does not match")
    parts["articles"] = articles
    return parts


def verify_against_test_prompts(prompts: Iterable[str]) -> dict:
    """Parse and re-render each released prompt; returns counts of exact matches and the first mismatch."""
    result = {"checked": 0, "exact": 0, "header_ok": 0, "footer_ok": 0, "first_mismatch": None}
    for text in prompts:
        result["checked"] += 1
        try:
            parts = parse_prompt(text)
        except ValueError as error:
            if result["first_mismatch"] is None:
                result["first_mismatch"] = {"index": result["checked"] - 1, "error": str(error)}
            continue
        result["header_ok"] += parts["header"] == HEADER
        result["footer_ok"] += parts["footer"] == FOOTER
        rendered = render_prompt(parts["question"], parts["background"], parts["criteria"], parts["today"],
                                 parts["close"], parts["articles"])
        if rendered == text:
            result["exact"] += 1
        elif result["first_mismatch"] is None:
            result["first_mismatch"] = {"index": result["checked"] - 1, "error": "re-render differs",
                                        "rendered_head": rendered[:300], "original_head": text[:300]}
    return result


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
_TODAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+00:00$")
_CLOSE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} 00:00:00\+00:00$")


def structural_signature(text: str) -> list[str]:
    """Section-level signature with content masked (parse, then describe each component's shape).

    Two prompts with the same signature share the header, section order and blank-line layout, the date-line
    shapes, the news block shape (numbered articles with an ISO or "Unknown Date" published date, a bare host as
    the source and a single-line summary, or the "No information found" block) and the final instruction. The
    number of articles and the text inside sections may differ. Raises ValueError when the template itself
    does not match.
    """
    parts = parse_prompt(text)
    signature = ["header:" + ("exact" if parts["header"] == HEADER else "DIFFERS"),
                 "question:" + ("single-line" if "\n" not in parts["question"] else "multi-line"),
                 "background:" + ("empty" if parts["background"] == "" else "text"),
                 "criteria:" + ("text" if parts["criteria"].strip() else "empty"),
                 "today:" + ("YYYY-MM-DD HH:MM:SS+00:00" if _TODAY_RE.match(parts["today"]) else "DIFFERS"),
                 "close:" + ("YYYY-MM-DD 00:00:00+00:00" if _CLOSE_RE.match(parts["close"]) else "DIFFERS")]
    articles = parts["articles"]
    if not articles:
        signature.append("news:No information found")
    else:
        published_ok = all(_ISO_DATE_RE.match(a.published) or a.published == UNKNOWN_DATE for a in articles)
        source_ok = all(re.match(r"^[a-z0-9.-]+$", a.source) for a in articles)
        summary_ok = all("\n" not in a.summary and a.summary.strip() for a in articles)
        ordered_ok = [a.published for a in order_articles(articles)] == [a.published for a in articles]
        signature.append("news:---/ARTICLES/[n] title (published on "
                         + ("<iso-ms-Z|Unknown Date>" if published_ok else "DIFFERS") + " by "
                         + ("<host>" if source_ok else "DIFFERS") + ") / Summary: "
                         + ("single-line" if summary_ok else "DIFFERS") + " x N, "
                         + ("newest-first" if ordered_ok else "UNORDERED") + "/----")
    signature.append("footer:" + ("exact" if parts["footer"] == FOOTER else "DIFFERS"))
    return signature


def structural_skeleton(text: str) -> list[str]:
    """Line-level skeleton with content masked, for diffing a training prompt against a test prompt."""
    lines = []
    for line in text.split("\n"):
        if line in (HEADER, FOOTER, "Question:", "Question Background:", "Resolution Criteria:", "<News>",
                    "</News>", "---", "----", "ARTICLES", NO_INFORMATION, ""):
            lines.append(line if line else "<blank>")
        elif line.startswith("Today's date: "):
            lines.append("Today's date: <YYYY-MM-DD HH:MM:SS+00:00>")
        elif line.startswith("Question close date: "):
            lines.append("Question close date: <YYYY-MM-DD 00:00:00+00:00>")
        elif re.match(r"^\[\d+\] .* \(published on .* by [^\s)]*\)$", line):
            lines.append("[n] <title> (published on <date> by <source>)")
        elif line.startswith("Summary: "):
            lines.append("Summary: <text>")
        else:
            lines.append("<text>")
    return lines
