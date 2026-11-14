"""LLM-free rules for structured pairs: threshold ladders, date ladders.

A proposition text is split into a template plus one varying element:
  threshold  one number (with its unit multiplier) and a comparator direction
             (up: above/over/at least/more than/exceed/>/+ ...; down: below/under/less than/...)
  date       one date expression preceded by a cumulative keyword (by/before/prior to/no later than)
Two propositions with the same template (and same direction) form a ladder rung pair.
  up threshold:   higher threshold  implies lower threshold
  down threshold: lower threshold   implies higher threshold
  date (by/before): earlier date    implies later date
Texts with negation / persistence words are rejected (they break or flip monotonicity).
"""

from __future__ import annotations

import calendar
import datetime as dt
import re

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
MONTHS["sept"] = 9
MON = r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\.?"

BAD = re.compile(r"\b(not|n't|no longer|never|without|still|remain|remains|stay|stays|continue|continues|survive|survives|"
                 r"last|lasts|keep|keeps|until|exactly|between|range|closest|nearest|first|highest|lowest|most|least|"
                 r"only|except|either|neither|nor|each|every|average|median)\b", re.I)
# "at least"/"at most" contain "least"/"most": handled by stripping comparator phrases before the BAD check
UP = [r"at least", r"more than", r"greater than", r"higher than", r"larger than", r"over", r"above", r"exceeds?",
      r"exceeding", r"surpass(?:es)?", r">=", r"≥", r">", r"or more", r"or higher", r"or above", r"\+",
      r"reach(?:es)?", r"hits?", r"↑"]
DOWN = [r"at most", r"less than", r"fewer than", r"lower than", r"below", r"under", r"<=", r"≤", r"<", r"or less",
        r"or fewer", r"or lower", r"or below", r"drops? below", r"falls? below", r"dips? below", r"down to", r"↓"]
WEAK_UP = (r"reach(?:es)?", r"hits?")  # "hit $X" can mean a fall to X: not used for prices
SELF_REF = re.compile(r"\bthis (market|question)\b|\bwhich\b", re.I)
CUM = r"(?:by|before|prior to|no later than|by the end of|before the end of|by end of|before end of)"

NUM = re.compile(r"(?<![\w.])([$€£])?\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?"
                 r"(k|m|b|bn|t|thousand|million|billion|trillion|%|percent)?(?![\w])", re.I)
MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9, "t": 1e12,
        "trillion": 1e12}

DATE_PATTERNS = [
    # (regex, kind)
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "iso"),
    (re.compile(rf"\b(?:the\s+)?end\s+of\s+({MON})(?:,?\s+(\d{{4}}))?", re.I), "end_month"),
    (re.compile(rf"\b({MON})\s+(\d{{1,2}})(?!\d)(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?", re.I), "mon_day"),
    (re.compile(rf"\b(\d{{1,2}})(?!\d)(?:st|nd|rd|th)?\s+(?:of\s+)?({MON})(?:,?\s+(\d{{4}}))?", re.I), "day_mon"),
    (re.compile(r"\b(?:the\s+)?end\s+of\s+(?:Q([1-4])\s+)?(\d{4})\b", re.I), "end_year"),
    (re.compile(r"\bQ([1-4])\s+(\d{4})\b", re.I), "quarter"),
    (re.compile(rf"\b({MON})(?:,?\s+(\d{{4}}))?(?![\w])", re.I), "month"),
    (re.compile(r"(?<![\w$.,])(20[2-3]\d)(?![\w%.,])"), "year"),
]


def _month_end(y: int, m: int) -> dt.date:
    return dt.date(y, m, calendar.monthrange(y, m)[1])


def _infer_year(m: int, d: int | None, ref: dt.date) -> int:
    """First occurrence of month(/day) on or after ref (the market's creation date)."""
    y = ref.year
    day = d or calendar.monthrange(y, m)[1]
    try:
        cand = dt.date(y, m, min(day, calendar.monthrange(y, m)[1]))
    except ValueError:
        return y
    return y if cand >= ref else y + 1


def find_dates(text: str, ref: dt.date) -> list[tuple[int, int, dt.date]]:
    """Non-overlapping date expressions: (start, end, resolved end-of-period date)."""
    taken: list[tuple[int, int, dt.date]] = []

    def free(s, e):
        return all(e <= a or s >= b for a, b, _ in taken)

    for rx, kind in DATE_PATTERNS:
        for mt in rx.finditer(text):
            s, e = mt.span()
            if not free(s, e):
                continue
            g = mt.groups()
            try:
                if kind == "iso":
                    d = dt.date(int(g[0]), int(g[1]), int(g[2]))
                elif kind == "end_month":
                    m = MONTHS[g[0].lower().rstrip(".")]
                    y = int(g[1]) if g[1] else _infer_year(m, None, ref)
                    d = _month_end(y, m)
                elif kind == "mon_day":
                    m = MONTHS[g[0].lower().rstrip(".")]
                    day = int(g[1])
                    y = int(g[2]) if g[2] else _infer_year(m, day, ref)
                    d = dt.date(y, m, day)
                elif kind == "day_mon":
                    m = MONTHS[g[1].lower().rstrip(".")]
                    day = int(g[0])
                    y = int(g[2]) if g[2] else _infer_year(m, day, ref)
                    d = dt.date(y, m, day)
                elif kind == "end_year":
                    y = int(g[1])
                    d = _month_end(y, 3 * int(g[0])) if g[0] else dt.date(y, 12, 31)
                elif kind == "quarter":
                    d = _month_end(int(g[1]), 3 * int(g[0]))
                elif kind == "month":
                    word = g[0].lower().rstrip(".")
                    if word == "may" and not g[1]:
                        continue  # "may" the verb
                    m = MONTHS[word]
                    y = int(g[1]) if g[1] else _infer_year(m, None, ref)
                    d = _month_end(y, m)
                else:  # year
                    d = dt.date(int(g[0]), 12, 31)
            except (ValueError, KeyError):
                continue
            taken.append((s, e, d))
    return sorted(taken)


def _strip_comparators(text: str) -> str:
    return re.sub(r"\bat (least|most)\b", " ", text, flags=re.I)


def date_key(text: str, ref: dt.date):
    """(template, date) if text has exactly one cumulative date expression, else None."""
    if BAD.search(_strip_comparators(text)) or SELF_REF.search(text):
        return None
    ds = find_dates(text, ref)
    cum = [(s, e, d) for s, e, d in ds if re.search(CUM + r"\s*(?:the\s+)?$", text[:s], re.I)]
    if len(cum) != 1:
        return None
    s, e, d = cum[0]
    others = [x for x in ds if x[0] != s]
    tmpl = text[:s] + "<D>" + text[e:]
    # other dates stay literal in the template (normalized)
    return re.sub(r"\s+", " ", tmpl.lower()).strip(" ?.!"), d, bool(others)


def _has(t: str, words: list[str]) -> bool:
    for w in words:
        if w in (">=", "≥", ">", "<=", "≤", "<"):
            if re.search(re.escape(w) + r"\s?[$€£]?\d", t):
                return True
        elif w == r"\+":
            if re.search(r"\d\s?[kmb%]?\+", t):
                return True
        elif re.search(r"(?<![\w])" + w + r"(?![\w])", t):
            return True
    return False


def _direction(text: str) -> str | None:
    t = text.lower()
    up, down = _has(t, UP), _has(t, DOWN)
    if up == down:
        return None
    return "up" if up else "down"


def numbers(text: str, ref: dt.date):
    """Numbers outside date expressions: (start, end, value)."""
    dates = find_dates(text, ref)
    out = []
    for mt in NUM.finditer(text):
        s, e = mt.span()
        if any(not (e <= a or s >= b) for a, b, _ in dates):
            continue
        cur, digits, suf = mt.groups()
        v = float(digits.replace(",", ""))
        if not cur and not suf and re.fullmatch(r"(19|20)\d\d", digits):
            continue  # a year
        if suf and suf.lower() in MULT:
            v *= MULT[suf.lower()]
        out.append((s, e, v, (cur or "") + ("%" if suf and suf.lower() in ("%", "percent") else "")))
    return out


def threshold_key(text: str, direction_text: str, ref: dt.date):
    """(template, value, direction, unit) if text has exactly one number outside dates."""
    if BAD.search(_strip_comparators(text)) or re.search(r"\d\s*(-|–|to)\s*\$?\d", text):
        return None
    if SELF_REF.search(text) or SELF_REF.search(direction_text):
        return None
    ns = numbers(text, ref)
    if len(ns) != 1:
        return None
    s, e, v, unit = ns[0]
    d = _direction(direction_text)
    if d is None:
        return None
    strong = [w for w in UP if w not in WEAK_UP]
    if d == "up" and not _has(direction_text.lower(), strong) and (unit == "$" or re.search(r"price|\$|stock|oil|gold|silver|bitcoin|btc|eth", direction_text, re.I)):
        return None  # only hit/reach on a price: direction ambiguous
    tmpl = text[:s] + "<N>" + text[e:]
    return re.sub(r"\s+", " ", tmpl.lower()).strip(" ?.!"), v, d, unit
