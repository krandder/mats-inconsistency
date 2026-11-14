"""As-of market context (price history and comments) for proposition prompts.

The market-context dataset (``/home/agent/data/jev-market-context-20260923``)
adds a ``context`` object to every proposition.  Everything in it was
timestamped strictly before the pair's forecast time ``t`` (00:00 UTC of the
as-of date, one week before the earlier closing week).  This module holds the
pure helpers shared by the dataset builder and the prompt renderers:

* ``tiptap_text``: Manifold rich-text JSON to plain text;
* ``price_at`` / ``price_history`` / ``trade_counts``: bets to as-of prices;
* ``select_comments``: newest comments within a count and character budget;
* ``context_block``: the text inserted under a question block.

``CONTEXT_MODES`` lists the renderer options.  ``"none"`` reproduces the old
prompt exactly; every other mode uses the as-of criteria text from the
context (the text as it stood at ``t``) instead of the current text.
"""

from __future__ import annotations

import datetime as dt
import re

CONTEXT_MODES = ("none", "asof_text", "prices", "comments", "prices+comments")
MAX_COMMENTS = 15
COMMENT_CHAR_BUDGET = 2000
MAX_COMMENT_CHARS = 400
HISTORY_POINTS = 8


# --- rich text --------------------------------------------------------------

_BLOCKS = {"paragraph", "heading", "blockquote", "bulletList", "orderedList", "listItem", "codeBlock",
           "horizontalRule", "doc"}


def tiptap_text(node) -> str:
    """Plain text of a Manifold TipTap document (or a plain string)."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node.strip()
    blocks: list[str] = []

    def inline(n) -> str:
        kind = n.get("type")
        if kind == "text":
            return n.get("text", "")
        if kind == "hardBreak":
            return "\n"
        if kind in ("mention", "contract-mention"):
            return "@" + str((n.get("attrs") or {}).get("label") or "")
        if kind == "image":
            return "[image]"
        if kind == "tiptapTweet":
            return "[tweet]"
        if kind == "iframe":
            return f"({(n.get('attrs') or {}).get('src', '')})"
        if kind == "linkPreview":
            return "[link preview]"
        return "".join(inline(c) for c in n.get("content") or [])

    def block(n) -> None:
        kind = n.get("type")
        children = n.get("content") or []
        if kind in _BLOCKS and kind not in ("paragraph", "heading", "codeBlock"):
            for child in children:
                if child.get("type") in _BLOCKS:
                    block(child)
                else:
                    text = inline(child).strip()
                    if text:
                        blocks.append(text)
            return
        text = inline(n).strip()
        if text:
            blocks.append(text)

    if isinstance(node, dict):
        block(node)
    return "\n\n".join(blocks).strip()


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --- prices -----------------------------------------------------------------

def price_at(bets: list[dict], when_ms: int, initial: float) -> float:
    """probAfter of the last bet created strictly before ``when_ms`` (bets sorted by time)."""
    value = initial
    for bet in bets:
        if bet["created_time"] >= when_ms:
            break
        if bet.get("prob_after") is not None:
            value = bet["prob_after"]
    return value


def price_history(bets: list[dict], start_ms: int, t_ms: int, initial: float,
                  points: int = HISTORY_POINTS) -> list[dict]:
    """``points`` evenly spaced samples from market creation to ``t`` (both included).

    The creation point is the initial probability; every later sample uses only
    bets created strictly before the sample time, and the last sample is ``t``.
    """
    if t_ms <= start_ms:
        return [{"date": iso_date(start_ms), "p": round(initial, 4)}]
    out = [{"date": iso_date(start_ms), "p": round(initial, 4)}]
    for k in range(1, points):
        when = start_ms + (t_ms - start_ms) * k // (points - 1)
        out.append({"date": iso_date(when if k < points - 1 else t_ms), "p": round(price_at(bets, when, initial), 4)})
    return out


def counts_as_trade(bet: dict, t_ms: int) -> bool:
    """A non-redemption bet that exchanged mana before ``t``.

    Market orders count at creation.  Limit orders count only once a fill
    happened before ``t`` (the stored amount is cumulative and may include
    later fills, so it is not used).
    """
    if bet["created_time"] >= t_ms or bet.get("is_redemption"):
        return False
    if bet.get("limit_prob") is None:
        return bool(bet.get("amount"))
    return bet.get("first_fill_time") is not None and bet["first_fill_time"] < t_ms


def trade_counts(bets: list[dict], t_ms: int) -> tuple[int, int]:
    trades = [b for b in bets if counts_as_trade(b, t_ms)]
    return len(trades), len({b.get("user_id") for b in trades})


# --- comments ---------------------------------------------------------------

def select_comments(comments: list[dict], *, max_comments: int = MAX_COMMENTS,
                    budget: int = COMMENT_CHAR_BUDGET, max_chars: int = MAX_COMMENT_CHARS) -> list[dict]:
    """Newest comments first until ``max_comments`` or the character budget; returned oldest first.

    ``comments`` carry ``created_time`` (ms), ``date``, ``user`` and ``text``.
    Each comment is cut to ``max_chars``; at least one comment is kept when any exist.
    """
    chosen, used = [], 0
    for c in sorted(comments, key=lambda c: c["created_time"], reverse=True):
        text = squash(c["text"])
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + " [...]"
        line = f"[{c['date']}] {c['user']}: {text}"
        if chosen and used + len(line) > budget:
            break
        chosen.append({**c, "text": text})
        used += len(line) + 1
        if len(chosen) >= max_comments:
            break
    return sorted(chosen, key=lambda c: c["created_time"])


# --- rendering --------------------------------------------------------------

def iso_date(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).date().isoformat()


def pct(p: float) -> str:
    value = 100 * p
    if 0 < value < 1:
        return "<1%"
    if 99 < value < 100:
        return ">99%"
    return f"{value:.0f}%"


MAX_SIBLINGS = 12
MAX_SIBLING_CHARS = 60


def sibling_text(siblings: list[dict], limit: int = MAX_SIBLINGS) -> str:
    """Compact "option price" list; resolved options show their resolution."""
    parts = []
    for s in siblings[:limit]:
        text = squash(s["text"])
        if len(text) > MAX_SIBLING_CHARS:
            text = text[:MAX_SIBLING_CHARS].rstrip() + "..."
        state = f"resolved {s['resolved_before_t']}" if s.get("resolved_before_t") else pct(s["p_t"])
        parts.append(f"{text} {state}")
    if len(siblings) > limit:
        parts.append(f"{len(siblings) - limit} more")
    return "; ".join(parts)


def check_mode(mode: str) -> str:
    if mode not in CONTEXT_MODES:
        raise ValueError(f"unknown context mode {mode!r}; expected one of {CONTEXT_MODES}")
    return mode


def context_block(context: dict, mode: str) -> str:
    """Market lines appended to a question block; empty for ``none``/``asof_text``."""
    check_mode(mode)
    if mode in ("none", "asof_text"):
        return ""
    as_of = context["as_of"]
    if not context.get("open_at_t", True):
        what = "option" if context.get("answer_id") else "market"
        return f"Market as of {as_of}: this {what} had not opened yet, so there is no price or comment history."
    lines: list[str] = []
    if mode in ("prices", "prices+comments"):
        price = context["price"]
        target = "this option" if context.get("answer_id") else "YES"
        lines.append(f"Market as of {as_of} (Manifold, play money): probability of {target} {pct(price['p_t'])}; "
                     f"{price['n_bets']} trades by {price['n_traders']} traders since the market opened on "
                     f"{context['created_date']}.")
        history = ", ".join(f"{point['date']} {pct(point['p'])}" for point in context["price_history"])
        lines.append(f"Price history: {history}.")
        siblings = context.get("sibling_prices")
        if siblings:
            lines.append(f"Other options as of {as_of}: {sibling_text(siblings)}.")
    if mode in ("comments", "prices+comments"):
        shown = context["comments"]
        total = context["n_comments_before_t"]
        if shown:
            lines.append(f"Comments before {as_of} (newest {len(shown)} of {total}, oldest first):")
            lines += [f"[{c['date']}] {c['user']}: {c['text']}" for c in shown]
        else:
            lines.append(f"Comments before {as_of}: none.")
    return "\n".join(lines)


def asof_row(row: dict, mode: str) -> tuple[dict, str | None]:
    """The proposition row to render and the criteria override for a mode.

    For ``none`` the row is returned unchanged (old prompt).  Otherwise the
    question, option and criteria text come from the context (as of ``t``).
    """
    check_mode(mode)
    if mode == "none":
        return row, None
    context = row.get("context")
    if not context:
        raise ValueError(f"proposition {row.get('id')} has no market context; use a market-context dataset")
    shown = dict(row)
    shown["parent_question"] = context.get("question_at_t") or row.get("parent_question")
    if row.get("answer_text"):
        shown["answer_text"] = context.get("answer_text_at_t") or row.get("answer_text")
    return shown, context["criteria"]["text"]
