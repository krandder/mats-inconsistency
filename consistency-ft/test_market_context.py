"""Market-context renderer, as-of helpers and the leakage rules of the dataset builder."""

import importlib.util
import sys
from pathlib import Path

import pytest

from training import market_context as mc
from training.supervised_pairs import FORM_ORDER, PairExample, render_prompt
from training.verbalized_baseline import question_block, user_prompt

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("build_market_context", ROOT / "scripts/build_market_context.py")
builder = importlib.util.module_from_spec(spec)
sys.modules["build_market_context"] = builder
spec.loader.exec_module(builder)

T = builder.t_ms_of("2025-10-20")
DAY = 86_400_000


def doc(*paragraphs):
    return {"type": "doc", "content": [{"type": "paragraph", "content": [{"type": "text", "text": p}]}
                                       for p in paragraphs]}


def context(open_at_t=True, answer_id=None, comments=None):
    return {"as_of": "2025-10-20", "t_ms": T, "open_at_t": open_at_t, "answer_id": answer_id,
            "not_open_reason": None if open_at_t else "market_created_at_or_after_t",
            "created_date": "2025-09-01", "question_at_t": "Old question?", "answer_text_at_t": None,
            "criteria": {"text": "As-of rules. Update 2025-10-01: clarification before t.", "source": "edit_history"},
            "price": {"p_t": 0.25, "n_bets": 12, "n_traders": 5} if open_at_t else None,
            "price_history": [{"date": "2025-09-01", "p": 0.5}, {"date": "2025-10-20", "p": 0.25}] if open_at_t else [],
            "n_comments_before_t": len(comments or []),
            "comments": comments or []}


def example(with_context=True):
    rows = {
        "a": {"id": "a", "parent_id": "a", "parent_question": "Current question?", "answer_text": None,
              "closing_week": "2025-10-27", "criteria_text": "Current rules.\n\nUpdate 2025-11-01: it happened."},
        "b": {"id": "b", "parent_id": "b", "parent_question": "Other?", "answer_text": None,
              "closing_week": "2025-11-03", "criteria_text": "Other rules."},
    }
    if with_context:
        rows["a"]["context"] = context(comments=[{"date": "2025-10-10", "user": "u1", "text": "Looks unlikely.",
                                                  "created_time": T - 10 * DAY}])
        rows["b"]["context"] = context(open_at_t=False)
    queries = {form: {"query_id": f"x-q{i:02d}", "form": form} for i, form in enumerate(FORM_ORDER)}
    return PairExample(pair={"pair_id": "x", "p_id": "a", "q_id": "b", "observed_query_id": "x-q00"},
                       queries=queries, propositions=rows, labels={})


def test_tiptap_text():
    node = {"type": "doc", "content": [
        {"type": "heading", "content": [{"type": "text", "text": "Rules"}]},
        {"type": "paragraph", "content": [{"type": "text", "text": "Hi "},
                                          {"type": "mention", "attrs": {"label": "bob"}}, {"type": "hardBreak"},
                                          {"type": "image"}]},
        {"type": "paragraph"},
        {"type": "bulletList", "content": [{"type": "listItem", "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "item"}]}]}]}]}
    assert mc.tiptap_text(node) == "Rules\n\nHi @bob\n[image]\n\nitem"
    assert mc.tiptap_text("plain ") == "plain"
    assert mc.tiptap_text(None) == ""


def test_price_uses_only_bets_strictly_before_t():
    bets = [{"created_time": T - 2 * DAY, "prob_after": 0.4}, {"created_time": T - 1, "prob_after": 0.3},
            {"created_time": T, "prob_after": 0.99}]
    assert mc.price_at(bets, T, 0.5) == 0.3
    assert mc.price_at([], T, 0.5) == 0.5
    history = mc.price_history(bets, T - 7 * DAY, T, 0.5, points=8)
    assert len(history) == 8 and history[0]["p"] == 0.5 and history[-1] == {"date": "2025-10-20", "p": 0.3}


def test_trade_counts_limit_orders_count_only_after_a_fill_before_t():
    bets = [{"created_time": T - 5, "amount": 10, "user_id": "u1", "limit_prob": None},
            {"created_time": T - 4, "amount": 5, "user_id": "u1", "limit_prob": None, "is_redemption": 1},
            {"created_time": T - 3, "amount": 50, "user_id": "u2", "limit_prob": 0.3, "first_fill_time": T + 5},
            {"created_time": T - 2, "amount": 50, "user_id": "u3", "limit_prob": 0.3, "first_fill_time": T - 1},
            {"created_time": T + 1, "amount": 50, "user_id": "u4", "limit_prob": None}]
    assert mc.trade_counts(bets, T) == (2, 2)


def test_select_comments_budget_and_order():
    comments = [{"created_time": T - k * DAY, "date": f"d{k}", "user": "u", "text": "x" * 300} for k in range(1, 30)]
    chosen = mc.select_comments(comments, max_comments=15, budget=2000, max_chars=400)
    assert 1 <= len(chosen) <= 7
    assert [c["created_time"] for c in chosen] == sorted(c["created_time"] for c in chosen)
    assert chosen[-1]["created_time"] == T - DAY  # the newest is kept
    assert len(mc.select_comments(comments[:20], budget=10**6)) == 15
    long = mc.select_comments([{"created_time": 1, "date": "d", "user": "u", "text": "y" * 1000}])
    assert long[0]["text"].endswith("[...]") and len(long[0]["text"]) < 420


def test_default_prompt_unchanged_by_context_field():
    with_ctx, without = example(True), example(False)
    for form in FORM_ORDER:
        assert render_prompt(with_ctx, form) == render_prompt(without, form)
        assert render_prompt(with_ctx, form, "none") == render_prompt(without, form)
        assert user_prompt(with_ctx, form, "v6_direct") == user_prompt(without, form, "v6_direct")


@pytest.mark.parametrize("mode", ["prices", "comments", "prices+comments", "asof_text"])
def test_context_modes(mode):
    ex = example(True)
    for text in (render_prompt(ex, "(P) AND (Q)", mode), user_prompt(ex, "(P) AND (Q)", "v6_direct", context=mode)):
        assert "Old question?" in text and "Current question?" not in text
        assert "As-of rules. Update 2025-10-01: clarification before t." in text
        assert "it happened" not in text
        assert ("probability of YES 25%" in text) == (mode in ("prices", "prices+comments"))
        assert ("2025-09-01 50%, 2025-10-20 25%" in text) == (mode in ("prices", "prices+comments"))
        assert ("[2025-10-10] u1: Looks unlikely." in text) == (mode in ("comments", "prices+comments"))
        assert ("had not opened yet" in text) == (mode != "asof_text")
        assert text.count("Assume today is 2025-10-20") == 1


def test_bad_mode_and_missing_context():
    with pytest.raises(ValueError):
        render_prompt(example(True), "P", "everything")
    with pytest.raises(ValueError):
        render_prompt(example(False), "P", "prices")
    with pytest.raises(ValueError):
        question_block(example(False).propositions["a"], context="comments")


class FakeIndex:
    def __init__(self, contract, edits=(), bets=(), comments=(), answers=None):
        self.contracts = {"c": contract}
        self.edits = {"c": list(edits)}
        self.answers = answers or {}
        self.chain_breaks = {}
        self._bets, self._comments = list(bets), list(comments)

    def bets(self, cid, answer_id):
        return [b for b in self._bets if b.get("answer_id") == answer_id]

    def comments_before(self, cid, t_ms):
        return [c for c in self._comments if c["created_time"] < t_ms]


def contract(created=T - 30 * DAY, resolved=T + 30 * DAY, description=doc("Now rules.")):
    return {"resolution_time": resolved, "close_time": resolved, "data": {
        "createdTime": created, "initialProbability": 0.5, "question": "Now?", "description": description}}


PROP = {"id": "c", "parent_id": "c", "parent_question": "Now?", "answer_text": None, "criteria_text": "Now rules."}


def test_builder_drops_resolved_at_or_before_t():
    for resolved in (T, T - DAY):
        ctx, reason, _ = builder.build_context(FakeIndex(contract(resolved=resolved)), PROP, "2025-10-20")
        assert ctx is None and reason == "market_resolved_at_or_before_t"
    resolution_edit = {"id": 1, "created_time": T - DAY, "keys": ["isResolved", "resolution"], "data": {}}
    ctx, reason, _ = builder.build_context(FakeIndex(contract(), edits=[resolution_edit]), PROP, "2025-10-20")
    assert reason == "resolution_event_before_t"


def test_builder_text_at_t_from_pre_image_and_no_post_t_records():
    edits = [{"id": 1, "created_time": T - DAY, "keys": ["description"], "data": {"question": "Q0", "description": doc("v0")}},
             {"id": 2, "created_time": T + DAY, "keys": ["description", "question"],
              "data": {"question": "Q1", "description": doc("v1 at t")}}]
    bets = [{"created_time": T - DAY, "prob_after": 0.2, "amount": 5, "user_id": "u", "limit_prob": None},
            {"created_time": T + DAY, "prob_after": 0.9, "amount": 5, "user_id": "v", "limit_prob": None}]
    comments = [{"id": "k1", "created_time": T - DAY, "date": "2025-10-19", "user": "u", "text": "before",
                 "edited_after_t": False},
                {"id": "k2", "created_time": T + DAY, "date": "2025-10-21", "user": "u", "text": "after",
                 "edited_after_t": False}]
    ctx, reason, audit = builder.build_context(FakeIndex(contract(), edits, bets, comments), PROP, "2025-10-20")
    assert reason is None and ctx["open_at_t"]
    assert ctx["criteria"]["text"] == "v1 at t" and ctx["question_at_t"] == "Q1"
    assert ctx["price"]["p_t"] == 0.2 and ctx["price"]["n_bets"] == 1
    assert [c["text"] for c in ctx["comments"]] == ["before"]
    assert "resolution_time" not in str(ctx)


def test_builder_keeps_market_created_after_t_without_history():
    ctx, reason, _ = builder.build_context(FakeIndex(contract(created=T + DAY)), PROP, "2025-10-20")
    assert reason is None and ctx["open_at_t"] is False and ctx["price"] is None and ctx["comments"] == []
    assert ctx["not_open_reason"] == "market_created_at_or_after_t"


def test_sibling_prices_rendered_in_price_modes():
    ctx = context(answer_id="o1")
    ctx["sibling_prices"] = [{"answer_id": "o2", "text": "Other lab", "p_t": 0.4, "resolved_before_t": None},
                             {"answer_id": "o3", "text": "Third", "p_t": 0.5, "resolved_before_t": "NO"}]
    assert "Other options as of 2025-10-20: Other lab 40%; Third resolved NO." in mc.context_block(ctx, "prices")
    assert "Other options" not in mc.context_block(ctx, "comments")
    many = [{"text": f"o{k}", "p_t": 0.1, "resolved_before_t": None} for k in range(15)]
    assert mc.sibling_text(many).endswith("3 more")


def test_sibling_prices_use_only_bets_before_t():
    idx = FakeIndex(contract(), bets=[
        {"answer_id": "o2", "created_time": T - DAY, "prob_after": 0.3},
        {"answer_id": "o2", "created_time": T + DAY, "prob_after": 0.9}],
        answers={"o2": {"contract_id": "c", "created_time": T - 5 * DAY, "resolution_time": None, "text": "B"},
                 "o3": {"contract_id": "c", "created_time": T + DAY, "resolution_time": None, "text": "late"}})
    got = builder.sibling_prices(idx, "c", "o1", T, {})
    assert [(s["text"], s["p_t"]) for s in got] == [("B", 0.3)]
