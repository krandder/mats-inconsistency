"""Tests for the market game runner (harness/market_game.py + scripts)."""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from harness import market_game as mg  # noqa: E402
from scripts import score_market_game as score  # noqa: E402

TURTEL_PROMPT = """You are an expert superforecaster, familiar with Structured Analytic Techniques as well as Superforecasting by Philip Tetlock and related work. Predict the probability that the following question will be resolved as true/yes. You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES.

Question:
{text}

Question Background:


Resolution Criteria:
This market resolves to "Yes" if {text}

Today's date: 2025-02-14 16:00:04+00:00
Question close date: 2025-02-19 00:00:00+00:00



We have retrieved the following information for this question:
<News>
---
ARTICLES
[1] Article about {text} (published on 2025-02-13T12:00:00.000Z by example.com)
Summary: Evidence for question {index}.
----
</News>

Think about your answer and output your final prediction (a number between 0 and 1) with an asterisk at the beginning and end of the decimal (Ex: *<probability>*)."""

TEXTS = [
    'Will "Conclave" win Best Costume Design at the 2025 Oscars?',
    'Will "Wicked" win Best Costume Design at the 2025 Oscars?',
    'Will "Nosferatu" win Best Costume Design at the 2025 Oscars?',
    'Will the following market resolve to "Duke"? Duke vs. Wake Forest',
    'Will the following market resolve to "Wake Forest"? Duke vs. Wake Forest',
    "Will the highest temperature in NYC be between 42-43°F on February 22?",
    "Will the highest temperature in NYC be between 44-45°F on February 22?",
    "Will the highest temperature in NYC be 46°F or higher on February 22?",
    "Will the highest temperature in NYC be between 42-43°F on February 23?",
    "Bitcoin above $97,000 on February 21?",
    "Bitcoin above $99,000 on February 28?",
    "Will Trump say \"Ukraine\" 20+ times during Macron presser today?",
    "Will Trump say \"Ukraine\" 10+ times during Macron presser today?",
    "Will Canada win the NHL 4 Nations Face-Off?",
    "Will Sweden win the NHL 4 Nations Face-Off?",
    "Will Apple launch an iPhone SE on February 19?",
    "Dave Portnoy in jail before March?",
    "Will Ireland, Norway, or Spain accept Gazan refugees in February?",
    "Will Manchester City win on 2025-02-23?",
    "Will Manchester City win on 2025-02-26?",
]
RESOLUTIONS = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]


def synthetic_questions(count: int = len(TEXTS)) -> list[mg.Question]:
    rows = []
    for i in range(count):
        text = TEXTS[i % len(TEXTS)] if i < len(TEXTS) else f"Synthetic question number {i}?"
        rows.append({"question_id": f"polymarket_0x{i:04x}" + "ab" * 30, "question_text": text,
                     "question_close_date": "2025-02-19", "volume": 1000.0 + i,
                     "resolution": RESOLUTIONS[i % len(RESOLUTIONS)], "prompt": TURTEL_PROMPT.format(text=text, index=i),
                     "prediction_polymarket": 0.7 if RESOLUTIONS[i % len(RESOLUTIONS)] else 0.2,
                     "prediction_o1": 0.5, "prediction_remax": 0.4, "reasoning_remax": "..."})
    return [mg.question_from_row(r) for r in rows]




class EngineTests(unittest.TestCase):
    def test_bootstrap_and_marginal_trade(self):
        ids = [f"v{i}" for i in range(5)]
        market = mg.Market.fresh(ids, liquidity=100.0, max_width=8)
        self.assertEqual(market.prices(), {v: 0.5 for v in ids})
        fills, rejected = market.execute([mg.Trade("v0", 0.7)], None)
        self.assertEqual(rejected, [])
        self.assertEqual(len(fills), 1)
        fill = fills[0]
        self.assertAlmostEqual(fill["pre"], 0.5)
        self.assertAlmostEqual(fill["post"], 0.7, places=9)
        self.assertAlmostEqual(fill["cost"], 100.0 * math.log(0.5 / 0.3), places=5)
        self.assertAlmostEqual(fill["kl"], mg.binary_kl(0.7, 0.5))
        self.assertAlmostEqual(market.price("v0"), 0.7, places=9)
        self.assertAlmostEqual(market.price("v1"), 0.5)
        self.assertEqual(market.version, 1)

    def test_conditional_trade_is_i_projection(self):
        ids = [f"v{i}" for i in range(4)]
        market = mg.Market.fresh(ids)
        market.execute([mg.Trade("v1", 0.6)], None)
        fills, rejected = market.execute([mg.Trade("v0", 0.9, (("v1", "yes"),))], None)
        self.assertEqual(rejected, [])
        # only the context slice moved; P(v1) unchanged; marginal reprices by mixture
        self.assertAlmostEqual(market.price("v0", {"v1": "yes"}), 0.9, places=9)
        self.assertAlmostEqual(market.price("v0", {"v1": "no"}), 0.5, places=9)
        self.assertAlmostEqual(market.price("v1"), 0.6, places=9)
        self.assertAlmostEqual(market.price("v0"), 0.9 * 0.6 + 0.5 * 0.4, places=9)
        self.assertEqual(market.links["v0"], {"v1"})
        self.assertEqual(market.links["v1"], {"v0"})
        self.assertAlmostEqual(fills[0]["pre"], 0.5)
        self.assertAlmostEqual(fills[0]["post_marginal"], 0.74, places=9)
        self.assertGreaterEqual(market.stats()["treewidth"], 1.0)
        view = market.view(["v0"])
        self.assertEqual(view["variables"]["v0"]["conditionals"][0]["other"], "v1")
        # snapshot round trip preserves structure and prices
        clone = mg.Market.from_snapshot(market.snapshot())
        self.assertEqual(clone.prices(), market.prices())
        self.assertAlmostEqual(clone.price("v0", {"v1": "yes"}), 0.9, places=9)
        self.assertEqual(clone.links, market.links)

    def test_width_budget_rejection_is_recorded(self):
        ids = [f"v{i}" for i in range(6)]
        market = mg.Market.fresh(ids, max_width=1)
        fills, rejected = market.execute([mg.Trade("v0", 0.6, (("v1", "yes"), ("v2", "yes")))], None)
        self.assertEqual(fills, [])
        self.assertEqual(rejected[0]["reason"], "engine_error")
        self.assertIn("treewidth", rejected[0]["detail"])
        self.assertEqual(market.prices(), {v: 0.5 for v in ids})  # state untouched

    def test_kl_budget_enforcement(self):
        ids = [f"v{i}" for i in range(4)]
        market = mg.Market.fresh(ids)
        trades = [mg.Trade("v0", 0.7), mg.Trade("v1", 0.7), mg.Trade("v2", 0.7), mg.Trade("v3", 0.51)]
        unit = mg.binary_kl(0.7, 0.5)
        fills, rejected = market.execute(trades, budget=2.5 * unit)
        self.assertEqual([f["trade"]["variable"] for f in fills], ["v0", "v1"])
        self.assertEqual([r["trade"]["variable"] for r in rejected], ["v2", "v3"])
        self.assertTrue(all(r["reason"] == "kl_budget" for r in rejected))
        self.assertLessEqual(fills[-1]["kl_cumulative"], 2.5 * unit)
        self.assertAlmostEqual(market.price("v3"), 0.5)  # a cheap trade after the cut is still rejected

    def test_budget_schedule(self):
        self.assertIsNone(mg.budget_for_pass(1, 8))
        self.assertAlmostEqual(mg.budget_for_pass(2, 8), 2.4)
        self.assertAlmostEqual(mg.budget_for_pass(3, 7), 2.1)
        self.assertAlmostEqual(mg.budget_for_pass(1, 8, first_pass_uncapped=False), 2.4)


class DatasetTests(unittest.TestCase):
    def test_split_turtel_prompt(self):
        question = synthetic_questions(1)[0]
        self.assertTrue(question.evidence.startswith("Question Background:"))
        self.assertTrue(question.evidence.endswith("</News>"))
        self.assertIn("Today's date: 2025-02-14 16:00:04+00:00", question.evidence)
        self.assertEqual(question.today, "2025-02-14 16:00:04+00:00")
        self.assertNotIn("Think about your answer", question.evidence)
        self.assertNotIn("expert superforecaster", question.evidence)

    def test_table1_verification(self):
        cache = mg.QUESTION_CACHE
        if not mg.PARQUET_PATH.exists():
            self.skipTest("test set parquet not mirrored")
        try:
            questions = mg.load_questions(mg.PARQUET_PATH, cache)
        except RuntimeError as exc:
            self.skipTest(str(exc))
        self.assertEqual(len(questions), 1265)
        result = mg.verify_table1(questions)
        self.assertTrue(result["ok"], result)
        self.assertAlmostEqual(result["computed"]["prediction_polymarket"]["brier"], 0.151, delta=0.002)
        self.assertAlmostEqual(result["computed"]["prediction_o1"]["brier"], 0.202, delta=0.002)
        self.assertAlmostEqual(result["computed"]["prediction_remax"]["brier"], 0.191, delta=0.002)
        self.assertTrue(all(q.today for q in questions))
        self.assertTrue(all(q.evidence.endswith("</News>") for q in questions))


class GroupingTests(unittest.TestCase):
    def test_siblings_and_groups(self):
        questions = synthetic_questions()
        grouper = mg.Grouper(questions)
        kinds = {tuple(sorted(s.members)): s for s in grouper.sibling_sets}
        by_text = {q.text: q.question_id for q in questions}
        award = tuple(sorted(by_text[t] for t in TEXTS[0:3]))
        versus = tuple(sorted(by_text[t] for t in TEXTS[3:5]))
        bins = tuple(sorted(by_text[t] for t in TEXTS[5:8]))
        self.assertIn(award, kinds)
        self.assertEqual(kinds[award].kind, "award")
        self.assertTrue(kinds[award].exclusive)
        self.assertIn(versus, kinds)
        self.assertEqual(kinds[versus].kind, "versus")
        self.assertIn(bins, kinds)
        self.assertEqual(kinds[bins].kind, "bins")  # the "or higher" top bin joins the between-bins of the same date
        other_date = by_text[TEXTS[8]]
        self.assertNotIn(other_date, kinds[bins].members)
        thresholds = tuple(sorted(by_text[t] for t in TEXTS[11:13]))
        self.assertIn(thresholds, kinds)
        self.assertEqual(kinds[thresholds].kind, "series")
        self.assertFalse(kinds[thresholds].exclusive)
        nhl = tuple(sorted(by_text[t] for t in TEXTS[13:15]))
        self.assertIn(nhl, kinds)
        bitcoin = {by_text[TEXTS[9]], by_text[TEXTS[10]]}
        self.assertFalse(any(bitcoin <= set(s.members) for s in grouper.sibling_sets))  # different dates
        groups1 = grouper.groups_for_pass(1, 8, seed=0)
        groups2 = grouper.groups_for_pass(2, 8, seed=0)
        self.assertEqual(sorted(len(g) for g in groups1), [6, 7, 7])
        self.assertEqual(sorted(q for g in groups1 for q in g), sorted(q.question_id for q in questions))
        for members in (award, versus, bins):
            self.assertTrue(any(set(members) <= set(g) for g in groups1), members)
        self.assertNotEqual(groups1, groups2)
        self.assertEqual(groups1, grouper.groups_for_pass(1, 8, seed=0))  # deterministic

    def test_group_sizes(self):
        self.assertEqual(mg.group_sizes(1265, 8), [8] * 152 + [7] * 7)
        self.assertEqual(sum(mg.group_sizes(1265, 8)), 1265)
        self.assertEqual(mg.group_sizes(20, 8), [7, 7, 6])
        self.assertEqual(mg.group_sizes(0, 8), [])


class PromptTests(unittest.TestCase):
    def test_prompt_determinism_and_content(self):
        questions = synthetic_questions(3)
        ids = [q.question_id for q in questions]
        market = mg.Market.fresh(ids)
        market.execute([mg.Trade(ids[0], 0.62), mg.Trade(ids[1], 0.8, ((ids[0], "yes"),))], None)
        view = market.view(ids)
        kwargs = {"session_id": "p2-g001", "pass_index": 2, "liquidity": 100.0}
        a = mg.build_prompt(questions, view, 2.4, **kwargs)
        b = mg.build_prompt(questions, market.view(ids), 2.4, **kwargs)
        self.assertEqual(a, b)
        self.assertEqual(mg.sha256_text(a), mg.sha256_text(b))
        self.assertIn("Do not use any knowledge from after that date, even if you have it; pretend you do not.", a)
        self.assertIn("eps_total = 2.400 nats", a)
        self.assertIn("b = 100", a)
        self.assertIn("called off", a)
        self.assertIn("log p_new(w) - log p_old(w)", a)
        for question in questions:
            self.assertIn(f"(full id {question.question_id})", a)
            self.assertIn(f"question: {question.text}", a)
            self.assertIn(question.evidence, a)
        self.assertIn("current market probability of \"yes\": 0.620", a)
        self.assertIn("Today's date: 2025-02-14 16:00:04+00:00", a)
        self.assertIn("P(this=yes | other=yes) = 0.800", a)
        self.assertIn('{"trades": [{"variable": "q1", "target": 0.xx, "context": [{"variable": "q2", "outcome": "yes"}]}]}', a)
        self.assertIn("At most 2 trades per question", a)
        self.assertIn("within [0.01, 0.99]", a)
        self.assertNotIn("PREVIOUS ATTEMPT REJECTED", a)
        uncapped = mg.build_prompt(questions, view, None, session_id="p1-g000", pass_index=1)
        self.assertIn("this pass is uncapped", uncapped)
        retry = mg.build_prompt(questions, view, 2.4, feedback=["unknown_variable: 'x'"], **kwargs)
        self.assertIn("PREVIOUS ATTEMPT REJECTED", retry)
        self.assertIn("- unknown_variable: 'x'", retry)
        self.assertNotEqual(mg.sha256_text(retry), mg.sha256_text(a))

    def test_developer_rule_frozen(self):
        self.assertIn("Do not use tools", mg.DEVELOPER_RULE)
        self.assertEqual(len(mg.sha256_text(mg.DEVELOPER_RULE)), 64)


class ParserTests(unittest.TestCase):
    def test_clean_block_with_prose(self):
        text = 'Rationale first.\n```json\n{"trades": [{"variable": "a", "target": 0.6}]}\n```\nThanks.'
        trades, errors = mg.parse_trades(text)
        self.assertEqual(trades, [{"variable": "a", "target": 0.6}])
        self.assertEqual(errors, [])

    def test_last_of_two_blocks_wins(self):
        text = ('Example: ```json\n{"trades": [{"variable": "wrong", "target": 0.1}]}\n```\n'
                'Final:\n```json\n{"trades": [{"variable": "right", "target": 0.9}]}\n```')
        trades, errors = mg.parse_trades(text)
        self.assertEqual(trades[0]["variable"], "right")

    def test_trailing_comma_rejected_cleanly(self):
        text = '```json\n{"trades": [{"variable": "a", "target": 0.6},]}\n```'
        trades, errors = mg.parse_trades(text)
        self.assertIsNone(trades)
        self.assertTrue(errors and "invalid JSON" in errors[0])

    def test_inline_fallback_and_broken_last_block(self):
        text = 'No fences here but {"trades": [{"variable": "a", "target": 0.55}]} inline.'
        trades, errors = mg.parse_trades(text)
        self.assertEqual(trades[0]["target"], 0.55)
        text = '```json\n{"trades": [{"variable": "a", "target": 0.6}]}\n```\n```json\n{"trades": [oops]}\n```'
        trades, errors = mg.parse_trades(text)
        self.assertEqual(trades[0]["variable"], "a")  # falls back to the earlier valid block
        self.assertTrue(errors)

    def test_empty_and_missing(self):
        self.assertEqual(mg.parse_trades("")[0], None)
        trades, errors = mg.parse_trades("I decline to trade.")
        self.assertIsNone(trades)
        self.assertTrue(errors)
        trades, errors = mg.parse_trades('```json\n{"positions": []}\n```')
        self.assertIsNone(trades)
        self.assertIn("no \"trades\" list", errors[0])


class ValidationTests(unittest.TestCase):
    def test_reasons(self):
        ids = ["polymarket_0xaaaa" + "1" * 60, "polymarket_0xbbbb" + "2" * 60, "polymarket_0xcccc" + "3" * 60]
        raw = [
            {"variable": ids[0], "target": 0.6},
            {"variable": "polymarket_0xzzzz", "target": 0.6},
            {"variable": ids[1], "target": 1.2},
            {"variable": ids[1], "target": "high"},
            {"variable": ids[1], "target": 0.4, "context": [{"variable": ids[1], "outcome": "yes"}]},
            {"variable": ids[1], "target": 0.4, "context": [{"variable": ids[0], "outcome": "yes"}, {"variable": ids[0], "outcome": "no"}]},
            {"variable": ids[1], "target": 0.4, "context": [{"variable": ids[0], "outcome": "maybe"}]},
            {"variable": ids[1], "target": 0.4, "context": [{"variable": "polymarket_0xdddd" + "4" * 60, "outcome": "yes"}]},
            {"variable": ids[0][:20], "target": 0.65},  # unique prefix resolves
            {"variable": ids[0], "target": 0.7},  # third trade on ids[0]
            "not an object",
            {"variable": ids[2], "target": 0.5, "context": "bad"},
            {"variable": ids[2], "target": 0.01, "context": None},
        ]
        result = mg.validate_trades(raw, ids)
        reasons = [mg.reason_code(v.error) if v.error else "ok" for v in result]
        self.assertEqual(reasons, ["ok", "unknown_variable", "target_range", "target_not_number", "context_self",
                                   "context_duplicate", "context_bad_outcome", "context_unknown_variable", "ok",
                                   "too_many_trades", "malformed", "context_malformed", "ok"])
        self.assertEqual(result[8].trade.variable, ids[0])
        self.assertEqual(result[12].trade.target, 0.01)


class ProviderTests(unittest.TestCase):
    def test_mock_is_deterministic_and_parseable(self):
        questions = synthetic_questions(8)
        ids = [q.question_id for q in questions]
        provider = mg.MockProvider({q.question_id: q.resolution for q in questions}, seed=1, garbage_rate=0.0)
        request = mg.SessionRequest("p1-g000", 1, 1, 0, "prompt", mg.DEVELOPER_RULE, ids, {q: 0.5 for q in ids}, None)
        a, b = provider.complete(request), provider.complete(request)
        self.assertEqual(a.text, b.text)
        trades, errors = mg.parse_trades(a.text)
        self.assertEqual(errors, [])
        validated = mg.validate_trades(trades, ids)
        self.assertTrue(any(v.trade for v in validated))

    def test_mock_garbage_is_rejected_then_retry_succeeds(self):
        questions = synthetic_questions(4)
        ids = [q.question_id for q in questions]
        provider = mg.MockProvider({q.question_id: q.resolution for q in questions}, seed=0, garbage_rate=1.0)
        first = mg.SessionRequest("s", 1, 1, 0, "prompt", mg.DEVELOPER_RULE, ids, {q: 0.5 for q in ids}, None)
        self.assertIsNone(mg.parse_trades(provider.complete(first).text)[0])
        second = mg.SessionRequest("s", 2, 1, 0, "prompt", mg.DEVELOPER_RULE, ids, {q: 0.5 for q in ids}, None)
        self.assertIsNotNone(mg.parse_trades(provider.complete(second).text)[0])

    def test_codex_command_matches_harness(self):
        prompt, empty, output = "PROMPT TEXT", Path("/tmp/x/empty"), Path("/tmp/x/final.txt")
        cmd, env, stdin = mg.codex_command("astra", prompt, empty, output, "high", environ={"PATH": "/bin", "OPENAI_API_KEY": "x", "TMUX": "y"})
        self.assertEqual(cmd[0], mg.CODEX_EXECUTABLE)
        self.assertEqual(cmd[1:4], ["exec", "--ephemeral", "--json"])
        self.assertIn("--model", cmd)
        self.assertEqual(cmd[cmd.index("--model") + 1], "gpt-6-astra")
        self.assertIn('model_reasoning_effort="high"', cmd)
        self.assertIn("developer_instructions=" + json.dumps(mg.DEVELOPER_RULE), cmd)
        self.assertEqual(cmd[-3:], ["-o", str(output), "-"])
        self.assertEqual(stdin, prompt)
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("TMUX", env)
        self.assertEqual(env["FLEET_ACTOR"], "consistency-" + mg.CAMPAIGN)
        luna, _, _ = mg.codex_command("luna", prompt, empty, output, "low", environ={})
        self.assertEqual(luna[luna.index("--model") + 1], "gpt-5.6-luna")
        try:
            from harness import run_native_checks as native
        except Exception as exc:  # pragma: no cover - optional dependency (requests)
            self.skipTest(f"harness.run_native_checks not importable: {exc}")
        for provider in ("astra", "luna"):
            expected_cmd, expected_env, expected_stdin = native.native_command(
                provider, prompt, empty, output, "medium", rule=mg.DEVELOPER_RULE, campaign=mg.CAMPAIGN)
            got_cmd, got_env, got_stdin = mg.codex_command(provider, prompt, empty, output, "medium")
            self.assertEqual(got_cmd, expected_cmd)
            self.assertEqual(got_stdin, expected_stdin)
            self.assertEqual(got_env, expected_env)

    def test_decode_codex_events(self):
        events = [{"type": "thread.started", "thread_id": "t-123"},
                  {"type": "item.completed", "item": {"type": "reasoning", "text": "..."}},
                  {"type": "item.completed", "item": {"type": "agent_message", "text": "final ```json\n{\"trades\": []}\n```"}},
                  {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}}]
        stdout = "\n".join(json.dumps(e) for e in events) + "\nnot json\n"
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / "final.txt"
            decoded = mg.decode_codex_events(stdout, final)
            self.assertEqual(decoded["errors"], [])
            self.assertEqual(decoded["thread_id"], "t-123")
            self.assertEqual(decoded["usage"]["output_tokens"], 5)
            self.assertIn('"trades"', decoded["text"])
            final.write_text("different")
            decoded = mg.decode_codex_events(stdout, final)
            self.assertIn("native_final_file_or_message_mismatch", decoded["errors"])
            self.assertEqual(decoded["text"], "different")
        bad = "\n".join(json.dumps(e) for e in events[:2] + [{"type": "turn.failed"}])
        decoded = mg.decode_codex_events(bad, None)
        self.assertIn("native_turn_failed", decoded["errors"])
        self.assertIn("native_completion_not_established", decoded["errors"])

    def test_dry_run_plan(self):
        provider = mg.CodexProvider("astra", effort="high", work_root=Path(tempfile.mkdtemp()))
        request = mg.SessionRequest("p1-g000", 1, 1, 0, "hello", mg.DEVELOPER_RULE, ["a"], {"a": 0.5}, None)
        plan = provider.dry_run(request)
        self.assertIn("--model gpt-6-astra", plan["command_line"])
        self.assertEqual(plan["stdin"], "hello")
        self.assertTrue(plan["output_file"].endswith("final.txt"))


def run_arena(out: Path, questions, passes=2, workers=1, seed=0, resume=False, snapshot_every=50,
              budget_factor=mg.BUDGET_FACTOR, **mock) -> mg.Arena:
    options = {"flip_rate": 0.0, "jitter": 0.0, "garbage_rate": 0.0, "invalid_rate": 0.0,
               "conditional_rate": 0.0, "big_jump_rate": 0.0, "step": 0.5}
    options.update(mock)
    config = mg.RunConfig(out=str(out), provider="mock", n_per_session=4, passes=passes, workers=workers, seed=seed,
                          snapshot_every=snapshot_every, budget_factor=budget_factor, mock_options=options)
    provider = mg.MockProvider({q.question_id: q.resolution for q in questions}, seed=seed, **options)
    arena = mg.Arena(config, questions, provider, log=lambda m: None)
    arena.prepare(resume=resume)
    arena.run()
    return arena


class QuotaGateTests(unittest.TestCase):
    def test_pause_then_continue(self):
        with tempfile.TemporaryDirectory() as tmp:
            flag = Path(tmp) / "flag"
            # prints PAUSE on the first two calls, then nothing
            command = f"if [ ! -f {flag} ]; then echo 1 > {flag}; echo PAUSE; elif [ \"$(cat {flag})\" = 1 ]; then echo 2 > {flag}; echo PAUSE; fi"
            messages = []
            gate = mg.QuotaGate(command, poll=0.01, min_interval=0.0, log=messages.append)
            gate.wait()
            self.assertEqual(gate.pauses, 2)
            self.assertEqual(len(messages), 2)
            gate.wait()
            self.assertEqual(gate.pauses, 2)
        self.assertEqual(mg.QuotaGate(None).wait(), None)


class ArenaTests(unittest.TestCase):
    def test_full_loop_scores_drop_each_pass(self):
        questions = synthetic_questions(12)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            arena = run_arena(out, questions, passes=3, workers=3)
            self.assertEqual(arena.stats["sessions"], 9)
            self.assertEqual(arena.stats["failed"], 0)
            records = [json.loads(l) for l in (out / "sessions.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 9)
            self.assertTrue(all(r["status"] == "completed" for r in records))
            self.assertTrue(all(len(r["prompt_sha256"]) == 64 for r in records))
            self.assertTrue(all("snapshot_version" in r and "timings" in r and "raw_response" in r for r in records))
            for p in (1, 2, 3):
                self.assertTrue((out / "snapshots" / f"pass-{p}.json").exists())
            result = score.score_run(out)
            briers = [row["brier"] for row in result["per_pass"]]
            self.assertEqual(len(briers), 3)
            self.assertLess(briers[0], result["baselines"]["constant_0.5"]["brier"])
            self.assertLess(briers[1], briers[0])
            self.assertLess(briers[2], briers[1])
            self.assertTrue(all(row["brier_improvement"] > 0 for row in result["per_pass"]))
            sessions = result["sessions"]
            self.assertEqual(sessions["validity_rate"], 1.0)
            self.assertTrue(sessions["telescoping"]["unlinked_ok"])
            self.assertGreater(sessions["payoff"]["total"], 0)
            self.assertEqual(sessions["payoff"]["share_positive"], 1.0)
            self.assertIsNotNone(result["herding"]["pooled_autocorrelation"])
            text = score.markdown(result)
            self.assertIn("| 3 |", text)

    def test_rejections_retries_and_budget_in_log(self):
        questions = synthetic_questions(8)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            arena = run_arena(out, questions, passes=2, workers=2, invalid_rate=1.0, conditional_rate=0.5,
                              big_jump_rate=0.5, garbage_rate=1.0, budget_factor=0.05)
            records = [json.loads(l) for l in (out / "sessions.jsonl").read_text().splitlines()]
            self.assertTrue(any(r["status"] == "retry" for r in records))
            retried = [r for r in records if r["attempt"] == 2]
            self.assertTrue(retried and all(r["feedback"] for r in retried))
            reasons = {x["reason"] for r in records for x in r["rejected"]}
            self.assertTrue(reasons & {"target_range", "unknown_variable", "context_self", "context_duplicate"})
            self.assertIn("kl_budget", reasons)
            pass2 = [r for r in records if r["pass"] == 2 and r["status"] == "completed"]
            self.assertTrue(all(r["kl_used"] <= r["kl_budget"] + 1e-9 for r in pass2))
            result = score.score_run(out)
            self.assertLess(result["sessions"]["validity_rate"], 1.0)
            self.assertGreater(result["sessions"]["conditional_trades_executed"], 0)
            self.assertGreater(result["sessions"]["telescoping"]["linked_variables"], 0)

    def test_resume_reload_equals_snapshot(self):
        questions = synthetic_questions(12)
        with tempfile.TemporaryDirectory() as tmp:
            reference = run_arena(Path(tmp) / "ref", questions, passes=2, workers=1, conditional_rate=0.3)
            partial_dir = Path(tmp) / "partial"
            partial = run_arena(partial_dir, questions, passes=1, workers=1, conditional_rate=0.3)
            resumed = run_arena(partial_dir, questions, passes=2, workers=1, resume=True, conditional_rate=0.3)
            self.assertEqual(resumed.market.prices(), reference.market.prices())
            self.assertEqual(resumed.market.snapshot()["engine"], reference.market.snapshot()["engine"])
            self.assertEqual(resumed.stats["sessions"], reference.stats["sessions"])
            records = [json.loads(l) for l in (partial_dir / "sessions.jsonl").read_text().splitlines()]
            self.assertEqual(len({r["session_id"] for r in records}), len(records))  # no session ran twice
            # replay path: latest.json every 2 sessions, pass snapshot removed
            replay_dir = Path(tmp) / "replay"
            run_arena(replay_dir, questions, passes=1, workers=1, snapshot_every=2, conditional_rate=0.3)
            (replay_dir / "snapshots" / "pass-1.json").unlink()
            config = mg.RunConfig(out=str(replay_dir), provider="mock", n_per_session=4, passes=1, workers=1, seed=0,
                                  snapshot_every=2, mock_options=partial.config.mock_options)
            arena = mg.Arena(config, questions, partial.provider, log=lambda m: None)
            arena.prepare(resume=True)
            self.assertEqual(arena.market.prices(), partial.market.prices())
            self.assertEqual(arena.completed, partial.completed)
            # a resume with a different configuration is refused
            bad = mg.RunConfig(out=str(replay_dir), provider="mock", n_per_session=5, passes=1, seed=0,
                               mock_options=partial.config.mock_options)
            with self.assertRaises(ValueError):
                mg.Arena(bad, questions, partial.provider, log=lambda m: None).prepare(resume=True)

    def test_select_questions_limit(self):
        questions = synthetic_questions(12)
        subset = mg.select_questions(questions, 5, seed=3)
        self.assertEqual(len(subset), 5)
        self.assertEqual(subset, mg.select_questions(questions, 5, seed=3))
        self.assertEqual(mg.select_questions(questions, None, 0), questions)


if __name__ == "__main__":
    unittest.main()
