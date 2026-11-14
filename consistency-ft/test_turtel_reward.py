"""Unit tests for the Turtel reward parser, guardrails and modified-GRPO advantages (stdlib only)."""

import math
import unittest

from training.turtel_prompt import (FOOTER, HEADER, Article, order_articles, parse_prompt, render_prompt,
                                    structural_signature)
from training.turtel_reward import (GuardrailConfig, expected_calibration_error, gibberish_proportion,
                                    group_advantages, non_english_proportion, parse_probability,
                                    score_completion, soft_brier)

REASONING = ("Okay, so I need to figure out the probability that Apple launches an iPhone SE on February 19. "
             "Several outlets report a launch event on that date, and the supply-chain reporting is consistent, "
             "but Gurman suggests the SE might be held for a bigger event in March. Base rates for rumoured Apple "
             "dates are mixed: announcements often slip by a week or two. Weighing the recent reports more heavily, "
             "I lean towards a launch but with real uncertainty about the exact date.")


class ParseProbabilityTests(unittest.TestCase):
    def test_asterisk_formats(self):
        self.assertEqual(parse_probability("<think>x</think>\n\n*0.35*"), 0.35)
        self.assertEqual(parse_probability("*.35*"), 0.35)
        self.assertEqual(parse_probability("*0*"), 0.0)
        self.assertEqual(parse_probability("*1*"), 1.0)
        self.assertEqual(parse_probability("*35%*"), 0.35)
        self.assertEqual(parse_probability("**0.35**"), 0.35)
        self.assertEqual(parse_probability("**Final Answer:** *0.72*"), 0.72)
        self.assertEqual(parse_probability("*0.35*."), 0.35)
        self.assertEqual(parse_probability("</think> *0.000*"), 0.0)

    def test_last_asterisk_wins_and_answer_region_preferred(self):
        text = "<think>maybe *0.9* at first, then lower</think>\nI'll say *0.2*\nFinal: *0.25*"
        self.assertEqual(parse_probability(text), 0.25)
        text = "<think>I considered *0.9*</think>\nProbability: 0.4"
        self.assertEqual(parse_probability(text), 0.4)

    def test_probability_label(self):
        self.assertEqual(parse_probability("Probability: 0.42"), 0.42)
        self.assertEqual(parse_probability("probability = 0.42"), 0.42)
        self.assertEqual(parse_probability("**Probability**: 42%"), 0.42)
        self.assertEqual(parse_probability("The probability that it happens is 0.15"), 0.15)

    def test_last_number_on_final_line(self):
        self.assertEqual(parse_probability("reasoning here\nFinal answer: 0.7"), 0.7)
        self.assertEqual(parse_probability("reasoning\nI'd put it at 0.65 or so, I'll go with 0.6"), 0.6)
        self.assertEqual(parse_probability("reasoning\nAround 35% likely"), 0.35)
        self.assertEqual(parse_probability("one line 0.55"), 0.55)
        self.assertIsNone(parse_probability("reasoning\nsee article 1 and 0 others"))
        self.assertEqual(parse_probability("reasoning\nsee article 1; I estimate 0.2"), 0.2)

    def test_unparseable(self):
        self.assertIsNone(parse_probability(""))
        self.assertIsNone(parse_probability("I cannot answer this."))
        self.assertIsNone(parse_probability("reasoning\nthe answer is 1.5"))
        self.assertIsNone(parse_probability("reasoning\n*2.5*"))
        self.assertIsNone(parse_probability("<think>" + "x " * 100))

    def test_out_of_range_asterisk_falls_through(self):
        self.assertEqual(parse_probability("*150%* no wait\nProbability: 0.3"), 0.3)

    def test_truncation_at_16000_chars(self):
        long = "<think>" + "reasoning " * 1700 + "</think>\n*0.3*"
        self.assertGreater(len(long), 16000)
        self.assertIsNone(parse_probability(long))
        self.assertEqual(parse_probability(long, max_chars=len(long)), 0.3)


class GuardrailTests(unittest.TestCase):
    def test_non_english(self):
        self.assertEqual(non_english_proportion("all english here"), 0.0)
        self.assertGreater(non_english_proportion("we think 这个问题 很难 决定"), 0.3)
        self.assertEqual(non_english_proportion("déjà vu café"), 0.0)

    def test_gibberish(self):
        self.assertLess(gibberish_proportion(REASONING), 0.05)
        self.assertGreater(gibberish_proportion("xkq zzzzzzz qwrtp bcdfg hjklm nprst " * 4), 0.5)
        loop = " ".join(["the market will resolve yes because"] * 20)
        self.assertGreater(gibberish_proportion(loop), 0.8)

    def test_score_parsed(self):
        result = score_completion(f"<think>{REASONING}</think>\n\n*0.3*", 1)
        self.assertTrue(result.parsed)
        self.assertAlmostEqual(result.brier, 0.49)
        self.assertTrue(result.explains_answer)
        self.assertAlmostEqual(result.penalty, 0.0)
        self.assertAlmostEqual(result.reward, -0.49)

    def test_score_unparsed_strict(self):
        result = score_completion(f"<think>{REASONING}</think>\n\nno idea", 0)
        self.assertFalse(result.parsed)
        self.assertEqual(result.brier, 1.0)
        self.assertAlmostEqual(result.reward, -1.0)

    def test_missing_explanation_penalty(self):
        config = GuardrailConfig(lambda_miss=0.25)
        result = score_completion("<think></think>\n*0.5*", 1, config)
        self.assertFalse(result.explains_answer)
        self.assertAlmostEqual(result.penalty, 0.25)
        self.assertAlmostEqual(result.reward, -0.25 - 0.25)

    def test_language_penalty_scales(self):
        config = GuardrailConfig(lambda_lang=0.5, lambda_miss=0.0)
        chinese = " ".join(["这是一个中文推理"] * 40)
        result = score_completion(f"<think>{chinese}</think>\n*0.5*", 1, config)
        self.assertGreater(result.non_english_proportion, 0.9)
        self.assertGreater(result.penalty, 0.45)

    def test_truncated_for_scoring_flag(self):
        result = score_completion("<think>" + "w " * 9000 + "</think>\n*0.4*", 1)
        self.assertTrue(result.truncated_for_scoring)
        self.assertFalse(result.parsed)


class AdvantageTests(unittest.TestCase):
    def test_group_mean_without_std(self):
        rewards = [-0.04, -0.64, -1.0, -0.16, -0.25, -0.25, -0.25, -0.25]
        adv = group_advantages(rewards, 4)
        self.assertAlmostEqual(sum(adv[:4]), 0.0)
        self.assertAlmostEqual(adv[0], -0.04 - (-0.46))
        self.assertEqual(adv[4:], [0.0, 0.0, 0.0, 0.0])
        # Not normalised: scaling rewards scales advantages.
        adv2 = group_advantages([r * 3 for r in rewards], 4)
        self.assertAlmostEqual(adv2[0], 3 * adv[0])

    def test_group_size_mismatch(self):
        with self.assertRaises(ValueError):
            group_advantages([0.0, 1.0, 2.0], 2)


class MetricTests(unittest.TestCase):
    def test_soft_brier(self):
        self.assertAlmostEqual(soft_brier([(None, 1), (0.5, 0)]), 0.25)
        self.assertAlmostEqual(soft_brier([(1.0, 1), (0.0, 0)]), 0.0)
        self.assertTrue(math.isnan(soft_brier([])))

    def test_ece(self):
        self.assertAlmostEqual(expected_calibration_error([(0.95, 1), (0.95, 1), (0.05, 0)]), 0.05)


class PromptTemplateTests(unittest.TestCase):
    def test_round_trip(self):
        articles = order_articles([
            Article("Older", "2025-01-01T00:00:00.000Z", "a.com", "s1"),
            Article("Unknown", "Unknown Date", "b.org", "s2"),
            Article("Newer", "2025-02-01T12:00:00.000Z", "c.net", "s3")])
        self.assertEqual([a.title for a in articles], ["Unknown", "Newer", "Older"])
        text = render_prompt("Will X?", "", "Resolves yes if X.\n\nSecond paragraph.", "2025-02-14 16:00:04+00:00",
                             "2025-02-19 00:00:00+00:00", articles)
        parts = parse_prompt(text)
        self.assertEqual(parts["header"], HEADER)
        self.assertEqual(parts["footer"], FOOTER)
        self.assertEqual(parts["articles"], articles)
        self.assertEqual(render_prompt(parts["question"], parts["background"], parts["criteria"], parts["today"],
                                       parts["close"], parts["articles"]), text)
        empty = render_prompt("Will X?", "", "c", "2025-02-14 16:00:04+00:00", "2025-02-19 00:00:00+00:00", [])
        self.assertIn("<News>\nNo information found\n</News>", empty)
        self.assertEqual(parse_prompt(empty)["articles"], [])
        self.assertEqual(structural_signature(text)[0], "header:exact")
        self.assertEqual(structural_signature(empty)[-2], "news:No information found")


if __name__ == "__main__":
    unittest.main()
