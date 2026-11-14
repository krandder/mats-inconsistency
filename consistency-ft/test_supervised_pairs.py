import unittest

from training.supervised_pairs import (
    FORM_ORDER,
    render_prompt,
    relation_penalties,
    validate_pair,
)


def fixture_pair():
    pair = {"pair_id": "pair-1", "p_id": "p", "q_id": "q", "observed_query_id": "pair-1-q07"}
    propositions = {
        "p": {"id": "p", "parent_question": "The first coin is heads.",
               "answer_text": "Heads", "criteria_text": "Resolve from the published coin record.\n\nUpdate 2025-10-22: resolved YES, LEAK-ME"},
        "q": {"id": "q", "parent_question": "The second coin is heads.",
               "answer_text": "Heads", "criteria_text": "Resolve from the published coin record.\n\nUpdate 2025-10-22: resolved YES, LEAK-ME"},
    }
    signs = {"P": (1, 0), "NOT P": (-1, 0), "Q": (0, 1), "NOT Q": (0, -1),
             "(P) AND (Q)": (1, 1), "(P) OR (Q)": (1, 1),
             "(P) AND (NOT Q)": (1, -1), "(P) OR (NOT Q)": (1, -1),
             "(NOT P) AND (Q)": (-1, 1), "(NOT P) OR (Q)": (-1, 1),
             "(NOT P) AND (NOT Q)": (-1, -1), "(NOT P) OR (NOT Q)": (-1, -1)}
    rows = [{"query_id": f"pair-1-q{index:02d}", "pair_id": "pair-1", "p_id": "p", "q_id": "q",
             "form": form, "p_sign": signs[form][0], "q_sign": signs[form][1]}
            for index, form in enumerate(FORM_ORDER)]
    # Deliberately scramble rows: semantic names must survive row reordering.
    rows = rows[7:] + rows[:7]
    labels = {"pair-1-q07": 1}
    return pair, rows, labels, propositions


class SupervisedPairTests(unittest.TestCase):
    def test_full_text_prompt_excludes_outcome_and_map_is_named(self):
        pair, rows, labels, propositions = fixture_pair()
        example = validate_pair(pair, rows, labels, propositions)
        self.assertEqual(set(example.relation_map), set(FORM_ORDER))
        self.assertEqual(example.relation_map["P"], "pair-1-q00")
        prompt = render_prompt(example, "P")
        self.assertIn("The first coin is heads", prompt)
        self.assertIn("published coin record", prompt)
        self.assertIn("Option: Heads", prompt)  # option name defines the proposition
        self.assertNotIn("LEAK-ME", prompt)  # creator edit stripped

    def test_incomplete_or_duplicate_forms_fail_closed(self):
        pair, rows, labels, propositions = fixture_pair()
        with self.assertRaises(ValueError):
            validate_pair(pair, rows[:-1], labels, propositions)
        duplicate = rows[:-1] + [dict(rows[0], query_id="pair-1-duplicate")]
        with self.assertRaises(ValueError):
            validate_pair(pair, duplicate, labels, propositions)

    def test_relation_penalties_use_names_and_zero_for_consistent_values(self):
        probabilities = {name: 0.5 for name in FORM_ORDER}
        terms = relation_penalties(probabilities)
        self.assertAlmostEqual(float(terms["total"]), 0.0)
        probabilities["P"] = 0.8
        probabilities["NOT P"] = 0.2
        probabilities["Q"] = 0.3
        probabilities["NOT Q"] = 0.7
        probabilities["(P) AND (Q)"] = 0.24
        probabilities["(P) OR (Q)"] = 0.86
        probabilities["(P) AND (NOT Q)"] = 0.56
        probabilities["(P) OR (NOT Q)"] = 0.94
        probabilities["(NOT P) AND (Q)"] = 0.06
        probabilities["(NOT P) OR (Q)"] = 0.44
        probabilities["(NOT P) AND (NOT Q)"] = 0.14
        probabilities["(NOT P) OR (NOT Q)"] = 0.76
        self.assertAlmostEqual(float(relation_penalties(probabilities)["total"]), 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
