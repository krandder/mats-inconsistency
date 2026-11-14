import json
import tempfile
import unittest
from pathlib import Path

from training.supervised_pairs import FORM_ORDER, LABEL_FILES, load_split, proposition_text, validate_pair

SIGNS = {"P": (1, 0), "NOT P": (-1, 0), "Q": (0, 1), "NOT Q": (0, -1),
         "(P) AND (Q)": (1, 1), "(P) OR (Q)": (1, 1),
         "(P) AND (NOT Q)": (1, -1), "(P) OR (NOT Q)": (1, -1),
         "(NOT P) AND (Q)": (-1, 1), "(NOT P) OR (Q)": (-1, 1),
         "(NOT P) AND (NOT Q)": (-1, -1), "(NOT P) OR (NOT Q)": (-1, -1)}


def write_split(root: Path, split: str) -> None:
    directory = root / split
    directory.mkdir(parents=True)
    pairs, queries, one, full = [], [], [], []
    for index in range(2):
        pair_id, p_id, q_id = f"{split}-pair-{index}", f"p{index}", f"q{index}"
        rows = [{"query_id": f"{pair_id}-q{col:02d}", "pair_id": pair_id, "p_id": p_id, "q_id": q_id,
                 "form": form, "p_sign": SIGNS[form][0], "q_sign": SIGNS[form][1]}
                for col, form in enumerate(FORM_ORDER)]
        observed = rows[5]["query_id"]  # "(P) OR (Q)": a compound, so singles never label it
        pairs.append({"pair_id": pair_id, "p_id": p_id, "q_id": q_id, "p_parent": p_id, "q_parent": q_id,
                      "observed_query_id": observed, "pair_weight": 0.5})
        queries.extend(rows)
        one.append({"query_id": observed, "label": 1})
        full.extend({"query_id": r["query_id"], "label": int(c % 2 == 0)} for c, r in enumerate(rows))
    props = [{"id": f"{k}{i}", "parent_question": f"{k} {i} happens?", "criteria_text": "Resolves YES.",
              "answer_text": "LEAK"} for i in range(2) for k in ("p", "q")]
    singles = [{"id": r["id"], "label": 1} for r in props]
    for name, rows in (("pairs.jsonl", pairs), ("queries.jsonl", queries), ("propositions.jsonl", props),
                       ("one-label-targets.jsonl", one), ("full-label-targets.jsonl", full),
                       ("single-targets.jsonl", singles)):
        (directory / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


class LabelModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_split(self.root, "train")

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_label_mode_labels_only_the_observed_form(self):
        ex = load_split(self.root, "train", labels_file=LABEL_FILES["one"])
        self.assertEqual(ex[0].observed_form, "(P) OR (Q)")
        self.assertEqual(ex[0].labeled_forms, {"(P) OR (Q)": 1})

    def test_full_label_mode_labels_all_twelve_forms(self):
        ex = load_split(self.root, "train", labels_file=LABEL_FILES["full"])
        self.assertEqual(set(ex[0].labeled_forms), set(FORM_ORDER))

    def test_single_mode_labels_p_and_q_only(self):
        ex = load_split(self.root, "train", labels_file=LABEL_FILES["single"])
        self.assertEqual(ex[0].labeled_forms, {"P": 1, "Q": 1})

    def test_observed_label_required_only_for_one_label_mode(self):
        pair = {"pair_id": "x", "p_id": "p0", "q_id": "q0", "observed_query_id": "x-q05"}
        rows = [{"query_id": f"x-q{c:02d}", "pair_id": "x", "p_id": "p0", "q_id": "q0", "form": f,
                 "p_sign": SIGNS[f][0], "q_sign": SIGNS[f][1]} for c, f in enumerate(FORM_ORDER)]
        props = {"p0": {"parent_question": "a"}, "q0": {"parent_question": "b"}}
        with self.assertRaises(ValueError):
            validate_pair(pair, rows, {"x-q00": 1}, props, require_observed=True)
        validate_pair(pair, rows, {"x-q00": 1}, props, require_observed=False)
        with self.assertRaises(ValueError):
            validate_pair(pair, rows, {}, props, require_observed=False)

    def test_long_criteria_are_capped_and_answer_text_never_used(self):
        text = proposition_text({"parent_question": "Q?", "criteria_text": "x" * 5000 + "\n\nUpdate 2025-10-01: LEAK", "answer_text": "OpenAI"})
        self.assertLess(len(text), 1700)
        self.assertNotIn("LEAK", text)
        self.assertIn("Option: OpenAI", text)
        self.assertTrue(text.endswith("[...]"))


if __name__ == "__main__":
    unittest.main()
