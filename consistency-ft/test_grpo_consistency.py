import math
import unittest

from training.grpo_consistency import FLOOR, incoherence, score_batch
from training.supervised_pairs import FORM_ORDER

COHERENT = {"P": 0.7, "NOT P": 0.3, "Q": 0.4, "NOT Q": 0.6, "(P) AND (Q)": 0.28, "(P) OR (Q)": 0.82,
            "(P) AND (NOT Q)": 0.42, "(P) OR (NOT Q)": 0.88, "(NOT P) AND (Q)": 0.12, "(NOT P) OR (Q)": 0.58,
            "(NOT P) AND (NOT Q)": 0.18, "(NOT P) OR (NOT Q)": 0.72}
LABELS = {"P": 1, "NOT P": 0, "Q": 0, "NOT Q": 1, "(P) AND (Q)": 0, "(P) OR (Q)": 1, "(P) AND (NOT Q)": 1,
          "(P) OR (NOT Q)": 1, "(NOT P) AND (Q)": 0, "(NOT P) OR (Q)": 0, "(NOT P) AND (NOT Q)": 0,
          "(NOT P) OR (NOT Q)": 1}


def batch(override=None, g=2):
    pid, forms, labels, texts = [], [], [], []
    for f in FORM_ORDER:
        for k in range(g):
            pid.append("x"); forms.append(f); labels.append(LABELS[f])
            p = (override or {}).get((f, k), COHERENT[f])
            texts.append("reasoning\nProbability: garbage" if p is None else f"Some reasoning.\nProbability: {p}")
    return pid, forms, labels, texts


class T(unittest.TestCase):
    def test_coherent_set_has_zero_incoherence(self):
        self.assertLess(incoherence(COHERENT)[0], 1e-4)

    def test_accuracy_reward_is_log_score(self):
        r, ex = score_batch("acc", 1.0, *batch(), 2)
        self.assertAlmostEqual(r[0], math.log(0.7), places=6)  # P, label 1

    def test_consistency_prefers_coherent_answer(self):
        r, ex = score_batch("cons", 1.0, *batch({("NOT P", 1): 0.7}), 2)
        i0 = 2  # NOT P rollout 0 (0.3, coherent), rollout 1 (0.7, incoherent)
        self.assertGreater(r[i0], r[i0 + 1])
        self.assertGreater(r[i0], -0.01)

    def test_unparseable_floor_below_everything(self):
        for arm in ("acc", "cons", "acc_cons"):
            r, ex = score_batch(arm, 1.0, *batch({("P", 0): None, ("Q", 1): 0.99}), 2)
            self.assertEqual(r[0], FLOOR[arm])
            self.assertTrue(all(x > FLOOR[arm] for j, x in enumerate(r) if j != 0))

    def test_acc_cons_is_sum(self):
        ra, _ = score_batch("acc", 1.0, *batch({("NOT P", 1): 0.7}), 2)
        rc, _ = score_batch("cons", 1.0, *batch({("NOT P", 1): 0.7}), 2)
        rb, _ = score_batch("acc_cons", 1.0, *batch({("NOT P", 1): 0.7}), 2)
        for a, c, b in zip(ra, rc, rb):
            self.assertAlmostEqual(a + c, b, places=9)

    def test_rejects_partial_pairs(self):
        pid, forms, labels, texts = batch()
        with self.assertRaises(ValueError):
            score_batch("acc", 1.0, pid[:-1], forms[:-1], labels[:-1], texts[:-1], 2)


if __name__ == "__main__":
    unittest.main()
