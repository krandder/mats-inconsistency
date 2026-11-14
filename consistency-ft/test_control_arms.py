"""Control arms for the consistency experiment: prior pull and permuted-form penalty."""
import random
import unittest

from training.supervised_pairs import (FORM_ORDER, FORM_PRIOR, form_permutation, permuted_relation_penalties,
                                       prior_pull_penalty, relation_penalties)


def coherent(seed: int) -> dict[str, float]:
    """Twelve form probabilities implied by a random joint distribution over (P, Q)."""
    rng = random.Random(seed)
    w = [rng.random() + 0.05 for _ in range(4)]
    tt, tf, ft, ff = (x / sum(w) for x in w)
    p, q = tt + tf, tt + ft
    return {"P": p, "NOT P": 1 - p, "Q": q, "NOT Q": 1 - q,
            "(P) AND (Q)": tt, "(P) OR (Q)": 1 - ff, "(P) AND (NOT Q)": tf, "(P) OR (NOT Q)": 1 - ft,
            "(NOT P) AND (Q)": ft, "(NOT P) OR (Q)": 1 - tf,
            "(NOT P) AND (NOT Q)": ff, "(NOT P) OR (NOT Q)": 1 - tt}


class PriorPullTests(unittest.TestCase):
    def test_zero_at_the_prior(self):
        self.assertEqual(FORM_PRIOR["P"], 0.5)
        self.assertEqual(FORM_PRIOR["(NOT P) AND (Q)"], 0.25)
        self.assertEqual(FORM_PRIOR["(P) OR (NOT Q)"], 0.75)
        self.assertEqual(prior_pull_penalty(dict(FORM_PRIOR)), 0.0)

    def test_positive_away_from_the_prior_and_differentiable(self):
        import torch
        values = {f: torch.tensor(FORM_PRIOR[f] + 0.1, requires_grad=True) for f in FORM_ORDER}
        loss = prior_pull_penalty(values)
        self.assertAlmostEqual(float(loss), 0.01, places=6)
        loss.backward()
        self.assertAlmostEqual(float(values["P"].grad), 2 * 0.1 / 12, places=6)


class PermutedPenaltyTests(unittest.TestCase):
    def test_permutation_is_fixed_and_not_identity(self):
        perm = form_permutation()
        self.assertEqual(perm, form_permutation(20260923))
        self.assertEqual(sorted(perm.values()), sorted(FORM_ORDER))
        self.assertGreater(sum(perm[f] != f for f in FORM_ORDER), 6)

    def test_differs_from_real_penalty_on_coherent_input(self):
        for seed in range(5):
            values = coherent(seed)
            self.assertAlmostEqual(relation_penalties(values)["total"], 0.0, places=12)
            self.assertGreater(permuted_relation_penalties(values)["total"], 1e-4)

    def test_zero_on_permuted_coherent_input(self):
        perm = form_permutation()
        for seed in range(5):
            target = coherent(seed)
            # Place coherent value target[f] where the permuted penalty reads slot f.
            values = {perm[f]: target[f] for f in FORM_ORDER}
            self.assertAlmostEqual(permuted_relation_penalties(values)["total"], 0.0, places=12)
            self.assertGreater(relation_penalties(values)["total"], 1e-4)


if __name__ == "__main__":
    unittest.main()
