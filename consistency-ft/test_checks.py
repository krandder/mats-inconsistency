#!/usr/bin/env python3
"""Small dependency-free formula check: python3 test_checks.py."""

import math

from common import violation


CASES = {
    "Negation": ({"P": 0.7, "not_P": 0.5}, 0.2),
    "Paraphrase": ({"P": 0.7, "para_P": 0.4}, 0.3),
    "Consequence": ({"P": 0.7, "cons_P": 0.5}, 0.2),
    "And": ({"P": 0.8, "Q": 0.7, "P_and_Q": 0.4}, 0.1),
    "Or": ({"P": 0.7, "Q": 0.4, "P_or_Q": 0.6}, 0.1),
    "AndOr": ({"P": 0.7, "Q": 0.4, "P_and_Q": 0.2, "P_or_Q": 0.8}, 0.1),
    "But": ({"P": 0.6, "Q_and_not_P": 0.3, "P_or_Q": 0.7}, 0.2),
    "Cond": ({"P": 0.8, "Q_given_P": 0.5, "P_and_Q": 0.3}, 0.1),
    "CondCond": ({"P": 0.8, "Q_given_P": 0.5, "R_given_P_and_Q": 0.5,
                   "P_and_Q_and_R": 0.1}, 0.1),
    "ExpEvidence": ({"P": 0.6, "Q": 0.5, "P_given_Q": 0.8,
                      "P_given_not_Q": 0.2}, 0.1),
}

for check_type, (probabilities, expected) in CASES.items():
    actual = violation(check_type, probabilities)
    assert math.isclose(actual, expected, abs_tol=1e-12), (check_type, actual, expected)

assert violation("Consequence", {"P": 0.4, "cons_P": 0.5}) == 0
assert math.isclose(violation("And", {"P": 0.8, "Q": 0.3, "P_and_Q": 0.4}), 0.1)
assert math.isclose(violation("Or", {"P": 0.3, "Q": 0.4, "P_or_Q": 0.8}), 0.1)

print(f"PASS: {len(CASES)} check formulas")
