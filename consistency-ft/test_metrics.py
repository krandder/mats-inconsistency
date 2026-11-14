#!/usr/bin/env python3
"""Offline exploitability checks: python3 test_metrics.py."""

import math

from common import CHECK_FIELDS
from metrics import exploit


COHERENT = {
    "Negation": {"P": 0.3, "not_P": 0.7},
    "Paraphrase": {"P": 0.3, "para_P": 0.3},
    "Consequence": {"P": 0.3, "cons_P": 0.6},
    "And": {"P": 0.4, "Q": 0.5, "P_and_Q": 0.2},
    "Or": {"P": 0.4, "Q": 0.5, "P_or_Q": 0.7},
    "AndOr": {"P": 0.4, "Q": 0.5, "P_and_Q": 0.2, "P_or_Q": 0.7},
    "But": {"P": 0.4, "Q_and_not_P": 0.3, "P_or_Q": 0.7},
    "Cond": {"P": 0.4, "Q_given_P": 0.5, "P_and_Q": 0.2},
    "CondCond": {"P": 0.4, "Q_given_P": 0.5, "R_given_P_and_Q": 0.5,
                 "P_and_Q_and_R": 0.1},
    "ExpEvidence": {"P": 0.4, "Q": 0.5, "P_given_Q": 0.6,
                    "P_given_not_Q": 0.2},
}

INCOHERENT = {
    "Negation": {"P": 0.1, "not_P": 0.1},
    "Paraphrase": {"P": 0.2, "para_P": 0.8},
    "Consequence": {"P": 0.8, "cons_P": 0.2},
    "And": {"P": 0.8, "Q": 0.7, "P_and_Q": 0.2},
    "Or": {"P": 0.8, "Q": 0.7, "P_or_Q": 0.4},
    "AndOr": {"P": 0.7, "Q": 0.6, "P_and_Q": 0.5, "P_or_Q": 0.4},
    "But": {"P": 0.6, "Q_and_not_P": 0.3, "P_or_Q": 0.5},
    "Cond": {"P": 0.8, "Q_given_P": 0.5, "P_and_Q": 0.2},
    "CondCond": {"P": 0.8, "Q_given_P": 0.5, "R_given_P_and_Q": 0.5,
                   "P_and_Q_and_R": 0.1},
    "ExpEvidence": {"P": 0.8, "Q": 0.5, "P_given_Q": 0.2,
                    "P_given_not_Q": 0.2},
}

assert COHERENT.keys() == INCOHERENT.keys() == CHECK_FIELDS.keys()
for check_type in CHECK_FIELDS:
    coherent = exploit(check_type, COHERENT[check_type])
    incoherent = exploit(check_type, INCOHERENT[check_type])
    assert coherent >= 0 and incoherent >= 0, (check_type, coherent, incoherent)
    assert math.isclose(coherent, 0.0, abs_tol=1e-6), (check_type, coherent)
    assert incoherent > 1e-6, (check_type, incoherent)

assert math.isclose(exploit("Negation", {"P": 0.3, "not_P": 0.7}), 0.0, abs_tol=1e-6)
large = exploit("Negation", {"P": 0.1, "not_P": 0.1})
small = exploit("Negation", {"P": 0.4, "not_P": 0.45})
assert math.isclose(large, 0.8, abs_tol=1e-6)
assert math.isclose(small, 0.15, abs_tol=1e-6)
assert large > small > 0

# Moving twice as far from the same coherent prices cannot reduce exploitability.
base = {"P": 0.3, "not_P": 0.7}
direction = {"P": -0.1, "not_P": -0.1}
scaled = [
    exploit("Negation", {key: base[key] + scale * direction[key] for key in base})
    for scale in (0.0, 1.0, 2.0)
]
assert scaled[0] <= scaled[1] <= scaled[2]

print(f"PASS: exploitability for {len(CHECK_FIELDS)} check types")
