"""Arbitrage exploitability for probabilistic-consistency checks."""

from itertools import combinations, product

from common import CHECK_FIELDS

try:
    from scipy.optimize import linprog
except ImportError:  # pragma: no cover - exercised on dependency-free installs
    linprog = None


def _outcomes(check_type):
    """Return role-ordered outcomes; None marks an unresolved conditional."""
    if check_type == "Negation":
        return [(x, 1 - x) for x in (0, 1)]
    if check_type == "Paraphrase":
        return [(x, x) for x in (0, 1)]
    if check_type == "Consequence":
        return [(p, q) for p, q in product((0, 1), repeat=2) if not p or q]

    outcomes = []
    variables = 3 if check_type == "CondCond" else 2
    for values in product((0, 1), repeat=variables):
        p, q = values[:2]
        if check_type == "And":
            outcome = (p, q, p & q)
        elif check_type == "Or":
            outcome = (p, q, p | q)
        elif check_type == "AndOr":
            outcome = (p, q, p & q, p | q)
        elif check_type == "But":
            outcome = (p, q & (1 - p), p | q)
        elif check_type == "Cond":
            outcome = (p, q if p else None, p & q)
        elif check_type == "CondCond":
            r = values[2]
            outcome = (p, q if p else None, r if p & q else None, p & q & r)
        elif check_type == "ExpEvidence":
            outcome = (p, q, p if q else None, p if not q else None)
        else:
            raise ValueError(f"unknown check type: {check_type}")
        outcomes.append(outcome)
    return outcomes


def _solve(matrix, values):
    """Solve a tiny square system by Gaussian elimination."""
    a = [list(row) + [value] for row, value in zip(matrix, values)]
    for col in range(len(a)):
        pivot = max(range(col, len(a)), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-12:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [value / scale for value in a[col]]
        for row in range(len(a)):
            if row != col:
                scale = a[row][col]
                a[row] = [x - scale * y for x, y in zip(a[row], a[col])]
    return [row[-1] for row in a]


def _vertex_lp(coefficients):
    """Maximize t over t <= c·alpha and alpha in [-1, 1] by vertices."""
    n = len(coefficients[0])
    constraints = [([-value for value in row] + [1.0], 0.0) for row in coefficients]
    for i in range(n):
        upper = [0.0] * (n + 1)
        lower = [0.0] * (n + 1)
        upper[i], lower[i] = 1.0, -1.0
        constraints.extend(((upper, 1.0), (lower, 1.0)))

    best = 0.0  # alpha = 0 guarantees zero profit
    for active in combinations(constraints, n + 1):
        point = _solve([row for row, _ in active], [value for _, value in active])
        if point is not None and all(
            sum(x * y for x, y in zip(row, point)) <= value + 1e-9
            for row, value in constraints
        ):
            best = max(best, point[-1])
    return best


def exploit(check_type: str, p: dict[str, float]) -> float:
    """Return the optimal adversary's worst-case guaranteed profit."""
    fields = CHECK_FIELDS.get(check_type)
    if fields is None:
        raise ValueError(f"unknown check type: {check_type}")
    coefficients = [
        [0.0 if value is None else p[field] - value for field, value in zip(fields, outcome)]
        for outcome in _outcomes(check_type)
    ]

    if linprog is not None:
        result = linprog(
            [0.0] * len(fields) + [-1.0],
            A_ub=[[-value for value in row] + [1.0] for row in coefficients],
            b_ub=[0.0] * len(coefficients),
            bounds=[(-1.0, 1.0)] * len(fields) + [(None, None)],
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"exploitability LP failed: {result.message}")
        value = -result.fun
    else:
        value = _vertex_lp(coefficients)
    return max(0.0, float(value))
