"""Sanity check of the circuit -> ReLU-network reduction (Theorem 4.1): random Boolean circuits, exact acceptance
probability by enumeration, Gaussian-input mean of the ReLU network by Monte Carlo. Variants: with biases ('bias'),
bias-free positively homogeneous ('homog'), and with Gaussian noise on every unit plus restore layers ('noisy').
usage: capp_check.py [n_circuits]"""
import itertools, math, sys
import numpy as np

rng = np.random.default_rng(11)
relu = lambda t: np.maximum(t, 0.0)


def random_circuit(m, s):
    """gates: list of (op, i, j) over wires 0..m-1 (inputs) and m.. (gates); output = last gate."""
    gates = []
    for g in range(s):
        nw = m + g
        op = rng.choice(['AND', 'OR', 'NOT'], p=[0.4, 0.4, 0.2])
        i = int(rng.integers(nw)); j = int(rng.integers(nw))
        gates.append((op, i, j))
    return gates


def eval_bool(gates, m, u):
    w = list(u)
    for op, i, j in gates:
        a, b = w[i], w[j]
        w.append(a & b if op == 'AND' else (a | b if op == 'OR' else 1 - a))
    return w[-1]


def relu_net_eval(gates, m, X, K, mode, sigma=0.0):
    """Evaluate the ReLU realization on a batch X (B, m+1); column 0 is x_0 (used for r in 'homog' mode).
    Wires hold values in [0, unit]; unit = 1 ('bias', 'noisy') or r = |x_0| ('homog').
    Gate units: AND = relu(a+b-unit); OR = relu(a+b) - relu(a+b-unit); NOT = relu(unit-a).
    'noisy': every relu unit gets N(0, sigma^2) noise and each gate output is restored by
    clamp(2v - 1/2, 0, 1) = relu(2v - 1/2) - relu(2v - 3/2) (two more noisy units)."""
    B = X.shape[0]
    nz = (lambda shape: sigma * rng.standard_normal(shape)) if mode == 'noisy' else (lambda shape: 0.0)
    if mode == 'homog':
        unit = relu(X[:, 0]) + relu(-X[:, 0])
    else:
        unit = np.ones(B)
    restore = lambda v: (relu(2 * v - 0.5) + nz(B)) - (relu(2 * v - 1.5) + nz(B))
    w = []
    for i in range(m):
        xi = X[:, i + 1]
        v = (relu(K * xi) + nz(B)) - (relu(K * xi - unit) + nz(B))
        w.append(restore(v) if mode == 'noisy' else v)
    for op, i, j in gates:
        a, b = w[i], w[j]
        if op == 'AND':
            v = relu(a + b - unit) + nz(B)
        elif op == 'OR':
            v = (relu(a + b) + nz(B)) - (relu(a + b - unit) + nz(B))
        else:
            v = relu(unit - a) + nz(B)
        w.append(restore(v) if mode == 'noisy' else v)
    return w[-1]


def main():
    nc = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    m, s, K = 10, 40, 1000.0
    M = 400_000
    for c in range(nc):
        gates = random_circuit(m, s)
        p = np.mean([eval_bool(gates, m, u) for u in itertools.product([0, 1], repeat=m)])
        X = rng.standard_normal((M, m + 1))
        fb = relu_net_eval(gates, m, X, K, 'bias')
        fh = relu_net_eval(gates, m, X, K, 'homog')
        fn = relu_net_eval(gates, m, X, K, 'noisy', sigma=0.02)
        Er = math.sqrt(2 / math.pi)
        print(f"circuit {c}: exact p = {p:.4f} | bias net E f = {fb.mean():.4f} +- {fb.std() / math.sqrt(M):.4f} "
              f"(range [{fb.min():.2f},{fb.max():.2f}]) | homog E f / E|x0| = {fh.mean() / Er:.4f} +- "
              f"{fh.std() / math.sqrt(M) / Er:.4f} | noisy(sigma=0.02) E f = {fn.mean():.4f} +- {fn.std() / math.sqrt(M):.4f}"
              f" | bound m/(K sqrt(2pi)) = {m / (K * math.sqrt(2 * math.pi)):.4f}")


if __name__ == '__main__':
    main()
