# Route to existence/uniqueness of the stationary sample measure (13:30Z)

Options considered:
1. Harris / Foster-Lyapunov (Meyn-Tweedie; Hairer-Mattingly 2011) for the annealed chain h -> relu(W h) + sigma xi
   with V = |h|^d, d in (0,2) (E g^{d/2} < 1 by strict Jensen) and Gaussian minorisation. Gives a unique ANNEALED
   invariant law and geometric ergodicity, but says nothing about fixed weights (the quenched sample measure).
2. Iterated random functions, Diaconis-Freedman (1999): needs E log Lip(F) < 0. Fails: Lip(F_l) = ||W_l||_op ~ 2.4-2.8.
3. Markov chains in random environment (Kifer 1986; Orey 1991; Lovas-Rasonyi 2021): quenched limits under random
   drift and minorisation constants; heavier machinery, constants random, citations unverified.
4. Pairwise fractional-moment contraction (Elton/Steinsaltz-style "contractive on average for pairs"), proved
   directly here from the exact arc-cosine identity (Lemma 1) and strict Jensen (Lemma 2). Gives the quenched
   pullback, a.s. synchronisation with the noise fixed (random point attractor), and uniqueness among tempered families.
Chosen: 4 (self-contained, gives the quenched object Kelvin asked about, and the synchronisation that coupling from
the past uses), with 1 as the annealed cross-check. The exact one-dimensional radius chain (FORMAL item 3) is a
by-product of the same rotation-invariance argument.
