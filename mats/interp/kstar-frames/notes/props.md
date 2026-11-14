# Two exact statements about carrying the k*-frame through layers (2026-10-31 ~00:05Z; revised ~02:55Z after the GPT-6 review)

Notation: z_c pre-activation of layer c, D_c its gate mask, J_c = d a_L / d z_c, B_c = D_c W_c D_{c-1} (the layer map
between active coordinates), so J_{c-1} = J_c B_c (because J_c = J_c D_c). F_c = a top-k right singular frame of J_c.

## Proposition 1 (dual covariance). PROVEN (linear algebra).
(a) If J_c has rank exactly k, the row space of J_{c-1} is B_c^T range(F_c); so the top-k right frame one layer
earlier is orth(B_c^T F_c) (rank k iff B_c^T is injective on range F_c).
(b) The inverse-transpose ("dual") transport (B_c^T)^+ maps B_c^T range(F_c) onto P_{range B_c} range(F_c). So it
returns F_c exactly iff range(F_c) lies in range(B_c), i.e. F_c has no component in ker(B_c^T) within range(D_c), a
subspace of dimension m_c - rank(B_c) = max(0, m_c - m_{c-1}) for generic weights. When the active count grows from
layer c-1 to c, those directions at layer c are invisible from layer c-1.
(c) The push-forward orth(B_c F_{c-1}) = orth(B_c B_c^T F_c) returns range(F_c) only if range(F_c) is invariant under
B_c B_c^T (and B_c B_c^T keeps its rank k there), which fails for generic Gaussian weights.
(Review, 02:35Z: (a) needs J_c of rank exactly k; when B_c^T loses rank on range(F_c) the previous layer's top-k frame
includes an arbitrary completion. The measured k* is a thresholded effective rank, so finite-horizon top-k* frames are
only approximately inverse-transpose covariant; the measured one-step overlaps below 1 are that approximation.)
Proof. (a) row(J_c B_c) = B_c^T row(J_c) and row(J_c) = range(F_c). (b) (B^T)^+ B^T = P_{range(B)} (the orthogonal
projector onto the column space of B); generic rank of the m_c x m_{c-1} Gaussian block is min(m_c, m_{c-1}).
(c) Substitute (a). QED
For full-rank J_c the relation in (a) holds only approximately (the spectrum below the k-th singular value mixes in);
how well dual transport does on the nets is measured (Q1a).

## Proposition 2 (no forward rule beats chance by more than one direction). PROVEN for sigma = 0, conditionally on an
## invariant spectral event and for a fixed (or past-measurable) k.
Setting of Theorem C (lyapunov lane): noiseless He-ReLU net, W_l iid N(0, 2/n), the future weights W_{c+1}, ..., W_L
independent of the past (x, W_1, ..., W_c); a_c != 0. Let e_c = a_c / |a_c| and B a past-measurable orthonormal basis
of range(D_c) ∩ e_c-perp (dimension m_c - 1 >= 2). Let S be any subspace of range(D_c) of dimension d that is a
measurable function of the past alone (for example the top-k left singular frame of the past product, which QR
push-forward converges to, or any frame built from the state and the weights up to layer c), with k fixed or a
function of the past. Let G be the event that J_c has a spectral gap at the cutoff, s_k(J_c) > s_{k+1}(J_c) (so the
top-k right frame F_c is unique; futures of rank below k are excluded). G depends only on singular values, so it
is invariant. Then, given the past and G,
  E[ tr(P_{F_c} P_S) | past, G ] <= |P_S e_c|^2 + k (d - |P_S e_c|^2) / (m_c - 1) <= 1 + k d / (m_c - 1).
The bound allows a radial advantage of at most one direction; it is not "no better than chance".
For d = k the mean overlap O = tr(P_F P_S)/k is at most k/(m_c - 1) + 1/k: chance plus one direction.
Proof. Given the past, M = W_{c+1} B and g = W_{c+1} e_c are independent, M has iid N(0, 2/n) entries, and the whole
future trajectory (z_{c+1} = |a_c| g, the gates D_{c+1}, ..., D_L and W_{c+2}, ..., W_L) depends on W_{c+1} only
through g. In the coordinates (e_c, B) of range(D_c), J_c [e_c, B] = [Q g, Q D_{c+1} M] with Q = D_L W_L ... W_{c+2}
D_{c+1}. For a fixed orthogonal O of size m_c - 1, M O has the law of M and is independent of (g, W_{c+2..L}), so
J_c [e_c, B] diag(1, O) has the same conditional law as J_c [e_c, B]. Hence the conditional law of the right frame R
(in these coordinates) is invariant under diag(1, O), and so is E[P_R | past]; by Schur's lemma (m_c - 1 >= 2) it equals
diag(alpha, beta I) with alpha in [0, 1] and alpha + (m_c - 1) beta = k. Then E tr(P_R P_S) = alpha s + beta (d - s)
with s = |P_S e_c|^2 <= 1, and beta <= k / (m_c - 1). The conditioning event depends only on singular values, which
are invariant, and on G the frame is a well-defined (equivariant) function of J_c, so no tie-breaking rule is needed.
QED
Checks: `logs/t1_prop2.log` averages over inputs of two fixed nets (a sanity check only); `logs/t2_prop2_cond.log`
fixes the past and resamples 400 futures for 6 pasts at n = 32 and 64: every conditional mean is 0.7-0.8 below the
bound. Putting e_c into S changes nothing because the past frame already contains it (|P_S e_c|^2 >= 0.9998); the
radial weight alpha measured by the re-check is 0.30-0.43 (GPT-6 re-check, 03:21Z).
Meaning: the directions that will matter downstream are, apart from the state's own direction, uniformly random with
respect to everything the forward pass has computed so far. Predicting them needs the future weights. Rules that use
the future weights (marginal gates, references, learned maps, per-path backward passes) are not covered and are
measured. With noise (sigma > 0) the exact structure breaks; measured.
