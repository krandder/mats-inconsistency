### 6.1 Attention is the merge of Gaussian messages, with the prior as a sink (PROVEN, identity)

Let tokens j carry Gaussian messages about y with precision tau_j and mean mu_j. A head with logits
L_j = log tau_j + c, values mu_j e_mu, plus a sink token with logit log tau_0 + c and value mu_0 e_mu + e_0, outputs
the posterior mean in the e_mu channel and a_0 = tau_0 / sum_{j >= 0} tau_j = tau_0 x (posterior variance) in the e_0
channel (the sum includes the sink). Without the sink (or another channel carrying the normaliser) the merged message
loses its total precision and is not closed under further merges. With a Gaussian prior N(mu_0, 1/tau_0), the sink is
the prior. In a multi-level merge the sink must enter once (at the root), or be subtracted at intermediate nodes;
otherwise merging two sunk subtrees counts tau_0 twice. Oracle GLS in the tiny-PFN world (reference b) is exactly one
such head. Exponential-family messages merge the same way in natural parameters, with an MLP undoing the
normalisation (construction; exact up to MLP approximation, which degrades as the sink weight a_0 ~ 1/N shrinks).
Attention sinks as such were named by Xiao et al. (arXiv:2309.17453); the identity with the Bayesian prior is the
point here.
