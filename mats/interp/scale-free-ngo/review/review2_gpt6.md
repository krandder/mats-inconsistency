# GPT-6 (gpt-6-astra, high) re-check of draft-v2, 2026-10-29

1. **Fixed.** Lee–Kondor–Ngo is now correctly described as log-score welfare aggregation, distinct from evidence fusion; its headline possibility results match the source.
2. **Fixed.** Well-foundedness is explicitly a structural analogy to one example; `[exact]` now applies to the shared-error model.
3. **Partly fixed.** The Knightian-demarcation misreading is removed, but T7’s “per member” cost still omits the logarithmic factor for simultaneous routing and the distinction between a pairwise lower bound and an all-pairs construction.
4. **Fixed.** The identification of Ngo’s wealth/truth objection with the mixture/product distinction is removed.
5. **Fixed.** Open-latent minimality is now restricted to the joint posterior and rich outside data; this concerns information retained, not uniquely required securities or coordinates.
6. **Partly fixed.** “Toward their current posterior (nothing counts twice)” states the intention but supplies no replacement, refinement-ratio, or public-history-conditioned trading rule.
7. **Partly fixed.** Myopia, risk neutrality, full support and net profit are restored; unrestricted feasible trades and a mechanism for settling latent-state securities remain unstated.
8. **Partly fixed.** Binary outcomes and finite rounds are restored, and “only”/complete-market claims are removed; unconditional independence still does not universally preclude truthful equilibrium, and D1’s extension beyond the cited game remains unqualified.
9. **Fixed.** Known Gaussian precision and minus curvature make the displayed correspondence exact.
10. **Partly fixed.** The \(\pi^{1-N}\) correction is correct **up to normalization, on positive prior support**; Dietrich–List Theorem 3 confirms the multiplicative **class**, with fixed calibration and domain \((\mathcal P^0)^n\). The normalized-weight impossibility still needs a sufficiently rich evidence family: if every signal is uninformative, \(P_i=\pi\), weights summing to one reproduce Bayes for every observation.
11. **Fixed.** The categorical “no mechanism” claim is removed.
12. **Fixed.** The architecture and convergence hypothesis are appropriately tentative.
13. **Partly fixed.** Honesty is now an assumption and the statistic is labeled correlational, but “do weight sources by inferred precision” still claims more mechanism than the correlation establishes.
14. **Fixed.** The unsupported continuous-action/no-cascade claim is removed.
15. **Fixed.** D2 is explicitly conjectural, uses fixed-width slots, and no longer treats failure of linear decoding as refuting separator sufficiency.
16. **Fixed.** Overlapping subagents and the restricted scope of the comparison are acknowledged.
17. **Fixed.** “Typically” and “not necessarily” are restored; the purported answer to RogerDearnaley is removed.
18. **Partly fixed.** Bibliographic identifiers and Dietrich–List’s text are supplied, but primary passages for Chen, Laskey and Wainwright–Jordan remain absent; Ostrovsky’s attribution to Geanakoplos–Polemarchakis is still omitted.

Four new findings:

**A. The hierarchical-coalitions quotation is accurate; its routing interpretation needs a boundary.**  
Draft: “routing; rounds | coalition formation (iterated vote-trading ‘produces hierarchical coalitions’)”; subsequently, “Forming his clusters is, in inference, a routing cost.”

[Source: Ngo’s March post](/home/agent/workspaces/mats/work/scale-free-ngo/sources/ngo-2025-03-21-scale-free.txt:96): “Pairs of subagents with different preferences are incentivized to switch both their votes to a compromise option. Iterating this process many times produces hierarchical coalitions.”

This concerns bargaining over **values and compromise options**. It supplies no claim that the resulting coalitions identify statistical dependence groups or reduce inference separators. The table’s “interpretation” label helps, but the subsequent “is” overstates the connection. Describe grouping by statistical dependence as your proposed inference analogue.

**B. The shortened belief-web description drops the agent-defining trust condition.**  
Draft: “an agent as ‘a densely-connected region’ of a belief web.”

[Source: *Agents as Webs of Beliefs*](/home/agent/workspaces/mats/work/scale-free-ngo/sources/ngo-M39Z2CvyfaxZdaxR4-Agents_as_Webs_of_Beliefs_2026-06-27T21.txt:58): “An ‘agent’ could then just be a densely-connected region of that belief web **which trusts updates from within that region much more than updates from outside it**…”

The quoted fragment is verbatim, but v2 removes the trust qualification that v1 retained elsewhere. Density alone is not his proposed criterion, and dense internal connectivity does not imply a small separator. Restore both the tentative “could” and the internal-versus-external trust condition.

**C. The revision removes T11d’s Gaussian-model qualification.**  
Draft: “sizes may grow (and must, like log N, for bounded noisy messages, T11d).”

[Source: our definition, §5.7](/home/agent/workspaces/mats/outbox/scale-free-theory-2026-10-29.md:490) assumes Gaussian observations, a bounded message with independent Gaussian channel noise, and an estimate whose “mean-squared error [is] at most \(c v_N\),” where \(v_N=1/(1+NI)\).

Bounded noisy messages alone do not force logarithmic growth. The bound depends on maintaining accuracy relative to the shrinking posterior variance under the specified model. V1 explicitly retained the Gaussian qualification; restore it and the accuracy requirement.

**D. The revised citation-check statement contradicts its citation register.**  
Draft: “other citations by abstract.”

[Source: CITATIONS.md](/home/agent/workspaces/mats/work/scale-free-ngo/CITATIONS.md) explicitly lists Hanson 2007 as “record only; the LMSR identity in the note is our algebra.” Consequently, “other citations by abstract” overstates the recorded verification. Preserve the distinction between abstract checks, bibliographic records and independently derived algebra.

Verdict: 11 findings fixed, 7 partly fixed; the comparison is substantially improved but still needs the remaining qualifications and four corrections above.
