# Your consistency requirement on amplified predictions, tested as a data-efficiency device (wf13/consistency)

Model: Claude Fable 5.2 (`claude-fable-5-2[1m]`). 2026-11-10, 16:25-19:15Z. CPU only. Words follow `wf9/FRAME.md` section 5.

## 0. In plain words

The context first. On 10 Nov you said your consistency requirement is on the amplified predictions: amplified P(X)
times P(Y given X) should agree with P(Y) times P(X given Y) in the limit of compute, any variance over orders is a
signal, and the point is data efficiency. Only a consistency-over-compute loss had been tried. I tested your form on
the parity world's trained network: the learned aggregator (your trust head: it reads K sampled trajectories and
forecasts a check) extended to "any check given any subset of the others", trained with 64 to 20,480 real check
vectors, with and without a consistency term that sees no real outcome. No GPU was free, so compute is 1 and 4
trajectories per forecast (not 256) and two worlds.

1. **Your cross-order form did not buy data efficiency here.** As a loss on pairs or on whole orders it changed
   accuracy by between +15 % and -23 % with 64 or 256 real vectors, nothing that held in both worlds, and cost 6-30 %
   at a strong weight once data were plentiful.
2. **Anchored to the simple world model's score it helped a little, and only where data were fewest**, as you
   said: 6-17 % less error with 64 or 256 real vectors, nothing from 1,024. But training on outcomes sampled from M
   does about the same, with no real outcome at all it matches about 1,000 real vectors, and with that little
   real data every version is still worse than one pass of M.
3. **At this compute the variance over orders is sampling noise.** Each amplified number comes from its own sampled
   trajectories; two orders disagree about as much as two draws of one order. It did not fall from 1 to 4
   trajectories. M itself is nearly consistent, because it says little.
4. **The October failure reappears.** The loss removes the variance by making the forecast stop reading its
   trajectories: error on the informative questions goes back to one pass of M. Anchoring to the score does not
   prevent it; a small weight does.
5. **Where your idea does work (found afterwards, both worlds):** when real outcomes score only one kind of question
   (here: only questions with 25 or more checks revealed), the predictor is wild on all the others, and either
   cross-order term repairs that: error 0.071 / 0.283 becomes 0.032-0.040, against 0.029 with real outcomes
   everywhere. Consistency carries knowledge from large contexts down to small ones, never the other way.

I don't know: what happens at 64 or 256 trajectories, where the noise may fall below the real order effect; whether a
predictor with many free numbers per question differs; how to estimate the order variance without the sampling noise.
