"""Builds CATALOGUE.md and catalogue/<id>.json from the entries below.

One entry per question.  Numbers are means with 95% t-intervals over seeds
(5 unless stated) and come from the summary files named in `files`.
Status: Gain / Informative / No gain / Problem / Running.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
E = []


def entry(id, question, setup, result, status, files):
    E.append(dict(id=id, question=question, setup=setup, result=result, status=status, files=files))


# ---------------------------------------------------------------- toy 1 / audit
entry("E-T1-1", "Does an unregulated Kelly market of trader agents keep many agents alive?",
      "10 agents, each reads 1-3 of 12 synthetic features, trains an online logistic model on its own log loss and "
      "bets half its wealth in an LMSR (b=0.5). Fees per feature read and compute cost; bankrupt agents are replaced.",
      "No. Wealth concentrates on one agent: 1.8±1.0 alive at the end, effective number of agents 1.2±0.6. "
      "Output log loss 0.543±0.018 vs 0.459 for logistic regression on all features (Bayes 0.436). "
      "Pay vs true value: Spearman 0.12±0.14.",
      "Problem", ["econ.py", "results/summary.json"])
entry("E-T1-2", "With a regulator paying by counterfactuals, does pay follow each agent's true value?",
      "Same 10 agents report truthful logits; the regulator fits a combiner and pays each epoch's pot by one rule. "
      "True value = Shapley of the information in the union of the agents' features.",
      "Spearman(pay, true value): LOO 0.81±0.10, MSR one order 0.54±0.14, MSR over 16 orders 0.68±0.05, exact Shapley 0.76±0.04, "
      "Kelly 0.12±0.14. Budget: LOO pays 0.58±0.13 of the pot; the MSR rules pay 1.02-1.03. All 10 agents alive under MSR-16 and Shapley; "
      "5.2±2.8 under LOO. Output log loss 0.52-0.54, still about 0.06 nats worse than logistic regression.",
      "Informative", ["econ.py", "results/summary.json", "charts/toy1_pay_vs_value.json", "charts/toy1_logloss.json"])
entry("E-T1-3", "How do read fees and compute cost change the economy?",
      "MSR-16 and Kelly, fee per feature read in {0, 0.004, 0.012, 0.03} x compute per step in {0, 0.004, 0.012, 0.03}, 3 seeds. "
      "Also fees burned instead of paid to feature owners.",
      "At fee 0.03 or compute 0.03 no agent survives and the output falls to the prior (0.693). Best log loss 0.520±0.069 at "
      "fee=compute=0.004. Burning the fees leaves 3.4±1.4 agents alive (10 when owners are paid) and pay tracks value worse (0.30±0.36 vs 0.68±0.05).",
      "Informative", ["econ.py", "results/summary.json"])
entry("E-AUD-1", "How do the pay rules treat substitutes and complements?",
      "10 designed agents, one feature each: x0 and its noisy copy x4, x1 and its copy x5, a complementary pair x6/x7 "
      "(useless alone, strong together), weak x2/x3, noise x8/x9. Pay in nats per epoch.",
      "Substitutes: Shapley pays x0 16.1 and its copy 9.2 (true values 19.0 and 11.8). LOO pays both copies about 0 or less "
      "(x4: -1.9). MSR in one order gives the credit to whichever comes first (x4: -1.4 forward, 21.5 reversed). "
      "Complements: every rule pays x6 and x7 below zero (Shapley -0.6 and -1.3) while their true values are +7.1 and +7.3. "
      "An agent that reports only its own belief carries almost none of the pair's information.",
      "Problem", ["econ.py", "results/summary.json", "charts/audit_categories.json"])
entry("E-AUD-2", "How many random orders does sampled Shapley need?",
      "Same designed agents; relative L1 error of MSR averaged over K orders vs exact Shapley.",
      "Error 0.83±0.07 at K=1, 0.36±0.05 at K=4, 0.18±0.01 at K=16, 0.09±0.01 at K=64. Rank agreement with true value: "
      "LOO 0.23±0.17, MSR one order 0.28±0.21, MSR-16 0.66±0.15, exact Shapley 0.69±0.14.",
      "Informative", ["econ.py", "results/summary.json", "charts/msrK_error.json"])
entry("E-T2-1", "With two layers and royalties, does money reach the useful first-layer agents?",
      "8 first-layer agents; 4 readers each read 3-4 of them and pass half of their positive earnings upstream, split by a rule.",
      "Equal-split royalties pay noise agents nearly as much as useful ones (0.89±0.48 vs 1.04±0.21); Spearman with true value 0.20±0.26; "
      "log loss 0.657±0.071. Splitting by Shapley pays noise agents -0.008±0.011 and useful ones 1.55±0.17; Spearman 0.71±0.16; log loss 0.579±0.025. "
      "Kelly readers collapse to 1.4±0.7 alive.",
      "Gain", ["econ.py", "results/summary.json", "charts/toy2_pay_vs_value.json", "charts/toy2_useless_died.json"])

# ---------------------------------------------------------------- experiment 0
entry("E0-1", "Is a hidden unit needed, or can an agent be just some of the weights?",
      "Softmax regression, 8 inputs, 3 classes. Agents own one weight (24 agents), one input's weights (8), or one class's weights (3). "
      "Trade = additive logit contribution; ledger, compute cost per active weight, bankruptcy and respawn. 800 rounds of 250 fresh examples.",
      "Weights work as agents at every granularity; the ledger balances in every run. With leave-one-out pay, each agent's own-pay gradient "
      "is exactly the global gradient: at zero cost the weights match backprop to 0.000 relative difference.",
      "Gain", ["exp0.py", "results/exp0_summary.json"])
entry("E0-2", "Does training each agent on its own Shapley pay reach the backprop solution?",
      "Same model; each agent follows the gradient of its own sampled-Shapley pay (a dropout-like objective). Cost 0.01 per active weight.",
      "No, except at the coarsest grain. Input-column agents: test log loss 0.808±0.003 vs 0.743±0.001 for backprop (Bayes 0.738); "
      "weights 68% apart. The copy of x0 gets 0.73 of x0's weight (backprop: 0.04), and the complementary pair is underweighted "
      "(0.37 vs 1.70). Single-weight agents: 0.808±0.006. Class-row agents (3): 0.743±0.001, same as backprop. MSR in one order: 0.815±0.019.",
      "Problem", ["exp0.py", "results/exp0_summary.json", "charts/exp0_logloss.json", "charts/exp0_weights.json"])
entry("E0-3", "When agents are weights, does their pay follow their true value?",
      "True value = Shapley of the refit game (best log score using only the coalition's weights). Pay averaged over the second half of training.",
      "Shapley-trained column agents: Spearman 0.92±0.02. Shapley pay evaluated at the backprop solution: 0.80±0.05; IG pay there: 0.82±0.03. "
      "LOO pays 2.1 times the pot and gives each complement 0.52, the value of the whole pair; for class-row agents it pays only 0.85.",
      "Informative", ["exp0.py", "results/exp0_summary.json", "charts/exp0_pay_vs_truth.json"])
entry("E0-4", "Do useless weights go to zero or go bankrupt?",
      "Noise inputs x6, x7. Cost per active weight 0, 0.01 or 0.05 per round (a useful column earns about 0.03-0.25 per round).",
      "At cost 0.01 noise agents go bankrupt repeatedly (6.8±0.6 deaths per run vs 1.2±1.0 for all useful agents together, Shapley arm), "
      "but their weights do not go to zero: the two noise inputs keep a total |w| of 0.10±0.04, the same as backprop with the same L1. At 0.05 the cost also kills useful agents (68±1 deaths per run) "
      "and log loss rises to 0.823.",
      "Informative", ["exp0.py", "results/exp0_summary.json"])

entry("E0-5", "Is there a pay rule between leave-one-out and Shapley that both trains well and pays fairly?",
      "Semivalue: every other agent is present independently with probability pi (pi near 1 is LOO, whose gradient is backprop; "
      "pi = 0.5 is the Banzhaf value, close to Shapley here). Payments rescaled to sum to the pot. Column and single-weight agents, cost 0.01.",
      "A clean dial. Column agents, pi = 0.5 / 0.8 / 0.95: test log loss 0.808±0.004 / 0.777±0.001 / 0.750±0.001 (backprop 0.743); "
      "Spearman(pay, true value) 0.91±0.02 / 0.84±0.01 / 0.71±0.02. Weight on the copy of x0: 0.74 / 0.53 / 0.17 of x0's. "
      "Single-weight agents give the same picture (0.807 / 0.767 / 0.748).",
      "Informative", ["exp0.py", "results/exp0_semi_summary.json"])

# ---------------------------------------------------------------- IG / Aumann-Shapley
entry("E-IG-1", "Can Aumann-Shapley pricing (integrated gradients) replace Shapley pay at scale?",
      "Designed agents with additive logits (informative, substitutes, complements, noise, harmful), binary and 3-class, "
      "n = 10, 100, 1000; 3 seeds. IG with m midpoint steps vs exact Shapley (n=10) or 400-1000 sampled orders.",
      "IG with m=20 is off from Shapley by 0.21±0.08 (n=10), 0.25±0.04 (n=100), 0.14±0.02 (n=1000) relative L1; Spearman 0.94-0.98. "
      "Sampled Shapley needs K=64 orders to match: 704 / 6,464 / 64,064 passes against IG's 20. Budget gap of IG: 0.15-1.1% at m=20, "
      "at most 0.2% at m=50.",
      "Gain", ["exp_ig.py", "results/ig_summary.json", "charts/ig_error_vs_shapley.json"])
entry("E-IG-2", "Where do IG and Shapley pay disagree?",
      "Pay by category of designed agent, n = 10, exact Shapley.",
      "IG pays complements about twice what Shapley pays (37±17 vs 18±12 nats per agent, binary), substitutes about 30% less "
      "(23 vs 34) and informative agents about 13% less; noise and harmful agents agree within about 2%.",
      "Informative", ["exp_ig.py", "results/ig_summary.json", "charts/ig_categories.json"])
entry("E-IG-3", "Can pay be recomputed only every N steps on a sample of examples?",
      "Online softmax regression, 10 input agents, 1000 steps of 512 examples; IG pay every N steps on s examples, scaled up.",
      "Cumulative pay error vs full recomputation: 5.5±3.0% (binary) and 3.8±1.1% (3-class) at N=10, s=256, which costs 5%; "
      "7.4±6.9% at N=50, s=512 (2% of the cost); 22±12% at N=50, s=32.",
      "Gain", ["exp_ig.py", "results/ig_summary.json", "charts/ig_online_subsample.json"])

# ---------------------------------------------------------------- hidden units (MLP) and rescaling
entry("E-MLP-1", "Does training each hidden unit on its own Shapley pay match backprop?",
      "One hidden layer of 32 tanh units, 4 of them wired to noise only; a unit owns its incoming weights, bias and outgoing weight. "
      "Pay = Shapley (16 orders, zero-out ablation) of the batch log-score gain; each unit follows the gradient of its own pay "
      "(8 predecessor sets). Synthetic stream, 1200 rounds, cost 0.0005 per connection; wdbc 70/30 split plus 4 noise columns.",
      "No. Synthetic test log loss 0.609±0.012 vs backprop 0.435±0.004 (logistic regression 0.433, true model 0.4355). The function is right "
      "but about 2.8 times too confident: one fitted output scale (0.36±0.03) brings it to 0.435, and its logit gap to backprop then "
      "(0.13±0.04) equals the gap between two backprop runs (0.13±0.03). This is dropout without the test-time rescale. "
      "wdbc: 0.55±0.31 vs 0.104±0.043; the rescale does not rescue it (0.32±0.37), it also overfits.",
      "Problem", ["exp_mlp.py", "results/mlp_summary.json", "charts/mlp_logloss_synth.json", "charts/mlp_logloss_wdbc.json"])
entry("E-MLP-2", "Do the other pay rules train hidden units as well as backprop?",
      "Same network; units follow the gradient of their own LOO pay or their fixed-order MSR pay.",
      "LOO gives exactly the backprop function at cost 0 (logit gap 0.000) and 0.435 at cost 0.0005, but it pays out only 21% of the pot "
      "and useful units go bankrupt 142±24 times per run. Fixed-order MSR reaches 0.439±0.005 with 1-2 surviving units "
      "(29±6 connections vs 386 for backprop): the first unit in the order takes all the value.",
      "Informative", ["exp_mlp.py", "results/mlp_summary.json", "charts/mlp_logloss_synth.json"])
entry("E-MLP-3", "Does a hidden unit's pay track its true value?",
      "Backprop-trained network. True value = unit Shapley with 200 orders on held-out data (Neuron Shapley style); pay from 16 orders on the training batch.",
      "Spearman(pay, true value): Shapley-16 0.965±0.027, LOO 0.84±0.10, fixed-order MSR 0.83±0.13 (synthetic); Shapley-16 0.969±0.016 on wdbc. "
      "Zero-out and mean-ablation truths agree (1.00 synthetic, 0.96 wdbc). Inside the over-confident Shapley-trained network LOO pay "
      "ranks units backwards (-0.85±0.10).",
      "Gain", ["exp_mlp.py", "results/mlp_summary.json", "charts/mlp_pay_vs_value.json"])
entry("E-MLP-4", "Do useless hidden units go bankrupt?",
      "4 unit slots wired only to noise; bankrupt units respawn with fresh weights.",
      "Yes under Shapley pay: at any cost above zero every noise slot went bankrupt and stayed dead after respawn (4.8-5 bankruptcies per run "
      "over the 4 slots) and ended with zero outgoing weight. Backprop only shrinks them (|outgoing weight| 0.016 vs 0.24 for useful units).",
      "Gain", ["exp_mlp.py", "results/mlp_summary.json"])
entry("E-MLP-5", "Do two copies of the same hidden unit share its pay?",
      "Copy the highest-paid unit and halve the outgoing weight of both copies, so the function does not change.",
      "Shapley pays each copy 0.52±0.03 and 0.51±0.04 of the original's pay (sum 1.03). LOO pays each 0.27 (sum 0.54): it under-pays redundancy. "
      "Fixed-order MSR splits 0.58/0.42 depending on which copy comes first.",
      "Gain", ["exp_mlp.py", "results/mlp_summary.json"])
entry("E-MLP-6", "Does unit-level Shapley pay work with two hidden layers?",
      "16 + 16 tanh units, 800 rounds, synthetic; all 32 units are players in one game on the network's log score.",
      "The first layer gets 0.52±0.01 of the Shapley pay against a true share of 0.54±0.02; Spearman(pay, true value) 0.97±0.03. "
      "Training on own Shapley pay shows the same over-confidence: 0.593, or 0.449 after one output scale.",
      "Informative", ["exp_mlp.py", "results/mlp_summary.json"])
entry("E-RS-1", "Is counterfactual pay unchanged when one unit is rescaled without changing the function?",
      "ReLU network, 16 units, backprop-trained. Each unit is split into an in-agent (incoming weights) and an out-agent (outgoing weight). "
      "Multiply the incoming weights by c and divide the outgoing weight by c, c in {0.1, 10}; top 6 units, 5 seeds.",
      "Yes. The function is identical (largest logit change 2e-15) and Shapley, mean-ablation Shapley and LOO pay are unchanged (ratio 1.000).",
      "Informative", ["exp_rescale.py", "results/rescale_summary.json", "charts/mlp_rescale_ratio.json"])
entry("E-RS-2", "Can pay or cost based on activation size be gamed by rescaling?",
      "Same rescaling. Rules that use magnitude: pay split by share of activation, readers paying per unit of activation read, L2 cost on weights.",
      "Yes. At c = 10 the in-agent's activation-share pay rises 6.13±0.18 times and the other in-agents lose 39%, for the same function. "
      "Per-activation payments move 10 times. L2 cost moves 100 times onto the in-agent at c = 10, or onto the reader at c = 0.1.",
      "Problem", ["exp_rescale.py", "results/rescale_summary.json", "charts/mlp_rescale_ratio.json"])
entry("E-RS-3", "Does normalising activations stop the rescaling game?",
      "Standardise each activation to unit standard deviation, with the statistics recomputed after the rescale (and, as a check, frozen before it).",
      "Mostly. Activation-share pay, per-activation payments and the reader's L2 cost become exactly invariant. Two gaps remain: L2 on the "
      "incoming weights still moves 100 times (the in-agent can escape it by shrinking, at no cost to the function), and with statistics frozen "
      "before the rescale the game is back (6.19 times).",
      "Gain", ["exp_rescale.py", "results/rescale_summary.json", "charts/mlp_rescale_ratio.json"])

# ---------------------------------------------------------------- information vs computation
entry("E-SPLIT-1", "Does Shapley pay an input and a processor by scarcity or by effort?",
      "Logit = a·x times input gates times processor gates. (a) 1 input read by k duplicate processors, (b) 1 processor reading k duplicate "
      "inputs, (c) one of each. Duplicates either substitute (any one suffices) or add up (each carries 1/k). Exact Shapley and IG (m = 20, 50); 5 seeds.",
      "By scarcity. In (a) the input's Shapley share is 0.50, 0.667 and 0.833 for k = 1, 2, 5 when any one processor suffices, and "
      "0.50, 0.607±0.003, 0.672±0.005 when they add up; (b) is the mirror image (input 0.333 and 0.167 at k = 2, 5). (c) splits 0.50/0.50, "
      "which is the symmetry axiom. IG agrees in the substitute case (0.59 and 0.78) but gives exactly 0.50 in every additive case: the logit "
      "grows as t squared along the straight path, so IG cannot see replaceability there.",
      "Informative", ["exp_split.py", "results/split_summary.json", "charts/split_scarcity.json"])
entry("E-SPLIT-2", "How much of the pay reaches the inputs, and does a layer-by-layer (Owen) split differ from flat Shapley?",
      "ReLU network 12 inputs -> 8 -> 4 -> 1, trained by backprop; two first-layer units read only noise features. Players: inputs and units. "
      "Flat Shapley and flat IG over all players; nested Owen value (each unit's union holds what it reads; an input read by several units is "
      "split into one player per path, i.e. Winter's levels value on the unrolled tree). Output bias fixed at the prior; 5 seeds.",
      "Share of the total going to inputs / layer 1 / layer 2: flat Shapley 0.61±0.06 / 0.46±0.59 / -0.07±0.59; IG 0.31±0.02 / 0.63±0.31 / "
      "0.06±0.32; nested Owen 0.19±0.07 / 0.10±0.34 / 0.71±0.34. In the nested split each unit keeps about 0.7 of its union's value, since "
      "it can veto the union, so the inputs' share shrinks at every layer. Input pay ranks inputs like their drop-feature value (Spearman: "
      "flat 0.89±0.10, Owen 0.85±0.10, IG 0.78±0.10) but only 0.53-0.61 like their information Shapley; the substitute pair gets about 0 "
      "where information Shapley gives it 0.067. Noise-only units get -0.002±0.003.",
      "Informative", ["exp_split.py", "results/split_summary.json", "charts/split_layers.json"])
entry("E-SPLIT-3", "Does training each unit on its own flat-Shapley or Owen pay work in a two-layer network?",
      "1000 rounds, ledger checked in code, bankruptcy and respawn. Arms: backprop, own flat-Shapley pay (dropout-like), own Owen pay "
      "(exact gradient for one sampled nested order, pay every 10 rounds).",
      "No. Test log loss at cost 0: backprop 0.440±0.001, Shapley 0.499±0.013, Owen 0.495±0.023 (Bayes 0.4355). At cost 0.002 per "
      "connection both economy arms collapse towards the prior (0.671 and 0.639; about 11.5 bankruptcies per 6 useful first-layer slots) "
      "while backprop stays at 0.439.",
      "No gain", ["exp_split.py", "results/split_summary.json", "charts/split_training.json"])


def build():
    # results from other scripts can add entries through catalogue_extra.json
    extra = os.path.join(HERE, "results", "catalogue_extra.json")
    if os.path.exists(extra):
        E.extend(json.load(open(extra)))
    os.makedirs(os.path.join(HERE, "catalogue"), exist_ok=True)
    # keep catalogue/<id>.json entries written by other agents
    known = {e["id"] for e in E}
    for fn in sorted(os.listdir(os.path.join(HERE, "catalogue"))):
        if fn.endswith(".json") and fn[:-5] not in known:
            e = json.load(open(os.path.join(HERE, "catalogue", fn)))
            if all(k in e for k in ("id", "question", "setup", "result", "status", "files")):
                E.append(e)
                known.add(e["id"])
    # keep "## <id>: ..." sections that other agents appended to CATALOGUE.md by hand
    kept = []
    old = os.path.join(HERE, "CATALOGUE.md")
    if os.path.exists(old):
        cur = None
        for line in open(old).read().split("\n"):
            if line.startswith("## "):
                sid = line[3:].split(":")[0].strip()
                cur = [] if sid not in known else None
                if cur is not None:
                    kept.append(cur)
            if cur is not None:
                cur.append(line)
    lines = ["# Trader economy: experiment catalogue", "",
             "One small experiment per question. Status: Gain, Informative, No gain, Problem, Running. "
             "Numbers are means with 95% t-intervals over seeds. Each entry also exists as `catalogue/<id>.json`. "
             "Design note: `../trader-economy-note-20260929.md`. Built by `catalogue.py`.", "",
             "| id | question | status |", "|---|---|---|"]
    for e in E:
        lines.append(f"| {e['id']} | {e['question']} | {e['status']} |")
    for e in E:
        json.dump(e, open(os.path.join(HERE, "catalogue", e["id"] + ".json"), "w"), indent=1)
        lines += ["", f"## {e['id']}: {e['question']}", "", f"Status: {e['status']}", "",
                  f"Setup: {e['setup']}", "", f"Result: {e['result']}", "",
                  "Files: " + ", ".join(f"`{f}`" for f in e["files"])]
    for sec in kept:
        lines += [""] + sec
    open(os.path.join(HERE, "CATALOGUE.md"), "w").write("\n".join(lines).rstrip("\n") + "\n")
    print(len(E), "entries,", len(kept), "hand-appended sections kept")


if __name__ == "__main__":
    build()
