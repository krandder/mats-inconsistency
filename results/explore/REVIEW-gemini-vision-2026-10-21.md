# Guardian-of-the-Vision Adversarial Audit: `aggbench/`, `L1–L10`, `expagg*`, and `trader_transformer.py` (`2026-10-21`)

Serves: adversarial design-fit and code-level audit against Kelvin's verbatim Trader-Transformer architecture (`research-orientation-2026-10-20.md` & `research-orientation-2026-10-20-addendum.md` lines 38–53).
Builds on: `/home/agent/workspaces/mats/outbox/stage1-design-fit-review-2026-10-21.md`, `/home/agent/workspaces/mats/outbox/overnight-plan-2026-10-21.md`, `aggbench` commit `a168388`.

---

## 1. Executive Summary & Kelvin's Non-Negotiable Architecture Standard

Kelvin's explicit Stage-1 design (`research-orientation-2026-10-20-addendum.md`, lines 38–53, *"A Transformer Where Each Token Is a Trader"*) specifies five architectural questions that every main-line experiment must directly test:

> - **`(1)` Trader-as-Token Causal Sequence (`layout="trader"`):** *"Start with the simplest faithful version: each trader's information is one token (`[report_i, key_i]` or local summary), arriving in a random order, with a causal mask over traders and a readout at the final layer of the last trader."*
> - **`(A)` Cross-Trader Attention Schedule (`attn_layers`):** *"How much cross-trader self-attention is needed across layers? Compare attention at every layer, only the last layer, only the first + last, every second layer, and none (independent traders summed at readout)."*
> - **`(B)` Prefix vs. Final-Token Supervision (`loss`):** *"Can we train without supervision on every token? A language model gets a loss at every prefix; here the label is the ground-truth outcome $Y$, and supervising every prefix on the same $Y$ may reward early tokens for guessing before the information is there. Compare final-token-only loss, every-prefix loss on $Y$, and every-prefix loss against the prefix Bayes posterior (when available as a diagnostic)."*
> - **`(C)` Trader-Direction Residual Stream (`token_state`):** *"Can we add a residual stream in the token/trader direction rather than only in depth? A recurrent or Highway-like state $s_t = g(s_{t-1}, h_t^{(L)})$ (the CEO's running summary as traders walk into the room) vs standard KV-cache attention over all prior traders."*
> - **`(D)` 3-D `(depth, token-index, trader-index)` Grid (`layout="grid" | "axial" | "grid_full"`):** *"Once single-token-per-trader works, extend to a multi-token-per-trader grid `(depth, token-index, trader-index)` where a head at `(l, t, r)` attends causally over earlier tokens of the same trader (`t' <= t, r' = r`) or earlier traders at the same token (`t' = t, r' <= r`)."*

Every run is gated by the four-part disclosure check (`overnight-plan-2026-10-21.md`):
- **`(a)`** What a token is (`[obs_i, own_model_i, peer_keys_i]` or per-piece token),
- **`(b)`** What a layer is (self-attention + MLP with residual stream in depth and/or across traders),
- **`(c)`** What crosses between traders and its width ($d_{\text{model}} \gg 1$, never just a 1-D scalar report unless explicitly labeled a scalar baseline),
- **`(d)`** Regret to exact Bayes **and** to the `full_context_transformer` bar across `E2`, `Z1`, `Z5`, `Z6` under `alignment="unaligned"`.

---

## 2. Adversarial Audit of Earlier `L1–L10` and `expagg*` Runs (Design-Fit Findings `DF1`–`DF4`)

| ID | Severity | Runs Affected | Violated Kelvin Quote | Evidence (`file:line`) | Status / Action Taken |
|---|---|---|---|---|---|
| **`DF1`** | **HIGH** | `L1` (`l1-learned-keys`), `L2` (`l2-budgeted-voi`), `L4` (`l4-reliability`), `L5` (`l5-routing`), `L7` (`l7-lmsr`), `L9` (`l9-order-var`), `L10` (`l10-scaling`), `expaggscalar` | *"each trader's information is one token (`[report_i, key_i]` or local summary)... When reports are 1D scalars, overlap is unidentifiable"* (`addendum.md:40`, `orientation.md:77-92`) | `aggbench/models.py:55-182` (`LearnedKeyClearing`, `ReliabilityClearing`), `benchmarks/explore_aggbench_scalar_diagnosis.py:92-215`: only the 1-D scalar `batch.reports["logit"]` (`shape [B, N]`) crosses between agents; raw local observations `batch.obs` never enter the aggregator. On `E2/E3`, even `oracle_clearing` is trapped at `0.0166` regret (`union_identified = 3.1%`). | Relabeled `expaggscalar` and `agg_scalar_summary` as `● Informative (Scalar-Report Baseline; Aligned-Slot Side-Line Diagnostic)`. Main-line work uses rich trader tokens (`full_context` / `piece_tokens`). |
| **`DF2`** | **CRITICAL** | `L3` (`l3-highdim-messages`), `L6` (`l6-decentralized`), `expagghighdim`, `agg_highdim_summary`, `expaggkylevoi` | *"Without aligned slot indices, the network must learn to match pieces from their keys rather than reading off coordinate positions"* (`aggbench/envs.py:124-169`, `stage1-design-fit-review-2026-10-21.md` `D1`–`D12`) | `benchmarks/explore_aggbench_highdim_messages.py:55-68` and `benchmarks/explore_aggbench_kyle_voi_3param.py:86-102` ran under pre-`1c2e780` `alignment="aligned"`, where `obs[:, :, k]` and `keys[:, :, 2k:2k+2]` share global piece index $k \in \{0..11\}$ across all agents. An unlearned 0-parameter coordinate-aligned Gram skeleton (`unlearned_0param_dv12`) already gets `0.0000` regret on `E2`, `E3-sybil`, and `E3-collude (full)`. | Relabeled `expagghighdim`, `agg_highdim_summary`, and `expaggkylevoi` as `● Informative (Aligned-Slot Diagnostic / Giveaway Control; Side-Line Exploratory)` and **withdrew** the `Decided:` line in `expagghighdim` and `agg_highdim_summary`. Fixed `D11` environment descriptions in `expaggkylevoi/README.md`. |
| **`DF3`** | **HIGH** | `expaggscalar`, `expagghighdim`, `expaggopaque`, `expaggkylevoi`, `aggbench/models.py:55-268` | *"Start with the simplest faithful version: each trader's information is one token... with a causal mask over traders and a readout at the final layer of the last trader"* (`addendum.md:40-41`) | `benchmarks/explore_aggbench_opaque_reports_clearing.py:114-210` (`OpaqueReportGramClear`) and `aggbench/models.py:66-92` (`LearnedKeyClearing`) hard-wire an explicit matrix inversion `torch.linalg.solve(G + lam * I, ...)` rather than testing whether multi-layer causal self-attention over trader tokens (`TraderTransformer`) learns de-duplication, reliability, and complementary synthesis (`Z6` XOR). | Stopped all side-line Gram-skeleton runners (`expagge4`, `expaggzoo` cancelled before training); all new runs use `aggbench/trader_transformer.py` directly. |
| **`DF4`** | **MEDIUM** | `L8` (`l8-amplification`) | *"Can we train without supervision on every token?... Compare final-token-only loss, every-prefix loss on $Y$, and every-prefix loss against the prefix Bayes posterior"* (`addendum.md:44-46`) | `L8` distilled $Q=4$ active subset selection into $Q=2$ on scalar `LearnedKeyClearing`, rather than testing prefix distillation (`loss="distill"` vs `"final"` vs `"every_prefix"`) inside a causal Trader-Transformer (`aggbench/trader_transformer.py:296-309`). | Tested directly via Axis `(B)` (`loss in {"final", "every_prefix", "distill"}`) in `aggbench/trader_transformer.py`. |

---

## 3. Code-Level Forensic Audit of Newly Landed `aggbench/trader_transformer.py` & `aggbench/transformer.py` (`a168388`)

We audited every line of `aggbench/trader_transformer.py`, `aggbench/transformer.py`, `aggbench/envs.py`, and `aggbench/zoo.py` at commit `a168388`. Below are five concrete `file:line` findings (`V1`–`V5`) that directly impact `nearbayes-20261021` and the `(A)/(B)/(C)/grid` sweep.

---

### Finding `V1` (`CRITICAL` — `TraderTransformer` in `aggbench/trader_transformer.py:271-351` Has No `device` Support and Runs 100% on CPU Even on an L4 GPU VM)

- **Evidence (`file:line`):**
  - Compare `FullContextTransformer` in `aggbench/transformer.py:100,115-117,140-141,178` (which accepts `device: str | None = None`, defaults to `"cuda" if torch.cuda.is_available() else "cpu"`, and moves `self.module`, `x`, `y`, and `perm` onto `self.device`) against `TraderTransformer` in `aggbench/trader_transformer.py:271-351`:
    ```python
    # aggbench/trader_transformer.py:293-339
    def fit(self, batch: Batch, visibility: str = "full") -> "TraderTransformer":
        torch.set_num_threads(min(2, os.cpu_count() or 1))
        torch.manual_seed(self.seed)
        x = self._tokens(batch, visibility)
        y = torch.as_tensor(batch.truth, dtype=torch.float32)
        ...
        self.module = _TraderTransformerModule(...)  # stays on CPU!
        ...
        for _ in range(self.epochs):
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = self._permute_traders(x[idx], gen)
                ...
                out = self.module(xb)  # 100% CPU execution with threads=2!
    ```
- **Impact:**
  - When `TraderTransformer` or `evaluate_trader_transformer` (`trader_transformer.py:354-404`) is launched on a spot L4 GPU VM, **PyTorch never touches the L4 GPU** — it runs on 2 CPU threads (`torch.set_num_threads(2)`), making 140-epoch or `layout="grid"` ($96 \times 96$ attention across $n_{\text{train}} = 8000..32000$) orders of magnitude slower!
- **Exact Fix:**
  - Add `device: str | None = None` (`self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")`) to `TraderTransformer.__init__`, move `x`, `y`, `pb`, and `self.module` to `self.device` in `.fit()` (`with torch.no_grad():` on `.predict_prefix()`), or move `model.module.to(device)` and input tensors to `cuda` in the GPU runner.

---

### Finding `V2` (`CRITICAL` — `full_context(..., own_model=True)` in `aggbench/transformer.py:40-42` and `trader_transformer.py:141` Leaks Noiseless `key.full` Under `visibility="none"` and `"noisy"` on `Z2`, `Z5`, `Z6`, and Scrambles Group One-Hots on `Z1`/`Z4`)

- **Evidence (`file:line`):**
  1. In `aggbench/transformer.py:40-42` (called by `TraderTransformer._tokens` at `trader_transformer.py:141` when `layout == "trader"`):
     ```python
     if own_model and batch.own_model is not None:
         own = np.nan_to_num(batch.own_model, nan=0.0) / 4.0
         parts.append(own.astype(np.float32))
     ```
  2. Where does `batch.own_model` come from? Look at `aggbench/envs.py:93` and `144-169`:
     ```python
     # aggbench/envs.py:93
     own = self.keys["full"].copy() if self.own_model is None else self.own_model.copy()
     ```
     And in `apply_alignment` (`aggbench/envs.py:128-130, 146-153`):
     ```python
     blocks = slot_layout(batch)
     if blocks is None:
         return batch.with_alignment("aligned")
     ```
  3. Now check `key_slot_blocks` across `Z1`–`Z6` in `aggbench/zoo.py`:
     - **`Z2` (`SharedNuisanceEnv`, `zoo.py:216`), `Z5` (`RegimeSwitchEnv`, `zoo.py:390`), `Z6` (`ComplementaryXOREnv`, `zoo.py:454`):** None of these classes define `key_slot_blocks` (`key_slot_blocks = ()` inherited from `Env`, `envs.py:173`), so `slot_layout(batch)` is `None`!
     - Consequently, on `Z2`, `Z5`, and `Z6`, `batch.own_model` is **literally a verbatim copy of `batch.keys["full"]`** (`envs.py:93`)!
     - When `TraderTransformer` (`layout="trader"`) or `FullContextTransformer` is evaluated at **`visibility="none"`** or **`visibility="noisy"`** on `Z2`, `Z5`, or `Z6`, `full_context(..., own_model=True)` concatenates the **exact, noiseless `batch.keys["full"]`** (via `batch.own_model`) into every trader's token!
     - Specifically on `Z6` (`ComplementaryXOREnv`, `zoo.py:482`), `keys["full"]` is `pair_id` (the exact 3-dim one-hot pair identity `eye[pair_idx]`), and on `Z5` (`RegimeSwitchEnv`, `zoo.py:424`), `keys["full"]` contains the exact regime-2 sensitivity `a[:, :, 1]`. Because `own_model` is included unconditionally at `visibility="none"` and `"noisy"`, the `visibility="none"` vs `"noisy"` vs `"full"` comparison in `layout="trader"` is completely short-circuited on `Z2`, `Z5`, and `Z6` (`visibility="none"` receives the exact noiseless key in `own_model`, while `visibility="noisy"` receives *both* the noiseless key in `own_model` and a noisy copy in `keys["noisy"]`!).
  4. Conversely, look at what `apply_alignment` (`envs.py:146-149`) does to `own_model` on **`Z1` (`CommonCauseEnv`, `zoo.py:194`, `key_slot_blocks = ((0, 3, 1),)`):**
     - In `Z1`, `key_full` (`dim=5`) is `[group_onehot_0, group_onehot_1, group_onehot_2, tau_i, rho_parent]`.
     - Because `key_slot_blocks = ((0, 3, 1),)`, `alignment_transforms` (`envs.py:148-149`) applies an **independent random permutation $\pi_i \in S_3$ to each agent $i$'s 3-dim group one-hot vector** in `own_model` (while permuting `keys["full"]` by a shared episode permutation $\pi_{\text{ep}} \in S_3$).
     - Permuting a 3-dim one-hot vector `[1, 0, 0]` by an independent uniform permutation $\pi_i \in S_3$ for each agent $i$ destroys all cross-agent group information in coordinates `0..2` of `own_model` (turning it into iid uniform one-hot noise across agents!). The **only** reason group identity survives in `Z1` at `visibility="none"` is that coordinate 4 (`rho_parent = st["rho"][g]`, a continuous float drawn per group) sits outside `key_slot_blocks`!
- **Impact:**
  1. On `Z2`, `Z5`, `Z6`, `visibility="none"` and `"noisy"` in `full_context` (`layout="trader"`) are not actually testing missing or noisy keys unless `own_model` is excluded or properly distinguished from public keys (`peer_keys=False, own_model=False` for `visibility="none"`, or `own_model` defined as private local sensor calibration distinct from shared pair/group IDs).
  2. In `piece_tokens` (`trader_transformer.py:109-131`, used by `layout in {"grid", "axial", "grid_full"}`), `batch.own_model` is **ONLY** read when ` batch.obs.shape[1] != 12` (`trader_transformer.py:121-125`), and in that `else` branch `batch.keys[visibility]` is **completely ignored**! Look at `trader_transformer.py:120-125`:
     ```python
     if k != 12 or batch.own_model is None or batch.own_model.shape[2] < 2 * k:
         # Fallback for non-piece envs: single token per trader from full_context.
         obs = np.nan_to_num(batch.obs, nan=0.0)
         mask = (~np.isnan(batch.obs)).astype(np.float32)
         own = np.nan_to_num(batch.own_model, nan=0.0) if batch.own_model is not None else np.zeros((b, n, 0), dtype=np.float32)
         tok = np.concatenate([obs, mask, own], axis=-1).astype(np.float32)
         return torch.as_tensor(tok[:, :, None, :])
     ```
     Notice that in `piece_tokens`:
     - On `E2` (`k == 12`), `piece_tokens` reads `batch.obs` and `batch.own_model`, and **never reads `batch.keys[visibility]` at all** (`visibility` is an unused parameter in `piece_tokens`!). Thus `evaluate_trader_transformer(..., layout="grid", visibility=...)` runs the **exact same tensor** for `visibility="none"`, `"noisy"`, and `"full"`!

---

### Finding `V3` (`HIGH` — `piece_tokens` in `aggbench/trader_transformer.py:109-131` Always Includes Noiseless `own` (`keys["full"]`) Even at `visibility="none"` and `"noisy"`, While `full_context` in `transformer.py:46-52` Appends Pre-Computed `batch.reports["logit"]`)

- **Evidence (`file:line`):**
  1. In `aggbench/trader_transformer.py:116-122`:
     ```python
     obs, keys, _, own = batch.agent_inputs(visibility, alignment=alignment, own_model=True)
     n, N, K = obs.shape
     T = max_pieces or K
     feat = [obs[..., None], own.reshape(n, N, 2, K).transpose(0, 1, 3, 2)]
     if keys.shape[-1]:
         feat.append(keys.reshape(n, N, 2, K).transpose(0, 1, 3, 2))
     ```
     Because `own` is `apply_alignment(..., self.keys["full"].copy(), T)` (`envs.py:93,97`), every cell `[present, obs_k, own_cov_k, own_mu_k, ...]` in `piece_tokens` already contains the **exact noiseless piece coverage and piece identity `(cov_k, cov_k * mu_k)`** at coordinates `2:4`, even when `visibility="none"` or `"noisy"`! When `visibility="full"`, coordinates `4:6` simply duplicate coordinates `2:4` (`keys == own`), and when `visibility="noisy"`, coordinates `4:6` append a noisy copy alongside the noiseless copy at `2:4`.
  2. Meanwhile, in `aggbench/transformer.py:46-48` (`full_context`, used when `layout="trader"`):
     ```python
     if batch.kind == "bernoulli":
         rep = batch.reports["logit"][..., None] / 4
     ```
     `layout="trader"` receives the pre-computed exact local Bayesian report `batch.reports["logit"]` inside every token (on `Z1`, `oracle_clearing == bayes == 0.000`, so `rep` + `key` already suffices without reading `obs`), whereas `piece_tokens` (`layout in {"grid", "axial", "grid_full"}`) omits `batch.reports["logit"]` and only supports `E2/E3` (`"cov" in batch.oracle`, line 114).

---

### Finding `V4` (`HIGH` — In `layout="trader"`, `nn.Linear(in_dim, width)` on Concatenated `unaligned` `E2` Slots Breaks Intra-Trader Slot Permutation Invariance $S_{12}$)

- **Evidence (`file:line`):**
  - Under `alignment="unaligned"` on `E2`/`E3` (`envs.py:148-151`), each trader $i$'s 12 observation slots `obs[b, i, 0..11]` and 12 local model slots `own_model[b, i, 0..23]` are scrambled by an **independent random permutation $\pi_i \in S_{12}$ per trader per episode** ($12! = 479{,}001{,}600$ permutations).
  - In `full_context` (`transformer.py:29-55`) and `_TraderTransformerModule` (`trader_transformer.py:197`), those 12 scrambled slots are flattened into a 48-dim vector `[obs_0..11, mask_0..11, own_0..23, ...]` and fed into a single unstructured `self.embed = nn.Linear(in_dim, width)`.
  - For a flat `nn.Linear(in_dim, width)` to compute even a simple permutation-invariant intra-trader quantity (or match piece $\mu_s$ across slots), its weights across all 12 slot positions must converge to identical or equivariant sub-blocks from finite data. At $n_{\text{train}} = 4000$, this broken intra-trader $S_{12}$ symmetry leaves `full_context_transformer` at `0.025–0.027` regret on `E2 (unaligned)` (vs `0.017` for scalar `oracle_clearing`).
- **Diagnostic Prediction for `layout="grid_full"` vs `layout="trader"`:**
  - In `piece_tokens` (`trader_transformer.py:127`), each slot $s \in \{0..11\}$ is tokenized as a **4-dim per-piece vector `[obs_s, seen_s, c_s, (c*mu)_s]`** and embedded via a **shared** `nn.Linear(4, width)` across all 12 pieces (`trader_transformer.py:197`). That automatically restores per-piece weight sharing!

---

### Finding `V5` (`HIGH` — In `layout="grid"` and `"axial"` (`trader_transformer.py:127, 154-156`), `trd_axis` Restricts Cross-Trader Attention to Matching Local Rank `t' == t`, Which Mismatches Unaligned Pieces)

- **Evidence (`file:line`):**
  - Look at how `piece_tokens` (`trader_transformer.py:126-130`) orders the $K=12$ piece tokens within each trader $r$:
    ```python
    # Order seen pieces first (sorted by seen mask descending, preserving random local order).
    order = np.argsort(-seen, axis=-1, kind="stable")
    ```
    Suppose Trader $r_1$ sees 3 pieces $\{A, B, C\}$ at local ranks $t \in \{0, 1, 2\}$ (and ranks $t \in \{3..11\}$ are empty padding with `seen=0`), while Trader $r_2$ sees 4 pieces $\{D, A, E, F\}$ where the shared piece $A$ lands at local rank $t = 1$ (not $t = 0$!).
  - Now look at `grid_masks` (`trader_transformer.py:155-156`):
    ```python
    tok_axis = ~((r[:, None] == r[None]) & (t[None] <= t[:, None]))
    trd_axis = ~((t[:, None] == t[None]) & (r[None] <= r[:, None]))
    ```
  - Under `trd_axis` (used by half the heads in `layout="grid"` and odd layers in `layout="axial"`), cell `(t, r)` at Trader $r_2$ can **only** attend to earlier Trader $r_1$ at the **exact same local rank `t' == t`**!
    - Since piece identity is randomly permuted inside each trader (`alignment="unaligned"`), rank $t$ in Trader $r_2$ almost never holds the same piece as rank $t$ in Trader $r_1$ (and for $t \ge L_{r_1}$, rank $t$ in Trader $r_1$ is an empty zero-padded slot!).
    - By contrast, `layout="grid_full"` (`full_cone`, `trader_transformer.py:157`: `~((t[None] <= t[:, None]) & (r[None] <= r[:, None]))`) allows cell `(t, r)` (and in particular the readout cell `(T-1, R-1) = (11, N-1)`) to attend across **all** piece ranks $t' \le t$ of all earlier traders $r' \le r$, enabling direct content-based key matching (`mu_{r_1, t_1} == mu_{r_2, t_2}`) across different local ranks $t_1 \neq t_2$!
  - Furthermore, in `tok_axis` (`t' <= t, r' = r`), piece rank $t=0$ within Trader $r$ cannot attend to piece ranks $t' > 0$ of the *same* trader $r$; only rank $t = T-1 = 11$ (which is an empty zero-padded slot whenever a trader sees fewer than 12 pieces!) sees all of Trader $r$'s pieces.

---

## 4. Summary of Required Fixes & Experimental Plan for Tonight

1. **Completed `#3840` README Relabeling:**
   - Updated `results/explore/expagghighdim/README.md`, `results/explore/agg_highdim_summary/README.md`, `results/explore/expaggscalar/README.md`, `results/explore/agg_scalar_summary/README.md`, and `results/explore/expaggkylevoi/README.md` (`● Informative`, aligned-slot giveaway / scalar / side-line labels, `Builds on:` lines added, `Decided:` withdrawn, `D11` environment formulas fixed).
2. **Urgent Alert to `opus-mats` for `nearbayes-20261021`:**
   - `V1`: Move `TraderTransformer` to `cuda` on L4 VMs (`trader_transformer.py` currently runs 100% on CPU with 2 threads!).
   - `V2` & `V3`: Note that `own_model` in `full_context` leaks `keys["full"]` into `visibility="none"`/`"noisy"` on `Z2/Z5/Z6`, and `piece_tokens` ignores `visibility` completely.
   - `V4` & `V5`: On `E2 (unaligned)`, `layout="grid_full"` (or intra-trader permutation-invariant piece embedding before cross-trader attention) avoids both the flat `nn.Linear(48, width)` $S_{12}$ bottleneck (`V4`) and the rank-locked `t' == t` cross-trader mask mismatch (`V5`).
3. **Our Spot-L4 `(A)/(B)/(C)/grid` Sweep (`results/explore/trader-transformer-abc-grid-20261021/`):**
   - Run the systematic `(A)` `attn_layers`, `(B)` `loss` (`final`, `every_prefix`, `distill` + prefix curve), `(C)` `token_state` (`kv`, `recurrent`, `feedback`), and `(D)` `layout` (`trader`, `grid`, `axial`, `grid_full`) sweep on spot L4 GPU (`CUDA` enabled!) across `E2`, `Z1`, `Z5`, `Z6` (`alignment="unaligned"`, `visibility in {"none", "noisy", "full"}`, 3 seeds), with the `(a)–(d)` gate reported in full.
