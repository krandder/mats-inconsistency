"""
Self-Test Suite and Demo Report Generator for `src/eval/elo_league.py`
and `src/eval/mini_eci.py`.

Run as a test suite:
  pytest /home/agent/workspaces/inconsistency-search/src/eval/test_eval.py
or standalone (runs all self-tests and generates the calibrated demo report):
  python3 -m src.eval.test_eval
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.elo_league import (
    COMPUTE_BUDGETS,
    DEFAULT_ANCHOR_ELO,
    ELO_SCALE,
    MULTIPLAYER_ENVS,
    AgentSpec,
    BayesianEloLeague,
    LeagueFitResult,
    MultiplayerGameRecord,
    btd_probabilities,
    simulate_calibrated_match,
)
from src.eval.mini_eci import (
    DEFAULT_ANCHOR_MINI_ECI,
    SINGLE_AGENT_BENCHMARKS,
    MiniECIFitResult,
    MiniECIFitter,
    simulate_single_agent_benchmarks,
)


def build_reference_population() -> List[AgentSpec]:
    """
    Build a representative multi-family agent population spanning all 9 compute
    budgets B in {0, 1, 2, 4, 8, 16, 32, 64, 128}.
    """
    specs: List[AgentSpec] = [
        AgentSpec("1-pass", "greedy-anchor", 0),
    ]
    scaling_families: List[Tuple[str, str]] = [
        ("Flat-MC P3", "unweighted-mix"),
        ("PUCT", "key-space-puct"),
        ("Hardcoded-Beam P8", "sync-beam-chk"),
        ("Learned-KV-Tree", "select-node-step-looped-T3"),
        ("Fractal-Prospectus-Tree", "hindsight-plan-looped-T3"),
        ("LIR-Masked", "detached-belief-rl"),
        ("Token-Masked", "shared-trunk-dcpo"),
        ("Fused", "joint-outcome-rl"),
    ]
    for fam, alg in scaling_families:
        for b in COMPUTE_BUDGETS:
            specs.append(AgentSpec(fam, alg, b))
    return specs


def test_btd_probabilities_and_parameter_recovery() -> None:
    """Verify exact BTD probability axioms and MAP parameter recovery."""
    p_a, p_b, p_d = btd_probabilities(0.8, -0.2, nu=0.35, first_mover_gamma=0.15, first_player_sign=1)
    assert abs((p_a + p_b + p_d) - 1.0) < 1e-12
    assert p_a > p_b > 0.0 and p_d > 0.0

    # Symmetry check when signs and skills are flipped
    p_a2, p_b2, p_d2 = btd_probabilities(-0.2, 0.8, nu=0.35, first_mover_gamma=0.15, first_player_sign=-1)
    assert abs(p_a - p_b2) < 1e-12
    assert abs(p_b - p_a2) < 1e-12
    assert abs(p_d - p_d2) < 1e-12

    # Parameter recovery on a 4-agent synthetic league
    league = BayesianEloLeague(anchor_elo=1000.0, prior_sigma=3.0)
    agents = [
        league.register_agent("1-pass", "anchor", 0),
        league.register_agent("PUCT", "puct", 16),
        league.register_agent("Learned-KV-Tree", "kv-tree", 64),
        league.register_agent("Fractal-Prospectus-Tree", "prospectus", 128),
    ]
    true_thetas = {
        agents[0].agent_id: 0.0,
        agents[1].agent_id: 0.5,
        agents[2].agent_id: 1.1,
        agents[3].agent_id: 1.7,
    }
    true_nu = 0.25
    true_gamma = 0.20

    def custom_sim(env: str, a: AgentSpec, b: AgentSpec, sign: int, seed: int) -> Tuple[float, bool]:
        pa, pb, _ = btd_probabilities(
            true_thetas[a.agent_id],
            true_thetas[b.agent_id],
            nu=true_nu,
            first_mover_gamma=true_gamma,
            first_player_sign=sign,
        )
        rng = np.random.default_rng(seed)
        u = float(rng.random())
        if u < float(pa):
            return 1.0, False
        if u < float(pa + pb):
            return 0.0, False
        return 0.5, True

    league.run_round_robin_tournament(
        env_name="Connect-4",
        agent_specs=agents,
        match_fn=custom_sim,
        games_per_pair=120,
        seed=777,
    )
    fit = league.fit_environment("Connect-4", n_bootstrap=100, seed=777)

    # Check anchor is exactly 1000.0
    r0 = fit.get_rating(agents[0])
    assert abs(r0.elo - 1000.0) < 1e-9

    # Check rank order matches ground-truth theta strictly
    elos = [fit.get_rating(a).elo for a in agents]
    assert elos[0] < elos[1] < elos[2] < elos[3], f"Expected strict ordering, got {elos}"

    # Check true Elo values fall within or very close to 95% bootstrap CIs
    for a in agents[1:]:
        r = fit.get_rating(a)
        true_elo = 1000.0 + ELO_SCALE * true_thetas[a.agent_id]
        assert abs(r.elo - true_elo) < 45.0, f"Elo recovery error too large: {r.elo} vs {true_elo}"
        assert r.elo_ci_low < r.elo < r.elo_ci_high


def test_multiplayer_games_and_swiss_tournament() -> None:
    """Verify N-player game decomposition (Galanis-Market & Proposal-Poker) and Swiss pairing."""
    league = BayesianEloLeague(anchor_elo=1000.0)
    specs = [
        league.register_agent("1-pass", "anchor", 0),
        league.register_agent("Fused", "joint-rl", 32),
        league.register_agent("Token-Masked", "dcpo", 32),
        league.register_agent("LIR-Masked", "detached-rl", 32),
        league.register_agent("Fractal-Prospectus-Tree", "prospectus", 32),
        league.register_agent("Learned-KV-Tree", "kv-tree", 32),
    ]

    # Add 4-player Proposal-Poker tables
    rng = np.random.default_rng(1234)
    for table_idx in range(40):
        table_specs = list(rng.choice(specs, size=4, replace=False))
        payoffs = tuple(
            float(
                (0.5 if s.model_family in ("LIR-Masked", "Fractal-Prospectus-Tree") else 0.0)
                - (0.4 if s.model_family == "Fused" else 0.0)
                + 0.15 * rng.normal()
            )
            for s in table_specs
        )
        rec = MultiplayerGameRecord(
            env_name="Proposal-Poker",
            agent_ids=tuple(s.agent_id for s in table_specs),
            payoffs=payoffs,
            seed=table_idx,
        )
        league.add_multiplayer_game(rec)

    fit_poker = league.fit_environment("Proposal-Poker", n_bootstrap=60, seed=99)
    assert fit_poker.num_matches == 40 * 6  # 4 choose 2 = 6 pairs per table
    assert (
        fit_poker.get_rating(specs[3]).elo > fit_poker.get_rating(specs[1]).elo
    ), "LIR-Masked must beat Fused RL in Proposal-Poker"

    # Also test Swiss tournament on Hex-5x5
    swiss_records = league.run_swiss_tournament(
        env_name="Hex-5x5",
        agent_specs=specs,
        match_fn=simulate_calibrated_match,
        num_rounds=5,
        games_per_pairing=4,
        seed=303,
    )
    assert len(swiss_records) == 5 * (len(specs) // 2) * 4
    fit_hex = league.fit_environment("Hex-5x5", n_bootstrap=60, seed=101)
    assert fit_hex.win_prob_matrix.shape == (len(specs), len(specs))
    assert fit_hex.dominance_matrix.shape == (len(specs), len(specs))


def test_mini_eci_rasch_2pl_and_compute_scaling() -> None:
    """
    Test joint Mini-ECI 1D IRT / Rasch-2PL fitting across all 4 single-agent
    benchmarks and all 5 multiplayer Elo environments over B in {0..128}.
    """
    # Use a compact subset of families across all 9 compute budgets B in {0..128}
    specs: List[AgentSpec] = [AgentSpec("1-pass", "greedy-anchor", 0)]
    for fam, alg in [
        ("Flat-MC P3", "unweighted-mix"),
        ("PUCT", "key-space-puct"),
        ("Learned-KV-Tree", "select-node-step-looped-T3"),
        ("Fractal-Prospectus-Tree", "hindsight-plan-looped-T3"),
        ("Fused", "joint-outcome-rl"),
    ]:
        for b in COMPUTE_BUDGETS:
            specs.append(AgentSpec(fam, alg, b))

    league = BayesianEloLeague(anchor_elo=1000.0)
    for s in specs:
        league.register_agent(s.model_family, s.search_algorithm, s.compute_budget_B)

    for env_idx, env in enumerate(MULTIPLAYER_ENVS):
        league.run_round_robin_tournament(
            env_name=env,
            agent_specs=specs,
            match_fn=simulate_calibrated_match,
            games_per_pair=2,
            seed=500 + env_idx * 100,
        )

    env_fits = league.fit_all_environments(n_bootstrap=40, seed=888)
    assert set(MULTIPLAYER_ENVS).issubset(set(env_fits.keys()))
    assert "ALL" in env_fits

    # Fit Mini-ECI combining single-agent benchmarks + multiplayer Elo ladders
    fitter = MiniECIFitter(anchor_mini_eci=DEFAULT_ANCHOR_MINI_ECI)
    sa_records = simulate_single_agent_benchmarks(
        agent_specs=specs,
        tasks=SINGLE_AGENT_BENCHMARKS,
        seeds=(101, 102, 103),
    )
    for r in sa_records:
        fitter.add_single_agent_record(
            model_family=r.model_family,
            search_algorithm=r.search_algorithm,
            compute_budget_B=r.compute_budget_B,
            task_name=r.task_name,
            kl=r.kl,
            brier=r.brier,
            witness_rate=r.witness_rate,
            seed=r.seed,
        )

    for env in MULTIPLAYER_ENVS:
        fitter.add_league_result(env_fits[env])

    eci_result = fitter.fit(n_bootstrap=60, seed=42)

    # 1. Verify 1-pass B=0 anchor is calibrated at exactly 100.0
    anchor_score = eci_result.scores["1-pass::greedy-anchor@B=0"]
    assert abs(anchor_score.mini_eci - 100.0) < 1e-9

    # 2. Verify all 17 IRT items (4 single-agent tasks * 3 metrics + 5 multiplayer Elo ladders)
    assert len(eci_result.items) == 4 * 3 + 5
    assert eci_result.explained_variance_r2 > 0.85, (
        f"Expected high 1D IRT explained variance, got {eci_result.explained_variance_r2:.4f}"
    )
    assert eci_result.leave_one_task_out_r > 0.85, (
        f"Expected high leave-one-task-out correlation, got {eci_result.leave_one_task_out_r:.4f}"
    )

    # 3. Verify compute-scaling curves span B in {0, 1, 2, 4, 8, 16, 32, 64, 128}
    fpt_curve = eci_result.scaling_curves["Fractal-Prospectus-Tree::hindsight-plan-looped-T3"]
    kv_curve = eci_result.scaling_curves["Learned-KV-Tree::select-node-step-looped-T3"]
    puct_curve = eci_result.scaling_curves["PUCT::key-space-puct"]
    mc_curve = eci_result.scaling_curves["Flat-MC P3::unweighted-mix"]
    fused_curve = eci_result.scaling_curves["Fused::joint-outcome-rl"]

    assert tuple(fpt_curve.budgets) == COMPUTE_BUDGETS
    assert fpt_curve.kl_is_monotone and kv_curve.kl_is_monotone
    assert not mc_curve.kl_is_monotone and not fused_curve.kl_is_monotone
    assert (
        fpt_curve.mini_ecis[-1] > kv_curve.mini_ecis[-1] > puct_curve.mini_ecis[-1] > mc_curve.mini_ecis[-1]
    )


def generate_demo_report(
    output_dir: Path = Path("/home/agent/workspaces/inconsistency-search/results"),
    seed: int = 20261018,
) -> Tuple[Dict[str, LeagueFitResult], MiniECIFitResult, Path, Path]:
    """
    Run the full population tournament across all 5 multiplayer environments and
    all 4 single-agent benchmarks over `B in {0, 1, 2, 4, 8, 16, 32, 64, 128}`,
    and emit a self-verifying Markdown + JSON report in `results/`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_reference_population()

    league = BayesianEloLeague(anchor_elo=1000.0)
    for s in specs:
        league.register_agent(s.model_family, s.search_algorithm, s.compute_budget_B)

    for env_idx, env in enumerate(MULTIPLAYER_ENVS):
        league.run_round_robin_tournament(
            env_name=env,
            agent_specs=specs,
            match_fn=simulate_calibrated_match,
            games_per_pair=2,
            seed=seed + env_idx * 1000,
        )

    league_results = league.fit_all_environments(n_bootstrap=100, seed=seed)

    fitter = MiniECIFitter(anchor_mini_eci=100.0)
    sa_records = simulate_single_agent_benchmarks(
        agent_specs=specs,
        tasks=SINGLE_AGENT_BENCHMARKS,
        seeds=(101, 102, 103),
    )
    for r in sa_records:
        fitter.add_single_agent_record(
            model_family=r.model_family,
            search_algorithm=r.search_algorithm,
            compute_budget_B=r.compute_budget_B,
            task_name=r.task_name,
            kl=r.kl,
            brier=r.brier,
            witness_rate=r.witness_rate,
            seed=r.seed,
        )
    for env in MULTIPLAYER_ENVS:
        fitter.add_league_result(league_results[env])

    eci_result = fitter.fit(n_bootstrap=100, seed=seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    json_path = output_dir / "mini_eci_elo_demo_results.json"
    md_path = output_dir / "mini_eci_elo_demo_report.md"

    payload = {
        "provenance": {
            "generator_script": "src/eval/test_eval.py",
            "reproduce_command": "python3 -m src.eval.test_eval",
            "seed": seed,
            "generated_at_utc": timestamp,
            "compute_budgets": list(COMPUTE_BUDGETS),
            "single_agent_benchmarks": list(SINGLE_AGENT_BENCHMARKS),
            "multiplayer_environments": list(MULTIPLAYER_ENVS),
        },
        "mini_eci_summary": {
            "anchor_agent_id": eci_result.anchor_agent_id,
            "anchor_mini_eci": eci_result.anchor_mini_eci,
            "explained_variance_r2": eci_result.explained_variance_r2,
            "leave_one_task_out_r": eci_result.leave_one_task_out_r,
            "bootstrap_replicates": eci_result.bootstrap_replicates,
        },
        "irt_items": {k: v.to_dict() for k, v in eci_result.items.items()},
        "scaling_curves": {
            k: {
                "model_family": v.model_family,
                "search_algorithm": v.search_algorithm,
                "budgets": v.budgets,
                "mini_ecis": v.mini_ecis,
                "ci_lows": v.ci_lows,
                "ci_highs": v.ci_highs,
                "mean_kls": v.mean_kls,
                "mean_briers": v.mean_briers,
                "mean_witness_rates": v.mean_witness_rates,
                "mean_elos": v.mean_elos,
                "kappa_per_doubling": v.kappa_per_doubling,
                "r_squared": v.r_squared,
                "eci_is_monotone": v.eci_is_monotone,
                "kl_is_monotone": v.kl_is_monotone,
                "total_eci_gain_B128": v.total_eci_gain_B128,
                "relative_kl_reduction_B128": v.relative_kl_reduction_B128,
            }
            for k, v in eci_result.scaling_curves.items()
        },
        "leaderboard": [s.to_dict() for s in eci_result.leaderboard()],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Build Markdown report
    lines: List[str] = [
        "# Mini-ECI & Multiplayer Bayesian Elo League — Evaluation Report",
        "",
        f"- **Provenance Check:** Generated by `python3 -m src.eval.test_eval` (`src/eval/test_eval.py`, seed `{seed}`) at `{timestamp}`.",
        f"- **Data Artifact:** `results/mini_eci_elo_demo_results.json`",
        f"- **1D Rasch-2PL IRT Explained Variance ($R^2$):** `{eci_result.explained_variance_r2:.4f}` | **Leave-One-Task-Out $r$:** `{eci_result.leave_one_task_out_r:.4f}`",
        f"- **Anchor Baseline:** `{eci_result.anchor_agent_id}` $\\equiv$ **`Mini-ECI = {eci_result.anchor_mini_eci:.1f}`**, **`Elo = {DEFAULT_ANCHOR_ELO:.1f}`**",
        "",
        "## 1. Test-Time Compute Scaling Curves ($B \\in \\{0, 1, 2, 4, 8, 16, 32, 64, 128\\}$)",
        "",
        "| Model Family | Search Algorithm | $B=0$ | $B=4$ | $B=16$ | $B=64$ | $B=128$ | Slope $\\kappa$ (/doubling) | KL Cut ($B=128$) | Anytime KL Monotone |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for key, curve in sorted(
        eci_result.scaling_curves.items(),
        key=lambda kv: kv[1].mini_ecis[-1],
        reverse=True,
    ):
        b_to_eci = dict(zip(curve.budgets, curve.mini_ecis))
        b0 = b_to_eci.get(0, 100.0)
        b4 = b_to_eci.get(4, b0)
        b16 = b_to_eci.get(16, b0)
        b64 = b_to_eci.get(64, b0)
        b128 = b_to_eci.get(128, b0)
        mono_str = "PASS" if curve.kl_is_monotone else "FAIL (oversearch)"
        lines.append(
            f"| `{curve.model_family}` | `{curve.search_algorithm}` | "
            f"{b0:.1f} | {b4:.1f} | {b16:.1f} | {b64:.1f} | **{b128:.1f}** | "
            f"`+{curve.kappa_per_doubling:.2f}` | `{100.0 * curve.relative_kl_reduction_B128:.1f}%` | {mono_str} |"
        )

    lines.extend(
        [
            "",
            "## 2. Top Frontier Agents ($B = 128$ vs. Baseline Anchor $B = 0$)",
            "",
            "| Rank | Agent ID | Mini-ECI [95% CI] | Single-Agent Sub-ECI | Multiplayer Elo [95% CI] | Mean KL $\\downarrow$ | Mean Brier $\\downarrow$ | Mean Witness $\\uparrow$ |",
            "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    all_elo_fit = league_results["ALL"]
    shown_ranks = 0
    for score in eci_result.leaderboard():
        if score.compute_budget_B in (0, 16, 64, 128) and shown_ranks < 18:
            shown_ranks += 1
            elo_r = all_elo_fit.ratings[score.agent_id]
            lines.append(
                f"| {shown_ranks} | `{score.agent_id}` | "
                f"**{score.mini_eci:.1f}** `[{score.mini_eci_ci_low:.1f}, {score.mini_eci_ci_high:.1f}]` | "
                f"{score.single_agent_sub_eci:.1f} | "
                f"{elo_r.elo:.1f} `[{elo_r.elo_ci_low:.1f}, {elo_r.elo_ci_high:.1f}]` | "
                f"{score.mean_kl:.4f} | {score.mean_brier:.4f} | {100.0 * score.mean_witness_rate:.1f}% |"
            )

    lines.extend(
        [
            "",
            "## 3. Fitted Rasch-2PL Item Parameters Across All 9 Environments",
            "",
            "| Item ID | Domain | Task | Metric | Discrimination $a_k$ | Difficulty $b_k$ | 1D Factor $R^2$ |",
            "| :--- | :--- | :--- | :--- | :---: | :---: | :---: |",
        ]
    )
    for item_id, item in sorted(eci_result.items.items()):
        lines.append(
            f"| `{item.item_id}` | `{item.domain}` | `{item.task_name}` | `{item.metric_name}` | "
            f"{item.discrimination_a:.3f} | {item.difficulty_b:+.3f} | {item.r_squared_with_theta:.4f} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return league_results, eci_result, md_path, json_path


def run_all_tests() -> None:
    print("[1/4] Testing Bradley-Terry-Davidson probabilities & MAP Elo recovery...")
    test_btd_probabilities_and_parameter_recovery()
    print("      PASS")

    print("[2/4] Testing N-player market games (Proposal-Poker) & Swiss tournament (Hex-5x5)...")
    test_multiplayer_games_and_swiss_tournament()
    print("      PASS")

    print("[3/4] Testing Mini-ECI 1D Rasch-2PL IRT fitter & B=0..128 compute scaling...")
    test_mini_eci_rasch_2pl_and_compute_scaling()
    print("      PASS")

    print("[4/4] Generating full population tournament & Mini-ECI demo report...")
    _, eci_res, md_path, json_path = generate_demo_report()
    print(f"      PASS — Explained Variance R^2 = {eci_res.explained_variance_r2:.4f}, LOTO r = {eci_res.leave_one_task_out_r:.4f}")
    print(f"      Report written to: {md_path}")
    print(f"      JSON written to:   {json_path}")


if __name__ == "__main__":
    run_all_tests()
