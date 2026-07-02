from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import compare_policies, simulate_policy
from params import GameState, SimulationParams, StrategyParams, TeamParams


def test_time_zero_positive_lead_always_holds() -> None:
    params = SimulationParams(
        state=GameState(time_remaining=0, lead=3, possession="trailing"),
        n_trials=20_000,
        seed=11,
    )

    foul = simulate_policy(params, "foul")
    defend = simulate_policy(params, "defend")

    assert foul.win_probability == 1.0
    assert defend.win_probability == 1.0


def test_time_zero_tie_uses_overtime_probability() -> None:
    params = SimulationParams(
        state=GameState(time_remaining=0, lead=0, possession="trailing"),
        strategy=StrategyParams(ot_win_prob=0.25),
        n_trials=80_000,
        seed=12,
    )

    result = simulate_policy(params, "defend")

    assert 0.245 <= result.win_probability <= 0.255


def test_perfect_leader_free_throws_make_foul_war_strong_when_no_pressure_turnovers() -> None:
    params = SimulationParams(
        state=GameState(time_remaining=8, lead=3, possession="trailing"),
        teams=TeamParams(
            leading_ft_pct=1.0,
            closing_lineup_ft_pct=1.0,
            inbound_turnover_rate=0.0,
            trailing_ft_pct=0.75,
            ft_miss_oreb_pct=0.0,
        ),
        strategy=StrategyParams(
            p_foul_shooter=0.0,
            p_made_three_and_one=0.0,
            time_to_foul_min=1.0,
            time_to_foul_max=1.0,
            retaliatory_foul_min=1.0,
            retaliatory_foul_max=1.0,
        ),
        n_trials=40_000,
        seed=13,
    )

    result = simulate_policy(params, "foul")

    assert result.win_probability > 0.995


def test_average_inputs_short_clock_foul_up_three_has_positive_edge() -> None:
    params = SimulationParams(
        state=GameState(time_remaining=5, lead=3, possession="trailing"),
        n_trials=80_000,
        seed=14,
    )

    comparison = compare_policies(params)

    assert comparison.delta > 0.005


def test_average_inputs_45_seconds_not_large_foul_edge() -> None:
    params = SimulationParams(
        state=GameState(time_remaining=45, lead=3, possession="trailing"),
        n_trials=80_000,
        seed=15,
    )

    comparison = compare_policies(params)

    assert comparison.delta < 0.03
