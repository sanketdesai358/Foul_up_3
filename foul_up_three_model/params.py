"""Shared model parameters and league defaults for the foul-up-three simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal


LeagueName = Literal["NBA", "NCAA"]
Possession = Literal["trailing", "leading"]
PolicyName = Literal["foul", "defend"]


@dataclass(frozen=True)
class LeagueRules:
    """Clock and foul rules that differ by league."""

    name: LeagueName = "NBA"
    shot_clock: float = 24.0
    offensive_rebound_clock: float = 14.0
    stop_clock_after_made_fg_under: float = 120.0
    bonus_ft_count: int = 2
    default_fouls_to_give: int = 0


@dataclass(frozen=True)
class TeamParams:
    """Team-level probabilities used by the end-game model.

    All probability values are expressed from 0.0 to 1.0.
    """

    trailing_ft_pct: float = 0.78
    trailing_three_pct: float = 0.36
    contested_three_modifier: float = 0.88
    trailing_two_pct: float = 0.53
    trailing_turnover_rate: float = 0.075
    live_ball_oreb_pct: float = 0.24
    ft_miss_oreb_pct: float = 0.14
    putback_two_pct: float = 0.58
    putback_three_pct: float = 0.33
    putback_kickout_three_rate: float = 0.23
    intentional_miss_execution_pct: float = 0.72
    leading_ft_pct: float = 0.79
    closing_lineup_ft_pct: float = 0.84
    use_closing_lineup_ft: bool = False
    defensive_rebound_pct: float = 0.76
    inbound_turnover_rate: float = 0.025

    @property
    def effective_leading_ft_pct(self) -> float:
        """Return the free-throw percentage used for the leader in the foul war."""

        if self.use_closing_lineup_ft:
            return self.closing_lineup_ft_pct
        return self.leading_ft_pct


@dataclass(frozen=True)
class StrategyParams:
    """Policy and late-game behavior parameters."""

    foul_lead_min: int = 3
    foul_time_max: float = 45.0
    p_foul_shooter: float = 0.035
    p_made_three_and_one: float = 0.004
    clean_foul_probability: float = 0.961
    time_to_foul_min: float = 2.0
    time_to_foul_max: float = 4.0
    retaliatory_foul_min: float = 1.0
    retaliatory_foul_max: float = 3.0
    foul_to_give_clock_burn_min: float = 1.0
    foul_to_give_clock_burn_max: float = 3.0
    ot_win_prob: float = 0.50
    max_transitions: int = 24


@dataclass(frozen=True)
class GameState:
    """Initial scoreboard and rules state for a scenario."""

    time_remaining: float = 12.0
    lead: int = 3
    possession: Possession = "trailing"
    leading_timeouts: int = 2
    trailing_timeouts: int = 2
    leading_fouls_to_give: int = 0
    trailing_fouls_to_give: int = 0
    leading_bonus: bool = True
    trailing_bonus: bool = True


@dataclass(frozen=True)
class SimulationParams:
    """Full input bundle for one Monte Carlo comparison."""

    league: LeagueRules = LeagueRules()
    teams: TeamParams = TeamParams()
    strategy: StrategyParams = StrategyParams()
    state: GameState = GameState()
    n_trials: int = 100_000
    seed: int = 7


NBA_RULES = LeagueRules()
NCAA_RULES = LeagueRules(
    name="NCAA",
    shot_clock=30.0,
    offensive_rebound_clock=20.0,
    stop_clock_after_made_fg_under=60.0,
    bonus_ft_count=2,
    default_fouls_to_give=0,
)


def league_rules(name: LeagueName) -> LeagueRules:
    """Return default rules for the requested league."""

    return NCAA_RULES if name == "NCAA" else NBA_RULES


def as_flat_dict(params: SimulationParams) -> dict[str, object]:
    """Flatten nested dataclasses for CSV export and UI display."""

    flat: dict[str, object] = {}
    for prefix, value in (
        ("league", params.league),
        ("teams", params.teams),
        ("strategy", params.strategy),
        ("state", params.state),
    ):
        for key, val in asdict(value).items():
            flat[f"{prefix}_{key}"] = val
    flat["n_trials"] = params.n_trials
    flat["seed"] = params.seed
    return flat


def with_state(params: SimulationParams, **state_updates: object) -> SimulationParams:
    """Return a parameter bundle with selected initial game-state fields changed."""

    return replace(params, state=replace(params.state, **state_updates))


def with_team(params: SimulationParams, **team_updates: object) -> SimulationParams:
    """Return a parameter bundle with selected team assumptions changed."""

    return replace(params, teams=replace(params.teams, **team_updates))


def with_strategy(params: SimulationParams, **strategy_updates: object) -> SimulationParams:
    """Return a parameter bundle with selected strategy assumptions changed."""

    return replace(params, strategy=replace(params.strategy, **strategy_updates))
