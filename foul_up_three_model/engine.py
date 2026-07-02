"""Vectorized Monte Carlo engine for late-game foul-up-three strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from params import PolicyName, SimulationParams


@dataclass(frozen=True)
class SimulationResult:
    """Aggregate result from simulating one policy."""

    policy: PolicyName
    win_probability: float
    standard_error: float
    wins: int
    trials: int


@dataclass(frozen=True)
class PolicyComparison:
    """Head-to-head comparison between the foul and defend policies."""

    foul: SimulationResult
    defend: SimulationResult
    delta: float
    ci_low: float
    ci_high: float

    @property
    def recommendation(self) -> str:
        """Plain-English decision label from the confidence interval and point estimate."""

        if self.ci_low > 0:
            return "Foul: simulated edge is positive with 95% confidence."
        if self.ci_high < 0:
            return "Defend: simulated edge favors straight defense."
        if self.delta > 0:
            return "Lean foul: edge is positive, but the interval crosses zero."
        if self.delta < 0:
            return "Lean defend: edge is negative, but the interval crosses zero."
        return "No clear edge under these assumptions."


def compare_policies(params: SimulationParams) -> PolicyComparison:
    """Simulate both policies with common random seeds and return the WP delta.

    The 95% interval uses the normal approximation for two independent binomial
    proportions. Common seeds reduce visual noise but the interval remains
    conservative for dashboard interpretation.
    """

    foul = simulate_policy(params, "foul", seed_offset=0)
    defend = simulate_policy(params, "defend", seed_offset=10_000_003)
    delta = foul.win_probability - defend.win_probability
    se_delta = float(np.sqrt(foul.standard_error**2 + defend.standard_error**2))
    return PolicyComparison(
        foul=foul,
        defend=defend,
        delta=delta,
        ci_low=delta - 1.96 * se_delta,
        ci_high=delta + 1.96 * se_delta,
    )


def simulate_policy(
    params: SimulationParams,
    policy: PolicyName,
    seed_offset: int = 0,
    return_wins: bool = False,
) -> SimulationResult | np.ndarray:
    """Simulate one policy and return wins or aggregate win probability.

    The state is held in NumPy arrays and updated possession by possession:
    scoreboard lead, time remaining, possession, fouls to give, and active mask.
    """

    n = int(params.n_trials)
    rng = np.random.default_rng(params.seed + seed_offset)
    lead = np.full(n, float(params.state.lead))
    time = np.full(n, float(params.state.time_remaining))
    possession = np.full(n, 0 if params.state.possession == "trailing" else 1, dtype=np.int8)
    leading_fouls_to_give = np.full(n, int(params.state.leading_fouls_to_give), dtype=np.int8)
    trailing_fouls_to_give = np.full(n, int(params.state.trailing_fouls_to_give), dtype=np.int8)
    active = time > 0

    if params.state.time_remaining <= 0:
        wins = _resolve_finished(rng, lead, params.strategy.ot_win_prob)
        return wins if return_wins else _result(policy, wins)

    for _ in range(params.strategy.max_transitions):
        if not np.any(active):
            break

        trailing_has_ball = active & (possession == 0)
        if np.any(trailing_has_ball):
            foul_now = (
                trailing_has_ball
                & (policy == "foul")
                & (lead >= params.strategy.foul_lead_min)
                & (time <= params.strategy.foul_time_max)
            )
            defend_now = trailing_has_ball & ~foul_now
            if np.any(foul_now):
                _apply_leading_intentional_foul(
                    rng,
                    params,
                    foul_now,
                    lead,
                    time,
                    possession,
                )
            if np.any(defend_now):
                _apply_defensive_possession(
                    rng,
                    params,
                    defend_now,
                    lead,
                    time,
                    possession,
                    leading_fouls_to_give,
                )

        leading_has_ball = active & (possession == 1)
        if np.any(leading_has_ball):
            _apply_leading_possession_under_pressure(
                rng,
                params,
                leading_has_ball,
                lead,
                time,
                possession,
                trailing_fouls_to_give,
            )

        active = time > 0

    wins = _resolve_finished(rng, lead, params.strategy.ot_win_prob)
    return wins if return_wins else _result(policy, wins)


def scenario_grid(
    params: SimulationParams,
    times: Iterable[int],
    leads: Iterable[int],
) -> list[dict[str, float | int | str]]:
    """Compare policies over a time-by-lead grid for heatmaps and CSV export."""

    from dataclasses import replace

    rows: list[dict[str, float | int | str]] = []
    for lead_value in leads:
        for time_value in times:
            grid_params = replace(
                params,
                state=replace(params.state, lead=int(lead_value), time_remaining=float(time_value)),
            )
            comparison = compare_policies(grid_params)
            rows.append(
                {
                    "lead": int(lead_value),
                    "time_remaining": int(time_value),
                    "wp_foul": comparison.foul.win_probability,
                    "wp_defend": comparison.defend.win_probability,
                    "delta": comparison.delta,
                    "ci_low": comparison.ci_low,
                    "ci_high": comparison.ci_high,
                    "recommendation": comparison.recommendation,
                }
            )
    return rows


def convergence_series(
    params: SimulationParams,
    policy: PolicyName,
    checkpoints: Iterable[int],
) -> list[dict[str, float | int | str]]:
    """Estimate win probability at increasing trial counts for diagnostics."""

    from dataclasses import replace

    rows: list[dict[str, float | int | str]] = []
    for trials in checkpoints:
        p = replace(params, n_trials=int(trials))
        result = simulate_policy(p, policy)
        assert isinstance(result, SimulationResult)
        rows.append(
            {
                "policy": policy,
                "trials": int(trials),
                "win_probability": result.win_probability,
                "standard_error": result.standard_error,
            }
        )
    return rows


def _result(policy: PolicyName, wins: np.ndarray) -> SimulationResult:
    wp = float(np.mean(wins))
    trials = int(wins.size)
    se = float(np.sqrt(max(wp * (1.0 - wp), 0.0) / max(trials, 1)))
    return SimulationResult(policy=policy, win_probability=wp, standard_error=se, wins=int(np.sum(wins)), trials=trials)


def _resolve_finished(rng: np.random.Generator, lead: np.ndarray, ot_win_prob: float) -> np.ndarray:
    wins = lead > 0
    tied = lead == 0
    if np.any(tied):
        wins = wins.copy()
        wins[tied] = rng.random(np.sum(tied)) < ot_win_prob
    return wins


def _clock_burn(
    rng: np.random.Generator,
    mask: np.ndarray,
    time: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    idx = np.flatnonzero(mask)
    burn = rng.uniform(low, high, idx.size)
    burn = np.minimum(burn, time[idx])
    time[idx] -= burn
    return idx


def _apply_leading_intentional_foul(
    rng: np.random.Generator,
    params: SimulationParams,
    mask: np.ndarray,
    lead: np.ndarray,
    time: np.ndarray,
    possession: np.ndarray,
) -> None:
    idx = _clock_burn(
        rng,
        mask,
        time,
        params.strategy.time_to_foul_min,
        params.strategy.time_to_foul_max,
    )
    if idx.size == 0:
        return

    still_time = idx[time[idx] > 0]
    if still_time.size == 0:
        return

    made_and_one = rng.random(still_time.size) < params.strategy.p_made_three_and_one
    if np.any(made_and_one):
        made_idx = still_time[made_and_one]
        lead[made_idx] -= 3.0
        lead[made_idx] -= _made_free_throws(rng, made_idx.size, 1, params.teams.trailing_ft_pct)

    remaining = still_time[~made_and_one]
    shooter_foul = rng.random(remaining.size) < params.strategy.p_foul_shooter
    if np.any(shooter_foul):
        shot_idx = remaining[shooter_foul]
        lead[shot_idx] -= _made_free_throws(rng, shot_idx.size, 3, params.teams.trailing_ft_pct)

    clean_idx = remaining[~shooter_foul]
    if clean_idx.size:
        first_makes = rng.random(clean_idx.size) < params.teams.trailing_ft_pct
        lead[clean_idx] -= first_makes.astype(float)
        should_miss = lead[clean_idx] >= 2.0
        execute_miss = should_miss & (rng.random(clean_idx.size) < params.teams.intentional_miss_execution_pct)
        second_makes = (~execute_miss) & (rng.random(clean_idx.size) < params.teams.trailing_ft_pct)
        lead[clean_idx] -= second_makes.astype(float)

        miss_idx = clean_idx[execute_miss]
        if miss_idx.size:
            oreb = rng.random(miss_idx.size) < params.teams.ft_miss_oreb_pct
            oreb_idx = miss_idx[oreb]
            if oreb_idx.size:
                kick_three = rng.random(oreb_idx.size) < params.teams.putback_kickout_three_rate
                three_idx = oreb_idx[kick_three]
                two_idx = oreb_idx[~kick_three]
                if three_idx.size:
                    lead[three_idx] -= 3.0 * (
                        rng.random(three_idx.size) < params.teams.putback_three_pct
                    )
                if two_idx.size:
                    lead[two_idx] -= 2.0 * (
                        rng.random(two_idx.size) < params.teams.putback_two_pct
                    )

    possession[still_time] = 1


def _apply_defensive_possession(
    rng: np.random.Generator,
    params: SimulationParams,
    mask: np.ndarray,
    lead: np.ndarray,
    time: np.ndarray,
    possession: np.ndarray,
    leading_fouls_to_give: np.ndarray,
) -> None:
    if np.any(mask & (leading_fouls_to_give > 0) & (time > 6.0)):
        ftg_mask = mask & (leading_fouls_to_give > 0) & (time > 6.0)
        ftg_idx = _clock_burn(
            rng,
            ftg_mask,
            time,
            params.strategy.foul_to_give_clock_burn_min,
            params.strategy.foul_to_give_clock_burn_max,
        )
        leading_fouls_to_give[ftg_idx] -= 1

    idx = np.flatnonzero(mask & (time > 0))
    if idx.size == 0:
        return

    duration = _trailing_possession_duration(rng, params, lead[idx], time[idx])
    burn = np.minimum(duration, time[idx])
    time[idx] -= burn

    live_idx = idx[time[idx] >= 0]
    if live_idx.size == 0:
        return

    turnover = rng.random(live_idx.size) < params.teams.trailing_turnover_rate
    turnover_idx = live_idx[turnover]
    if turnover_idx.size:
        possession[turnover_idx] = 1

    shot_idx = live_idx[~turnover]
    if shot_idx.size == 0:
        return

    p_three = _three_attempt_probability(lead[shot_idx], time[shot_idx])
    is_three = rng.random(shot_idx.size) < p_three
    made_three = is_three & (
        rng.random(shot_idx.size)
        < params.teams.trailing_three_pct * params.teams.contested_three_modifier
    )
    made_two = (~is_three) & (rng.random(shot_idx.size) < params.teams.trailing_two_pct)
    lead[shot_idx] -= 3.0 * made_three.astype(float)
    lead[shot_idx] -= 2.0 * made_two.astype(float)

    made = made_three | made_two
    possession[shot_idx[made]] = 1

    miss_idx = shot_idx[~made]
    if miss_idx.size:
        effective_oreb = np.clip(
            (params.teams.live_ball_oreb_pct + (1.0 - params.teams.defensive_rebound_pct)) / 2.0,
            0.0,
            1.0,
        )
        oreb = rng.random(miss_idx.size) < effective_oreb
        oreb_idx = miss_idx[oreb]
        dreb_idx = miss_idx[~oreb]
        possession[dreb_idx] = 1
        if oreb_idx.size:
            time[oreb_idx] = np.maximum(time[oreb_idx] - rng.uniform(1.0, 3.0, oreb_idx.size), 0.0)
            kick_three = rng.random(oreb_idx.size) < params.teams.putback_kickout_three_rate
            three_idx = oreb_idx[kick_three]
            two_idx = oreb_idx[~kick_three]
            if three_idx.size:
                lead[three_idx] -= 3.0 * (
                    rng.random(three_idx.size) < params.teams.putback_three_pct
                )
            if two_idx.size:
                lead[two_idx] -= 2.0 * (
                    rng.random(two_idx.size) < params.teams.putback_two_pct
                )
            possession[oreb_idx] = 1


def _apply_leading_possession_under_pressure(
    rng: np.random.Generator,
    params: SimulationParams,
    mask: np.ndarray,
    lead: np.ndarray,
    time: np.ndarray,
    possession: np.ndarray,
    trailing_fouls_to_give: np.ndarray,
) -> None:
    idx = np.flatnonzero(mask & (time > 0))
    if idx.size == 0:
        return

    inbound_turnover = rng.random(idx.size) < params.teams.inbound_turnover_rate
    turnover_idx = idx[inbound_turnover]
    if turnover_idx.size:
        time[turnover_idx] = np.maximum(time[turnover_idx] - rng.uniform(0.5, 1.5, turnover_idx.size), 0.0)
        possession[turnover_idx] = 0

    secure_idx = idx[~inbound_turnover]
    if secure_idx.size == 0:
        return

    foul_idx = _clock_burn(
        rng,
        np.isin(np.arange(time.size), secure_idx),
        time,
        params.strategy.retaliatory_foul_min,
        params.strategy.retaliatory_foul_max,
    )
    foul_idx = foul_idx[time[foul_idx] >= 0]
    if foul_idx.size == 0:
        return

    has_foul_to_give = trailing_fouls_to_give[foul_idx] > 0
    if np.any(has_foul_to_give):
        ftg_idx = foul_idx[has_foul_to_give]
        trailing_fouls_to_give[ftg_idx] -= 1
        possession[ftg_idx] = 1

    ft_idx = foul_idx[~has_foul_to_give]
    if ft_idx.size:
        lead[ft_idx] += _made_free_throws(rng, ft_idx.size, params.league.bonus_ft_count, params.teams.effective_leading_ft_pct)
        possession[ft_idx] = 0


def _made_free_throws(
    rng: np.random.Generator,
    n_trials: int,
    attempts: int,
    ft_pct: float,
) -> np.ndarray:
    return np.sum(rng.random((n_trials, attempts)) < ft_pct, axis=1).astype(float)


def _trailing_possession_duration(
    rng: np.random.Generator,
    params: SimulationParams,
    lead: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    urgency = np.clip((9.0 - lead) / 6.0, 0.25, 1.0)
    late = time <= 12.0
    very_late = time <= 6.0
    mean = np.where(time > 30.0, 9.0 - 2.0 * urgency, 6.5 - 2.5 * urgency)
    mean = np.where(late, 4.0 - 1.5 * urgency, mean)
    mean = np.where(very_late, np.maximum(time - 0.4, 0.4), mean)
    duration = rng.gamma(shape=2.0, scale=np.maximum(mean, 0.5) / 2.0)
    duration = np.minimum(duration, np.minimum(time, params.league.shot_clock))
    return np.maximum(duration, 0.2)


def _three_attempt_probability(lead: np.ndarray, time: np.ndarray) -> np.ndarray:
    p = np.where(lead <= 3.0, 0.62, 0.44)
    p = np.where(lead >= 5.0, 0.52, p)
    p = np.where(time <= 15.0, p + 0.18, p)
    p = np.where(time <= 7.0, p + 0.12, p)
    p = np.where(lead <= 2.0, p - 0.12, p)
    return np.clip(p, 0.12, 0.96)
