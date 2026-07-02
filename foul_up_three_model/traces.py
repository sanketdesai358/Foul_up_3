"""Readable single-game traces for auditing model logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from params import PolicyName, SimulationParams


@dataclass(frozen=True)
class TraceResult:
    """A narrated simulated end-game path."""

    policy: PolicyName
    won: bool
    final_lead: int
    lines: list[str]


def sample_traces(params: SimulationParams, policy: PolicyName, count: int = 6) -> list[TraceResult]:
    """Generate human-readable traces using the same assumptions as the engine.

    This intentionally uses a scalar simulation so each event can be narrated.
    It is for auditability, not dashboard-speed estimation.
    """

    rng = np.random.default_rng(params.seed + (101 if policy == "foul" else 202))
    return [_single_trace(rng, params, policy, i + 1) for i in range(count)]


def _single_trace(
    rng: np.random.Generator,
    params: SimulationParams,
    policy: PolicyName,
    trace_num: int,
) -> TraceResult:
    lead = int(params.state.lead)
    time = float(params.state.time_remaining)
    possession = params.state.possession
    leading_ftg = int(params.state.leading_fouls_to_give)
    trailing_ftg = int(params.state.trailing_fouls_to_give)
    lines = [f"Trace {trace_num}: start up {lead}, {time:.1f}s, {possession} ball."]

    for _ in range(params.strategy.max_transitions):
        if time <= 0:
            break
        if possession == "trailing":
            foul_now = (
                policy == "foul"
                and lead >= params.strategy.foul_lead_min
                and time <= params.strategy.foul_time_max
            )
            if foul_now:
                burn = min(time, rng.uniform(params.strategy.time_to_foul_min, params.strategy.time_to_foul_max))
                time -= burn
                lines.append(f"{time:.1f}s: leader fouls before the shot after {burn:.1f}s.")
                if time <= 0:
                    break
                r = rng.random()
                if r < params.strategy.p_made_three_and_one:
                    lead -= 3
                    made_ft = rng.random() < params.teams.trailing_ft_pct
                    lead -= int(made_ft)
                    lines.append(f"{time:.1f}s: disaster foul, made three plus {'made' if made_ft else 'missed'} FT. Lead {lead}.")
                elif r < params.strategy.p_made_three_and_one + params.strategy.p_foul_shooter:
                    makes = int(np.sum(rng.random(3) < params.teams.trailing_ft_pct))
                    lead -= makes
                    lines.append(f"{time:.1f}s: accidental three-shot foul, trailing makes {makes}/3. Lead {lead}.")
                else:
                    first = rng.random() < params.teams.trailing_ft_pct
                    lead -= int(first)
                    miss_intentionally = lead >= 2 and rng.random() < params.teams.intentional_miss_execution_pct
                    if miss_intentionally:
                        oreb = rng.random() < params.teams.ft_miss_oreb_pct
                        lines.append(f"{time:.1f}s: makes first, intentionally misses second, OREB={oreb}. Lead {lead}.")
                        if oreb:
                            three = rng.random() < params.teams.putback_kickout_three_rate
                            make = rng.random() < (params.teams.putback_three_pct if three else params.teams.putback_two_pct)
                            lead -= (3 if three else 2) * int(make)
                            lines.append(f"{time:.1f}s: putback {'three' if three else 'two'} {'falls' if make else 'misses'}. Lead {lead}.")
                    else:
                        second = rng.random() < params.teams.trailing_ft_pct
                        lead -= int(second)
                        lines.append(f"{time:.1f}s: clean foul, trailing makes {int(first) + int(second)}/2. Lead {lead}.")
                possession = "leading"
            else:
                if leading_ftg > 0 and time > 6:
                    burn = min(time, rng.uniform(params.strategy.foul_to_give_clock_burn_min, params.strategy.foul_to_give_clock_burn_max))
                    time -= burn
                    leading_ftg -= 1
                    lines.append(f"{time:.1f}s: leader burns a foul to give.")
                duration = min(time, rng.uniform(2.0, 9.0 if time > 15 else 5.0))
                time -= duration
                turnover = rng.random() < params.teams.trailing_turnover_rate
                if turnover:
                    possession = "leading"
                    lines.append(f"{time:.1f}s: trailing team turns it over. Lead {lead}.")
                else:
                    p_three = 0.80 if lead <= 3 and time <= 15 else 0.55
                    three = rng.random() < p_three
                    make_pct = params.teams.trailing_three_pct * params.teams.contested_three_modifier if three else params.teams.trailing_two_pct
                    made = rng.random() < make_pct
                    lead -= (3 if three else 2) * int(made)
                    possession = "leading"
                    lines.append(f"{time:.1f}s: trailing {'3PA' if three else '2PA'} {'made' if made else 'missed'}. Lead {lead}.")
        else:
            if rng.random() < params.teams.inbound_turnover_rate:
                time = max(0.0, time - rng.uniform(0.5, 1.5))
                possession = "trailing"
                lines.append(f"{time:.1f}s: pressure forces an inbound turnover. Lead {lead}.")
                continue
            burn = min(time, rng.uniform(params.strategy.retaliatory_foul_min, params.strategy.retaliatory_foul_max))
            time -= burn
            if trailing_ftg > 0:
                trailing_ftg -= 1
                lines.append(f"{time:.1f}s: trailing team uses a foul to give.")
                possession = "leading"
            else:
                makes = int(np.sum(rng.random(params.league.bonus_ft_count) < params.teams.effective_leading_ft_pct))
                lead += makes
                possession = "trailing"
                lines.append(f"{time:.1f}s: leader fouled, makes {makes}/{params.league.bonus_ft_count}. Lead {lead}.")

    if lead == 0:
        won = bool(rng.random() < params.strategy.ot_win_prob)
        lines.append(f"0.0s: tied game, overtime coin uses {params.strategy.ot_win_prob:.0%}; leader {'wins' if won else 'loses'}.")
    else:
        won = lead > 0
        lines.append(f"0.0s: final margin from leader perspective {lead}; leader {'wins' if won else 'loses'}.")
    return TraceResult(policy=policy, won=won, final_lead=lead, lines=lines)
