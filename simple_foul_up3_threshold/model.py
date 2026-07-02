"""Simple Monte Carlo threshold model for late-game intentional fouling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Hardcoded assumptions, intentionally not exposed in the UI.
N_TRIALS = 50_000
P_SHOOTER_FOUL = 0.05  # Intentional foul accidentally becomes 3 free throws.
FOUL_SECONDS = 3.0  # Clock consumed to commit any intentional foul.
SHOT_SECONDS = 10.0  # Defended possession length, capped by remaining time.
PUTBACK_CONVERSION = 1.00  # In this toy model, FT-miss OREBs become scramble scores.
PUTBACK_TIE_RATE = 0.30  # Converted putbacks are 3-point tie plays this often.
TRAILING_TWO_PCT = 0.50  # Down 1 or 2, trailing team attacks for a two.
LEADING_TWO_PCT = 0.15  # Clock-kill offense ends in a low-value late-clock two.
INBOUND_TURNOVER = 0.05  # Pressure turnover after the leader receives the ball.
TRAILING_FOUL_BACK_UNDER = 30.0  # Trailer fouls the leader below this time.
OT_WIN_PROB = 0.50  # Tied at 0:00 goes to overtime coin flip.
MAX_EVENTS = 30


@dataclass(frozen=True)
class Inputs:
    """The only user-facing inputs for the model."""

    trailing_ft: float = 0.78
    trailing_three: float = 0.36
    leading_ft: float = 0.78
    ft_miss_oreb: float = 0.14
    trials: int = N_TRIALS
    seed: int = 7


def sweep(inputs: Inputs, margin: int = 3) -> dict[str, np.ndarray]:
    """Sweep 1-60 seconds and return foul-minus-defend curves.

    `up_delta` is from the leading team's perspective when up by `margin`
    with the trailing team starting with the ball. `down_delta` is from the
    trailing team's perspective when down by `margin` with the leader starting
    with the ball.
    """

    times = np.arange(1, 61)
    up_foul, up_defend, down_foul, down_defend = [], [], [], []
    for t in times:
        up_foul.append(simulate_up_margin(t, "foul", inputs, margin))
        up_defend.append(simulate_up_margin(t, "defend", inputs, margin))
        down_foul.append(simulate_down_margin(t, "foul", inputs, margin))
        down_defend.append(simulate_down_margin(t, "defend", inputs, margin))
    up_foul = np.array(up_foul)
    up_defend = np.array(up_defend)
    down_foul = np.array(down_foul)
    down_defend = np.array(down_defend)
    return {
        "time": times,
        "up_foul": up_foul,
        "up_defend": up_defend,
        "up_delta": up_foul - up_defend,
        "down_foul": down_foul,
        "down_defend": down_defend,
        "down_delta": down_foul - down_defend,
    }


def crossover(time: np.ndarray, delta: np.ndarray) -> tuple[int | None, float | None]:
    """Return the largest time where fouling beats the alternative."""

    good = np.flatnonzero(delta > 0)
    if good.size == 0:
        return None, None
    i = int(good[-1])
    return int(time[i]), float(delta[i])


def simulate_up3(seconds: int, policy: str, inputs: Inputs) -> float:
    """Backward-compatible helper for the original up-3 test case."""

    return simulate_up_margin(seconds, policy, inputs, margin=3)


def simulate_down3(seconds: int, policy: str, inputs: Inputs) -> float:
    """Backward-compatible helper for the original down-3 test case."""

    return simulate_down_margin(seconds, policy, inputs, margin=3)


def simulate_up_margin(seconds: int, policy: str, inputs: Inputs, margin: int) -> float:
    """Win probability for the team leading by 3 with the trailer starting on ball.

    FOUL means the leader fouls immediately whenever the trailing team has the
    ball. DEFEND means the leader never intentionally fouls.
    """

    rng = np.random.default_rng(inputs.seed + margin * 10_000 + seconds * 101 + (0 if policy == "foul" else 1))
    wins = _simulate(seconds, policy, "trailing", inputs, rng, margin=margin)
    return float(wins.mean())


def simulate_down_margin(seconds: int, policy: str, inputs: Inputs, margin: int) -> float:
    """Win probability for the team trailing by `margin` with the leader on ball.

    FOUL means the trailing team fouls the leader immediately. DEFEND means it
    plays for a stop until the under-30-seconds foul-back rule applies.
    """

    rng = np.random.default_rng(inputs.seed + margin * 10_000 + seconds * 103 + (2 if policy == "foul" else 3))
    leader_wins = _simulate(
        seconds,
        "defend",
        "leading",
        inputs,
        rng,
        margin=margin,
        trailer_opening_policy=policy,
    )
    return float(1.0 - leader_wins.mean())


def _simulate(
    seconds: int,
    leader_policy: str,
    start_possession: str,
    inputs: Inputs,
    rng: np.random.Generator,
    margin: int,
    trailer_opening_policy: str | None = None,
) -> np.ndarray:
    lead = np.full(inputs.trials, float(margin))
    time = np.full(inputs.trials, float(seconds))
    possession = np.full(inputs.trials, 0 if start_possession == "trailing" else 1, dtype=np.int8)

    for _ in range(MAX_EVENTS):
        active = time > 0
        if not active.any():
            break
        trailing = active & (possession == 0)
        if trailing.any():
            if leader_policy == "foul":
                _leader_fouls_trailer(rng, inputs, trailing, lead, time, possession)
            else:
                _trailer_runs_offense(rng, inputs, trailing, lead, time, possession)
        leading = (time > 0) & (possession == 1)
        if leading.any():
            _leader_has_ball(rng, inputs, leading, lead, time, possession, trailer_opening_policy)
            trailer_opening_policy = None

    return _leader_wins(rng, lead)


def _leader_fouls_trailer(rng, inputs: Inputs, mask, lead, time, possession) -> None:
    idx = np.flatnonzero(mask)
    time[idx] = np.maximum(0.0, time[idx] - FOUL_SECONDS)
    live = idx[time[idx] > 0]
    if live.size == 0:
        return
    shooter = rng.random(live.size) < P_SHOOTER_FOUL
    if shooter.any():
        sidx = live[shooter]
        lead[sidx] -= (rng.random((sidx.size, 3)) < inputs.trailing_ft).sum(axis=1)
    clean = live[~shooter]
    if clean.size:
        first = rng.random(clean.size) < inputs.trailing_ft
        lead[clean] -= first
        still_down_2 = lead[clean] >= 2
        second_make = (~still_down_2) & (rng.random(clean.size) < inputs.trailing_ft)
        lead[clean] -= second_make
        miss_idx = clean[still_down_2]
        if miss_idx.size:
            oreb = rng.random(miss_idx.size) < inputs.ft_miss_oreb
            put = miss_idx[oreb & (rng.random(miss_idx.size) < PUTBACK_CONVERSION)]
            if put.size:
                three = rng.random(put.size) < PUTBACK_TIE_RATE
                lead[put] -= np.where(three, 3.0, 2.0)
    possession[live] = 1


def _trailer_runs_offense(rng, inputs: Inputs, mask, lead, time, possession) -> None:
    idx = np.flatnonzero(mask)
    time[idx] = np.maximum(0.0, time[idx] - np.minimum(time[idx], SHOT_SECONDS))
    live = idx
    down = lead[live]
    take_three = down >= 3
    made_three = take_three & (rng.random(live.size) < inputs.trailing_three)
    made_two = (~take_three) & (rng.random(live.size) < TRAILING_TWO_PCT)
    lead[live] -= 3.0 * made_three + 2.0 * made_two
    possession[live] = 1


def _leader_has_ball(rng, inputs: Inputs, mask, lead, time, possession, opening_policy: str | None) -> None:
    idx = np.flatnonzero(mask)
    turnover = rng.random(idx.size) < INBOUND_TURNOVER
    if turnover.any():
        tidx = idx[turnover]
        time[tidx] = np.maximum(0.0, time[tidx] - 1.0)
        possession[tidx] = 0
    secure = idx[~turnover]
    if secure.size == 0:
        return
    foul_now = (opening_policy == "foul") | (time[secure] < TRAILING_FOUL_BACK_UNDER)
    if np.isscalar(foul_now):
        foul_now = np.full(secure.size, bool(foul_now))
    fidx = secure[foul_now]
    if fidx.size:
        time[fidx] = np.maximum(0.0, time[fidx] - FOUL_SECONDS)
        live = fidx[time[fidx] > 0]
        if live.size:
            lead[live] += (rng.random((live.size, 2)) < inputs.leading_ft).sum(axis=1)
            possession[live] = 0
    didx = secure[~foul_now]
    if didx.size:
        time[didx] = np.maximum(0.0, time[didx] - np.minimum(time[didx], SHOT_SECONDS))
        made = rng.random(didx.size) < LEADING_TWO_PCT
        lead[didx] += 2.0 * made
        possession[didx] = 0


def _leader_wins(rng: np.random.Generator, lead: np.ndarray) -> np.ndarray:
    wins = lead > 0
    tied = lead == 0
    if tied.any():
        wins = wins.copy()
        wins[tied] = rng.random(tied.sum()) < OT_WIN_PROB
    return wins
