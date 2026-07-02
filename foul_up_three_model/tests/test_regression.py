from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import compare_policies
from params import GameState, SimulationParams


def test_foul_advantage_grows_as_clock_shrinks_for_up_three_average_case() -> None:
    long_clock = SimulationParams(
        state=GameState(time_remaining=45, lead=3, possession="trailing"),
        n_trials=60_000,
        seed=21,
    )
    short_clock = SimulationParams(
        state=GameState(time_remaining=5, lead=3, possession="trailing"),
        n_trials=60_000,
        seed=21,
    )

    long_delta = compare_policies(long_clock).delta
    short_delta = compare_policies(short_clock).delta

    assert short_delta > long_delta
