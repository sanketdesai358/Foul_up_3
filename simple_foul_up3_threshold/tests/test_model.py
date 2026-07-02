from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import Inputs, simulate_up3


def test_one_second_up_three_fouling_beats_defending() -> None:
    inputs = Inputs(trials=80_000, seed=11)

    assert simulate_up3(1, "foul", inputs) > simulate_up3(1, "defend", inputs)


def test_45_seconds_and_above_defending_clearly_beats_fouling() -> None:
    inputs = Inputs(trials=80_000, seed=12)

    for seconds in [45, 50, 55, 60]:
        assert simulate_up3(seconds, "defend", inputs) > simulate_up3(seconds, "foul", inputs) + 0.02
