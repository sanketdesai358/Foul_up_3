# Foul Up 3+ End-Game Strategy Model

This project simulates whether a basketball team leading by 3 to 8 points late in a game should intentionally foul or play straight defense. It generalizes the classic final-possession "foul up 3" question to end-game windows with multiple possessions and a retaliatory fouling war.

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Run tests:

```powershell
pytest
```

## Files

- `engine.py`: vectorized NumPy Monte Carlo simulator and policy comparison functions.
- `params.py`: dataclasses for league rules, team assumptions, strategy settings, and initial game state.
- `app.py`: Streamlit dashboard.
- `charts.py`: Plotly chart builders.
- `traces.py`: readable scalar play-by-play traces for auditability.
- `tests/`: pytest coverage for degenerate cases and a regression sanity check.

## Model Assumptions

The simulator tracks the leading team's margin, clock, possession, fouls to give, and bonus-style two-shot free throws. Ties at the buzzer are resolved by a configurable overtime win probability, defaulting to 50%.

When the leading team intentionally fouls:

- The foul burns configurable clock time before free throws.
- Most fouls are clean two-shot fouls.
- A configurable share become accidental three-shot fouls.
- A small configurable share become made-three-and-one events.
- If the trailing team is still down by at least 2 after the first free throw, it may intentionally miss the second.
- Intentional misses have a legal-execution probability, a distinct FT-miss offensive rebound rate, and putback or kick-out outcomes.
- After the foul sequence, the leading team inbounds against pressure, may turn it over, and is usually fouled back.

When the leading team defends:

- Trailing possession length depends on clock and margin.
- Shot mix shifts toward threes as the clock shrinks and the margin makes a three more valuable.
- Three-point accuracy is reduced by a contested/end-game modifier.
- Misses can produce live-ball offensive rebounds and immediate putback or kick-out chances.
- Fouls to give burn clock before the shot when available.

## Known Simplifications

- This is a possession-level model, not a player-tracking or lineup-level model.
- Timeouts and frontcourt advancement are included as state inputs but are not yet modeled with separate decision branches.
- The NBA final-two-minute clock stoppage is approximated through possession transitions rather than a full play-clock rules engine.
- Lane violations, reviews, substitutions, trapped inbound geometry, exact foul count progression, and individual player free-throw targeting are not modeled.
- Heatmap and break-even grids can use fewer trials than the headline scenario to keep the dashboard responsive.

## Validation Expectations

With league-average inputs:

- At 0 seconds, a positive lead always holds.
- With perfect leading-team free-throw shooting, the retaliatory foul sequence becomes nearly deterministic after pressure events are removed.
- Up 3 with only a few seconds left, fouling should usually show a small positive win-probability edge.
- Around 45 seconds up 3, defending should generally be near-neutral or favored unless assumptions strongly reward fouling.
