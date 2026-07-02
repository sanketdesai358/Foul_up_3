"""Streamlit dashboard for intentional-foul end-game strategy."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from charts import break_even_chart, convergence_chart, decision_heatmap, tornado_chart
from engine import compare_policies, convergence_series, scenario_grid
from params import GameState, SimulationParams, StrategyParams, TeamParams, league_rules
from traces import sample_traces


st.set_page_config(page_title="Foul Up 3+ Strategy", layout="wide")


def main() -> None:
    """Render the Streamlit app."""

    st.title("Foul Up 3+ End-Game Strategy")

    params = sidebar_params()
    comparison = compare_policies(params)

    metric_cols = st.columns(4)
    metric_cols[0].metric("WP if foul", f"{comparison.foul.win_probability:.2%}")
    metric_cols[1].metric("WP if defend", f"{comparison.defend.win_probability:.2%}")
    metric_cols[2].metric(
        "Foul edge",
        f"{comparison.delta:+.2%}",
        f"95% CI {comparison.ci_low:+.2%} to {comparison.ci_high:+.2%}",
    )
    metric_cols[3].metric("Recommendation", short_recommendation(comparison.recommendation))
    st.caption(comparison.recommendation)

    tabs = st.tabs(
        [
            "Decision Map",
            "Sensitivity",
            "Break-Even",
            "Traces",
            "Convergence",
            "Assumptions",
        ]
    )

    with tabs[0]:
        grid_trials = st.slider("Trials per grid cell", 2_000, 100_000, min(params.n_trials, 20_000), step=2_000)
        time_step = st.select_slider("Time grid step", options=[1, 3, 5], value=3)
        grid_params = replace(params, n_trials=grid_trials)
        grid = cached_grid(grid_params, time_step)
        st.plotly_chart(decision_heatmap(grid), use_container_width=True)
        st.download_button(
            "Download scenario grid CSV",
            grid.to_csv(index=False).encode("utf-8"),
            "foul_up_three_grid.csv",
            "text/csv",
        )

    with tabs[1]:
        sensitivity = cached_sensitivity(params)
        st.plotly_chart(tornado_chart(sensitivity), use_container_width=True)
        st.dataframe(
            sensitivity[["parameter", "low_delta", "base_delta", "high_delta", "impact"]],
            use_container_width=True,
            hide_index=True,
        )

    with tabs[2]:
        be_trials = st.slider("Break-even trials per point", 1_000, 30_000, min(params.n_trials, 5_000), step=1_000)
        be_params = replace(params, n_trials=be_trials)
        break_even = cached_break_even(be_params)
        st.plotly_chart(break_even_chart(break_even), use_container_width=True)
        st.dataframe(break_even, use_container_width=True, hide_index=True)

    with tabs[3]:
        trace_policy = st.radio("Trace policy", ["foul", "defend"], horizontal=True)
        for trace in sample_traces(params, trace_policy, count=6):
            with st.expander(f"{trace.policy.upper()} trace: {'win' if trace.won else 'loss'}, final lead {trace.final_lead}"):
                st.text("\n".join(trace.lines))

    with tabs[4]:
        checkpoints = [500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, min(params.n_trials, 100_000)]
        checkpoints = sorted(set(x for x in checkpoints if x <= params.n_trials and x > 0))
        conv_rows = pd.DataFrame(
            convergence_series(params, "foul", checkpoints)
            + convergence_series(params, "defend", checkpoints)
        )
        st.plotly_chart(convergence_chart(conv_rows), use_container_width=True)
        st.dataframe(conv_rows, use_container_width=True, hide_index=True)

    with tabs[5]:
        st.markdown(
            """
            This model is possession-level rather than player-level. It captures the foul-war loop,
            free-throw miss rebounds, pressure inbound turnovers, fouls to give, late shot selection,
            offensive rebounds, and overtime resolution. It does not model individual matchups,
            substitutions, reviews, lane violations, or every timeout/inbound branch.
            """
        )
        st.json(flat_params(params))


def sidebar_params() -> SimulationParams:
    """Collect sidebar controls into a SimulationParams object."""

    with st.sidebar:
        st.header("Game State")
        league_name = st.selectbox("League", ["NBA", "NCAA"], index=0)
        rules = league_rules(league_name)
        time_remaining = st.slider("Time remaining", 0, 60, 12)
        lead = st.slider("Leading margin", 3, 8, 3)
        possession = st.radio("Possession", ["trailing", "leading"], horizontal=True)
        leading_ftg = st.slider("Leader fouls to give", 0, 3, 0)
        trailing_ftg = st.slider("Trailer fouls to give", 0, 3, 0)
        leading_timeouts = st.slider("Leader timeouts", 0, 3, 2)
        trailing_timeouts = st.slider("Trailer timeouts", 0, 3, 2)

        st.header("Trailing Team")
        trailing_ft = st.slider("FT%", 0.40, 1.00, 0.78, 0.01)
        trailing_three = st.slider("3P%", 0.20, 0.55, 0.36, 0.01)
        contested_mod = st.slider("Contested 3 modifier", 0.60, 1.10, 0.88, 0.01)
        trailing_two = st.slider("2P%", 0.35, 0.70, 0.53, 0.01)
        turnover = st.slider("Turnover rate", 0.00, 0.25, 0.075, 0.005)
        live_oreb = st.slider("Live-ball OREB%", 0.05, 0.45, 0.24, 0.01)
        ft_oreb = st.slider("FT-miss OREB%", 0.02, 0.35, 0.14, 0.01)
        putback_two = st.slider("Putback 2P%", 0.30, 0.80, 0.58, 0.01)
        putback_three = st.slider("Kick-out 3P%", 0.20, 0.55, 0.33, 0.01)
        kickout_rate = st.slider("Kick-out three rate", 0.00, 0.70, 0.23, 0.01)
        intentional_miss = st.slider("Intentional-miss execution", 0.20, 1.00, 0.72, 0.01)

        st.header("Leading Team")
        leading_ft = st.slider("FT%", 0.40, 1.00, 0.79, 0.01)
        use_closing = st.checkbox("Use closing-lineup FT%", value=False)
        closing_ft = st.slider("Closing-lineup FT%", 0.40, 1.00, 0.84, 0.01)
        dreb = st.slider("Defensive rebound%", 0.55, 0.95, 0.76, 0.01)
        inbound_to = st.slider("Inbound turnover under pressure", 0.00, 0.15, 0.025, 0.005)

        st.header("Strategy")
        foul_lead_min = st.slider("Foul when lead >=", 3, 8, 3)
        foul_time_max = st.slider("Foul when time <=", 0, 60, 45)
        p_foul_shooter = st.slider("Accidental 3-shot foul", 0.000, 0.120, 0.035, 0.005)
        p_and_one = st.slider("Made-3 and-1 foul", 0.000, 0.030, 0.004, 0.001)
        foul_min = st.slider("Foul commit min seconds", 0.5, 8.0, 2.0, 0.5)
        foul_max = st.slider("Foul commit max seconds", foul_min, 10.0, max(4.0, foul_min), 0.5)
        retal_min = st.slider("Retaliatory foul min seconds", 0.2, 6.0, 1.0, 0.2)
        retal_max = st.slider("Retaliatory foul max seconds", retal_min, 8.0, max(3.0, retal_min), 0.2)
        ot_win = st.slider("OT win probability", 0.00, 1.00, 0.50, 0.01)

        st.header("Simulation")
        n_trials = st.slider("Trials", 1_000, 200_000, 100_000, step=1_000)
        seed = st.number_input("Random seed", min_value=0, max_value=2_000_000_000, value=7, step=1)

    teams = TeamParams(
        trailing_ft_pct=trailing_ft,
        trailing_three_pct=trailing_three,
        contested_three_modifier=contested_mod,
        trailing_two_pct=trailing_two,
        trailing_turnover_rate=turnover,
        live_ball_oreb_pct=live_oreb,
        ft_miss_oreb_pct=ft_oreb,
        putback_two_pct=putback_two,
        putback_three_pct=putback_three,
        putback_kickout_three_rate=kickout_rate,
        intentional_miss_execution_pct=intentional_miss,
        leading_ft_pct=leading_ft,
        closing_lineup_ft_pct=closing_ft,
        use_closing_lineup_ft=use_closing,
        defensive_rebound_pct=dreb,
        inbound_turnover_rate=inbound_to,
    )
    strategy = StrategyParams(
        foul_lead_min=foul_lead_min,
        foul_time_max=float(foul_time_max),
        p_foul_shooter=p_foul_shooter,
        p_made_three_and_one=p_and_one,
        clean_foul_probability=max(0.0, 1.0 - p_foul_shooter - p_and_one),
        time_to_foul_min=foul_min,
        time_to_foul_max=foul_max,
        retaliatory_foul_min=retal_min,
        retaliatory_foul_max=retal_max,
        ot_win_prob=ot_win,
    )
    state = GameState(
        time_remaining=float(time_remaining),
        lead=lead,
        possession=possession,
        leading_timeouts=leading_timeouts,
        trailing_timeouts=trailing_timeouts,
        leading_fouls_to_give=leading_ftg,
        trailing_fouls_to_give=trailing_ftg,
        leading_bonus=True,
        trailing_bonus=True,
    )
    return SimulationParams(league=rules, teams=teams, strategy=strategy, state=state, n_trials=n_trials, seed=int(seed))


@st.cache_data(show_spinner="Simulating grid...")
def cached_grid(params: SimulationParams, time_step: int) -> pd.DataFrame:
    """Cached scenario grid for the heatmap."""

    times = list(range(0, 46, time_step))
    if times[-1] != 45:
        times.append(45)
    rows = scenario_grid(params, times=times, leads=range(3, 9))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Running sensitivity analysis...")
def cached_sensitivity(params: SimulationParams) -> pd.DataFrame:
    """Vary important inputs around the base case and return tornado data."""

    specs = [
        ("Trailing FT%", "teams", "trailing_ft_pct", -0.06, 0.06, 0.35, 1.0),
        ("Trailing 3P%", "teams", "trailing_three_pct", -0.05, 0.05, 0.15, 0.65),
        ("Contested 3 mod", "teams", "contested_three_modifier", -0.08, 0.08, 0.50, 1.20),
        ("Leader FT%", "teams", "leading_ft_pct", -0.06, 0.06, 0.35, 1.0),
        ("FT-miss OREB%", "teams", "ft_miss_oreb_pct", -0.06, 0.06, 0.00, 0.45),
        ("Inbound TO%", "teams", "inbound_turnover_rate", -0.015, 0.015, 0.00, 0.20),
        ("Accidental shooter foul", "strategy", "p_foul_shooter", -0.02, 0.02, 0.00, 0.20),
        ("Time to foul max", "strategy", "time_to_foul_max", -1.0, 1.0, 0.5, 10.0),
    ]
    base_delta = compare_policies(params).delta
    rows = []
    for label, group, field, low_move, high_move, floor, ceiling in specs:
        obj = getattr(params, group)
        base_value = getattr(obj, field)
        low_value = min(max(base_value + low_move, floor), ceiling)
        high_value = min(max(base_value + high_move, floor), ceiling)
        low_params = replace(params, **{group: replace(obj, **{field: low_value})})
        high_params = replace(params, **{group: replace(obj, **{field: high_value})})
        low_delta = compare_policies(low_params).delta
        high_delta = compare_policies(high_params).delta
        rows.append(
            {
                "parameter": label,
                "base_delta": base_delta,
                "low_delta": low_delta,
                "high_delta": high_delta,
                "impact": max(abs(low_delta - base_delta), abs(high_delta - base_delta)),
            }
        )
    return pd.DataFrame(rows).sort_values("impact", ascending=False)


@st.cache_data(show_spinner="Solving break-even curves...")
def cached_break_even(params: SimulationParams) -> pd.DataFrame:
    """Find the approximate trailing 3P% where fouling begins to beat defending."""

    rows = []
    for lead in range(3, 9):
        for time_value in range(0, 46, 5):
            base = replace(params, state=replace(params.state, lead=lead, time_remaining=float(time_value)))
            low, high = 0.20, 0.55
            low_delta = compare_policies(replace(base, teams=replace(base.teams, trailing_three_pct=low))).delta
            high_delta = compare_policies(replace(base, teams=replace(base.teams, trailing_three_pct=high))).delta
            if low_delta >= 0:
                threshold = low
            elif high_delta < 0:
                threshold = None
            else:
                for _ in range(7):
                    mid = (low + high) / 2.0
                    mid_delta = compare_policies(replace(base, teams=replace(base.teams, trailing_three_pct=mid))).delta
                    if mid_delta >= 0:
                        high = mid
                    else:
                        low = mid
                threshold = high
            rows.append(
                {
                    "lead": str(lead),
                    "time_remaining": time_value,
                    "break_even_three_pct": threshold,
                }
            )
    return pd.DataFrame(rows)


def flat_params(params: SimulationParams) -> dict[str, object]:
    """Return nested settings as serializable dictionaries."""

    return {
        "league": params.league.__dict__,
        "teams": params.teams.__dict__,
        "strategy": params.strategy.__dict__,
        "state": params.state.__dict__,
        "n_trials": params.n_trials,
        "seed": params.seed,
    }


def short_recommendation(text: str) -> str:
    """Compact recommendation for a metric cell."""

    if text.startswith("Foul"):
        return "Foul"
    if text.startswith("Defend"):
        return "Defend"
    if text.startswith("Lean foul"):
        return "Lean foul"
    if text.startswith("Lean defend"):
        return "Lean defend"
    return "Neutral"


if __name__ == "__main__":
    main()
