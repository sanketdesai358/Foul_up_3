"""Minimal Streamlit UI for the simple foul threshold model."""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from model import Inputs, sweep


SMOOTH_SECONDS = 5
MIN_STABLE_EDGE = 0.002
MIN_WINDOW_SECONDS = 2


st.set_page_config(page_title="Simple Foul Thresholds", layout="wide")
st.title("Simple Foul Thresholds")
st.markdown(
    """
    This app is a simple end-game decision model for intentional fouling. It asks:
    if a team is leading or trailing by 3-6 points in the final minute, when does
    fouling create more win probability than playing straight defense?

    The model matters because late-game strategy is not just about one shot. Fouling
    changes the whole possession sequence: free throws, intentional misses, rebounds,
    inbound pressure, clock burn, and whether the opponent can still get a tying or
    extending shot. The charts compare those paths directly with Monte Carlo simulation.

    The headline windows use a smoothed foul edge so one noisy second does not become
    a fake rule. The faint lines show the raw simulation points; the thicker lines and
    shaded bands show the more stable decision signal.
    """
)

with st.sidebar:
    trailing_ft = st.slider("Trailing team FT%", 0.40, 1.00, 0.78, 0.01)
    trailing_three = st.slider("Trailing team 3P%", 0.20, 0.60, 0.36, 0.01)
    leading_ft = st.slider("Leading team FT%", 0.40, 1.00, 0.78, 0.01)
    ft_miss_oreb = st.slider("FT-miss offensive rebound%", 0.00, 0.40, 0.14, 0.01)


@st.cache_data(show_spinner="Running simulations...")
def cached_sweeps(trailing_ft: float, trailing_three: float, leading_ft: float, ft_miss_oreb: float) -> dict[int, dict]:
    inputs = Inputs(trailing_ft, trailing_three, leading_ft, ft_miss_oreb)
    return {margin: sweep(inputs, margin=margin) for margin in [3, 4, 5, 6]}


def moving_average(values, window: int = SMOOTH_SECONDS):
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(np.asarray(values), (pad_left, pad_right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def positive_windows(times, values) -> list[tuple[int, int]]:
    windows = []
    start = None
    for time, value in zip(times, values):
        if value > MIN_STABLE_EDGE and start is None:
            start = int(time)
        if (value <= MIN_STABLE_EDGE or time == times[-1]) and start is not None:
            end = int(time - 1 if value <= MIN_STABLE_EDGE else time)
            if end - start + 1 >= MIN_WINDOW_SECONDS:
                windows.append((start, end))
            start = None
    return windows


def format_windows(windows: list[tuple[int, int]]) -> str:
    if not windows:
        return "No stable window"
    labels = [f"{start}s" if start == end else f"{start}-{end}s" for start, end in windows]
    return ", ".join(labels[:3]) + ("..." if len(labels) > 3 else "")


def max_window_gain(times, values, windows: list[tuple[int, int]]) -> float | None:
    if not windows:
        return None
    best = None
    for start, end in windows:
        mask = (times >= start) & (times <= end)
        gain = float(values[mask].max())
        best = gain if best is None else max(best, gain)
    return best


def render_margin_tab(margin: int, result: dict) -> None:
    up_smooth = moving_average(result["up_delta"])
    down_smooth = moving_average(result["down_delta"])
    up_windows = positive_windows(result["time"], up_smooth)
    down_windows = positive_windows(result["time"], down_smooth)
    up_gain = max_window_gain(result["time"], up_smooth, up_windows)
    down_gain = max_window_gain(result["time"], down_smooth, down_windows)

    left, right = st.columns(2)
    left.metric(
        f"Up {margin}: stable leader-foul windows",
        format_windows(up_windows),
        "" if up_gain is None else f"Best smoothed gain {up_gain:+.2%}",
    )
    right.metric(
        f"Down {margin}, opponent ball: stable foul windows",
        format_windows(down_windows),
        "" if down_gain is None else f"Best smoothed gain {down_gain:+.2%}",
    )
    st.caption(
        f"Windows use a {SMOOTH_SECONDS}-second moving average and require at least "
        f"{MIN_STABLE_EDGE:.1%} foul edge for {MIN_WINDOW_SECONDS}+ seconds. "
        "The faint lines below are raw one-second simulation points."
    )

    fig = go.Figure()
    fig.add_scatter(
        x=result["time"],
        y=result["up_delta"],
        mode="lines",
        name=f"Up {margin}: raw edge",
        line={"color": "rgba(0, 102, 204, 0.25)"},
    )
    fig.add_scatter(
        x=result["time"],
        y=up_smooth,
        mode="lines",
        name=f"Up {margin}: smoothed edge",
        line={"color": "#0066cc", "width": 3},
    )
    fig.add_scatter(
        x=result["time"],
        y=result["down_delta"],
        mode="lines",
        name=f"Down {margin}: raw edge",
        line={"color": "rgba(102, 178, 255, 0.35)"},
    )
    fig.add_scatter(
        x=result["time"],
        y=down_smooth,
        mode="lines",
        name=f"Down {margin}: smoothed edge",
        line={"color": "#66b2ff", "width": 3},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    for start, end in up_windows:
        fig.add_vrect(x0=start, x1=end, fillcolor="#0066cc", opacity=0.08, line_width=0)
    for start, end in down_windows:
        fig.add_vrect(x0=start, x1=end, fillcolor="#66b2ff", opacity=0.08, line_width=0)
    fig.update_layout(
        title=f"Fouling Edge by Time Remaining: {margin}-Point Margin",
        xaxis_title="Seconds remaining",
        yaxis_title="WP(foul) - WP(defend)",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)

    wp_fig = go.Figure()
    wp_fig.add_scatter(
        x=result["time"],
        y=result["up_foul"],
        mode="lines",
        name=f"Up {margin}: WP if leader fouls",
    )
    wp_fig.add_scatter(
        x=result["time"],
        y=result["up_defend"],
        mode="lines",
        name=f"Up {margin}: WP if leader defends",
    )
    wp_fig.add_scatter(
        x=result["time"],
        y=result["down_foul"],
        mode="lines",
        name=f"Down {margin}: WP if trailer fouls",
        line={"dash": "dash"},
    )
    wp_fig.add_scatter(
        x=result["time"],
        y=result["down_defend"],
        mode="lines",
        name=f"Down {margin}: WP if trailer defends",
        line={"dash": "dash"},
    )
    wp_fig.update_layout(
        title=f"Win Probability by Policy: {margin}-Point Margin",
        xaxis_title="Seconds remaining",
        yaxis_title="Win probability",
        yaxis_tickformat=".0%",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    st.plotly_chart(wp_fig, use_container_width=True)

    bench_times = [45, 30, 20, 10, 5]
    bench = pd.DataFrame(
        {
            "seconds": bench_times,
            f"up_{margin}_leader_foul_edge": [
                float(result["up_delta"][result["time"] == t][0]) for t in bench_times
            ],
            f"down_{margin}_foul_edge": [
                float(result["down_delta"][result["time"] == t][0]) for t in bench_times
            ],
        }
    )
    st.subheader("Benchmark")
    st.dataframe(
        bench.style.format(
            {
                f"up_{margin}_leader_foul_edge": "{:+.2%}",
                f"down_{margin}_foul_edge": "{:+.2%}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


results = cached_sweeps(trailing_ft, trailing_three, leading_ft, ft_miss_oreb)
tabs = st.tabs([f"Down {margin}" for margin in [3, 4, 5, 6]])
for tab, margin in zip(tabs, [3, 4, 5, 6]):
    with tab:
        render_margin_tab(margin, results[margin])
