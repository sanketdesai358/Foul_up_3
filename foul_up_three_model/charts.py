"""Plotly chart builders for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def decision_heatmap(grid: pd.DataFrame) -> go.Figure:
    """Return a time-by-lead heatmap of foul-minus-defend win probability."""

    pivot = grid.pivot(index="lead", columns="time_remaining", values="delta").sort_index()
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "WP delta"},
            hovertemplate="Time: %{x}s<br>Lead: %{y}<br>Delta: %{z:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Contour(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            contours={"start": 0, "end": 0, "size": 1, "coloring": "none"},
            line={"color": "black", "width": 2},
            showscale=False,
            hoverinfo="skip",
            name="Break-even frontier",
        )
    )
    fig.update_layout(
        title="Foul Frontier",
        xaxis_title="Seconds remaining",
        yaxis_title="Leading team margin",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return fig


def tornado_chart(rows: pd.DataFrame) -> go.Figure:
    """Return a tornado chart showing input sensitivity around the base case."""

    plot_df = rows.copy().sort_values("impact", ascending=True)
    fig = go.Figure()
    fig.add_bar(
        y=plot_df["parameter"],
        x=plot_df["low_delta"] - plot_df["base_delta"],
        orientation="h",
        name="Low assumption",
        marker_color="#40798c",
        hovertemplate="%{y}<br>Move: %{x:.2%}<extra></extra>",
    )
    fig.add_bar(
        y=plot_df["parameter"],
        x=plot_df["high_delta"] - plot_df["base_delta"],
        orientation="h",
        name="High assumption",
        marker_color="#c8553d",
        hovertemplate="%{y}<br>Move: %{x:.2%}<extra></extra>",
    )
    fig.update_layout(
        title="Decision Sensitivity",
        xaxis_title="Change in foul edge",
        yaxis_title="",
        barmode="overlay",
        margin={"l": 130, "r": 20, "t": 50, "b": 40},
    )
    return fig


def break_even_chart(rows: pd.DataFrame) -> go.Figure:
    """Return a line chart of break-even trailing-team 3P% by time remaining."""

    fig = px.line(
        rows,
        x="time_remaining",
        y="break_even_three_pct",
        color="lead",
        markers=True,
        labels={
            "time_remaining": "Seconds remaining",
            "break_even_three_pct": "Break-even trailing 3P%",
            "lead": "Lead",
        },
        title="Break-Even Trailing 3P%",
    )
    fig.update_layout(margin={"l": 40, "r": 20, "t": 50, "b": 40})
    return fig


def convergence_chart(rows: pd.DataFrame) -> go.Figure:
    """Return a convergence diagnostic chart for WP versus trial count."""

    fig = px.line(
        rows,
        x="trials",
        y="win_probability",
        color="policy",
        markers=True,
        labels={"trials": "Trials", "win_probability": "Win probability"},
        title="Convergence Diagnostic",
    )
    fig.update_xaxes(type="log")
    fig.update_layout(margin={"l": 40, "r": 20, "t": 50, "b": 40})
    return fig
