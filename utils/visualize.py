import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def pnl_plot(
    time, y_value, action,
    *,
    top_title="Reward Trajectory",
    bottom_title="Action",
    y_label_top="Cummulative Reward",
    y_label_bottom="action",
    row_heights=(0.74, 0.26),
    vertical_spacing=0.02,
    height=620,
    vx=0.8,
    show_legend=True,
    show_rangeslider=False,
):
    # --- prepare data ---
    df = pd.DataFrame({
        "time": pd.to_datetime(time),
        "y": pd.to_numeric(y_value, errors="coerce"),
        "action": action
    }).dropna(subset=["time", "y"]).sort_values("time")

    # --- figure ---
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=list(row_heights),
        vertical_spacing=vertical_spacing,
        subplot_titles=(top_title, bottom_title)
    )

    # top: main series only
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["y"], mode="lines", name="Cummulative Reward"),
        row=1, col=1
    )

    # bottom: action as step (no background/shading)
    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["action"],
            mode="lines", name="action",
            line=dict(width=2),
            line_shape="hv"     # step
        ),
        row=2, col=1
    )

    if vx:
        fig.add_vline(
                x=df["time"].iloc[int(df.shape[0]*vx)], line_color="red", line_dash="dash", line_width=2,
                row=1, col=1
            )
        fig.add_vline(
                x=df["time"].iloc[int(df.shape[0]*vx)], line_color="red", line_dash="dash", line_width=2,
                row=2, col=1
            )

    # axes/layout
    uniq = np.unique(df["action"])
    if len(uniq) <= 8:
        fig.update_yaxes(tickmode="array", tickvals=uniq, row=2, col=1)

    fig.update_yaxes(title_text=y_label_top, row=1, col=1)
    fig.update_yaxes(title_text=y_label_bottom, row=2, col=1)

    fig.update_layout(
        height=height,
        hovermode="x unified",
        showlegend=show_legend,
        margin=dict(l=60, r=30, t=50, b=30),
    )
    fig.update_xaxes(matches="x", showgrid=True)
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=show_rangeslider)))

    return fig
