import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def pnl_plot(
    time, 
    y_value, 
    action,
    top_title="PnL",
    bottom_title="Action",
    y_label_top="Equity",
    y_label_bottom="Action",
    row_heights=(0.74, 0.26),
    vertical_spacing=0.02,
    height=620,
    vx=0.8,
    show_legend=True,
    show_rangeslider=False,
    save_path=None
):
    """
    Plot PnL/equity and actions in a 2-row subplot.
    
    Args:
        time: Time values (will be converted to datetime)
        y_value: Equity/PnL values
        action: Action values
        top_title: Title for top subplot
        bottom_title: Title for bottom subplot
        y_label_top: Y-axis label for top plot
        y_label_bottom: Y-axis label for bottom plot
        row_heights: Height ratios for subplots (top, bottom)
        vertical_spacing: Space between subplots
        height: Total figure height in pixels
        vx: Position for vertical line (as fraction of data length), None to disable
        show_legend: Whether to show legend
        show_rangeslider: Whether to show range slider
        save_path: Optional path to save figure (HTML or image)
    
    Returns:
        plotly figure object
    """
    # Prepare data
    df = pd.DataFrame({
        "time": pd.to_datetime(time),
        "y": pd.to_numeric(y_value, errors="coerce"),
        "action": action
    }).dropna(subset=["time", "y"]).sort_values("time")
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        row_heights=list(row_heights),
        vertical_spacing=vertical_spacing,
        subplot_titles=(top_title, bottom_title)
    )
    
    # Top subplot: Equity/PnL line
    fig.add_trace(
        go.Scatter(
            x=df["time"], 
            y=df["y"], 
            mode="lines", 
            name="Equity",
            line=dict(width=2, color='blue')
        ),
        row=1, col=1
    )
    
    # Bottom subplot: Action as step plot
    fig.add_trace(
        go.Scatter(
            x=df["time"], 
            y=df["action"],
            mode="lines", 
            name="Action",
            line=dict(width=2, color='green'),
            line_shape="hv"  # Step plot
        ),
        row=2, col=1
    )
    
    # Add vertical line if specified
    if vx is not None:
        vline_x = df["time"].iloc[int(df.shape[0] * vx)]
        fig.add_vline(
            x=vline_x, 
            line_color="red", 
            line_dash="dash", 
            line_width=2,
            row=1, col=1
        )
        fig.add_vline(
            x=vline_x, 
            line_color="red", 
            line_dash="dash", 
            line_width=2,
            row=2, col=1
        )
    
    # Configure y-axis for action plot (show discrete values if few unique actions)
    uniq_actions = np.unique(df["action"])
    if len(uniq_actions) <= 8:
        fig.update_yaxes(
            tickmode="array", 
            tickvals=uniq_actions, 
            row=2, col=1
        )
    
    # Update axis labels
    fig.update_yaxes(title_text=y_label_top, row=1, col=1)
    fig.update_yaxes(title_text=y_label_bottom, row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=height,
        hovermode="x unified",
        showlegend=show_legend,
        margin=dict(l=60, r=30, t=50, b=30),
        template='plotly_white'
    )
    
    # Configure x-axis
    fig.update_xaxes(matches="x", showgrid=True)
    fig.update_xaxes(rangeslider=dict(visible=show_rangeslider), row=2, col=1)
    
    # Save if path provided
    if save_path:
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)
    
    return fig


def losses_compare_plot(
    x: list,
    y_dict: dict,
    title: str = "Training Losses Comparison",
    xlabel: str = "Iterations",
    ylabel: str = "Loss Value",
    save_path: str = None
):
    """
    Plot multiple losses on the same axis for comparison using Plotly.
    
    Args:
        x: List of x-axis values (e.g., iterations)
        y_dict: Dictionary with keys as loss names and values as lists of loss values
                Example: {"train/loss": [...], "train/value_loss": [...]}
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional path to save the figure (HTML or image)
    
    Returns:
        plotly figure object
    """
    fig = go.Figure()
    
    # Add trace for each loss
    for loss_name, loss_values in y_dict.items():
        fig.add_trace(go.Scatter(
            x=x,
            y=loss_values,
            mode='lines',
            name=loss_name,
            line=dict(width=2),
            opacity=0.8
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, weight='bold')),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        hovermode='x unified',
        template='plotly_white',
        width=1200,
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Save if path provided
    if save_path:
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)
    
    return fig


def loss_reward_plot(
    x: list,
    y1: list,
    y2: list,
    y1_label: str = "Loss",
    y2_label: str = "Cumulative Reward",
    title: str = "Loss vs Reward During Training",
    xlabel: str = "Iterations",
    save_path: str = None
):
    """
    Plot loss and reward on dual y-axes using Plotly.
    
    Args:
        x: List of x-axis values (e.g., iterations)
        y1: List of loss values (left y-axis)
        y2: List of reward values (right y-axis)
        y1_label: Label for left y-axis (loss)
        y2_label: Label for right y-axis (reward)
        title: Plot title
        xlabel: X-axis label
        save_path: Optional path to save the figure (HTML or image)
    
    Returns:
        plotly figure object
    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add loss trace (primary y-axis)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y1,
            name=y1_label,
            line=dict(color='red', width=2),
            opacity=0.8
        ),
        secondary_y=False
    )
    
    # Add reward trace (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y2,
            name=y2_label,
            line=dict(color='blue', width=2),
            opacity=0.8
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, weight='bold')),
        hovermode='x unified',
        template='plotly_white',
        width=1200,
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Update axes labels
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=y1_label, secondary_y=False, title_font=dict(color='red'))
    fig.update_yaxes(title_text=y2_label, secondary_y=True, title_font=dict(color='blue'))
    
    # Save if path provided
    if save_path:
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)
    
    return fig

def plot_return(returns, title='Return Distribution Box Plot', split=None):
    """
    Create a box plot for return distribution using Plotly
    
    Parameters:
    -----------
    returns : pandas.Series, numpy.array, or list
        Array of return values
    title : str, optional
        Title of the plot (default: 'Return Distribution Box Plot')
    split : float, optional
        If provided, splits the data into two parts based on the ratio.
        For example, split=0.8 will create an 80/20 split (default: None)
    
    Returns:
    --------
    plotly.graph_objects.Figure : The Plotly figure object
    
    Example:
    --------
    >>> returns = [0.01, -0.02, 0.03, 0.015, -0.01, 0.02, -0.015, 0.025]
    >>> fig = plot_return(returns)
    >>> 
    >>> # With 80/20 split
    >>> fig = plot_return(returns, title='Split Returns', split=0.8)
    """
    
    # Convert to numpy array if needed
    if isinstance(returns, pd.Series):
        returns = returns.values
    elif isinstance(returns, list):
        returns = np.array(returns)
    
    # Create figure
    fig = go.Figure()
    
    if split is not None and 0 < split < 1:
        # Calculate split index
        split_idx = int(len(returns) * split)
        
        # Split the data
        returns_part1 = returns[:split_idx]
        returns_part2 = returns[split_idx:]
        
        # Add first box plot trace
        fig.add_trace(go.Box(
            y=returns_part1,
            name=f'First {split*100:.0f}%',
            marker=dict(
                color='rgba(59, 130, 246, 0.7)',
                line=dict(color='rgba(37, 99, 235, 1)', width=2)
            ),
            boxmean='sd',
            boxpoints='outliers'
        ))
        
        # Add second box plot trace
        fig.add_trace(go.Box(
            y=returns_part2,
            name=f'Last {(1-split)*100:.0f}%',
            marker=dict(
                color='rgba(239, 68, 68, 0.7)',
                line=dict(color='rgba(220, 38, 38, 1)', width=2)
            ),
            boxmean='sd',
            boxpoints='outliers'
        ))
        
        # Update layout for split view
        fig.update_layout(
            title={
                'text': title,
                'font': {'size': 20, 'family': 'Arial, sans-serif'},
                'x': 0.5,
                'xanchor': 'center'
            },
            yaxis_title='Return',
            xaxis_title='Data Split',
            yaxis=dict(
                tickformat='.4%',
                gridcolor='#e5e7eb'
            ),
            plot_bgcolor='#f9fafb',
            paper_bgcolor='white',
            height=600,
            width=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=100, b=60, l=80, r=40)
        )
    else:
        # Original single box plot
        fig.add_trace(go.Box(
            y=returns,
            name='Returns',
            marker=dict(
                color='rgba(59, 130, 246, 0.7)',
                line=dict(color='rgba(37, 99, 235, 1)', width=2)
            ),
            boxmean='sd',
            boxpoints='outliers'
        ))
        
        # Update layout for single view
        fig.update_layout(
            title={
                'text': title,
                'font': {'size': 20, 'family': 'Arial, sans-serif'},
                'x': 0.5,
                'xanchor': 'center'
            },
            yaxis_title='Return',
            yaxis=dict(
                tickformat='.4%',
                gridcolor='#e5e7eb'
            ),
            plot_bgcolor='#f9fafb',
            paper_bgcolor='white',
            height=600,
            width=800,
            showlegend=False,
            margin=dict(t=80, b=60, l=80, r=40)
        )

    # Display the plot
    fig.show()
    
    return fig

def multi_strategy_pnl_plot(
    df_list,
    strategy_names=None,
    title="Strategy PnL Comparison",
    y_label="Equity",
    height=620,
    vx=None,
    show_legend=True,
    show_rangeslider=False,
    colors=None,
    line_width=2,
    save_path=None
):
    """
    Plot PnL/equity for multiple strategies on a single graph.
    
    Args:
        df_list: List of DataFrames, each containing 'time' and 'equity' (or 'pnl') columns
        strategy_names: List of strategy names for legend (if None, uses Strategy 1, 2, etc.)
        title: Plot title
        y_label: Y-axis label
        height: Figure height in pixels
        vx: Position for vertical line (as fraction of data length), None to disable
        show_legend: Whether to show legend
        show_rangeslider: Whether to show range slider
        colors: List of colors for each strategy (if None, uses plotly defaults)
        line_width: Width of plot lines
        save_path: Optional path to save figure (HTML or image)
    
    Returns:
        plotly figure object
    """
    
    # Default strategy names if not provided
    if strategy_names is None:
        strategy_names = [f"Strategy {i+1}" for i in range(len(df_list))]
    
    # Default colors if not provided
    if colors is None:
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    
    # Create figure
    fig = go.Figure()
    
    # Track the earliest and latest dates across all strategies
    all_times = []
    
    # Add trace for each strategy
    for idx, df in enumerate(df_list):
        # Prepare data - handle different column names
        df_copy = df.copy()
        
        # Check for time column
        time_col = None
        for col in ['time', 'Time', 'date', 'Date', 'datetime', 'Datetime']:
            if col in df_copy.columns:
                time_col = col
                break
        if time_col is None:
            raise ValueError(f"DataFrame {idx} must have a time/date column")
        
        # Check for equity/pnl column
        equity_col = None
        for col in ['equity', 'Equity', 'pnl', 'PnL', 'value', 'Value', 'y']:
            if col in df_copy.columns:
                equity_col = col
                break
        if equity_col is None:
            raise ValueError(f"DataFrame {idx} must have an equity/pnl column")
        
        # Standardize column names
        df_copy['time'] = pd.to_datetime(df_copy[time_col])
        df_copy['y'] = pd.to_numeric(df_copy[equity_col], errors="coerce")
        
        # Clean and sort data
        df_clean = df_copy[['time', 'y']].dropna().sort_values('time')
        
        # Collect all times for vertical line calculation
        all_times.extend(df_clean['time'].tolist())
        
        # Add trace
        fig.add_trace(
            go.Scatter(
                x=df_clean['time'], 
                y=df_clean['y'], 
                mode='lines', 
                name=strategy_names[idx],
                line=dict(
                    width=line_width, 
                    color=colors[idx % len(colors)]
                ),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Time: %{x}<br>' +
                             'Value: %{y:.2f}<br>' +
                             '<extra></extra>'
            )
        )
    
    # Add vertical line if specified
    if vx is not None and all_times:
        all_times_sorted = sorted(all_times)
        vline_idx = int(len(all_times_sorted) * vx)
        vline_x = all_times_sorted[vline_idx]
        
        # Convert to timestamp value to avoid pandas Timestamp arithmetic issues
        vline_timestamp = pd.Timestamp(vline_x).value / 10**9  # Convert to Unix timestamp
        vline_datetime = pd.to_datetime(vline_timestamp, unit='s')
        
        # Add vertical line using shapes instead of add_vline to avoid the issue
        fig.add_shape(
            type="line",
            x0=vline_datetime,
            x1=vline_datetime,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(
                color="red",
                width=1.5,
                dash="dash"
            )
        )
        
        # Add annotation for the vertical line
        fig.add_annotation(
            x=vline_datetime,
            y=1,
            yref="paper",
            text="Test/Train",
            showarrow=False,
            yshift=10
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16)
        ),
        yaxis_title=y_label,
        xaxis_title="Time",
        height=height,
        hovermode='x unified',
        showlegend=show_legend,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        margin=dict(l=60, r=30, t=50, b=50),
        template='plotly_white',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            rangeslider=dict(visible=show_rangeslider)
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
        )
    )
    
    # Save if path provided
    if save_path:
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)
    
    return fig
