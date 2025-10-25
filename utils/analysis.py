import pandas as pd
import numpy as np

def analyze(df):
    """
    Analyze portfolio performance
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'equity' and 'action' columns
    
    Returns:
    --------
    dict : Dictionary containing statistical results and action counts
    """
    
    # Calculate returns (using equity percentage change)
    df['return'] = df['equity'].pct_change()
    
    # Remove first NaN value
    returns = df['return'].dropna()
    
    # =================
    # Check for all-zero returns
    # =================
    if len(returns) == 0 or (returns == 0).all():
        print("Warning: All returns are zero or no valid returns available.")
        print("Cannot calculate meaningful statistics.")
        
        # Return minimal stats
        action_counts = df['action'].value_counts().sort_index()
        action_percentages = (action_counts / len(df) * 100).round(2)
        
        action_stats = pd.DataFrame({
            'Action': action_counts.index,
            'Count': action_counts.values,
            'Percentage': action_percentages.values
        })
        
        print(action_stats)
        
        return {
            'stats_df': None,
            'action_stats': action_stats,
            'returns': returns,
            'metrics': None,
            'warning': 'All returns are zero'
        }
    
    # =================
    # Statistical Analysis
    # =================
    
    # Basic statistics
    mean_return = returns.mean()
    volatility = returns.std()
    
    # Annualized statistics
    hours_per_year = 365 * 24
    annualized_return = mean_return * hours_per_year
    annualized_volatility = volatility * np.sqrt(hours_per_year)
    
    # Check if volatility is zero
    if volatility == 0 or np.isnan(volatility):
        print("Warning: Volatility is zero. Sharpe Ratio cannot be calculated.")
        sharpe_ratio = np.nan
        annualized_sharpe = np.nan
    else:
        # Set risk-free rate (assume 2% annualized)
        annual_risk_free_rate = 0.05
        hourly_risk_free_rate = annual_risk_free_rate / hours_per_year
        
        # Calculate Sharpe Ratio
        excess_returns = returns - hourly_risk_free_rate
        sharpe_ratio = excess_returns.mean() / volatility
        annualized_sharpe = sharpe_ratio * np.sqrt(hours_per_year)
    
    # =================
    # Maximum Drawdown
    # =================
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    max_drawdown_pct = max_drawdown * 100
    
    # =================
    # Win Rate (action != 0 and return > 0)
    # =================
    # Calculate PnL for each period
    df['pnl'] = df['equity'].diff()
    
    # Filter for periods where action is not zero
    active_periods = df[df['action'] != 0].copy()
    
    if len(active_periods) > 0:
        # Count winning periods (positive PnL when action is not zero)
        winning_periods = len(active_periods[active_periods['pnl'] > 0])
        total_active_periods = len(active_periods)
        win_rate = (winning_periods / total_active_periods) * 100
    else:
        winning_periods = 0
        total_active_periods = 0
        win_rate = 0.0
    
    # Create statistics DataFrame
    stats_dict = {
        'Metric': [
            'Annual Mean Return',
            'Annual Volatility',
            'Annualized Sharpe Ratio (2% RF)',
            'Maximum Drawdown',
            'Win Rate (Active Periods)',
            'Winning Periods',
            'Total Active Periods'
        ],
        'Value': [
            f'{annualized_return:.6f}',
            f'{annualized_volatility:.6f}',
            f'{annualized_sharpe:.6f}' if not np.isnan(annualized_sharpe) else 'N/A',
            f'{max_drawdown:.6f}',
            f'{win_rate:.2f}%',
            f'{winning_periods}',
            f'{total_active_periods}'
        ],
        'Percentage': [
            f'{annualized_return * 100:.2f}%',
            f'{annualized_volatility * 100:.2f}%',
            'N/A',
            f'{max_drawdown_pct:.2f}%',
            f'{win_rate:.2f}%',
            'N/A',
            'N/A'
        ]
    }
    
    stats_df = pd.DataFrame(stats_dict)
    
    print("\n=== Performance Metrics ===")
    print(stats_df)

    # =================
    # Action Count Statistics
    # =================
    
    action_counts = df['action'].value_counts().sort_index()
    action_percentages = (action_counts / len(df) * 100).round(2)
    
    action_stats = pd.DataFrame({
        'Action': action_counts.index,
        'Count': action_counts.values,
        'Percentage': action_percentages.values
    })
    
    print("\n=== Action Statistics ===")
    print(action_stats)
    
    # Return results
    return {
        'stats_df': stats_df,
        'action_stats': action_stats,
        'returns': returns,
        'metrics': {
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'annualized_sharpe': annualized_sharpe,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'win_rate': win_rate,
            'winning_periods': winning_periods,
            'total_active_periods': total_active_periods
        }
    }

