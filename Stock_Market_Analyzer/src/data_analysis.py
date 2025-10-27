import numpy as np
import pandas as pd

def daily_returns(df):
    """
    Calculate daily returns as percentage change
    """
    try:
        returns = df.pct_change().dropna()
        # Handle potential division by zero or inf values
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        return returns
    except Exception as e:
        print(f"Error calculating daily returns: {e}")
        return pd.DataFrame()

def cumulative_returns(df):
    """
    Calculate cumulative returns: (current_price / initial_price) - 1
    """
    try:
        if df.empty:
            return df
        cumulative = (df / df.iloc[0]) - 1
        return cumulative
    except Exception as e:
        print(f"Error calculating cumulative returns: {e}")
        return pd.DataFrame()

def annualized_volatility(daily_returns): 
    """
    Calculate annualized volatility (assuming 252 trading days)
    The annualized volatility, tells you how much the stock’s price fluctuates over a year. 
    Volatility is a measure of risk.
    """
    try:
        if daily_returns.empty:
            return pd.Series()
        return daily_returns.std() * np.sqrt(252)
    except Exception as e:
        print(f"Error calculating annualized volatility: {e}")
        return pd.Series()

def correlation_matrix(daily_returns):
    """
    Calculate correlation matrix of daily returns.
    A correlation matrix is often used to see if stocks move in the same direction (positive correlation) 
    or in opposite directions (negative correlation).
    """
    try:
        if daily_returns.empty:
            return pd.DataFrame()
        return daily_returns.corr()
    except Exception as e:
        print(f"Error calculating correlation matrix: {e}")
        return pd.DataFrame()

def sharpe_ratio(daily_returns, annual_vol, risk_free=0):
    """
    The Sharpe ratio is a measure of the risk-adjusted return of an investment.
    """
    try:
        if daily_returns.empty or annual_vol.empty:
            return pd.Series()
        mean_daily = daily_returns.mean()
        return (mean_daily * 252 - risk_free) / annual_vol
    except Exception as e:
        print(f"Error calculating Sharpe ratio: {e}")
        return pd.Series()