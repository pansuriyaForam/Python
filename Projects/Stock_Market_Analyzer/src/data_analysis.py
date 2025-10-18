import numpy as np

def daily_returns(df):
    return df.pct_change().dropna()

def cumulative_returns(df):
    return (df / df.iloc[0]) - 1

def annualized_volatility(daily_returns):
    return daily_returns.std() * np.sqrt(252)

def correlation_matrix(daily_returns):
    return daily_returns.corr()

def sharpe_ratio(daily_returns, annual_vol, risk_free=0):
    mean_daily = daily_returns.mean()
    return (mean_daily * 252 - risk_free) / annual_vol