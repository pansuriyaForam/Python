import pandas as pd
import yfinance as yf

def load_data(tickers, start_date, end_date):
    """Download OHLCV data for given tickers and date range"""
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    return data

def clean_data(df):
    """Flatten MultiIndex, rename columns, drop NAs"""
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df = df.dropna()
    return df

def save_cleaned_data(df, filepath):
    """Save cleaned DataFrame to CSV"""
    df.to_csv(filepath, index = True)