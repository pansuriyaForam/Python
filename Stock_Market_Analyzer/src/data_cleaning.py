import pandas as pd
import yfinance as yf

def load_data(tickers, start_date, end_date):
    """Download OHLCV data for given tickers and date range"""
    try:
        # Convert single ticker to list
        if isinstance(tickers, str):
            tickers = [tickers]
            
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', progress=True)
        
        # If only one ticker, yfinance returns DataFrame without MultiIndex
        if len(tickers) == 1:
            # Add ticker prefix to columns
            data.columns = [f"{tickers[0]}_{col}" for col in data.columns]
        return data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def clean_data(df):
    """Flatten MultiIndex, rename columns, drop NAs"""
    try:
        # If the DataFrame is empty, it just returns the same empty DataFrame.
        if df.empty:
            return df
            
        # Check if we have a MultiIndex columns (multiple tickers)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
        # Single ticker, columns are already flattened
        
        # Drop rows with all NaN values
        df = df.dropna(how='all')
        
        # Forward fill and then drop any remaining NAs
        df = df.ffill().dropna()
        
        return df
        
    except Exception as e:
        print(f"Error cleaning data: {e}")
        return df

def save_cleaned_data(df, filepath):
    """Save cleaned DataFrame to CSV"""
    df.to_csv(filepath, index=True)