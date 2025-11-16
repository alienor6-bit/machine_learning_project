"""
Data loading and feature engineering
"""

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands


def download_bitcoin(start_date='2018-01-01'):
    """
    Download Bitcoin price data from Yahoo Finance
    """
    print("Downloading Bitcoin data...")
    df = yf.download('BTC-USD', start=start_date, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print(f"Downloaded {len(df)} days of data")
    return df


def add_technical_features(df):
    """
    Add technical indicators as features (Full Version)
    """
    data = pd.DataFrame(index=df.index)
    
    # --- Features from Price ---
    data['close'] = df['Close']
    data['high'] = df['High']
    data['low'] = df['Low']
    data['return_1d'] = df['Close'].pct_change(1)
    data['return_3d'] = df['Close'].pct_change(3)
    data['return_7d'] = df['Close'].pct_change(7)
    data['volatility_7d'] = data['return_1d'].rolling(7).std()
    data['volatility_30d'] = data['return_1d'].rolling(30).std()
    
    # SMA
    sma_10 = SMAIndicator(df['Close'], 10).sma_indicator()
    sma_20 = SMAIndicator(df['Close'], 20).sma_indicator()
    sma_50 = SMAIndicator(df['Close'], 50).sma_indicator()
    data['price_vs_sma10'] = (df['Close'] - sma_10) / sma_10
    data['price_vs_sma20'] = (df['Close'] - sma_20) / sma_20
    data['price_vs_sma50'] = (df['Close'] - sma_50) / sma_50
    
    # RSI
    data['rsi_14'] = RSIIndicator(df['Close'], 14).rsi()
    
    # MACD
    macd = MACD(df['Close'])
    data['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = BollingerBands(df['Close'], 20)
    data['bb_position'] = (df['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    data['bb_width'] = bb.bollinger_wband()
    
    # Price range
    data['high_low_range'] = (df['High'] - df['Low']) / df['Close']
    
    # --- Features from Volume ---
    data['volume'] = df['Volume']
    data['volume_change'] = df['Volume'].pct_change()
    data['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    print(f"Added {len(data.columns)} technical features")
    return data


def create_regression_targets(df, horizons=[1, 3, 7, 14]):
    """
    Create regression targets (future % volume change)
    """
    print(f"\nCreating regression targets for horizons: {horizons}")
    targets = pd.DataFrame(index=df.index)
    
    # Use a rolling average of volume to smooth it out
    vol_avg = df['Volume'].rolling(5).mean()
    
    for h in horizons:
        # Future average volume
        future_vol_avg = vol_avg.shift(-h)
        
        # Target: % change in future avg volume vs current avg volume
        targets[f'target_{h}d'] = (future_vol_avg - vol_avg) / vol_avg
        
        # Show stats
        print(f"  {h}d - Avg Change: {targets[f'target_{h}d'].mean():.2%}, "
              f"Std: {targets[f'target_{h}d'].std():.2%}")
    
    return targets


def prepare_dataset(start_date='2018-01-01', horizons=[1, 3, 7, 14]):
    """
    Main function to prepare complete dataset
    """
    print("="*60)
    print("DATA PREPARATION")
    print("="*60)
    
    raw_data = download_bitcoin(start_date)
    features = add_technical_features(raw_data)
    
    # Call the new regression target function
    targets = create_regression_targets(raw_data, horizons)
    
    dataset = pd.concat([features, targets], axis=1)
    
    initial_rows = len(dataset)
    dataset = dataset.dropna()
    print(f"\nRemoved {initial_rows - len(dataset)} NaN rows")
    print(f"Final dataset rows: {len(dataset)}")
    
    return dataset, horizons