"""
Data loading and feature engineering for Bitcoin prediction
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
    
    # Basic price features
    data['close'] = df['Close']
    data['high'] = df['High']
    data['low'] = df['Low']
    data['volume'] = df['Volume']
    
    # Returns (multiple timeframes)
    data['return_1d'] = df['Close'].pct_change(1)
    data['return_3d'] = df['Close'].pct_change(3)
    data['return_7d'] = df['Close'].pct_change(7)
    
    # Volatility
    data['volatility_7d'] = data['return_1d'].rolling(7).std()
    data['volatility_30d'] = data['return_1d'].rolling(30).std()
    
    # Volume features
    data['volume_change'] = df['Volume'].pct_change()
    data['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # Moving averages
    sma_10 = SMAIndicator(df['Close'], 10).sma_indicator()
    sma_20 = SMAIndicator(df['Close'], 20).sma_indicator()
    sma_50 = SMAIndicator(df['Close'], 50).sma_indicator()
    sma_200 = SMAIndicator(df['Close'], 200).sma_indicator()
    
    data['sma_10'] = sma_10
    data['sma_20'] = sma_20
    data['sma_50'] = sma_50
    data['sma_200'] = sma_200
    
    # Price relative to moving averages
    data['price_vs_sma10'] = (df['Close'] - sma_10) / sma_10
    data['price_vs_sma20'] = (df['Close'] - sma_20) / sma_20
    data['price_vs_sma50'] = (df['Close'] - sma_50) / sma_50
    
    # RSI (Relative Strength Index)
    data['rsi_7'] = RSIIndicator(df['Close'], 7).rsi()
    data['rsi_14'] = RSIIndicator(df['Close'], 14).rsi()
    data['rsi_21'] = RSIIndicator(df['Close'], 21).rsi()
    
    # MACD
    macd = MACD(df['Close'])
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = BollingerBands(df['Close'], 20)
    data['bb_upper'] = bb.bollinger_hband()
    data['bb_lower'] = bb.bollinger_lband()
    data['bb_middle'] = bb.bollinger_mavg()
    data['bb_position'] = (df['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
    
    # Price range
    data['high_low_range'] = (df['High'] - df['Low']) / df['Close']
    
    print(f"Added {len(data.columns)} technical features")
    return data


def create_classification_targets(df, horizons=[1, 3, 7, 14], percentile=33.3):
    """
    Create classification targets (0=Down, 1=Neutral, 2=Up)
    """
    print(f"\nCreating classification targets using {percentile:.1f}th percentile")
    targets = pd.DataFrame(index=df.index)
    
    for h in horizons:
        future_return = df['Close'].shift(-h) / df['Close'] - 1
        
        # Determine thresholds
        abs_returns = future_return.abs()
        
        # We need two thresholds for 3 classes
        thresh_low = abs_returns.quantile(percentile / 100)
        # This is not quite right for 33/33/33. Let's fix this logic.
        
        # NEW LOGIC for 33/33/33 split:
        thresh_down = future_return.quantile(percentile / 100) # e.g., 33.3rd percentile
        thresh_up = future_return.quantile(1 - (percentile / 100)) # e.g., 66.7th percentile
        
        # 0: Down, 1: Neutral, 2: Up
        targets[f'target_{h}d'] = 1
        targets.loc[future_return > thresh_up, f'target_{h}d'] = 2
        targets.loc[future_return < thresh_down, f'target_{h}d'] = 0
        
        # Show distribution
        counts = targets[f'target_{h}d'].value_counts().sort_index()
        total = len(targets[f'target_{h}d'].dropna())
        dist = {k: counts.get(k, 0) / total * 100 for k in [0, 1, 2]}
        print(f"  {h}d (Thresh: {thresh_down:.2%} / {thresh_up:.2%}) - Down: {dist[0]:.1f}%, Neut: {dist[1]:.1f}%, Up: {dist[2]:.1f}%")

    return targets


def prepare_dataset(start_date='2018-01-01', horizons=[1, 3, 7, 14], percentile=33.3):
    """
    Main function to prepare complete dataset
    """
    print("="*60)
    print("DATA PREPARATION")
    print("="*60)
    
    raw_data = download_bitcoin(start_date)
    features = add_technical_features(raw_data)
    targets = create_classification_targets(raw_data, horizons, percentile)
    
    dataset = pd.concat([features, targets], axis=1)
    
    # Remove NaN rows (from TAs and future targets)
    initial_rows = len(dataset)
    dataset = dataset.dropna()
    print(f"\nRemoved {initial_rows - len(dataset)} NaN rows")
    print(f"Final dataset rows: {len(dataset)}")
    
    return dataset, horizons


if __name__ == "__main__":
    dataset, horizons = prepare_dataset()
    print(f"\nDataset shape: {dataset.shape}")