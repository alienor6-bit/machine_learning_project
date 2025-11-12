import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
#from ta.volume import VolumeSMAIndicator
import numpy as np

def get_forex_data(pair="EURUSD=X", start="2015-01-01", end="2025-01-01"):
    """
    Download forex data for EUR/USD pair
    """
    print(f"Downloading {pair} data from {start} to {end}")

    # Download forex data
    data = yf.download(pair, start=start, end=end, interval="1d")

    if data.empty:
        raise ValueError(f"No data found for {pair}")

    print(f"Downloaded {len(data)} days of data")
    return data

def add_forex_indicators(df):
    """
    Add technical indicators specific to forex trading
    """
    # Ensure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Clean data - remove any rows with NaN values in OHLC
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

    if len(df) < 100:
        raise ValueError("Not enough data points after cleaning")

    indicators = pd.DataFrame(index=df.index)

    # Basic price data
    indicators['Open'] = df['Open']
    indicators['High'] = df['High']
    indicators['Low'] = df['Low']
    indicators['Close'] = df['Close']
    indicators['Volume'] = df['Volume']

    # Price-based features
    indicators['HL_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
    indicators['OC_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100

    # RSI (multiple timeframes)
    indicators['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()
    indicators['RSI_21'] = RSIIndicator(close=df['Close'], window=21).rsi()

    # MACD
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    indicators['MACD'] = macd.macd()
    indicators['MACD_Signal'] = macd.macd_signal()
    indicators['MACD_Histogram'] = macd.macd_diff()

    # Moving Averages (important for forex)
    indicators['SMA_10'] = SMAIndicator(close=df['Close'], window=10).sma_indicator()
    indicators['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    indicators['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    indicators['SMA_200'] = SMAIndicator(close=df['Close'], window=200).sma_indicator()

    indicators['EMA_12'] = EMAIndicator(close=df['Close'], window=12).ema_indicator()
    indicators['EMA_26'] = EMAIndicator(close=df['Close'], window=26).ema_indicator()
    indicators['EMA_50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()

    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    indicators['BB_Upper'] = bb.bollinger_hband()
    indicators['BB_Lower'] = bb.bollinger_lband()
    indicators['BB_Middle'] = bb.bollinger_mavg()
    indicators['BB_Width'] = (indicators['BB_Upper'] - indicators['BB_Lower']) / indicators['BB_Middle']
    indicators['BB_Position'] = (df['Close'] - indicators['BB_Lower']) / (indicators['BB_Upper'] - indicators['BB_Lower'])

    # Average True Range (volatility)
    indicators['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()

    # Stochastic Oscillator
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
    indicators['Stoch_K'] = stoch.stoch()
    indicators['Stoch_D'] = stoch.stoch_signal()

    # ADX (trend strength)
    indicators['ADX'] = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).adx()

    # Price changes and returns
    indicators['Price_Change'] = df['Close'].pct_change(fill_method=None)
    indicators['Price_Change_2d'] = df['Close'].pct_change(periods=2, fill_method=None)
    indicators['Price_Change_5d'] = df['Close'].pct_change(periods=5, fill_method=None)

    # Volume indicators (if volume data is available and meaningful)
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        #indicators['Volume_SMA'] = VolumeSMAIndicator(close=df['Close'], volume=df['Volume'], window=20).volume_sma()
        indicators['Volume_Ratio'] = df['Volume'] / indicators['Volume_SMA']

    # Distance from moving averages (normalized)
    eps = 1e-8
    indicators['Dist_SMA20'] = (df['Close'] - indicators['SMA_20']) / (indicators['SMA_20'] + eps)
    indicators['Dist_SMA50'] = (df['Close'] - indicators['SMA_50']) / (indicators['SMA_50'] + eps)
    indicators['Dist_EMA12'] = (df['Close'] - indicators['EMA_12']) / (indicators['EMA_12'] + eps)

    # Candlestick patterns (simplified)
    indicators['Doji'] = (abs(df['Close'] - df['Open']) / (df['High'] - df['Low'] + eps) < 0.1).astype(int)
    indicators['Hammer'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low'] + eps) > 0.6).astype(int)

    # Day of week effect (forex markets are affected by this)
    indicators['DayOfWeek'] = df.index.dayofweek
    indicators['IsMonday'] = (indicators['DayOfWeek'] == 0).astype(int)
    indicators['IsFriday'] = (indicators['DayOfWeek'] == 4).astype(int)

    print(f"Created {len(indicators.columns)} technical indicators")
    return indicators

def create_target_variables(df, horizons=[1, 5, 10, 20]):
    """
    Create target variables for forex prediction
    """
    targets = pd.DataFrame(index=df.index)

    for horizon in horizons:
        # Future return
        future_price = df['Close'].shift(-horizon)
        targets[f'Target_Return_{horizon}d'] = (future_price - df['Close']) / df['Close']

        # Future direction (binary classification)
        targets[f'Target_Direction_{horizon}d'] = (future_price > df['Close']).astype(int)

        # Future high/low within horizon
        future_high = df['High'].rolling(window=horizon, min_periods=1).max().shift(-horizon+1)
        future_low = df['Low'].rolling(window=horizon, min_periods=1).min().shift(-horizon+1)

        targets[f'Target_Max_Return_{horizon}d'] = (future_high - df['Close']) / df['Close']
        targets[f'Target_Min_Return_{horizon}d'] = (future_low - df['Close']) / df['Close']

    print(f"Created target variables for horizons: {horizons}")
    return targets

def prepare_forex_dataset():
    """
    Main function to prepare the complete forex dataset
    """
    # Download data
    raw_data = get_forex_data()

    # Add technical indicators
    indicators = add_forex_indicators(raw_data)

    # Create targets
    targets = create_target_variables(raw_data)

    # Combine everything
    complete_dataset = pd.concat([indicators, targets], axis=1)

    # Remove rows with NaN (usually the last few rows due to future targets)
    complete_dataset = complete_dataset.dropna()

    print(f"\nFinal dataset shape: {complete_dataset.shape}")
    print(f"Date range: {complete_dataset.index[0]} to {complete_dataset.index[-1]}")

    return complete_dataset

if __name__ == "__main__":
    # Test the functions
    dataset = prepare_forex_dataset()
    print("\nDataset summary:")
    print(dataset.describe())
