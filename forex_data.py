import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import numpy as np

def get_forex_data(pair="EURUSD=X", start="2015-01-01", end="2025-01-01"):
    """
    Download forex data with robust handling of yfinance column structures
    """
    print(f"Downloading {pair} data from {start} to {end}")

    # List of EUR/USD symbols to try
    symbols_to_try = [pair, "EUR=X", "EURUSD"]

    for symbol in symbols_to_try:
        try:
            print(f"  Attempting to download: {symbol}")

            # Try download with basic settings first
            data = yf.download(symbol, start=start, end=end, interval="1d",
                             auto_adjust=False, prepost=False, progress=False)

            if not data.empty:
                print(f"  ✓ Successfully downloaded {len(data)} rows with {symbol}")
                print(f"  Raw columns: {list(data.columns)}")

                # HANDLE MULTIINDEX COLUMNS
                if isinstance(data.columns, pd.MultiIndex):
                    print("  📊 Detected MultiIndex columns - processing...")

                    # The structure is typically: (column_name, ticker_symbol)
                    # We want to extract just the column names

                    # Method 1: Get the first level (column names)
                    clean_columns = data.columns.get_level_values(0)
                    data.columns = clean_columns
                    print(f"  ✓ Flattened to: {list(data.columns)}")

                # STANDARDIZE COLUMN NAMES
                column_mapping = {
                    'Adj Close': 'Adj_Close',
                    'adj close': 'Adj_Close',
                    'Open': 'Open',
                    'High': 'High',
                    'Low': 'Low',
                    'Close': 'Close',
                    'Volume': 'Volume'
                }

                # Apply mapping
                data = data.rename(columns=column_mapping)
                print(f"  📋 Standardized columns: {list(data.columns)}")

                # VERIFY WE HAVE ESSENTIAL DATA
                essential_cols = ['Open', 'High', 'Low', 'Close']
                available_essential = [col for col in essential_cols if col in data.columns]

                if len(available_essential) >= 1:  # At least Close price
                    print(f"  ✅ Found essential data: {available_essential}")

                    # Fill missing OHLC with Close if needed (common for some forex feeds)
                    if 'Close' in data.columns:
                        for col in ['Open', 'High', 'Low']:
                            if col not in data.columns:
                                print(f"  📝 Creating missing {col} from Close price")
                                data[col] = data['Close']

                    # Ensure Volume column exists (forex volume is often not meaningful)
                    if 'Volume' not in data.columns:
                        print("  📝 Adding dummy Volume column")
                        data['Volume'] = 0

                    # Remove any completely empty rows
                    data = data.dropna(how='all')

                    if len(data) > 0:
                        print(f"  ✅ Final data: {data.shape}")
                        print(f"  📅 Date range: {data.index[0]} to {data.index[-1]}")
                        print(f"  📊 Sample data:")
                        print(data[['Open', 'High', 'Low', 'Close']].head(2))
                        return data
                    else:
                        print(f"  ❌ No data remaining after cleaning")
                else:
                    print(f"  ❌ No essential price data found")

        except Exception as e:
            print(f"  ❌ Error with {symbol}: {str(e)[:100]}...")
            continue

    # If all symbols fail, create sample data for testing
    print(f"\n⚠️  Could not download real data - creating sample forex data for testing...")
    return create_sample_forex_data(start, end)

def create_sample_forex_data(start="2015-01-01", end="2025-01-01"):
    """
    Create realistic sample EUR/USD data for testing when real data is unavailable
    """
    print("Creating sample EUR/USD data...")

    # Generate realistic EUR/USD price series
    date_range = pd.date_range(start=start, end=end, freq='D')
    n_days = len(date_range)

    # Start at realistic EUR/USD level
    initial_price = 1.0800  # EUR/USD around 1.08

    # Generate realistic price movements
    np.random.seed(42)  # For reproducible results

    # Daily returns with realistic volatility for EUR/USD
    daily_returns = np.random.normal(0, 0.006, n_days)  # ~0.6% daily volatility

    # Add some trend and mean reversion
    trend = np.linspace(-0.05, 0.05, n_days)  # Slight trend over time
    mean_reversion = -0.1 * (np.cumsum(daily_returns) - np.mean(daily_returns))

    total_returns = daily_returns + trend/n_days + mean_reversion/n_days

    # Generate price series
    close_prices = [initial_price]
    for ret in total_returns[1:]:
        new_price = close_prices[-1] * (1 + ret)
        close_prices.append(max(0.8, min(1.5, new_price)))  # Realistic EUR/USD range

    # Create OHLC data
    data_dict = {'Close': close_prices}

    # Generate realistic OHLC from Close
    highs, lows, opens = [], [], []

    for i, close in enumerate(close_prices):
        # Daily range typically 0.2-1% for EUR/USD
        daily_range = abs(np.random.normal(0, 0.003))

        high = close * (1 + daily_range/2)
        low = close * (1 - daily_range/2)

        # Open is close to previous close with small gap
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, 0.001)  # Small overnight gap
            open_price = close_prices[i-1] * (1 + gap)
            open_price = max(low, min(high, open_price))  # Keep within range

        opens.append(open_price)
        highs.append(high)
        lows.append(low)

    # Create DataFrame
    sample_data = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': close_prices,
        'Adj_Close': close_prices,
        'Volume': [0] * n_days  # Forex volume not meaningful
    }, index=date_range)

    # Remove weekends (forex markets are 24/5)
    sample_data = sample_data[sample_data.index.dayofweek < 5]

    print(f"✅ Created sample data: {sample_data.shape}")
    print(f"📅 Date range: {sample_data.index[0]} to {sample_data.index[-1]}")
    print(f"💱 Price range: {sample_data['Close'].min():.4f} - {sample_data['Close'].max():.4f}")

    return sample_data

def add_forex_indicators(df):
    """
    Add comprehensive technical indicators for forex trading
    """
    print(f"Creating forex indicators for {len(df)} data points...")

    # Ensure we have minimum required data
    if len(df) < 250:  # Need ~1 year for 200-day MA
        print(f"⚠️  Warning: Only {len(df)} days of data. Some indicators may not be reliable.")

    # Clean data
    required_cols = ['Open', 'High', 'Low', 'Close']
    df_clean = df[required_cols].dropna()

    if len(df_clean) == 0:
        raise ValueError("No valid OHLC data after cleaning")

    print(f"✅ Using {len(df_clean)} clean data points")

    indicators = pd.DataFrame(index=df_clean.index)

    # === BASIC PRICE DATA ===
    indicators['Open'] = df_clean['Open']
    indicators['High'] = df_clean['High']
    indicators['Low'] = df_clean['Low']
    indicators['Close'] = df_clean['Close']

    # === MOMENTUM INDICATORS ===
    # RSI - key for forex
    indicators['RSI_14'] = RSIIndicator(close=df_clean['Close'], window=14).rsi()
    indicators['RSI_21'] = RSIIndicator(close=df_clean['Close'], window=21).rsi()

    # MACD - crucial for forex trends
    macd = MACD(close=df_clean['Close'])
    indicators['MACD'] = macd.macd()
    indicators['MACD_Signal'] = macd.macd_signal()
    indicators['MACD_Histogram'] = macd.macd_diff()

    # Stochastic - important for forex
    stoch = StochasticOscillator(high=df_clean['High'], low=df_clean['Low'], close=df_clean['Close'])
    indicators['Stoch_K'] = stoch.stoch()
    indicators['Stoch_D'] = stoch.stoch_signal()

    # === TREND INDICATORS ===
    # Moving averages - critical for forex
    indicators['SMA_20'] = SMAIndicator(close=df_clean['Close'], window=20).sma_indicator()
    indicators['SMA_50'] = SMAIndicator(close=df_clean['Close'], window=50).sma_indicator()
    indicators['EMA_12'] = EMAIndicator(close=df_clean['Close'], window=12).ema_indicator()
    indicators['EMA_26'] = EMAIndicator(close=df_clean['Close'], window=26).ema_indicator()

    # Only add SMA_200 if we have enough data
    if len(df_clean) >= 200:
        indicators['SMA_200'] = SMAIndicator(close=df_clean['Close'], window=200).sma_indicator()
    else:
        indicators['SMA_200'] = indicators['SMA_50']  # Use SMA_50 as proxy

    # === VOLATILITY INDICATORS ===
    # Bollinger Bands
    bb = BollingerBands(close=df_clean['Close'], window=20)
    indicators['BB_Upper'] = bb.bollinger_hband()
    indicators['BB_Lower'] = bb.bollinger_lband()
    indicators['BB_Middle'] = bb.bollinger_mavg()

    # ATR
    indicators['ATR'] = AverageTrueRange(high=df_clean['High'], low=df_clean['Low'], close=df_clean['Close']).average_true_range()

    # === PRICE MOVEMENTS ===
    indicators['Price_Change'] = df_clean['Close'].pct_change(fill_method=None)
    indicators['Price_Change_5d'] = df_clean['Close'].pct_change(periods=5, fill_method=None)

    # === DERIVED FEATURES ===
    eps = 1e-8

    # Price vs moving averages
    indicators['Price_vs_SMA20'] = (df_clean['Close'] - indicators['SMA_20']) / (indicators['SMA_20'] + eps)
    indicators['Price_vs_SMA50'] = (df_clean['Close'] - indicators['SMA_50']) / (indicators['SMA_50'] + eps)

    # Bollinger position
    bb_range = indicators['BB_Upper'] - indicators['BB_Lower']
    indicators['BB_Position'] = (df_clean['Close'] - indicators['BB_Lower']) / (bb_range + eps)
    indicators['BB_Width'] = bb_range / (indicators['BB_Middle'] + eps)

    # === TIME FEATURES ===
    indicators['DayOfWeek'] = df_clean.index.dayofweek
    indicators['IsMonday'] = (indicators['DayOfWeek'] == 0).astype(int)
    indicators['IsFriday'] = (indicators['DayOfWeek'] == 4).astype(int)

    # === CANDLESTICK PATTERNS ===
    indicators['Body_Size'] = abs(df_clean['Close'] - df_clean['Open']) / (df_clean['Close'] + eps)
    indicators['Is_Bullish'] = (df_clean['Close'] > df_clean['Open']).astype(int)

    print(f"✅ Created {len(indicators.columns)} forex indicators")
    return indicators

def create_forex_targets(df, horizons=[1, 5, 10]):
    """
    Create forex-specific target variables
    """
    print(f"Creating forex targets for horizons: {horizons}")

    targets = pd.DataFrame(index=df.index)

    for horizon in horizons:
        # Future return
        future_price = df['Close'].shift(-horizon)
        targets[f'Target_Return_{horizon}d'] = (future_price - df['Close']) / df['Close']

        # Future direction
        targets[f'Target_Direction_{horizon}d'] = (future_price > df['Close']).astype(int)

        # Future high/low
        if all(col in df.columns for col in ['High', 'Low']):
            future_high = df['High'].rolling(window=horizon).max().shift(-horizon+1)
            future_low = df['Low'].rolling(window=horizon).min().shift(-horizon+1)

            targets[f'Target_Max_Return_{horizon}d'] = (future_high - df['Close']) / df['Close']
            targets[f'Target_Min_Return_{horizon}d'] = (future_low - df['Close']) / df['Close']

    print(f"✅ Created {len(targets.columns)} target variables")
    return targets

def prepare_forex_dataset():
    """
    Main function to prepare complete EUR/USD dataset
    """
    print("="*60)
    print("PREPARING EUR/USD FOREX DATASET")
    print("="*60)

    try:
        # Step 1: Get forex data
        print("\n📊 Step 1: Downloading forex data...")
        raw_data = get_forex_data()

        # Step 2: Create indicators
        print("\n📈 Step 2: Creating technical indicators...")
        indicators = add_forex_indicators(raw_data)

        # Step 3: Create targets
        print("\n🎯 Step 3: Creating target variables...")
        targets = create_forex_targets(raw_data)

        # Step 4: Combine data
        print("\n🔄 Step 4: Combining data...")
        complete_dataset = pd.concat([indicators, targets], axis=1)

        # Step 5: Final cleaning
        print("\n🧹 Step 5: Final cleaning...")
        initial_rows = len(complete_dataset)
        complete_dataset = complete_dataset.dropna()
        final_rows = len(complete_dataset)

        print(f"✅ Removed {initial_rows - final_rows} rows with NaN")
        print(f"✅ Final dataset: {complete_dataset.shape}")
        print(f"📅 Date range: {complete_dataset.index[0]} to {complete_dataset.index[-1]}")

        # Show breakdown
        feature_cols = [col for col in complete_dataset.columns if not col.startswith('Target_')]
        target_cols = [col for col in complete_dataset.columns if col.startswith('Target_')]

        print(f"\n📋 Dataset composition:")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Targets: {len(target_cols)}")

        return complete_dataset

    except Exception as e:
        print(f"❌ Error in prepare_forex_dataset: {e}")
        raise

if __name__ == "__main__":
    try:
        dataset = prepare_forex_dataset()
        print(f"\n🎉 SUCCESS! Forex dataset ready with {dataset.shape[0]} rows and {dataset.shape[1]} columns")

        # Show sample
        print("\n📊 Sample data:")
        sample_cols = ['Close', 'RSI_14', 'MACD', 'Target_Return_1d', 'Target_Return_5d']
        available_cols = [col for col in sample_cols if col in dataset.columns]
        print(dataset[available_cols].tail())

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
