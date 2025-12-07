import pandas as pd
import yfinance as yf
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==================================================================================
# 1. DATA LOADING
# ==================================================================================
def download_data(start_date='2000-01-01', ticker='^FCHI'):
    """
    Downloads historical data from Yahoo Finance.

    Args:
        start_date (str): Start date for data (YYYY-MM-DD).
        ticker (str): Ticker symbol (e.g., '^FCHI' for CAC 40, 'TSLA' for Tesla).

    Returns:
        pd.DataFrame: DataFrame containing the 'Close' price.
    """
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=start_date, progress=False)

    # Handle MultiIndex columns if present (common issue with yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=['Close'])
    print(f"  ✓ {len(df)} rows downloaded")
    return df



# ==================================================================================
# 2. MACROECONOMIC FEATURES (Context)
# ==================================================================================
def add_macro_features(df):
    """
    Adds macroeconomic indicators to give the model market context.

    Features added:
    - QQQ (Nasdaq 100): Proxy for global tech sentiment.
    - SPY (S&P 500): Proxy for global market health.
    - VIX: Volatility Index (Fear Gauge).
    - DXY: US Dollar Index (Currency strength).
    """
    data = pd.DataFrame(index=df.index)

    # 1. Market Indices (QQQ & SPY)
    try:
        qqq = yf.download('QQQ', start='2000-01-01', progress=False)['Close']
        spy = yf.download('SPY', start='2000-01-01', progress=False)['Close']

        # Calculate daily returns for these indices
        data['qqq_return'] = qqq.pct_change().reindex(df.index, method='ffill')
        data['spy_return'] = spy.pct_change().reindex(df.index, method='ffill')

        # Calculate rolling correlation with our target asset (20-day window)
        # This tells the model if our asset is moving WITH or AGAINST the market
        aligned_qqq = qqq.reindex(df.index, method='ffill')
        data['qqq_corr'] = df['Close'].rolling(20).corr(aligned_qqq)

        print("  ✓ Market Indices (QQQ, SPY) fetched")
    except Exception as e:
        print(f"  ✗ Market Indices failed: {e}")
        data['qqq_return'] = 0
        data['spy_return'] = 0
        data['qqq_corr'] = 0

    # 2. VIX (Volatility Index)
    try:
        vix = yf.download('^VIX', start='2000-01-01', progress=False)['Close']
        data['vix'] = vix.reindex(df.index, method='ffill')
        data['vix_change'] = data['vix'].pct_change(5) # 5-day change in fear
        print("  ✓ VIX fetched")
    except:
        data['vix'] = 0
        data['vix_change'] = 0

    # 3. DXY (Dollar Index)
    try:
        dxy = yf.download('DX-Y.NYB', start='2000-01-01', progress=False)['Close']
        data['dxy'] = dxy.reindex(df.index, method='ffill')
        data['dxy_momentum'] = data['dxy'].pct_change(20) # 20-day trend of Dollar
        print("  ✓ DXY fetched")
    except:
        data['dxy'] = 0
        data['dxy_momentum'] = 0

    return data

# ==================================================================================
# 3. TECHNICAL INDICATORS (Price Action)
# ==================================================================================
def add_technical_features(df):
    """
    Computes technical indicators based on price and volume.

    Categories:
    - Momentum: RSI, Stochastic, ROC
    - Trend: MACD, ADX, Moving Averages
    - Volatility: Bollinger Bands, ATR
    """
    data = pd.DataFrame(index=df.index)
    close = df['Close']

    # --- Returns (The most basic feature) ---
    data['return_1d'] = close.pct_change(1)
    data['return_3d'] = close.pct_change(3)
    data['return_5d'] = close.pct_change(5)
    data['return_10d'] = close.pct_change(10)

    # --- Lag Features (Memory) ---
    # Give the model access to past returns explicitly
    for lag in [1, 2, 3, 5, 10]:
        data[f'return_lag_{lag}'] = data['return_1d'].shift(lag)

    # --- Rolling Statistics (Recent History) ---
    data['return_mean_5'] = data['return_1d'].rolling(5).mean()
    data['return_std_5'] = data['return_1d'].rolling(5).std() # Short-term volatility
    data['volatility_20d'] = data['return_1d'].rolling(20).std() # Monthly volatility

    # --- Moving Averages (Trend) ---
    sma_10 = SMAIndicator(close, 10).sma_indicator()
    sma_50 = SMAIndicator(close, 50).sma_indicator()
    sma_200 = SMAIndicator(close, 200).sma_indicator() # Long-term trend

    # Distance from MAs (Normalized)
    data['dist_sma10'] = (close - sma_10) / sma_10
    data['dist_sma50'] = (close - sma_50) / sma_50
    data['dist_sma200'] = (close - sma_200) / sma_200

    # --- RSI (Relative Strength Index) ---
    # Measures overbought (>70) or oversold (<30) conditions
    data['rsi_14'] = RSIIndicator(close, 14).rsi()
    data['rsi_overbought'] = (data['rsi_14'] > 70).astype(int)
    data['rsi_oversold'] = (data['rsi_14'] < 30).astype(int)

    # --- MACD (Moving Average Convergence Divergence) ---
    # Trend-following momentum indicator
    macd = MACD(close)
    data['macd_diff'] = macd.macd_diff() # Histogram
    data['macd_norm'] = data['macd_diff'] / close # Normalized for scale invariance

    # --- Bollinger Bands (Volatility) ---
    bb = BollingerBands(close, 20)
    data['bb_width'] = bb.bollinger_wband() # High width = High volatility
    # Position within bands (0 = Lower Band, 1 = Upper Band)
    data['bb_pos'] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())

    # --- ATR (Average True Range) ---
    # Absolute volatility measure
    try:
        atr = AverageTrueRange(df['High'], df['Low'], close, 14)
        data['atr_pct'] = atr.average_true_range() / close # Normalized ATR
    except:
        data['atr_pct'] = 0

    print(f"  ✓ {len([c for c in data.columns])} technical features created")
    return data


# ==================================================================================
# 4. DATASET PREPARATION
# ==================================================================================
def prepare_dataset(start_date='2000-01-01', horizons=[1, 5]):
    """
    Orchestrates the data creation pipeline.
    1. Download Data
    2. Check for Duplicates
    3. Add Macro Features
    4. Add Technical Features
    5. Detect Outliers
    6. Create Targets (Labels)
    """
    raw_data = download_data(start_date)
    if len(raw_data) < 200:
        raise ValueError("Insufficient data.")

    # --- Check for Duplicates ---
    duplicates = raw_data.duplicated().sum()
    print(f"  Duplicates found: {duplicates}")
    if duplicates > 0:
        raw_data = raw_data.drop_duplicates()
        print(f"  ✓ Removed {duplicates} duplicate rows")

    macro_features = add_macro_features(raw_data)
    technical_features = add_technical_features(raw_data)
    features = pd.concat([technical_features, macro_features], axis=1)

    # --- Detect Outliers (Z-score method) ---
    from scipy import stats
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    # Drop NaN temporarily for Z-score calculation
    features_clean = features[numeric_cols].dropna()
    if len(features_clean) > 0:
        z_scores = np.abs(stats.zscore(features_clean))
        outliers_per_feature = (z_scores > 3).sum(axis=0)
        total_outliers = outliers_per_feature.sum()
        print(f"  Outliers detected (Z-score > 3): {total_outliers} total")

    # --- Target Creation ---
    targets = pd.DataFrame(index=raw_data.index)
    for h in horizons:
        future_close = raw_data['Close'].shift(-h)
        # Binary Classification: 1 if Price Goes Up, 0 otherwise
        targets[f'target_{h}d'] = (future_close > raw_data['Close']).astype(int)
        # Store real return for backtesting later
        current_close = raw_data['Close']
        targets[f'real_return_{h}d'] = (future_close - current_close) / current_close

    dataset = pd.concat([features, targets], axis=1)

    # Drop NaN values created by lag features and indicators
    initial_len = len(dataset)
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()
    final_len = len(dataset)

    print(f"Data cleaning: {initial_len} to {final_len} rows")
    return dataset

def create_sequences(data, horizons, window_size=60):
    """
    Creates 3D sequences for LSTM input.
    Format: (Samples, Time Steps, Features)
    """
    feature_cols = [c for c in data.columns if not c.startswith('target_') and not c.startswith('real_return')]
    target_cols = [f'target_{h}d' for h in horizons]

    features = data[feature_cols].values
    targets = data[target_cols].values

    X, y = [], []
    for i in range(window_size, len(data)):
        # Take a window of 'window_size' past days
        X.append(features[i-window_size:i])
        # Take the target for the current day
        y.append(targets[i])

    print(f"  ✓ Sequences created: {len(X)} samples x {window_size} timesteps x {len(feature_cols)} features")
    return np.array(X), np.array(y), feature_cols

def split_temporal(X, y, test_size=0.2, val_size=0.1):
    """
    Splits data chronologically (Time Series Split).
    CRITICAL: Do NOT shuffle randomly, or you will leak future information!
    """
    n = len(X)
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))

    X_train, y_train = X[:val_idx], y[:val_idx]
    X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
    X_test, y_test = X[test_idx:], y[test_idx:]

    print(f"Split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def balance_training_set(X_train, y_train):
    """
    Undersamples the majority class to enforce 50/50 balance.
    NOTE: Disabled for Trend Following strategies on Stocks.
    """
    if y_train.ndim > 1:
        y_target = y_train[:, 0]
    else:
        y_target = y_train

    indices_0 = np.where(y_target == 0)[0]
    indices_1 = np.where(y_target == 1)[0]

    min_samples = min(len(indices_0), len(indices_1))
    indices_0_sel = np.random.choice(indices_0, min_samples, replace=False)
    indices_1_sel = np.random.choice(indices_1, min_samples, replace=False)

    balanced_idx = np.concatenate([indices_0_sel, indices_1_sel])
    np.random.shuffle(balanced_idx)

    print(f"  ✓ Balanced: {len(indices_0)} downs + {len(indices_1)} ups → {len(balanced_idx)} samples")
    return X_train[balanced_idx], y_train[balanced_idx]

def normalize_features(X_train, X_val, X_test):
    """
    Normalizes features to have Mean=0 and Std=1 (StandardScaler).
    CRITICAL: Fit scaler ONLY on Training data to avoid leakage.
    """
    n_samples, n_timesteps, n_features = X_train.shape
    X_train_flat = X_train.reshape(-1, n_features)

    scaler = StandardScaler()
    scaler.fit(X_train_flat)

    X_train_norm = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_norm = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_norm = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    print(f"  ✓ Features normalized (mean=0, std=1)")
    return X_train_norm, X_val_norm, X_test_norm, scaler

def apply_dimension_reduction(X_train, X_val, X_test, method='pca', n_components=0.95):
    """
    Reduces feature dimensionality to combat overfitting and improve training speed.

    Args:
        X_train, X_val, X_test: Normalized 3D arrays (Samples, Time, Features)
        method: 'pca' or 'select_k_best'
        n_components: For PCA (variance ratio) or SelectKBest (number of features)

    Returns:
        Reduced arrays (now 2D for traditional ML, or reshaped for LSTM)
    """
    # Flatten 3D to 2D for dimension reduction
    n_samples_train, n_timesteps, n_features = X_train.shape
    X_train_flat = X_train.reshape(n_samples_train, -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
        X_train_reduced = reducer.fit_transform(X_train_flat)
        X_val_reduced = reducer.transform(X_val_flat)
        X_test_reduced = reducer.transform(X_test_flat)

        print(f"  ✓ PCA: {X_train_flat.shape[1]} features → {reducer.n_components_} components")
        print(f"    Explained variance: {reducer.explained_variance_ratio_.sum():.1%}")

    return X_train_reduced, X_val_reduced, X_test_reduced, reducer
