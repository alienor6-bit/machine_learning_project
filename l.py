"""
Improved Forex Model Configuration
Based on analysis of poor initial results
"""

# === IMPROVED MODEL ARCHITECTURE ===

def build_improved_forex_lstm(input_shape, n_outputs, dropout=0.4):
    """
    Enhanced LSTM architecture for forex prediction
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras import layers
    from tensorflow.keras.regularizers import l2

    model = Sequential([
        # First LSTM layer with more units and regularization
        layers.LSTM(128, return_sequences=True, input_shape=input_shape,
                   dropout=dropout, recurrent_dropout=dropout,
                   kernel_regularizer=l2(0.001)),

        # Batch normalization for stable training
        layers.BatchNormalization(),

        # Second LSTM layer
        layers.LSTM(64, return_sequences=True,
                   dropout=dropout, recurrent_dropout=dropout,
                   kernel_regularizer=l2(0.001)),

        # Third LSTM layer
        layers.LSTM(32, dropout=dropout, recurrent_dropout=dropout),

        # Dense layers with regularization
        layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        layers.Dropout(dropout),
        layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        layers.Dropout(dropout),

        # Output layer
        layers.Dense(n_outputs)
    ])

    return model

# === IMPROVED TRAINING CONFIGURATION ===

def get_improved_training_config():
    """
    Better training configuration for forex
    """
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    optimizer = Adam(
        learning_rate=0.001,  # Lower learning rate
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=25,  # More patience
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,  # Reduce LR more often
            min_lr=1e-7,
            verbose=1
        )
    ]

    return optimizer, callbacks

# === IMPROVED DATA PREPROCESSING ===

def improve_feature_selection(dataset):
    """
    Select only the most predictive features for forex
    """
    import pandas as pd

    # Core forex indicators (remove noise)
    core_features = [
        # Price data
        'Close', 'Open', 'High', 'Low',

        # Key momentum indicators
        'RSI_14', 'RSI_21', 'MACD', 'MACD_Signal', 'MACD_Histogram',
        'Stoch_K', 'Stoch_D',

        # Trend indicators
        'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
        'Price_vs_SMA20', 'Price_vs_SMA50',

        # Volatility
        'BB_Upper', 'BB_Lower', 'BB_Position', 'BB_Width', 'ATR',

        # Price changes
        'Price_Change', 'Price_Change_5d',

        # Time features (important for forex)
        'DayOfWeek', 'IsMonday', 'IsFriday',

        # Pattern recognition
        'Body_Size', 'Is_Bullish'
    ]

    # Filter to available features
    available_features = [col for col in core_features if col in dataset.columns]
    target_cols = [col for col in dataset.columns if col.startswith('Target_')]

    selected_dataset = dataset[available_features + target_cols].copy()

    print(f"Feature selection: {len(dataset.columns)} → {len(available_features)} features")
    return selected_dataset

def add_forex_specific_features(dataset):
    """
    Add features specifically designed for forex prediction
    """
    import numpy as np

    df = dataset.copy()

    # === VOLATILITY REGIME DETECTION ===
    df['Volatility_Regime'] = (df['ATR'] > df['ATR'].rolling(20).mean()).astype(int)

    # === TREND STRENGTH ===
    df['Trend_Strength'] = abs(df['Price_vs_SMA20']) + abs(df['Price_vs_SMA50'])

    # === MOMENTUM CONFLUENCE ===
    rsi_bullish = (df['RSI_14'] > 50).astype(int)
    macd_bullish = (df['MACD'] > df['MACD_Signal']).astype(int)
    price_bullish = (df['Close'] > df['SMA_20']).astype(int)

    df['Momentum_Confluence'] = rsi_bullish + macd_bullish + price_bullish

    # === SUPPORT/RESISTANCE LEVELS ===
    df['Near_BB_Upper'] = (df['BB_Position'] > 0.8).astype(int)
    df['Near_BB_Lower'] = (df['BB_Position'] < 0.2).astype(int)

    # === VOLATILITY BREAKOUT ===
    df['Low_Volatility'] = (df['BB_Width'] < df['BB_Width'].rolling(20).quantile(0.2)).astype(int)
    df['High_Volatility'] = (df['BB_Width'] > df['BB_Width'].rolling(20).quantile(0.8)).astype(int)

    print(f"Added forex-specific features: {dataset.shape[1]} → {df.shape[1]} columns")
    return df

# === IMPROVED TARGET CREATION ===

def create_classification_targets(dataset, threshold_percentile=60):
    """
    Create better classification targets for forex
    """
    df = dataset.copy()

    for horizon in [1, 5, 10]:
        target_col = f'Target_Return_{horizon}d'
        if target_col in df.columns:

            # Calculate dynamic thresholds based on volatility
            abs_returns = abs(df[target_col])
            threshold = abs_returns.quantile(threshold_percentile / 100)

            # Strong up movement
            df[f'Target_Strong_Up_{horizon}d'] = (df[target_col] > threshold).astype(int)

            # Strong down movement
            df[f'Target_Strong_Down_{horizon}d'] = (df[target_col] < -threshold).astype(int)

            # Sideways market (no strong movement)
            df[f'Target_Sideways_{horizon}d'] = (
                (abs(df[target_col]) <= threshold).astype(int)
            )

    return df

# === ENSEMBLE PREDICTION ===

def create_ensemble_prediction(models, X_test):
    """
    Combine predictions from multiple models
    """
    import numpy as np

    predictions = []
    for model in models:
        pred = model.predict(X_test, verbose=0)
        predictions.append(pred)

    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)

    # Also get prediction confidence (standard deviation)
    pred_std = np.std(predictions, axis=0)

    return ensemble_pred, pred_std

# === USAGE EXAMPLE ===

def improved_forex_pipeline():
    """
    Example of how to use these improvements
    """

    # 1. Load and improve dataset
    from forex_data_robust import prepare_forex_dataset
    dataset = prepare_forex_dataset()

    # 2. Feature engineering
    dataset = add_forex_specific_features(dataset)
    dataset = improve_feature_selection(dataset)
    dataset = create_classification_targets(dataset)

    # 3. Use improved model
    # model = build_improved_forex_lstm(input_shape, n_outputs)
    # optimizer, callbacks = get_improved_training_config()

    print("✅ Improved forex pipeline ready")
    return dataset

if __name__ == "__main__":
    print("🚀 Forex Model Improvements")
    print("=" * 50)
    print("1. Enhanced LSTM architecture with regularization")
    print("2. Better feature selection (remove noise)")
    print("3. Forex-specific feature engineering")
    print("4. Classification targets for better signals")
    print("5. Ensemble methods for robustness")
    print("=" * 50)

    # Test improvements
    try:
        dataset = improved_forex_pipeline()
        print(f"\n✅ Improved dataset ready: {dataset.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")
