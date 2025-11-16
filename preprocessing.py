"""
Data preprocessing: sequences creation and normalization
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def create_sequences(data, horizons, window_size=60):
    """
    Create time series sequences for LSTM
    """
    print(f"\nCreating sequences with window_size={window_size}")
    
    feature_cols = [c for c in data.columns if not c.startswith('target_')]
    target_cols = [f'target_{h}d' for h in horizons]
    
    features = data[feature_cols].values
    targets = data[target_cols].values
    
    X, y = [], []
    
    # Note: range stops at len(data)
    for i in range(window_size, len(data)):
        X.append(features[i-window_size:i])
        y.append(targets[i]) # Target is at step 'i'
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"  X shape: {X.shape}") # (samples, window_size, n_features)
    print(f"  y shape: {y.shape}") # (samples, n_horizons)
    
    return X, y, feature_cols, target_cols


def split_temporal(X, y, test_size=0.2, val_size=0.1):
    """
    Split data temporally (no shuffling)
    """
    n = len(X)
    
    # Calculate split indices
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))
    
    X_train, y_train = X[:val_idx], y[:val_idx]
    X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
    X_test, y_test = X[test_idx:], y[test_idx:]
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_features(X_train, X_val, X_test):
    """
    Normalize feature sequences using MinMaxScaler
    Fit only on training data
    """
    print("\nNormalizing features...")
    
    n_samples, n_timesteps, n_features = X_train.shape
    
    # Reshape to 2D for scaling: (samples * timesteps, features)
    X_train_2d = X_train.reshape(-1, n_features)
    
    scaler = MinMaxScaler()
    scaler.fit(X_train_2d) # Fit ONLY on training data
    
    # Transform all splits
    X_train_norm = scaler.transform(X_train_2d).reshape(X_train.shape)
    X_val_norm = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_norm = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    
    print("Features normalized")
    
    return X_train_norm, X_val_norm, X_test_norm, scaler


if __name__ == "__main__":
    print("Preprocessing module - Use with main.py")