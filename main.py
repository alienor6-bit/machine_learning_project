"""
Main script for EUR/USD Forex Prediction using LSTM
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import pickle

# Add current directory to path to import our modules
sys.path.append('.')

# Import our custom modules
from forex_data import prepare_forex_dataset
from preprocessing import clean_indicators, create_sequences, normalize_sequences_split, prepare_train_test_split
from model import build_lstm_model, get_callbacks
from train import train_model
from evaluation import evaluate_model, plot_predictions, calculate_trading_metrics

def create_sequences_for_forex(df, window_size=60, horizons=[1, 5, 10, 20]):
    """
    Adapted create_sequences function for forex data
    """
    print(f"Creating sequences for forex data...")
    print(f"DataFrame shape: {df.shape}")

    # Target columns
    target_cols = [f'Target_Return_{h}d' for h in horizons]
    target_cols = [col for col in target_cols if col in df.columns]

    if not target_cols:
        raise ValueError(f"No target columns found. Available columns: {list(df.columns)}")

    # Feature columns (all except targets)
    feature_cols = [col for col in df.columns if 'Target_' not in col]

    print(f"Features: {len(feature_cols)}")
    print(f"Targets: {len(target_cols)}")

    # Prepare data
    features = df[feature_cols].values
    targets = df[target_cols].values

    # Create sequences
    X, y = [], []

    for i in range(window_size, len(df)):
        # Features: past window_size days
        seq_features = features[i-window_size:i]
        # Target: current day's target (future returns)
        seq_target = targets[i]

        # Check for NaN values
        if (not np.isnan(seq_features).any() and
            not np.isnan(seq_target).any()):
            X.append(seq_features)
            y.append(seq_target)

    X = np.array(X)
    y = np.array(y)

    print(f"Final sequences - X shape: {X.shape}, y shape: {y.shape}")
    return X, y, feature_cols, target_cols

def prepare_forex_sequences(dataset, window_size=60, test_size=0.2, val_size=0.1):
    """
    Prepare sequences and splits for forex data
    """
    print(f"\n{'='*50}")
    print("PRÉPARATION DES SÉQUENCES FOREX")
    print(f"{'='*50}")

    # Create sequences
    X, y, feature_cols, target_cols = create_sequences_for_forex(
        dataset,
        window_size=window_size
    )

    if len(X) == 0:
        raise ValueError("No sequences created. Check your data.")

    print(f"\nSéquences créées:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Targets: {len(target_cols)}")

    # Split data temporally (important for time series)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_test_split(
        X, y, test_size=test_size, val_size=val_size
    )

    # Normalize sequences
    X_train_norm, X_val_norm, X_test_norm, scaler = normalize_sequences_split(
        X_train, X_val, X_test, method='minmax'
    )

    # Package everything
    sequences_data = {
        'EURUSD': {
            'X_train': X_train_norm,
            'X_val': X_val_norm,
            'X_test': X_test_norm,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'target_cols': target_cols
        }
    }

    return sequences_data

def train_forex_model(sequences_data, epochs=100, batch_size=32):
    """
    Train LSTM model for forex prediction
    """
    print(f"\n{'='*50}")
    print("ENTRAÎNEMENT DU MODÈLE FOREX")
    print(f"{'='*50}")

    data = sequences_data['EURUSD']
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    n_outputs = y_train.shape[1]

    print(f"Input shape: {input_shape}")
    print(f"Output dimensions: {n_outputs}")

    model = build_lstm_model(input_shape, n_outputs, units=64, dropout=0.3)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Display model architecture
    model.summary()

    # Train model
    callbacks = get_callbacks(patience=20)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, history

def save_results(model, sequences_data, history, metrics, timestamp):
    """
    Save model and results
    """
    results_dir = f"/mnt/user-data/outputs/forex_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Save model
    model.save(f"{results_dir}/forex_lstm_model.h5")

    # Save scaler
    with open(f"{results_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(sequences_data['EURUSD']['scaler'], f)

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f"{results_dir}/training_history.csv", index=False)

    # Save metrics
    with open(f"{results_dir}/evaluation_metrics.pkl", 'wb') as f:
        pickle.dump(metrics, f)

    print(f"\nRésultats sauvegardés dans: {results_dir}")
    return results_dir

def main():
    """
    Main execution pipeline
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("🚀 DÉMARRAGE DU PIPELINE DE PRÉDICTION FOREX EUR/USD")
    print(f"Timestamp: {timestamp}")

    try:
        # Step 1: Prepare forex dataset
        print("\n" + "="*60)
        print("ÉTAPE 1: PRÉPARATION DES DONNÉES")
        print("="*60)

        dataset = prepare_forex_dataset()
        print(f"Dataset final: {dataset.shape}")
        print(f"Période: {dataset.index[0]} à {dataset.index[-1]}")

        # Step 2: Create sequences
        print("\n" + "="*60)
        print("ÉTAPE 2: CRÉATION DES SÉQUENCES")
        print("="*60)

        sequences_data = prepare_forex_sequences(
            dataset,
            window_size=60,  # 60 days of history
            test_size=0.2,   # 20% for testing
            val_size=0.1     # 10% for validation
        )

        # Step 3: Train model
        print("\n" + "="*60)
        print("ÉTAPE 3: ENTRAÎNEMENT")
        print("="*60)

        model, history = train_forex_model(
            sequences_data,
            epochs=100,
            batch_size=32
        )

        # Step 4: Evaluate model
        print("\n" + "="*60)
        print("ÉTAPE 4: ÉVALUATION")
        print("="*60)

        metrics = evaluate_model(model, sequences_data, ticker="EURUSD")

        # Step 5: Trading analysis
        print("\n" + "="*60)
        print("ÉTAPE 5: ANALYSE DE TRADING")
        print("="*60)

        # Get test predictions for trading analysis
        test_data = sequences_data['EURUSD']
        X_test, y_test = test_data['X_test'], test_data['y_test']
        y_pred = model.predict(X_test, verbose=0)

        trading_metrics = calculate_trading_metrics(y_test, y_pred)

        for horizon, metrics_dict in trading_metrics.items():
            print(f"\n{horizon}:")
            for metric, value in metrics_dict.items():
                print(f"  {metric}: {value:.4f}")

        # Step 6: Save results
        print("\n" + "="*60)
        print("ÉTAPE 6: SAUVEGARDE")
        print("="*60)

        results_dir = save_results(model, sequences_data, history, metrics, timestamp)

        print("\n🎉 PIPELINE TERMINÉ AVEC SUCCÈS!")
        print(f"📊 Résultats disponibles dans: {results_dir}")

        return model, sequences_data, history, metrics

    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    model, sequences_data, history, metrics = main()
