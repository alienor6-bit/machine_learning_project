"""
Main script for Bitcoin VOLUME prediction (Regression)
"""

import os
from datetime import datetime
import pickle

from data_loader import prepare_dataset
from preprocessing import (create_sequences, split_temporal, normalize_features, normalize_targets)
from model import (build_regressor, compile_model, get_callbacks )
from train import train_model, display_training_results
from evaluate import (predict_and_evaluate, show_sample_predictions )
import tensorflow as tf

def main():
    """Main execution pipeline"""
    tf.keras.utils.set_random_seed(42)
    print("="*60)
    print("BITCOIN VOLUME PREDICTION - REGRESSION")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # --- Configuration ---
    WINDOW_SIZE = 60
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    EPOCHS = 100
    BATCH_SIZE = 32
    HORIZONS = [1, 3, 7, 14]
    
    # Step 1: Prepare data
    dataset, horizons = prepare_dataset(
        start_date='2018-01-01',
        horizons=HORIZONS
    )
    
    # Step 2: Create sequences
    X, y, feature_cols, target_cols = create_sequences(
        dataset,
        horizons=horizons,
        window_size=WINDOW_SIZE
    )
    
    # Step 3: Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_temporal(
        X, y,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE
    )
    
    # Step 4: Normalize Features (X)
    X_train_norm, X_val_norm, X_test_norm, feature_scaler = normalize_features(
        X_train, X_val, X_test
    )
    
    # --- NEW Step 4.5: Normalize Targets (y) ---
    y_train_norm, y_val_norm, y_test_norm, target_scaler = normalize_targets(
        y_train, y_val, y_test
    )
    
    # Step 5: Build model
    model = build_regressor(
        input_shape=(X_train_norm.shape[1], X_train_norm.shape[2]),
        n_horizons=len(horizons),
        units=64,
        dropout=0.4
    )
    model = compile_model(model, learning_rate=0.0001)
    model.summary()
    
    # Step 6: Train
    callbacks = get_callbacks(patience=20)
    history = train_model(
        model,
        X_train_norm, y_train_norm, # Pass normalized data
        X_val_norm, y_val_norm,     # Pass normalized data
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    display_training_results(history)
    
    # Step 7: Evaluate
    results = predict_and_evaluate(
        model, 
        X_test_norm, y_test_norm, # Pass normalized test data
        horizons, 
        target_scaler # <-- Pass the scaler for inverse transform
    )
    
    # Step 8: Show predictions
    show_sample_predictions(results, horizons, n_samples=10)
    
    # Step 9: Save model and scalers
    print("\nStep 9: Saving model...")
    save_dir = f"saved_models/volume_regressor_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    model.save(f"{save_dir}/model.keras")
    
    # Save feature scaler
    with open(f"{save_dir}/feature_scaler.pkl", 'wb') as f:
        pickle.dump(feature_scaler, f)
        
    # --- CRITICAL: Save target scaler ---
    with open(f"{save_dir}/target_scaler.pkl", 'wb') as f:
        pickle.dump(target_scaler, f)
    
    print(f"\nModel and scalers saved to: {save_dir}")
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return model, results, history


if __name__ == "__main__":
    main()