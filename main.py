"""
Main script for Bitcoin price prediction
"""

import os
from datetime import datetime
import pickle
import numpy as np
# Pas besoin de class_weight ici
# from sklearn.utils import class_weight 

from data_loader import prepare_dataset
from preprocessing import create_sequences, split_temporal, normalize_features
from model import build_classifier, compile_model, get_callbacks
from train import train_model, display_training_results
from evaluate import predict_and_evaluate, show_sample_predictions, analyze_by_confidence


def main():
    """Main execution pipeline"""
    
    print("="*60)
    print("BITCOIN PRICE PREDICTION - CLASSIFICATION")
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
        horizons=HORIZONS,
        percentile=40 # <-- Vérifie que c'est bien 40 (ou 30)
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
    
    # Step 4: Normalize
    X_train_norm, X_val_norm, X_test_norm, scaler = normalize_features(
        X_train, X_val, X_test
    )
    
    # Step 5: Build model
    model = build_classifier(
        input_shape=(X_train_norm.shape[1], X_train_norm.shape[2]),
        n_horizons=len(horizons),
        units=64,
        dropout=0.3
    )
    model = compile_model(model, learning_rate=0.001)
    model.summary()
    
    # --- Il n'y a PAS de calcul de sample_weight ici ---

    # Step 6: Train
    print("\nStep 6: Training...")
    callbacks = get_callbacks(patience=20)
    
    # --- APPEL CORRECT ---
    # On passe X_val_norm et y_val séparément
    history = train_model(
        model,
        X_train_norm, y_train,
        X_val_norm, y_val, # <-- C'est ici
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
        # Pas de class_weight ou sample_weight
    )
    display_training_results(history)
    
    # Step 7: Evaluate
    results = predict_and_evaluate(model, X_test_norm, y_test, horizons)
    
    # Step 8: Analyze
    show_sample_predictions(results, horizons, n_samples=10)
    analyze_by_confidence(results, horizons)
    
    # Step 9: Save model
    save_dir = f"saved_models/bitcoin_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    model.save(f"{save_dir}/model.keras")
    
    with open(f"{save_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nModel saved to: {save_dir}")
    print("\nPIPELINE COMPLETED SUCCESSFULLY")
    
    return model, results, history


if __name__ == "__main__":
    main()