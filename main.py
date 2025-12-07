import os
import pickle
import numpy as np
import tensorflow as tf
from data import prepare_dataset, create_sequences, split_temporal, normalize_features
from model import build_classifier, compile_model, train_model, get_callbacks
from analysis import perform_eda, predict_and_evaluate, run_backtest
from additional_models import (run_logistic_regression,run_random_forest,run_xgboost,run_ensemble_model,compare_all_models
)

def main():
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)

    print("="*60)
    print("CAC 40 PREDICTION PROJECT - COMPLETE PIPELINE")
    print("="*60)

    # Configuration
    WINDOW_SIZE = 60
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    EPOCHS = 50
    BATCH_SIZE = 32
    HORIZONS = [1, 5]
    CONFIDENCE_THRESHOLD = 0.65

    # ========================================
    # STEP 1: DATA PREPARATION
    # ========================================
    print("\n" + "="*60)
    print("STEP 1: DATA PREPARATION")
    print("="*60)

    dataset = prepare_dataset(start_date='2000-01-01', horizons=HORIZONS)

    # ========================================
    # STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
    # ========================================
    print("\n" + "="*60)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("="*60)

    perform_eda(dataset)

    # Target Distribution Analysis
    target_5d = dataset['target_5d']
    n_up = (target_5d == 1).sum()
    n_down = (target_5d == 0).sum()

    print(f"\nTarget Distribution (5-day Horizon):")
    print(f"  Up (1):   {n_up} ({n_up/len(target_5d)*100:.1f}%)")
    print(f"  Down (0): {n_down} ({n_down/len(target_5d)*100:.1f}%)")

    if abs(n_up - n_down) / len(target_5d) > 0.1:
        print("  ⚠️  Dataset is IMBALANCED (>10% difference)")
    else:
        print("  ✓ Dataset is BALANCED")

    # ========================================
    # STEP 3: CREATE SEQUENCES & SPLIT
    # ========================================
    print("\n" + "="*60)
    print("STEP 3: SEQUENCE CREATION & DATA SPLIT")
    print("="*60)

    X, y, feature_cols = create_sequences(dataset, HORIZONS, WINDOW_SIZE)

    returns_cols = [f'real_return_{h}d' for h in HORIZONS]
    returns_aligned = dataset[returns_cols].values[WINDOW_SIZE:]

    X_train, X_val, X_test, y_train, y_val, y_test = split_temporal(X, y, TEST_SIZE, VAL_SIZE)
    test_idx = int(len(X) * (1 - TEST_SIZE))
    returns_test = returns_aligned[test_idx:]

    X_train_norm, X_val_norm, X_test_norm, scaler = normalize_features(X_train, X_val, X_test)

    # ========================================
    # STEP 3.5: DIMENSION REDUCTION (PCA)
    # ========================================
    print("\n" + "="*60)
    print("STEP 3.5: DIMENSION REDUCTION (PCA)")
    print("="*60)

    from data import apply_dimension_reduction

    X_train_pca, X_val_pca, X_test_pca, pca = apply_dimension_reduction(
        X_train_norm, X_val_norm, X_test_norm,
        method='pca',
        n_components=0.95
    )

    # ========================================
    # STEP 4: BASELINE MODELS (SIMPLE METHODS)
    # ========================================
    print("\n" + "="*60)
    print("STEP 4: BASELINE MODELS")
    print("="*60)
    print("Testing simple models to establish a performance baseline...")

    results_comparison = {}

    # Model 1: Logistic Regression
    lr_acc = run_logistic_regression(X_train_pca, y_train, X_test_pca, y_test, horizon_index=1)
    results_comparison['Logistic Regression'] = lr_acc

    # Model 2: Random Forest
    rf_acc, feature_importance = run_random_forest(X_train_pca, y_train, X_test_pca, y_test, horizon_index=1)
    results_comparison['Random Forest'] = rf_acc

    # Model 3: XGBoost
    xgb_acc = run_xgboost(X_train_pca, y_train, X_test_pca, y_test, horizon_index=1)
    results_comparison['XGBoost'] = xgb_acc

    # ========================================
    # STEP 5: LSTM MODEL (DEEP LEARNING)
    # ========================================
    print("\n" + "="*60)
    print("STEP 5: DEEP LEARNING (BIDIRECTIONAL LSTM)")
    print("="*60)

    model = build_classifier(
        input_shape=(X_train_norm.shape[1], X_train_norm.shape[2]),
        n_horizons=len(HORIZONS),
        units=24,
        dropout=0.2
    )
    model = compile_model(model, learning_rate=0.001)
    model.build(input_shape=(None, X_train_norm.shape[1], X_train_norm.shape[2]))

    print("\nModel Architecture:")
    model.summary()

    history = train_model(
        model, X_train_norm, y_train, X_val_norm, y_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=get_callbacks()
    )

    # LSTM Test Accuracy (Standard prediction without confidence filter)
    y_pred_lstm = model.predict(X_test_norm, verbose=0)
    y_pred_lstm_h = y_pred_lstm[:, 1] if y_pred_lstm.ndim > 1 else y_pred_lstm
    y_test_h = y_test[:, 1] if y_test.ndim > 1 else y_test
    lstm_pred_binary = (y_pred_lstm_h > 0.5).astype(int)

    from sklearn.metrics import accuracy_score
    lstm_acc = accuracy_score(y_test_h, lstm_pred_binary)
    results_comparison['LSTM (Bidirectional)'] = lstm_acc

    print(f"\nLSTM Test Accuracy (Standard): {lstm_acc:.2%}")

    # ========================================
    # STEP 6: ENSEMBLE MODEL
    # ========================================
    print("\n" + "="*60)
    print("STEP 6: ENSEMBLE MODEL (RF + XGBoost + LSTM)")
    print("="*60)

    ensemble_acc = run_ensemble_model(X_train_norm, y_train, X_test_norm, y_test, model, horizon_index=1)
    results_comparison['Ensemble (Voting)'] = ensemble_acc

    # ========================================
    # STEP 7: MODEL COMPARISON
    # ========================================
    print("\n" + "="*60)
    print("STEP 7: MODEL COMPARISON & ANALYSIS")
    print("="*60)

    compare_all_models(results_comparison)

    # ========================================
    # STEP 8: TRADING STRATEGY BACKTEST
    # ========================================
    print("\n" + "="*60)
    print("STEP 8: TRADING STRATEGY (Confidence Filter)")
    print("="*60)
    print("Now testing LSTM with Confidence Filter (65% threshold)...")

    results = predict_and_evaluate(
        model, X_test_norm, y_test, HORIZONS, returns_test,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

    run_backtest(results, horizon=5)

    # ========================================
    # STEP 9: WALK-FORWARD VALIDATION
    # ========================================
    print("\n" + "="*60)
    print("STEP 9: WALK-FORWARD VALIDATION")
    print("="*60)

    from walk_forward_validation import walk_forward_validation
    best_config = {
        'name': 'Best_LSTM',
        'units': 64,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'patience': 20,
        'confidence_threshold': 0.65
    }
    walk_forward_validation(best_config)

    # ========================================
    # STEP 10: SAVE MODELS
    # ========================================
    print("\n" + "="*60)
    print("STEP 10: SAVING MODELS")
    print("="*60)

    save_dir = f"saved_models/final"
    os.makedirs(save_dir, exist_ok=True)

    model.save(f"{save_dir}/lstm_model.keras")
    with open(f"{save_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    print(f"✓ LSTM model saved to: {save_dir}/lstm_model.keras")
    print(f"✓ Scaler saved to: {save_dir}/scaler.pkl")

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*60)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("="*60)
    print("\nGenerated Files:")
    print("  1. eda_target_dist.png - Target distribution")
    print("  2. eda_correlation.png - Feature correlation matrix")
    print("  3. model_comparison.png - Model accuracy comparison")
    print("  4. backtest_result.png - Trading backtest results")
    print("  5. results.txt - Walk-forward validation summary")
    print("\nKey Findings:")
    print(f"  Best Simple Model: {max(results_comparison, key=results_comparison.get)} ({max(results_comparison.values()):.2%})")
    print(f"  LSTM Performance: {lstm_acc:.2%}")
    print(f"  Ensemble Performance: {ensemble_acc:.2%}")
    print("\n✓ All models trained and evaluated successfully!")

if __name__ == "__main__":
    main()
