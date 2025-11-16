"""
Model evaluation for regression
"""

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def predict_and_evaluate(model, X_test, y_test_norm, horizons, target_scaler):
    """
    Make predictions and calculate regression metrics
    
    Args:
        model: Trained Keras model
        X_test: Test features (normalized)
        y_test_norm: Test targets (normalized)
        horizons: List of prediction horizons
        target_scaler: The fitted scaler for targets
    
    Returns:
        results: Dictionary with predictions and metrics
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION (REGRESSION)")
    print("="*60)
    
    # Make predictions (normalized)
    print("\nGenerating predictions...")
    y_pred_norm = model.predict(X_test, verbose=0)
    
    # --- CRITICAL: Inverse transform to get real values ---
    y_pred_real = target_scaler.inverse_transform(y_pred_norm)
    y_true_real = target_scaler.inverse_transform(y_test_norm)
    
    results = {}
    
    for i, h in enumerate(horizons):
        # Extract real values for this horizon
        y_true_h = y_true_real[:, i]
        y_pred_h = y_pred_real[:, i]
        
        # Calculate metrics
        r2 = r2_score(y_true_h, y_pred_h)
        mae = mean_absolute_error(y_true_h, y_pred_h)
        mse = mean_squared_error(y_true_h, y_pred_h)
        
        # Store results
        results[h] = {
            'y_true': y_true_h,
            'y_pred': y_pred_h,
            'r2_score': r2,
            'mae': mae,
            'mse': mse
        }
        
        # Display metrics
        print(f"\n{h}-day predictions:")
        print(f"  R-squared (R²): {r2:.3f}")
        print(f"  Mean Absolute Error (MAE): {mae:.2%}")
        print(f"  (Le modèle se trompe en moyenne de +/- {mae:.2%})")
        print(f"  Root Mean Squared Error (RMSE): {np.sqrt(mse):.2%}")
    
    return results


def show_sample_predictions(results, horizons, n_samples=10):
    """
    Display sample predictions (real values)
    
    Args:
        results: Results dictionary from predict_and_evaluate
        horizons: List of prediction horizons
        n_samples: Number of samples to show
    """
    print("\n" + "="*60)
    print(f"SAMPLE PREDICTIONS (last {n_samples})")
    print("="*60)
    
    for h in horizons:
        r = results[h]
        
        print(f"\n{h}-day predictions (% change in volume):")
        print(f"{'True Value':<15} {'Predicted Value':<15} {'Error':<10}")
        print("-" * 45)
        
        # Iterate from the end of the dataset
        for i in range(-n_samples, 0):
            true_val = r['y_true'][i]
            pred_val = r['y_pred'][i]
            error = true_val - pred_val
            
            print(f"{true_val:>+14.2%} {pred_val:>+16.2%} {error:>+9.2%}")