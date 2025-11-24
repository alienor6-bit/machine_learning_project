import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Fix for "main thread is not in main loop" error
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ==================================================================================
# 6. ANALYSIS & EVALUATION
# ==================================================================================

def perform_eda(dataset):
    """
    Exploratory Data Analysis (EDA).
    Visualizes target distribution and feature correlations.
    """
    print("Performing EDA...")
    # 1. Target Balance
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target_5d', data=dataset)
    plt.title('Target Distribution (5-day Horizon)')
    plt.savefig('eda_target_dist.png')
    plt.close()
    
    # 2. Correlation Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(dataset.corr(), cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.savefig('eda_correlation.png')
    plt.close()
    print("  ✓ EDA plots saved")

def run_baseline_model(X_train, y_train, X_test, y_test, horizon_index=1):
    """
    Runs a Random Forest baseline to compare against the LSTM.
    If LSTM doesn't beat this, the complexity isn't justified.
    """
    # Flatten 3D input (Samples, Time, Features) -> 2D (Samples, Time*Features)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Select target for specific horizon
    y_train_h = y_train[:, horizon_index] if y_train.ndim > 1 else y_train
    y_test_h = y_test[:, horizon_index] if y_test.ndim > 1 else y_test
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_flat, y_train_h)
    
    y_pred = clf.predict(X_test_flat)
    acc = accuracy_score(y_test_h, y_pred)
    return acc

def get_optimal_threshold(model, X_val, horizon_index=1):
    """
    Finds the best probability threshold (e.g., 0.6 instead of 0.5)
    to maximize accuracy on the validation set.
    """
    probs = model.predict(X_val, verbose=0)
    probs = probs[:, horizon_index] if probs.ndim > 1 else probs
    return 0.5 # Simplified for now

def predict_and_evaluate(model, X_test, y_test, horizons, returns_test, confidence_threshold=0.6):
    """
    Generates predictions and evaluates performance with a Confidence Filter.
    
    Confidence Filter:
    - Only trade if model probability > threshold (e.g., 0.65).
    - If probability is between 0.35 and 0.65, we stay in CASH.
    """
    probs = model.predict(X_test, verbose=0)
    results = {}
    
    for i, h in enumerate(horizons):
        p = probs[:, i] if probs.ndim > 1 else probs
        y_true = y_test[:, i] if y_test.ndim > 1 else y_test
        
        # Apply Confidence Threshold
        # 1 = Buy (High confidence Up)
        # 0 = Cash/Short (High confidence Down) - In Long-Only we treat this as Cash
        # -1 = No Trade (Low confidence)
        
        y_pred_filtered = np.zeros_like(p)
        y_pred_filtered[:] = -1 # Default to No Trade
        
        y_pred_filtered[p > confidence_threshold] = 1 # Buy
        y_pred_filtered[p < (1 - confidence_threshold)] = 0 # Sell/Cash
        
        # Calculate Accuracy only on traded samples
        mask = y_pred_filtered != -1
        if np.sum(mask) > 0:
            acc = accuracy_score(y_true[mask], y_pred_filtered[mask])
        else:
            acc = 0.0
            
        results[h] = {
            'accuracy': acc,
            'n_trades': np.sum(mask),
            'y_pred': y_pred_filtered,
            'returns': returns_test[:, i] if returns_test.ndim > 1 else returns_test
        }
        
        print(f"Horizon {h}d: Trades {np.sum(mask)}/{len(p)} ({np.mean(mask):.1%}) | Accuracy {acc:.2%}")
        
    return results

def run_backtest(results, horizon=5, transaction_cost=0.0002):
    """
    Simulates trading based on model predictions.
    
    Strategy:
    - Buy if Prediction = 1
    - Cash if Prediction = 0 (Long-Only)
    - Cash if Prediction = -1 (No Trade)
    """
    r = results[horizon]
    y_pred = r['y_pred']
    real_returns = np.nan_to_num(r['returns'], nan=0.0)
    
    capital_market = 1000.0
    capital_model = 1000.0
    
    hist_market = []
    hist_model = []
    
    n_trades = 0
    
    for i in range(len(y_pred)):
        ret = real_returns[i]
        
        # Market Performance (Buy & Hold)
        capital_market *= (1 + ret)
        
        # Model Performance
        daily_gain = 0
        if y_pred[i] == 1:
            # BUY
            daily_gain = ret - transaction_cost
            n_trades += 1
        else:
            # CASH (Stay out of market)
            # We do NOT short stocks in this strategy
            daily_gain = 0
            
        capital_model *= (1 + daily_gain)
        hist_market.append(capital_market)
        hist_model.append(capital_model)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(hist_market, label='Market (Buy & Hold)', alpha=0.6)
    plt.plot(hist_model, label='AI Model', linewidth=2)
    plt.title(f'Backtest (Horizon {horizon}d) - Initial Capital $1000')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('backtest_result.png')
    plt.close()
    
    print(f"Backtest H{horizon}: Market {((capital_market-1000)/1000):+.2%} | AI {((capital_model-1000)/1000):+.2%} | Trades {n_trades}")
