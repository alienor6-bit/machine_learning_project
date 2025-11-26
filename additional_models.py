"""
Additional Models for Comparison
=================================
This module implements simple baseline models to compare against the LSTM.
According to ML guidelines, we must compare our advanced model (LSTM) 
with simpler approaches.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================================================
# 1. SIMPLE MODELS (BASELINES)
# ==================================================================================

def run_logistic_regression(X_train, y_train, X_test, y_test, horizon_index=1):
    """
    Logistic Regression - The simplest classification algorithm.
    
    Why test this?
    - If LSTM doesn't beat this, deep learning is overkill.
    - Logistic Regression assumes linear relationships.
    """
    print("\n" + "="*60)
    print("MODEL 1: LOGISTIC REGRESSION")
    print("="*60)
    
    # Flatten 3D to 2D
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Select target
    y_train_h = y_train[:, horizon_index] if y_train.ndim > 1 else y_train
    y_test_h = y_test[:, horizon_index] if y_test.ndim > 1 else y_test
    
    # Train
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_flat, y_train_h)
    
    # Predict
    y_pred = clf.predict(X_test_flat)
    acc = accuracy_score(y_test_h, y_pred)
    
    print(f"Accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test_h, y_pred, target_names=['Down', 'Up']))
    
    return acc

def run_random_forest(X_train, y_train, X_test, y_test, horizon_index=1):
    """
    Random Forest - Strong non-linear baseline.
    
    Why test this?
    - Handles non-linear relationships well.
    - No hyperparameter tuning needed (robust defaults).
    - If RF beats LSTM, LSTM is not learning temporal patterns.
    """
    print("\n" + "="*60)
    print("MODEL 2: RANDOM FOREST")
    print("="*60)
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    y_train_h = y_train[:, horizon_index] if y_train.ndim > 1 else y_train
    y_test_h = y_test[:, horizon_index] if y_test.ndim > 1 else y_test
    
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train_flat, y_train_h)
    
    y_pred = clf.predict(X_test_flat)
    acc = accuracy_score(y_test_h, y_pred)
    
    print(f"Accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test_h, y_pred, target_names=['Down', 'Up']))
    
    # Feature Importance (Top 10)
    feature_importance = clf.feature_importances_
    top_10_idx = np.argsort(feature_importance)[-10:]
    
    print("\nTop 10 Most Important Features:")
    for idx in reversed(top_10_idx):
        print(f"  Feature {idx}: {feature_importance[idx]:.4f}")
    
    return acc, feature_importance

def run_xgboost(X_train, y_train, X_test, y_test, horizon_index=1):
    """
    XGBoost - State-of-the-art gradient boosting.
    
    Why test this?
    - Often beats neural networks on tabular data.
    - Industry standard for structured data.
    """
    print("\n" + "="*60)
    print("MODEL 3: XGBOOST")
    print("="*60)
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    y_train_h = y_train[:, horizon_index] if y_train.ndim > 1 else y_train
    y_test_h = y_test[:, horizon_index] if y_test.ndim > 1 else y_test
    
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    clf.fit(X_train_flat, y_train_h, verbose=False)
    
    y_pred = clf.predict(X_test_flat)
    acc = accuracy_score(y_test_h, y_pred)
    
    print(f"Accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test_h, y_pred, target_names=['Down', 'Up']))
    
    return acc

# ==================================================================================
# 2. ENSEMBLE MODEL (VOTING CLASSIFIER)
# ==================================================================================

def run_ensemble_model(X_train, y_train, X_test, y_test, lstm_model, horizon_index=1):
    """
    Ensemble: Combines Random Forest + XGBoost + LSTM predictions.
    
    Strategy:
    - Each model votes (0 or 1)
    - Final prediction = Majority vote (2 out of 3)
    
    Why?
    - Reduces individual model biases.
    - Often improves accuracy vs single models.
    """
    print("\n" + "="*60)
    print("MODEL 4: ENSEMBLE (RF + XGBoost + LSTM)")
    print("="*60)
    
    # Prepare data
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    y_train_h = y_train[:, horizon_index] if y_train.ndim > 1 else y_train
    y_test_h = y_test[:, horizon_index] if y_test.ndim > 1 else y_test
    
    # Train RF
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_flat, y_train_h)
    rf_pred = rf.predict(X_test_flat)
    
    # Train XGBoost
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss')
    xgb.fit(X_train_flat, y_train_h, verbose=False)
    xgb_pred = xgb.predict(X_test_flat)
    
    # LSTM predictions
    lstm_probs = lstm_model.predict(X_test, verbose=0)
    lstm_probs_h = lstm_probs[:, horizon_index] if lstm_probs.ndim > 1 else lstm_probs
    lstm_pred = (lstm_probs_h > 0.5).astype(int)
    
    # Majority Voting
    votes = np.array([rf_pred, xgb_pred, lstm_pred])
    ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=votes)
    
    # Evaluate
    acc = accuracy_score(y_test_h, ensemble_pred)
    
    print(f"Ensemble Accuracy: {acc:.2%}")
    print("\nIndividual Model Accuracies:")
    print(f"  Random Forest: {accuracy_score(y_test_h, rf_pred):.2%}")
    print(f"  XGBoost:       {accuracy_score(y_test_h, xgb_pred):.2%}")
    print(f"  LSTM:          {accuracy_score(y_test_h, lstm_pred):.2%}")
    
    print("\nClassification Report (Ensemble):")
    print(classification_report(y_test_h, ensemble_pred, target_names=['Down', 'Up']))
    
    return acc

# ==================================================================================
# 3. MODEL COMPARISON VISUALIZATION
# ==================================================================================

def compare_all_models(results_dict):
    """
    Creates a bar chart comparing all model accuracies.
    
    Args:
        results_dict: {'Model Name': accuracy}
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    # Sort by accuracy
    sorted_results = dict(sorted(results_dict.items(), key=lambda x: x[1], reverse=True))
    
    print(f"{'Model':<25} {'Accuracy'}")
    print("-"*40)
    for model, acc in sorted_results.items():
        print(f"{model:<25} {acc:>8.2%}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    models = list(sorted_results.keys())
    accuracies = list(sorted_results.values())
    
    colors = ['green' if acc > 0.55 else 'orange' if acc > 0.50 else 'red' for acc in accuracies]
    
    plt.barh(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('Accuracy')
    plt.title('Model Comparison (5-day Horizon)')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Random Baseline (50%)')
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.close()
    
    print("\n✓ Comparison chart saved: model_comparison.png")
    
    # Determine best model
    best_model = max(sorted_results, key=sorted_results.get)
    print(f"\n BEST MODEL: {best_model} ({sorted_results[best_model]:.2%})")
    