import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import precision_score, accuracy_score, recall_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, f1_score
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
    pres = precision_score(y_test_h, y_pred)
    rs = recall_score(y_test_h, y_pred)

    return acc, pres, rs

# def get_optimal_threshold(model, X_val, horizon_index=1):
#     """
#     Finds the best probability threshold (e.g., 0.6 instead of 0.5)
#     to maximize accuracy on the validation set.
#     """
#     probs = model.predict(X_val, verbose=0)
#     probs = probs[:, horizon_index] if probs.ndim > 1 else probs
#     return 0.5

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

def evaluate_classifier(y_true, y_pred, y_proba=None, model_name="Model", save_dir='.'):
    """
    Comprehensive classifier evaluation.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    if y_proba is not None:
        roc_auc = roc_auc_score(y_true, y_proba)
        print(f"ROC-AUC:   {roc_auc:.4f}")
    else:
        roc_auc = None

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Down', 'Up']))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(f'{save_dir}/confusion_matrix_{safe_name}.png', dpi=150)
    plt.close()

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc_auc}


def plot_roc_curves(results_dict, save_dir='.'):
    """
    Plots ROC curves for multiple models.
    """
    plt.figure(figsize=(8, 6))

    for model_name, data in results_dict.items():
        if 'y_proba' in data and data['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(data['y_true'], data['y_proba'])
            auc = roc_auc_score(data['y_true'], data['y_proba'])
            plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.500)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/roc_curves.png', dpi=150)
    plt.close()


def plot_training_history(history, save_dir='.'):
    """
    Plots training and validation loss/accuracy curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss Over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_title('Accuracy Over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_history.png', dpi=150)
    plt.close()
