"""
Visualizations Module - Essential Charts for ML Project
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

plt.style.use('seaborn-v0_8-whitegrid')


# =============================================================================
# 1. TARGET DISTRIBUTION
# =============================================================================
def plot_target_distribution(dataset, horizons=[1, 5], save_path='viz_target_dist.png'):
    """Visualizes class balance for each prediction horizon."""
    fig, axes = plt.subplots(1, len(horizons), figsize=(5*len(horizons), 4))
    if len(horizons) == 1: axes = [axes]
    
    for i, h in enumerate(horizons):
        target = dataset[f'target_{h}d']
        counts = target.value_counts().sort_index()
        axes[i].bar(['Down', 'Up'], counts.values, color=['#e74c3c', '#2ecc71'], edgecolor='black')
        for j, v in enumerate(counts.values):
            axes[i].text(j, v + 20, f'{v/len(target)*100:.1f}%', ha='center')
        axes[i].set_title(f'{h}-day Horizon')
        axes[i].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

# =============================================================================
# 2. CORRELATION WITH TARGET
# =============================================================================
def plot_correlation_analysis(dataset, top_n=15, save_path='viz_correlation.png'):
    """Shows top features correlated with target."""
    feature_cols = [c for c in dataset.columns if not c.startswith('target_') and not c.startswith('real_return')]
    
    if 'target_5d' not in dataset.columns:
        print("  No target_5d column found")
        return
    
    corr = dataset[feature_cols + ['target_5d']].corr()['target_5d'].drop('target_5d')
    top = corr.abs().nlargest(top_n)
    
    plt.figure(figsize=(10, 6))
    colors = ['#2ecc71' if corr[idx] > 0 else '#e74c3c' for idx in top.index]
    plt.barh(range(len(top)), [corr[idx] for idx in top.index], color=colors)
    plt.yticks(range(len(top)), top.index)
    plt.xlabel('Correlation with Target (5d)')
    plt.title('Top Features Correlated with Target')
    plt.axvline(x=0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

# =============================================================================
# 3. LEARNING CURVES (Overfitting Detection)
# =============================================================================
def plot_learning_curves(history, save_path='viz_learning_curves.png'):
    """Plots training vs validation loss and accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history.history['loss'], 'b-', label='Train')
    axes[0].plot(epochs, history.history['val_loss'], 'r-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    
    # Accuracy
    axes[1].plot(epochs, history.history['accuracy'], 'b-', label='Train')
    axes[1].plot(epochs, history.history['val_accuracy'], 'r-', label='Validation')
    axes[1].axhline(y=0.5, color='gray', linestyle='--', label='Random')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
    
    # Return overfitting info
    best_epoch = np.argmin(history.history['val_loss']) + 1
    return {'best_epoch': best_epoch}


# =============================================================================
# 4. ROC CURVES
# =============================================================================
def plot_roc_curves(y_true, probas_dict, save_path='viz_roc.png'):
    """Plots ROC curves for all models."""
    plt.figure(figsize=(8, 6))
    
    for name, y_proba in probas_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# 5. BACKTEST EQUITY CURVE
# =============================================================================
def plot_backtest(y_pred, returns, save_path='viz_backtest.png'):
    """Plots equity curves for AI strategy vs Buy & Hold."""
    capital_market, capital_ai = 1000.0, 1000.0
    hist_market, hist_ai = [1000], [1000]
    n_trades, wins = 0, 0
    
    # Step every 5 days to avoid overlapping 5-day returns
    for i in range(0, len(y_pred), 5):
        ret = returns[i] if not np.isnan(returns[i]) else 0
        
        # Buy & Hold: always invested
        capital_market *= (1 + ret)
        
        if y_pred[i] == 1:  # Buy signal
            gain = ret - 0.0002  # Transaction cost
            capital_ai *= (1 + gain)
            n_trades += 1
            if ret > 0: wins += 1
        # else: stay in cash (no gain, no loss)
        
        hist_market.append(capital_market)
        hist_ai.append(capital_ai)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Equity curves
    axes[0].plot(hist_market, label='Buy & Hold', alpha=0.7)
    axes[0].plot(hist_ai, label='AI Strategy', linewidth=2)
    axes[0].set_title('Equity Curves (5-day periods)')
    axes[0].set_xlabel('Period (5-day intervals)')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].legend()
    
    # Stats
    axes[1].axis('off')
    market_ret = (hist_market[-1] - 1000) / 1000 * 100
    ai_ret = (hist_ai[-1] - 1000) / 1000 * 100
    win_rate = wins / n_trades * 100 if n_trades > 0 else 0
    
    stats = f"""
    BACKTEST RESULTS
    ================
    AI Return:      {ai_ret:+.2f}%
    Market Return:  {market_ret:+.2f}%
    Outperformance: {ai_ret - market_ret:+.2f}%
    
    Trades: {n_trades}
    Win Rate: {win_rate:.1f}%
    """
    axes[1].text(0.1, 0.5, stats, fontsize=12, family='monospace', va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
    
    return {'ai_return': ai_ret, 'market_return': market_ret, 'win_rate': win_rate, 'n_trades': n_trades}