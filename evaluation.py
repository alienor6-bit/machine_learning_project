import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import matplotlib.pyplot as plt

def calculate_metrics(y_true, y_pred, horizons):
    """
    Calculate comprehensive metrics for time series forecasting
    """
    n_horizons = len(horizons)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    metrics = {}

    for i, horizon in enumerate(horizons):
        if i < y_true.shape[1] and i < y_pred.shape[1]:
            true_values = y_true[:, i]
            pred_values = y_pred[:, i]

            # Remove NaN values
            mask = ~(np.isnan(true_values) | np.isnan(pred_values))
            true_values = true_values[mask]
            pred_values = pred_values[mask]

            if len(true_values) > 0:
                # Regression metrics
                mse = mean_squared_error(true_values, pred_values)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(true_values, pred_values)

                # Direction accuracy
                true_direction = (true_values > 0).astype(int)
                pred_direction = (pred_values > 0).astype(int)
                direction_accuracy = accuracy_score(true_direction, pred_direction)

                # Custom forex metrics
                mean_return = np.mean(true_values)
                return_std = np.std(true_values)

                # Normalized metrics
                rmse_normalized = rmse / (return_std + 1e-8)
                mae_normalized = mae / (abs(mean_return) + 1e-8)

                metrics[f'{horizon}d'] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'Direction_Accuracy': direction_accuracy,
                    'RMSE_Normalized': rmse_normalized,
                    'MAE_Normalized': mae_normalized,
                    'Mean_True_Return': mean_return,
                    'Std_True_Return': return_std
                }

                print(f"    {horizon}d - RMSE: {rmse:.6f}, MAE: {mae:.6f}, Direction Acc: {direction_accuracy:.3f}")
            else:
                print(f"    {horizon}d - No valid predictions")

    return metrics

def evaluate_model(model, sequences_data, ticker="EURUSD"):
    """
    Evaluate the model on train/val/test splits and display comprehensive metrics
    """
    print(f"\n{'='*60}\nÉvaluation pour {ticker}\n{'='*60}")

    data = sequences_data[ticker]
    all_metrics = {}

    for split_name in ['train', 'val', 'test']:
        X_key = f'X_{split_name}'
        y_key = f'y_{split_name}'

        if X_key not in data or y_key not in data:
            print(f"\n{split_name.upper()}: Données manquantes")
            continue

        X, y_true = data[X_key], data[y_key]

        if len(X) == 0:
            print(f"\n{split_name.upper()}: Pas de données")
            continue

        print(f"\n{split_name.upper()}:")

        # Make predictions
        y_pred = model.predict(X, verbose=0)

        # Calculate metrics for each horizon
        horizons = ['1d', '5d', '10d', '20d']  # Adjust based on your targets
        metrics = calculate_metrics(y_true, y_pred, horizons)
        all_metrics[split_name] = metrics

        # Additional analysis for test set
        if split_name == 'test':
            analyze_predictions(y_true, y_pred, horizons)

    return all_metrics

def analyze_predictions(y_true, y_pred, horizons):
    """
    Detailed analysis of predictions
    """
    print("\n" + "-"*40)
    print("ANALYSE DÉTAILLÉE DES PRÉDICTIONS")
    print("-"*40)

    for i, horizon in enumerate(horizons):
        if i < y_true.shape[1] and i < y_pred.shape[1]:
            true_values = y_true[:, i]
            pred_values = y_pred[:, i]

            # Remove NaN values
            mask = ~(np.isnan(true_values) | np.isnan(pred_values))
            true_values = true_values[mask]
            pred_values = pred_values[mask]

            if len(true_values) > 0:
                print(f"\n{horizon} Horizon:")

                # Correlation
                correlation = np.corrcoef(true_values, pred_values)[0, 1]
                print(f"  Corrélation: {correlation:.4f}")

                # Profitable trades analysis
                profitable_signals = (pred_values > 0) & (true_values > 0)
                losing_signals = (pred_values > 0) & (true_values < 0)

                total_positive_signals = np.sum(pred_values > 0)
                if total_positive_signals > 0:
                    win_rate = np.sum(profitable_signals) / total_positive_signals
                    print(f"  Taux de réussite (signaux positifs): {win_rate:.3f}")

                # Return analysis
                avg_pred_return = np.mean(pred_values[pred_values > 0]) if np.any(pred_values > 0) else 0
                avg_true_return_on_signals = np.mean(true_values[pred_values > 0]) if np.any(pred_values > 0) else 0

                print(f"  Retour moyen prédit (signaux positifs): {avg_pred_return:.6f}")
                print(f"  Retour réel moyen (sur signaux positifs): {avg_true_return_on_signals:.6f}")

def plot_predictions(y_true, y_pred, horizon_idx=0, title="Prédictions vs Réalité"):
    """
    Plot predictions vs actual values
    """
    plt.figure(figsize=(12, 6))

    true_values = y_true[:, horizon_idx]
    pred_values = y_pred[:, horizon_idx]

    # Remove NaN values for plotting
    mask = ~(np.isnan(true_values) | np.isnan(pred_values))
    true_values = true_values[mask]
    pred_values = pred_values[mask]

    # Time series plot
    plt.subplot(1, 2, 1)
    plt.plot(true_values[:100], label='Réel', alpha=0.7)
    plt.plot(pred_values[:100], label='Prédit', alpha=0.7)
    plt.title(f'{title} - Série temporelle')
    plt.legend()
    plt.xlabel('Temps')
    plt.ylabel('Retour')

    # Scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(true_values, pred_values, alpha=0.5)
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Valeurs prédites')
    plt.title(f'{title} - Corrélation')

    plt.tight_layout()
    plt.show()

def calculate_trading_metrics(y_true, y_pred, initial_balance=10000, transaction_cost=0.0001):
    """
    Calculate trading-specific metrics
    """
    results = {}

    for i in range(y_pred.shape[1]):
        true_returns = y_true[:, i]
        predicted_returns = y_pred[:, i]

        # Simple trading strategy: buy if predicted return > 0
        positions = (predicted_returns > 0).astype(float)

        # Calculate returns
        strategy_returns = positions * true_returns

        # Apply transaction costs
        position_changes = np.abs(np.diff(positions, prepend=0))
        costs = position_changes * transaction_cost
        net_returns = strategy_returns - costs

        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + net_returns)
        final_balance = initial_balance * cumulative_returns[-1]

        # Calculate metrics
        total_return = cumulative_returns[-1] - 1
        volatility = np.std(net_returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = np.mean(net_returns) / np.std(net_returns) * np.sqrt(252) if np.std(net_returns) > 0 else 0

        max_drawdown = np.max(np.maximum.accumulate(cumulative_returns) - cumulative_returns) / np.maximum.accumulate(cumulative_returns).max()

        results[f'horizon_{i}'] = {
            'Total_Return': total_return,
            'Final_Balance': final_balance,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Number_of_Trades': np.sum(position_changes)
        }

    return results
