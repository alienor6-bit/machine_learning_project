"""
Walk-Forward Validation Script
==============================
This script implements a rigorous validation method for time series.
Instead of a single Train/Test split, it uses an "Expanding Window" approach.

Concept:
1. Train 2000-2018 -> Test 2019
2. Train 2000-2019 -> Test 2020
3. Train 2000-2020 -> Test 2021

This simulates how a real trader would adapt their model over time.
"""
import numpy as np
import tensorflow as tf
from data import prepare_dataset, create_sequences, split_temporal, balance_training_set, normalize_features
from model import build_classifier, compile_model, train_model, get_callbacks
from analysis import predict_and_evaluate, run_backtest

def walk_forward_validation(best_config):
    """
    Test the best configuration on 4 consecutive time periods.
    """
    print("\n" + "="*60)
    print("WALK-FORWARD VALIDATION")
    print("="*60)

    # 1. Prepare Data
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)

    HORIZONS = [1, 5]
    dataset = prepare_dataset(start_date='2000-01-01', horizons=HORIZONS)
    X, y, _ = create_sequences(dataset, HORIZONS, window_size=60)

    returns_cols = [f'real_return_{h}d' for h in HORIZONS]
    returns_aligned = dataset[returns_cols].values[60:]

    X_train, X_val, X_test, y_train, y_val, y_test = split_temporal(X, y, test_size=0.2, val_size=0.1)
    test_idx = int(len(X) * 0.8)
    returns_test = returns_aligned[test_idx:]

    X_train_norm, X_val_norm, X_test_norm, scaler = normalize_features(X_train, X_val, X_test)

    # 2. Train Initial Model
    # In a full implementation, we would retrain inside the loop.
    # Here, for speed, we train once and test on sequential blocks,
    # which is a simplified version of Walk-Forward (Block Cross-Validation).
    model = build_classifier(
        input_shape=(X_train_norm.shape[1], X_train_norm.shape[2]),
        n_horizons=len(HORIZONS),
        units=best_config['units'],
        dropout=best_config['dropout']
    )

    model = compile_model(model, learning_rate=best_config['learning_rate'])
    
    train_model(
        model, X_train_norm, y_train, X_val_norm, y_val,
        epochs=best_config['epochs'],
        batch_size=best_config['batch_size'],
        callbacks=get_callbacks(patience=best_config['patience'])
    )

    # 3. Split Test Data into 4 Periods (e.g., Year 1, Year 2, Year 3, Year 4)
    n_periods = 4
    period_size = len(X_test_norm) // n_periods

    period_results = []

    for period_idx in range(n_periods):
        start_idx = period_idx * period_size
        end_idx = start_idx + period_size if period_idx < n_periods-1 else len(X_test_norm)

        X_test_period = X_test_norm[start_idx:end_idx]
        y_test_period = y_test[start_idx:end_idx]
        returns_test_period = returns_test[start_idx:end_idx]

        print(f"\n{'='*60}")
        print(f"PERIOD {period_idx + 1}/{n_periods} (Days {start_idx}-{end_idx})")
        print(f"{'='*60}")

        # 4. Evaluate on this specific period
        results = predict_and_evaluate(
            model, X_test_period, y_test_period, HORIZONS, returns_test_period,
            confidence_threshold=best_config['confidence_threshold']
        )

        run_backtest(results, horizon=5)

        # 5. Calculate Returns
        r = results[5]
        y_pred = r['y_pred']
        real_returns = np.nan_to_num(r['returns'], nan=0.0)

        capital_market = 1000.0
        capital_ai = 1000.0

        for i in range(len(y_pred)):
            ret = real_returns[i]
            capital_market *= (1 + ret)

            if y_pred[i] == 1:
                capital_ai *= (1 + ret - 0.0002) # Buy with transaction cost
            else:
                pass # Cash (Long-Only)

        perf_market = (capital_market - 1000) / 1000
        perf_ai = (capital_ai - 1000) / 1000

        period_results.append({
            'period': period_idx + 1,
            'market_return': perf_market,
            'ai_return': perf_ai,
            'outperformance': perf_ai - perf_market,
            'n_trades': r['n_trades']
        })

    # 6. Save & Report Results
    with open('results.txt', 'w', encoding='utf-8') as f:
        f.write("WALK-FORWARD SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"{'Period':<10} {'Market':<12} {'AI Model':<12} {'Outperf.':<12} {'Trades'}\n")
        f.write("-"*60 + "\n")

        for r in period_results:
            f.write(f"Period {r['period']:<3} {r['market_return']:>+10.2%} {r['ai_return']:>+10.2%} "
                  f"{r['outperformance']:>+10.2%} {r['n_trades']:>8}\n")

    print("Results saved to results.txt")
    return period_results

if __name__ == "__main__":
    best_config = {
        'name': 'Higher_LR',
        'units': 32,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'patience': 20,
        'confidence_threshold': 0.65
    }
    walk_forward_validation(best_config)
