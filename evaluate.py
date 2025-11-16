"""
Model evaluation and prediction display
"""

import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def predict_and_evaluate(model, X_test, y_test, horizons):
    """
    Make predictions and calculate metrics
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Make predictions
    print("\nGenerating predictions...")
    # predictions_proba shape is (batch_size, n_horizons, 3_classes)
    predictions_proba = model.predict(X_test, verbose=0)
    
    results = {}
    
    for i, h in enumerate(horizons):
        # Extract predictions for this specific horizon
        y_pred_proba_horizon = predictions_proba[:, i, :] # Shape: (batch, 3)
        y_pred = np.argmax(y_pred_proba_horizon, axis=1)  # Shape: (batch,)
        y_true = y_test[:, i].astype(int)                 # Shape: (batch,)
        
        # Calculate confidence (max probability)
        confidence = np.max(y_pred_proba_horizon, axis=1)
        
        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # High confidence accuracy
        high_conf_threshold = 0.6
        high_conf_mask = confidence > high_conf_threshold
        if high_conf_mask.sum() > 0:
            high_conf_acc = accuracy_score(y_true[high_conf_mask], y_pred[high_conf_mask])
            high_conf_count = high_conf_mask.sum()
        else:
            high_conf_acc = 0.0
            high_conf_count = 0
        
        # Store results
        results[h] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba_horizon,
            'confidence': confidence,
            'accuracy': accuracy,
            'high_conf_accuracy': high_conf_acc,
            'high_conf_count': high_conf_count,
            'avg_confidence': confidence.mean()
        }
        
        # Display metrics
        print(f"\n{h}-day predictions:")
        print(f"  Overall accuracy: {accuracy:.2%}")
        print(f"  High confidence accuracy (>{high_conf_threshold:.0%}): {high_conf_acc:.2%}")
        print(f"  High confidence samples: {high_conf_count} / {len(y_test)} ({high_conf_count/len(y_test):.1%})")
        
        # Detailed classification report
        print(f"\n  Classification Report (0=Down, 1=Neutral, 2=Up):")
        print(classification_report(
            y_true, y_pred,
            target_names=['Strong Down', 'Neutral', 'Strong Up'],
            zero_division=0,
            digits=3
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"  Confusion Matrix:")
        print(f"  {cm}")
    
    return results


def show_sample_predictions(results, horizons, n_samples=10):
    """
    Display sample predictions
    """
    print("\n" + "="*60)
    print(f"SAMPLE PREDICTIONS (last {n_samples})")
    print("="*60)
    
    class_names = ['STRONG DOWN', 'NEUTRAL', 'STRONG UP']
    
    for h in horizons:
        r = results[h]
        
        print(f"\n{h}-day predictions:")
        print(f"{'True':<15} {'Predicted':<15} {'Confidence':<12} {'Correct':<10}")
        print("-" * 55)
        
        # Iterate from the end of the dataset
        for i in range(-n_samples, 0):
            true_class = class_names[r['y_true'][i]]
            pred_class = class_names[r['y_pred'][i]]
            conf = r['confidence'][i]
            correct = "✓" if r['y_true'][i] == r['y_pred'][i] else "✗"
            
            print(f"{true_class:<15} {pred_class:<15} {conf:.1%}         {correct}")


def analyze_by_confidence(results, horizons):
    """
    Analyze accuracy by confidence level
    """
    print("\n" + "="*60)
    print("ACCURACY BY CONFIDENCE LEVEL")
    print("="*60)
    
    confidence_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for h in horizons:
        r = results[h]
        
        print(f"\n{h}-day predictions:")
        print(f"{'Confidence >':<15} {'Count (% total)':<20} {'Accuracy':<12}")
        print("-" * 50)
        
        for threshold in confidence_thresholds:
            mask = r['confidence'] > threshold
            count = mask.sum()
            if count > 0:
                acc = accuracy_score(r['y_true'][mask], r['y_pred'][mask])
                pct = count / len(r['y_true']) * 100
                print(f"{threshold:<15.1f} {count:<8} ({pct:>5.1f}%)       {acc:.2%}")
            else:
                print(f"{threshold:<15.1f} 0        (  0.0%)       N/A")

if __name__ == "__main__":
    print("Evaluation module - Use with main.py")