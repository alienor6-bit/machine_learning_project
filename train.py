"""
Model training functions
"""

def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=100, batch_size=32, callbacks=None):
    """
    Train the LSTM regressor
    """
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=False,
        verbose=1
    )
    
    print("\nTraining completed!")
    
    return history


def display_training_results(history):
    """
    Display training results summary for regression
    """
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    # Find best epoch based on val_loss (val_mse)
    best_epoch = min(range(len(history.history['val_loss'])), 
                    key=lambda i: history.history['val_loss'][i])
    
    print(f"\nBest validation epoch: {best_epoch + 1}")
    
    # Get metrics from the best epoch
    best_val_loss = history.history['val_loss'][best_epoch]
    best_val_mae = history.history['val_mae'][best_epoch]
    train_loss = history.history['loss'][best_epoch]
    train_mae = history.history['mae'][best_epoch]

    print(f"  Training Loss (MSE): {train_loss:.4f}")
    print(f"  Training MAE: {train_mae:.4f}")
    print(f"  Validation Loss (MSE): {best_val_loss:.4f}")
    print(f"  Validation MAE: {best_val_mae:.4f}")