"""
Model training functions
"""

# --- SIGNATURE DE FONCTION CORRECTE ---
# Elle attend X_val et y_val, pas 'validation_data'
def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=100, batch_size=32, callbacks=None):
    """
    Train the LSTM classifier
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
        validation_data=(X_val, y_val), # <-- On crée le tuple ici
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        # Pas de sample_weight
        verbose=1
    )
    
    print("\nTraining completed!")
    
    return history


def display_training_results(history):
    """
    Display training results summary
    """
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    # Find best epoch based on val_loss
    best_epoch = min(range(len(history.history['val_loss'])), 
                    key=lambda i: history.history['val_loss'][i])
    
    print(f"\nBest validation epoch: {best_epoch + 1}")
    
    # Get metrics from the best epoch
    best_val_loss = history.history['val_loss'][best_epoch]
    best_val_acc = history.history['val_accuracy'][best_epoch]
    train_loss = history.history['loss'][best_epoch]
    train_acc = history.history['accuracy'][best_epoch]

    print(f"  Training loss: {train_loss:.4f}")
    print(f"  Training accuracy: {train_acc:.4f}")
    print(f"  Validation loss: {best_val_loss:.4f}")
    print(f"  Validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    print("Training module - Use with main.py")