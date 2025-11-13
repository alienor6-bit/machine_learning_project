from model import build_lstm_model, get_callbacks
import numpy as np

def train_model(ticker, sequences_data, epochs=100, batch_size=32):
    """
    Train LSTM model for a specific ticker/currency pair
    """
    print(f"\n{'='*60}\nEntraînement pour {ticker}\n{'='*60}")

    data = sequences_data[ticker]
    X_train, y_train, X_val, y_val = data['X_train'], data['y_train'], data['X_val'], data['y_val']

    # Validate data shapes
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError(f"No training data available for {ticker}")

    if len(X_val) == 0 or len(y_val) == 0:
        print("Warning: No validation data available")
        X_val, y_val = None, None

    # Build and compile model
    input_shape = (X_train.shape[1], X_train.shape[2])
    n_outputs = y_train.shape[1]

    model = build_lstm_model(input_shape, n_outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print(f"Input shape: {input_shape}")
    print(f"Outputs: {n_outputs} horizons")
    print(f"Training samples: {len(X_train)}")
    if X_val is not None:
        print(f"Validation samples: {len(X_val)}")

    # Prepare callbacks
    callbacks = get_callbacks(patience=15)

    # Train model
    validation_data = (X_val, y_val) if X_val is not None else None

    history = model.fit(
        X_train, y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, history

def train_multiple_models(sequences_data, epochs=100, batch_size=32):
    """
    Train models for multiple tickers/currency pairs
    """
    models = {}
    histories = {}

    for ticker in sequences_data.keys():
        print(f"\nTraining model for {ticker}...")
        try:
            model, history = train_model(ticker, sequences_data, epochs, batch_size)
            models[ticker] = model
            histories[ticker] = history
            print(f" Model for {ticker} trained successfully")
        except Exception as e:
            print(f" Error training model for {ticker}: {str(e)}")
            models[ticker] = None
            histories[ticker] = None

    return models, histories

def evaluate_training_history(history, ticker="Unknown"):
    """
    Analyze and display training history
    """
    print(f"\nAnalyse de l'entraînement pour {ticker}:")

    if history is None:
        print("Pas d'historique d'entraînement disponible")
        return

    history_dict = history.history

    final_epoch = len(history_dict['loss'])

    print(f"  Nombre d'époques: {final_epoch}")
    print(f"  Loss finale (train): {history_dict['loss'][-1]:.6f}")
    print(f"  MAE finale (train): {history_dict['mae'][-1]:.6f}")

    if 'val_loss' in history_dict:
        print(f"  Loss finale (val): {history_dict['val_loss'][-1]:.6f}")
        print(f"  MAE finale (val): {history_dict['val_mae'][-1]:.6f}")

        # Check for overfitting
        train_loss = history_dict['loss'][-1]
        val_loss = history_dict['val_loss'][-1]

        if val_loss > train_loss * 1.2:
            print(" Possible overfitting detected")
        elif val_loss < train_loss * 0.8:
            print(" Modèle semble bien généralisé")
        else:
            print(" Validation loss raisonnable")

    # Best epoch info
    if 'val_loss' in history_dict:
        best_epoch = np.argmin(history_dict['val_loss']) + 1
        best_val_loss = min(history_dict['val_loss'])
        print(f"  Meilleure époque: {best_epoch} (val_loss: {best_val_loss:.6f})")

def save_model_checkpoint(model, ticker, timestamp, save_dir="/mnt/user-data/outputs"):
    """
    Save model checkpoint
    """
    import os

    checkpoint_dir = f"{save_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_path = f"{checkpoint_dir}/{ticker}_model_{timestamp}.h5"
    model.save(model_path)

    print(f"Modèle sauvegardé: {model_path}")
    return model_path
