def train_model(ticker, sequences_data, epochs=100, batch_size=32):
    print(f"\n{'='*60}\nEntraînement pour {ticker}\n{'='*60}")

    data = sequences_data[ticker]
    X_train, y_train, X_val, y_val = data['X_train'], data['y_train'], data['X_val'], data['y_val']

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), y_train.shape[1])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print(f"Input shape: {(X_train.shape[1], X_train.shape[2])}")
    print(f"Outputs: {y_train.shape[1]} horizons")

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       epochs=epochs, batch_size=batch_size, callbacks=get_callbacks(), verbose=1)

    return model, history
