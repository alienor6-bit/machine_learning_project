def build_lstm_model(input_shape, n_outputs, units=64, dropout=0.2):
    return Sequential([
        layers.LSTM(units, return_sequences=True, input_shape=input_shape),
        layers.Dropout(dropout),
        layers.LSTM(units // 2),
        layers.Dropout(dropout),
        layers.Dense(32, activation='relu'),
        layers.Dense(n_outputs)
    ])
#add callback
