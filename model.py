from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def get_callbacks(patience=15):
    return [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

def build_lstm_model(input_shape, n_outputs, units=64, dropout=0.2):
    return Sequential([
        layers.LSTM(units, return_sequences=True, input_shape=input_shape),
        layers.Dropout(dropout),
        layers.LSTM(units // 2),
        layers.Dropout(dropout),
        layers.Dense(32, activation='relu'),
        layers.Dense(n_outputs)
    ])
