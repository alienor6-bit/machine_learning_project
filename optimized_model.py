from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

def build_improved_forex_lstm(input_shape, n_outputs, dropout=0.4):
    """
    Enhanced LSTM architecture for forex prediction
    """

    model = Sequential([
        # First LSTM layer with more units and regularization
        layers.LSTM(128, return_sequences=True, input_shape=input_shape,
                   dropout=dropout, recurrent_dropout=dropout,
                   kernel_regularizer=l2(0.001)),

        # Batch normalization for stable training
        layers.BatchNormalization(),

        # Second LSTM layer
        layers.LSTM(64, return_sequences=True,
                   dropout=dropout, recurrent_dropout=dropout,
                   kernel_regularizer=l2(0.001)),

        # Third LSTM layer
        layers.LSTM(32, dropout=dropout, recurrent_dropout=dropout),

        # Dense layers with regularization
        layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        layers.Dropout(dropout),
        layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        layers.Dropout(dropout),

        # Output layer
        layers.Dense(n_outputs)
    ])

    return model

# === IMPROVED TRAINING CONFIGURATION ===

def get_improved_training_config():
    """
    Better training configuration for forex
    """
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    optimizer = Adam(
        learning_rate=0.001,  # Lower learning rate
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=25,  # More patience
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,  # Reduce LR more often
            min_lr=1e-7,
            verbose=1
        )
    ]

    return optimizer, callbacks
