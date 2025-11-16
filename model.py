"""
LSTM model architecture for regression
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import tensorflow as tf # Importé pour la métrique MAE


def build_regressor(input_shape, n_horizons, units=64, dropout=0.4):
    """
    Build LSTM regressor for multi-horizon prediction
    """
    print("\nBuilding LSTM Regressor...")
    print(f"  Input shape: {input_shape}")
    print(f"  LSTM units: {units}")
    print(f"  Dropout: {dropout}")

    l2_reg = 0.001

    model = Sequential([
        # Input layer
        LSTM(
            units, 
            return_sequences=True, 
            input_shape=input_shape,
            kernel_regularizer=l2(l2_reg)
        ),
        Dropout(dropout),
        BatchNormalization(),
        
        # Hidden layer
        LSTM(
            units // 2,
            kernel_regularizer=l2(l2_reg)
        ),
        Dropout(dropout),
        
        # Dense layer
        Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout),
        
        # --- CRITICAL CHANGE: Output layer ---
        # No Reshape, No Softmax
        # Just one 'linear' output node per horizon
        Dense(n_horizons, activation='linear')
    ])
    
    return model


def compile_model(model, learning_rate=0.0001):
    """
    Compile model with optimizer and loss for regression
    """
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error', # <-- MSE for regression
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')] # <-- MAE is more interpretable
    )
    
    print(f"\nModel compiled with loss=MSE, metric=MAE, lr={learning_rate}")
    return model


def get_callbacks(patience=20, min_delta=0.0001):
    """
    Get training callbacks
    (monitoring val_loss, which is val_mse)
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            min_delta=min_delta
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks