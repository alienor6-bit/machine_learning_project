"""
LSTM model architecture for classification
(Final attempt: Simplified, heavily regularized)
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, 
    Reshape, Softmax
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2 # <-- IMPORT L2 REGULARIZER


def build_classifier(input_shape, n_horizons, units=64, dropout=0.5): 
    """
    Build a simplified, heavily regularized LSTM classifier
    """
    print("\nBuilding SIMPLIFIED and REGULARIZED LSTM classifier...")
    print(f"  Input shape: {input_shape}")
    print(f"  LSTM units: {units}")
    print(f"  Dropout: {dropout}")
    print(f"  L2 Regularization: 0.001")

    # L2 value (la "taxe" sur les poids)
    l2_reg = 0.001

    model = Sequential([
        # First LSTM layer with L2 reg
        LSTM(
            units, 
            return_sequences=True, 
            input_shape=input_shape,
            kernel_regularizer=l2(l2_reg) # <-- ADD L2
        ),
        Dropout(dropout),
        BatchNormalization(),
        
        # --- MODEL SIMPLIFIED: Removed one LSTM layer ---
        
        # Final LSTM layer
        LSTM(
            units // 2,
            kernel_regularizer=l2(l2_reg) # <-- ADD L2
        ),
        Dropout(dropout),
        
        # Dense layers
        Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)), # <-- ADD L2
        Dropout(dropout),
        
        # Output layer
        Dense(n_horizons * 3),
        Reshape((n_horizons, 3)),
        Softmax(axis=-1)
    ])
    
    return model


def compile_model(model, learning_rate=0.0001): # <-- Keep low learning rate
    """
    Compile model with optimizer and loss
    """
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\nModel compiled with learning rate: {learning_rate}")
    return model


def get_callbacks(patience=20, min_delta=0.0001):
    """
    Get training callbacks
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


if __name__ == "__main__":
    # Test model creation
    model = build_classifier(input_shape=(60, 20), n_horizons=4)
    model = compile_model(model)
    model.summary()