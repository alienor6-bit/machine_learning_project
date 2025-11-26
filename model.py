import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# ==================================================================================
# 5. MODEL ARCHITECTURE (LSTM)
# ==================================================================================
def build_classifier(input_shape, n_horizons, units=64, dropout=0.3):
    """
    Builds a Bidirectional LSTM model for time series classification.
    
    Architecture:
    1. Bidirectional LSTM Layer: Captures patterns from past to future and vice versa.
    2. BatchNormalization: Stabilizes learning.
    3. Dropout: Prevents overfitting by randomly turning off neurons.
    4. Dense Layers: Final classification decision.
    """
    model = Sequential([
        # LSTM Layer
        # return_sequences=False because we only want the output of the LAST timestep
        tf.keras.layers.Bidirectional(LSTM(
            units, 
            input_shape=input_shape, 
            return_sequences=False,
            kernel_regularizer=l2(0.001) # L2 Regularization to prevent overfitting
        )),
        BatchNormalization(),
        Dropout(dropout),
        
        # Dense Layer (Hidden)
        Dense(units // 2, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(dropout),
        
        # Output Layer
        # Sigmoid activation outputs a probability between 0 and 1
        Dense(n_horizons, activation='sigmoid')
    ])
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compiles the model with Optimizer and Loss Function.
    """
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0) # clipnorm prevents exploding gradients
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy', # Standard loss for binary classification
        metrics=['accuracy']
    )
    return model

def get_callbacks(patience=20):
    """
    Callbacks to optimize training:
    1. EarlyStopping: Stop training if validation loss stops improving.
    2. ReduceLROnPlateau: Lower learning rate if stuck in a local minimum.
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=patience // 2,
        min_lr=1e-6,
        verbose=1
    )
    
    return [early_stopping, reduce_lr]

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, callbacks=None, class_weight=None):
    """
    Executes the training loop.
    """
    print(f"Training model on {len(X_train)} samples...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True, # Shuffle batches to break correlation
        verbose=1,
        class_weight=class_weight
    )
    return history