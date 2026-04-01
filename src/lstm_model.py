"""
LSTM model definitions for cryptocurrency price prediction.

Provides builders for:
- Bidirectional LSTM (single-output price predictor)
- MC-Dropout Bidirectional LSTM (uncertainty-aware predictor)
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import LSTM, Bidirectional, Dense, Dropout
from keras.optimizers import Adam


def build_bidirectional_lstm(
    input_shape: tuple[int, int],
    units: int = 100,
    dropout_rate: float = 0.2,
    dense_units: int = 25,
    learning_rate: float = 1e-3,
) -> Model:
    """Build and compile a two-layer Bidirectional LSTM.

    Parameters
    ----------
    input_shape : (timesteps, features)
    units : int
        LSTM hidden units per direction.
    dropout_rate : float
    dense_units : int
        Units in the hidden dense layer.
    learning_rate : float
    """
    inputs = Input(shape=input_shape)

    x = Bidirectional(LSTM(units, return_sequences=True))(inputs)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(units, return_sequences=False))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipvalue=0.5),
        loss="mean_squared_error",
    )
    return model


def build_mc_dropout_lstm(
    input_shape: tuple[int, int],
    units: int = 100,
    dropout_rate: float = 0.2,
    dense_units: int = 25,
    output_steps: int = 1,
    learning_rate: float = 1e-3,
) -> Model:
    """Bidirectional LSTM with MC-Dropout for uncertainty estimation.

    Dropout layers use ``training=True`` at inference time so they
    remain active, producing stochastic forward passes.

    Parameters
    ----------
    output_steps : int
        Number of future time-steps to predict simultaneously.
    """
    inputs = Input(shape=input_shape)

    x = Bidirectional(LSTM(units, return_sequences=True))(inputs)
    x = Dropout(dropout_rate)(x, training=True)
    x = Bidirectional(LSTM(units, return_sequences=False))(x)
    x = Dropout(dropout_rate)(x, training=True)
    x = Dense(dense_units, activation="relu")(x)
    outputs = Dense(output_steps)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipvalue=0.5),
        loss="mean_squared_error",
    )
    return model


def get_callbacks(
    patience_stop: int = 5,
    patience_lr: int = 3,
    min_lr: float = 1e-5,
) -> list:
    """Standard callbacks: early stopping + learning-rate reduction."""
    return [
        EarlyStopping(monitor="val_loss", patience=patience_stop, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=patience_lr, min_lr=min_lr),
    ]


def predict_with_uncertainty(
    model: Model,
    X: np.ndarray,
    n_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Run *n_iterations* stochastic forward passes (MC-Dropout).

    Works with modern Keras by calling the model with ``training=True``
    (dropout layers built with ``training=True`` are always active).

    Returns
    -------
    mean : np.ndarray
        Mean prediction across iterations.
    std : np.ndarray
        Standard deviation (uncertainty) across iterations.
    """
    predictions = np.stack(
        [model(X, training=True).numpy() for _ in range(n_iterations)]
    )
    return predictions.mean(axis=0), predictions.std(axis=0)


def forecast_future(
    model: Model,
    last_window: np.ndarray,
    steps: int = 500,
    scaler_y=None,
) -> np.ndarray:
    """Auto-regressive rolling forecast.

    Parameters
    ----------
    model : Model
    last_window : np.ndarray
        Shape ``(lookback, features)``.  The seed window.
    steps : int
        How many future steps to predict.
    scaler_y : MinMaxScaler | None
        If provided, inverse-transforms predictions back to original scale.

    Returns
    -------
    np.ndarray of shape ``(steps,)`` with predicted values.
    """
    window = last_window.copy()
    preds = []

    for _ in range(steps):
        x_input = window[np.newaxis, :, :]
        pred_scaled = model.predict(x_input, verbose=0)

        if scaler_y is not None:
            pred_original = scaler_y.inverse_transform(pred_scaled)[0, 0]
        else:
            pred_original = pred_scaled[0, 0]

        preds.append(pred_original)

        new_row = np.zeros((1, window.shape[1]))
        new_row[0, 0] = pred_scaled[0, 0]
        window = np.concatenate([window[1:], new_row], axis=0)

    return np.array(preds)
