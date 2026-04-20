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

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None


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
    batch_size: int = 1024,
    show_progress: bool = False,
    progress_desc: str = "MC-Dropout",
) -> tuple[np.ndarray, np.ndarray]:
    """Run *n_iterations* stochastic forward passes (MC-Dropout).

    Works with modern Keras by calling the model with ``training=True``
    (dropout layers built with ``training=True`` are always active).

    Predictions are generated in batches to avoid allocating very large
    LSTM activation tensors when ``X`` contains many sequences.
    When ``show_progress=True``, a tqdm progress bar reports batch-level
    progress with elapsed time and ETA.

    Returns
    -------
    mean : np.ndarray
        Mean prediction across iterations.
    std : np.ndarray
        Standard deviation (uncertainty) across iterations.
    """
    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    X = np.asarray(X, dtype=np.float32)
    running_sum = None
    running_sq_sum = None
    n_batches = (len(X) + batch_size - 1) // batch_size
    progress = None

    if show_progress and tqdm is not None:
        progress = tqdm(
            total=n_iterations * n_batches,
            desc=progress_desc,
            unit="batch",
        )

    for _ in range(n_iterations):
        batch_predictions = []
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            batch_predictions.append(model(X[start:end], training=True).numpy())
            if progress is not None:
                progress.update(1)

        preds = np.concatenate(batch_predictions, axis=0).astype(np.float64, copy=False)

        if running_sum is None:
            running_sum = preds
            running_sq_sum = np.square(preds)
        else:
            running_sum += preds
            running_sq_sum += np.square(preds)

    mean = running_sum / n_iterations
    variance = (running_sq_sum / n_iterations) - np.square(mean)
    std = np.sqrt(np.maximum(variance, 0.0))

    if progress is not None:
        progress.close()

    return mean, std


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
