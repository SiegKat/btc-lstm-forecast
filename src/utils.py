"""
Plotting, evaluation, and scaling utilities.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.style.use("seaborn-v0_8-darkgrid")

FIGSIZE = (16, 6)
COLORS = {
    "train": "#1f77b4",
    "actual": "#2ca02c",
    "predicted": "#ff7f0e",
    "forecast": "#d62728",
    "ci": "#ff7f0e",
}


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute RMSE, MAE, and MAPE.

    Returns a dict with keys ``rmse``, ``mae``, ``mape``.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape}


def plot_predictions(
    train_values: np.ndarray,
    actual_values: np.ndarray,
    predicted_values: np.ndarray,
    title: str = "LSTM Price Prediction",
    xlabel: str = "Time Step",
    ylabel: str = "Price (USD)",
    figsize: tuple[int, int] = FIGSIZE,
) -> plt.Figure:
    """Plot training data, actual test prices, and predictions."""
    fig, ax = plt.subplots(figsize=figsize)
    train_x = np.arange(len(train_values))
    test_x = np.arange(len(train_values), len(train_values) + len(actual_values))

    ax.plot(train_x, train_values, color=COLORS["train"], label="Train", alpha=0.8)
    ax.plot(test_x, actual_values, color=COLORS["actual"], label="Actual", alpha=0.8)

    pred_x = test_x[: len(predicted_values)]
    ax.plot(pred_x, predicted_values, color=COLORS["predicted"], label="Predicted", linewidth=1.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_future_forecast(
    forecast: np.ndarray,
    confidence_std: Optional[np.ndarray] = None,
    title: str = "Future Price Forecast",
    xlabel: str = "Steps Ahead",
    ylabel: str = "Price (USD)",
    figsize: tuple[int, int] = FIGSIZE,
) -> plt.Figure:
    """Plot future forecast with optional confidence band."""
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(forecast))

    ax.plot(x, forecast, color=COLORS["forecast"], label="Forecast", linewidth=1.5)

    if confidence_std is not None:
        upper = forecast + 2 * confidence_std
        lower = forecast - 2 * confidence_std
        ax.fill_between(
            x, lower, upper,
            color=COLORS["ci"], alpha=0.15, label="95% CI",
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_training_history(
    history,
    title: str = "Training History",
    figsize: tuple[int, int] = FIGSIZE,
) -> plt.Figure:
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(history.history["loss"], label="Train Loss")
    ax.plot(history.history["val_loss"], label="Val Loss")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_portfolio(
    backtest_df,
    title: str = "RL Agent Portfolio Value",
    figsize: tuple[int, int] = FIGSIZE,
) -> plt.Figure:
    """Plot portfolio value over time from a backtest DataFrame."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(backtest_df["step"], backtest_df["portfolio_value"], linewidth=1.2)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.axhline(
        y=backtest_df["portfolio_value"].iloc[0],
        color="gray", linestyle="--", alpha=0.5, label="Initial Value",
    )
    ax.legend()
    fig.tight_layout()
    return fig


def print_metrics(metrics: dict[str, float]) -> None:
    """Pretty-print evaluation metrics."""
    print(f"  RMSE : {metrics['rmse']:,.2f}")
    print(f"  MAE  : {metrics['mae']:,.2f}")
    print(f"  MAPE : {metrics['mape']:.2f}%")
