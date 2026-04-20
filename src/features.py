"""
Feature engineering for cryptocurrency price data.

Adds technical indicators via ``pandas_ta`` and creates windowed
sequences suitable for LSTM input.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute common technical indicators and append them as columns.

    Requires columns: ``open``, ``high``, ``low``, ``close``, ``volume``.

    Indicators added:
    RSI, MACD, Signal Line, SMA-10, SMA-50, Bollinger Bands,
    ATR, OBV, Stochastic %K / %D.
    """
    df = df.copy()

    df["RSI"] = ta.rsi(df["close"], length=14)

    macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["MACD"] = macd_df["MACD_12_26_9"]
    df["Signal_Line"] = macd_df["MACDs_12_26_9"]

    df["SMA_10"] = ta.sma(df["close"], length=10)
    df["SMA_50"] = ta.sma(df["close"], length=50)

    bbands = ta.bbands(df["close"], length=20, std=2)
    df["Upper_Band"] = bbands["BBU_20_2.0"]
    df["Lower_Band"] = bbands["BBL_20_2.0"]

    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    df["OBV"] = ta.obv(df["close"], df["volume"])

    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    df["%K"] = stoch["STOCHk_14_3_3"]
    df["%D"] = stoch["STOCHd_14_3_3"]

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def prepare_sequences(
    df: pd.DataFrame,
    target_col: str = "close",
    lookback: int = 60,
    train_ratio: float = 0.8,
    feature_cols: Optional[list[str]] = None,
) -> dict:
    """Scale features and build (X, y) sequences for LSTM training.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame **with** technical indicators already computed.
    target_col : str
        Column to predict.
    lookback : int
        Number of past time-steps in each input window.
    train_ratio : float
        Fraction of data used for training.
    feature_cols : list[str] | None
        Columns to use as features.  ``None`` means all numeric columns
        except ``open_time``.

    Returns
    -------
    dict
        Keys: ``X_train``, ``y_train``, ``X_test``, ``y_test``,
        ``scaler_X``, ``scaler_y``, ``training_data_len``.

    Notes
    -----
    Feature and target scalers are fit on the training split only to
    avoid leaking future information into the training pipeline.
    """
    df = df.copy()
    if "open_time" in df.columns:
        df = df.drop(columns=["open_time"])

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if feature_cols is None:
        df = df.select_dtypes(include=[np.number]).copy()
    else:
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        df = df[feature_cols].copy()

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' is not present in the feature set.")

    if len(df) <= lookback:
        raise ValueError("DataFrame must contain more rows than lookback.")

    training_data_len = math.ceil(len(df) * train_ratio)
    if training_data_len <= lookback:
        raise ValueError("Training split must contain more rows than lookback.")

    train_df = df.iloc[:training_data_len].copy()

    scaler_X = MinMaxScaler()
    scaler_X.fit(train_df)
    scaled = scaler_X.transform(df)
    scaled_df = pd.DataFrame(scaled, columns=df.columns, index=df.index)

    target_idx = list(df.columns).index(target_col)
    scaler_y = MinMaxScaler()
    scaler_y.fit(train_df[[target_col]])

    train_block = scaled_df.iloc[:training_data_len]
    test_block = scaled_df.iloc[training_data_len - lookback :]

    def _make_xy(block: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        arr = block.values
        for i in range(lookback, len(arr)):
            X.append(arr[i - lookback : i])
            y.append(arr[i, target_idx])
        return np.array(X), np.array(y)

    X_train, y_train = _make_xy(train_block)
    X_test, y_test = _make_xy(test_block)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "training_data_len": training_data_len,
        "feature_names": list(df.columns),
    }
