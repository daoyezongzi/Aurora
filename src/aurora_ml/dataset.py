from __future__ import annotations

import numpy as np
import pandas as pd


def split_by_time(
    frame: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total:.6f}")

    if len(frame) < 3:
        raise ValueError("frame must contain at least 3 rows")

    n = len(frame)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Keep each split non-empty.
    train_end = max(train_end, 1)
    val_end = max(val_end, train_end + 1)
    val_end = min(val_end, n - 1)

    train_df = frame.iloc[:train_end].copy()
    val_df = frame.iloc[train_end:val_end].copy()
    test_df = frame.iloc[val_end:].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"split produced empty partition: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
    return train_df, val_df, test_df


def build_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if len(features) != len(labels):
        raise ValueError("features and labels must have same length")
    if len(features) < sequence_length:
        return np.empty((0, sequence_length, features.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)

    xs: list[np.ndarray] = []
    ys: list[float] = []
    for end_idx in range(sequence_length - 1, len(features)):
        start_idx = end_idx - sequence_length + 1
        xs.append(features[start_idx : end_idx + 1])
        ys.append(float(labels[end_idx]))

    x_array = np.asarray(xs, dtype=np.float32)
    y_array = np.asarray(ys, dtype=np.float32)
    return x_array, y_array
