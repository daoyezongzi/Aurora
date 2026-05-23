from __future__ import annotations

import pandas as pd

from aurora_ml.dataset import split_by_time


def test_time_split_has_no_future_leakage() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.date_range("2021-01-01", periods=100, freq="D").strftime("%Y-%m-%d"),
            "close": range(100),
        }
    )

    train_df, val_df, test_df = split_by_time(frame, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    assert train_df["trade_date"].max() < val_df["trade_date"].min()
    assert val_df["trade_date"].max() < test_df["trade_date"].min()
