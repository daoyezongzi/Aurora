from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from aurora_ml.config import AuroraConfig
from aurora_ml.utils import ensure_parent, normalize_date_to_yyyymmdd, sha256_file, today_yyyymmdd

BASE_COLUMNS = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "pct_chg"]


@dataclass(frozen=True)
class PreparedData:
    raw_df: pd.DataFrame
    train_df: pd.DataFrame
    raw_path: Path
    processed_path: Path
    raw_sha256: str


def prepare_data(config: AuroraConfig, refresh: bool, tushare_token: str | None) -> PreparedData:
    raw_path = config.data.raw_path
    processed_path = config.data.processed_path
    ensure_parent(raw_path)
    ensure_parent(processed_path)

    should_refresh = refresh or config.data.refresh_by_default or (not raw_path.exists())
    if should_refresh:
        if not tushare_token:
            raise ValueError("TUSHARE_TOKEN is required when raw data does not exist or refresh is requested.")
        raw_df = fetch_from_tushare(
            code=config.data.code,
            start_date=config.data.start_date,
            end_date=config.data.end_date,
            token=tushare_token,
        )
        raw_df.to_csv(raw_path, index=False, encoding="utf-8")
    else:
        raw_df = pd.read_csv(raw_path)

    if raw_df.empty:
        raise ValueError("raw data is empty")

    raw_hash = sha256_file(raw_path)
    expected_hash = config.data.expected_sha256
    if expected_hash and raw_hash != expected_hash:
        raise ValueError(
            f"SHA256 mismatch for {raw_path}: expected={expected_hash}, actual={raw_hash}. "
            "Update expected_sha256 only after verifying dataset source."
        )

    train_df = build_training_frame(raw_df, config)
    train_df.to_csv(processed_path, index=False, encoding="utf-8")

    return PreparedData(
        raw_df=raw_df,
        train_df=train_df,
        raw_path=raw_path,
        processed_path=processed_path,
        raw_sha256=raw_hash,
    )


def fetch_from_tushare(code: str, start_date: str, end_date: str, token: str) -> pd.DataFrame:
    import tushare as ts

    # Use token argument directly to avoid writing tk.csv under user home.
    pro = ts.pro_api(token=token)

    start = normalize_date_to_yyyymmdd(start_date)
    end = today_yyyymmdd() if end_date.strip().lower() == "latest" else normalize_date_to_yyyymmdd(end_date)
    fields = ",".join(BASE_COLUMNS)
    frame = pro.daily(ts_code=code, start_date=start, end_date=end, fields=fields)

    if frame is None or frame.empty:
        raise ValueError(f"no rows returned from tushare daily for {code}")

    ordered = frame.loc[:, BASE_COLUMNS].copy()
    ordered["trade_date"] = pd.to_datetime(ordered["trade_date"], format="%Y%m%d", errors="coerce")
    ordered = ordered.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    ordered["trade_date"] = ordered["trade_date"].dt.strftime("%Y-%m-%d")
    return ordered


def build_training_frame(raw_df: pd.DataFrame, config: AuroraConfig) -> pd.DataFrame:
    frame = raw_df.copy()
    required = {"trade_date", "close", "open", "high", "low", "vol", "pct_chg"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"raw data missing required columns: {missing}")

    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)

    numeric_cols = ["open", "high", "low", "close", "vol", "pct_chg"]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame["ma5"] = frame["close"].rolling(window=5, min_periods=5).mean()
    frame["ma10"] = frame["close"].rolling(window=10, min_periods=10).mean()
    frame[config.data.label_col] = (frame["close"].shift(-1) > frame["close"]).astype(int)

    keep_cols = ["trade_date"] + config.data.feature_cols + [config.data.label_col]
    frame = frame.loc[:, keep_cols].copy()
    frame = frame.dropna().reset_index(drop=True)
    frame["trade_date"] = frame["trade_date"].dt.strftime("%Y-%m-%d")

    if len(frame) < (config.split.sequence_length + 30):
        raise ValueError(
            f"not enough rows after feature building: {len(frame)} rows. "
            f"Need at least {config.split.sequence_length + 30} rows."
        )
    return frame
