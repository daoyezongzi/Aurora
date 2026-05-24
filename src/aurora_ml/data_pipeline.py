from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3

import numpy as np
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


def prepare_data(
    config: AuroraConfig,
    refresh: bool,
    tushare_token: str | None,
    feature_engine: str = "pandas",
) -> PreparedData:
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

    train_df = build_training_frame(raw_df, config, feature_engine=feature_engine)
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


def build_training_frame(raw_df: pd.DataFrame, config: AuroraConfig, feature_engine: str = "pandas") -> pd.DataFrame:
    engine = str(feature_engine).strip().lower()
    if engine == "pandas":
        return _build_training_frame_pandas(raw_df, config)
    if engine == "sql":
        return _build_training_frame_sql(raw_df, config)
    raise ValueError(f"unsupported feature_engine={feature_engine!r}; expected one of: pandas, sql")


def _normalize_raw_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
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
    return frame


def _finalize_training_frame(frame: pd.DataFrame, config: AuroraConfig) -> pd.DataFrame:
    keep_cols = ["trade_date"] + config.data.feature_cols + [config.data.label_col]
    frame = frame.loc[:, keep_cols].copy()
    frame = frame.dropna().reset_index(drop=True)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if len(frame) < (config.split.sequence_length + 30):
        raise ValueError(
            f"not enough rows after feature building: {len(frame)} rows. "
            f"Need at least {config.split.sequence_length + 30} rows."
        )
    return frame


def _build_training_frame_pandas(raw_df: pd.DataFrame, config: AuroraConfig) -> pd.DataFrame:
    frame = _normalize_raw_frame(raw_df)

    eps = 1e-9
    close_prev = frame["close"].shift(1)
    day_ret = frame["close"].pct_change()

    # Base moving averages.
    frame["ma5"] = frame["close"].rolling(window=5, min_periods=5).mean()
    frame["ma10"] = frame["close"].rolling(window=10, min_periods=10).mean()
    frame["ma20"] = frame["close"].rolling(window=20, min_periods=20).mean()

    # Return and volatility features.
    frame["ret_1"] = frame["close"].pct_change(periods=1)
    frame["ret_3"] = frame["close"].pct_change(periods=3)
    frame["ret_5"] = frame["close"].pct_change(periods=5)
    frame["volatility_5"] = day_ret.rolling(window=5, min_periods=5).std()
    frame["volatility_10"] = day_ret.rolling(window=10, min_periods=10).std()
    frame["volatility_20"] = day_ret.rolling(window=20, min_periods=20).std()

    # Price structure features.
    frame["intraday_range"] = (frame["high"] - frame["low"]) / (frame["close"].abs() + eps)
    frame["overnight_gap"] = (frame["open"] - close_prev) / (close_prev.abs() + eps)
    frame["close_ma5_gap"] = (frame["close"] - frame["ma5"]) / (frame["ma5"].abs() + eps)
    frame["close_ma10_gap"] = (frame["close"] - frame["ma10"]) / (frame["ma10"].abs() + eps)
    frame["close_ma20_gap"] = (frame["close"] - frame["ma20"]) / (frame["ma20"].abs() + eps)
    frame["close_pos_in_day"] = (frame["close"] - frame["low"]) / ((frame["high"] - frame["low"]).abs() + eps)

    # Volume features.
    frame["vol_ma5"] = frame["vol"].rolling(window=5, min_periods=5).mean()
    frame["vol_ma20"] = frame["vol"].rolling(window=20, min_periods=20).mean()
    frame["vol_ratio_5_20"] = frame["vol_ma5"] / (frame["vol_ma20"].abs() + eps)

    frame[config.data.label_col] = (frame["close"].shift(-1) > frame["close"]).astype(int)

    return _finalize_training_frame(frame, config)


def _build_training_frame_sql(raw_df: pd.DataFrame, config: AuroraConfig) -> pd.DataFrame:
    frame = _normalize_raw_frame(raw_df)
    frame = frame.loc[:, ["trade_date", "open", "high", "low", "close", "vol", "pct_chg"]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame.dropna(subset=["trade_date"]).reset_index(drop=True)

    query = """
WITH base AS (
    SELECT
        trade_date,
        CAST(open AS REAL) AS open,
        CAST(high AS REAL) AS high,
        CAST(low AS REAL) AS low,
        CAST(close AS REAL) AS close,
        CAST(vol AS REAL) AS vol,
        CAST(pct_chg AS REAL) AS pct_chg,
        LAG(close, 1) OVER (ORDER BY trade_date) AS close_prev,
        LEAD(close, 1) OVER (ORDER BY trade_date) AS close_next,
        (CAST(close AS REAL) / NULLIF(LAG(close, 1) OVER (ORDER BY trade_date), 0.0) - 1.0) AS day_ret,
        (CAST(close AS REAL) / NULLIF(LAG(close, 3) OVER (ORDER BY trade_date), 0.0) - 1.0) AS ret_3,
        (CAST(close AS REAL) / NULLIF(LAG(close, 5) OVER (ORDER BY trade_date), 0.0) - 1.0) AS ret_5
    FROM daily
),
feat AS (
    SELECT
        trade_date,
        open, high, low, close, vol, pct_chg,
        CASE WHEN COUNT(close) OVER (ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) >= 5
            THEN AVG(close) OVER (ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) END AS ma5,
        CASE WHEN COUNT(close) OVER (ORDER BY trade_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) >= 10
            THEN AVG(close) OVER (ORDER BY trade_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) END AS ma10,
        CASE WHEN COUNT(close) OVER (ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) >= 20
            THEN AVG(close) OVER (ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) END AS ma20,
        day_ret AS ret_1,
        ret_3,
        ret_5,
        CASE WHEN COUNT(day_ret) OVER (ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) >= 5
            THEN (AVG(day_ret * day_ret) OVER (ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)
                - AVG(day_ret) OVER (ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)
                * AVG(day_ret) OVER (ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW))
            END AS vol_var_5,
        CASE WHEN COUNT(day_ret) OVER (ORDER BY trade_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) >= 10
            THEN (AVG(day_ret * day_ret) OVER (ORDER BY trade_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)
                - AVG(day_ret) OVER (ORDER BY trade_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)
                * AVG(day_ret) OVER (ORDER BY trade_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW))
            END AS vol_var_10,
        CASE WHEN COUNT(day_ret) OVER (ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) >= 20
            THEN (AVG(day_ret * day_ret) OVER (ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
                - AVG(day_ret) OVER (ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
                * AVG(day_ret) OVER (ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW))
            END AS vol_var_20,
        (high - low) / (ABS(close) + 1e-9) AS intraday_range,
        (open - close_prev) / (ABS(close_prev) + 1e-9) AS overnight_gap,
        CASE WHEN COUNT(vol) OVER (ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) >= 5
            THEN AVG(vol) OVER (ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) END AS vol_ma5,
        CASE WHEN COUNT(vol) OVER (ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) >= 20
            THEN AVG(vol) OVER (ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) END AS vol_ma20,
        close_next
    FROM base
)
SELECT
    trade_date,
    open, high, low, close, vol, pct_chg,
    ma5, ma10, ma20,
    ret_1, ret_3, ret_5,
    vol_var_5, vol_var_10, vol_var_20,
    intraday_range,
    overnight_gap,
    (close - ma5) / (ABS(ma5) + 1e-9) AS close_ma5_gap,
    (close - ma10) / (ABS(ma10) + 1e-9) AS close_ma10_gap,
    (close - ma20) / (ABS(ma20) + 1e-9) AS close_ma20_gap,
    (close - low) / (ABS(high - low) + 1e-9) AS close_pos_in_day,
    vol_ma5,
    vol_ma20,
    vol_ma5 / (ABS(vol_ma20) + 1e-9) AS vol_ratio_5_20,
    CASE WHEN close_next > close THEN 1 ELSE 0 END AS target_up_raw
FROM feat
ORDER BY trade_date
"""

    with sqlite3.connect(":memory:") as conn:
        frame.to_sql("daily", conn, index=False, if_exists="replace")
        out = pd.read_sql_query(query, conn)

    for win in (5, 10, 20):
        var_col = f"vol_var_{win}"
        vol_col = f"volatility_{win}"
        out[var_col] = pd.to_numeric(out[var_col], errors="coerce")
        out[vol_col] = np.sqrt(np.clip(out[var_col].to_numpy(dtype=float), a_min=0.0, a_max=None))
    out = out.drop(columns=["vol_var_5", "vol_var_10", "vol_var_20"])
    out[config.data.label_col] = pd.to_numeric(out["target_up_raw"], errors="coerce").astype("Int64")
    out = out.drop(columns=["target_up_raw"])

    return _finalize_training_frame(out, config)
