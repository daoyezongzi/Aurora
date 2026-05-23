from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from aurora_ml.config import AuroraConfig
from aurora_ml.model import LSTMClassifier
from aurora_ml.utils import resolve_device


@dataclass(frozen=True)
class PredictResult:
    trade_date: str
    up_prob: float
    y_pred: int
    threshold: float


def predict_latest(config: AuroraConfig) -> PredictResult:
    processed_path = config.data.processed_path
    if not processed_path.exists():
        raise FileNotFoundError(
            f"{processed_path} not found. Run scripts/prepare_data.py before prediction."
        )

    frame = pd.read_csv(processed_path)
    seq_len = config.split.sequence_length
    feature_cols = config.data.feature_cols
    if len(frame) < seq_len:
        raise ValueError(f"not enough rows in processed data: need at least {seq_len}")

    checkpoint = torch.load(config.output.model_path, map_location="cpu")
    scaler = np.load(config.output.scaler_path, allow_pickle=True)
    mean = scaler["mean"].astype(np.float32)
    scale = scaler["scale"].astype(np.float32)

    model = LSTMClassifier(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    device = resolve_device(config.project.device_policy)
    model = model.to(device)

    recent = frame.iloc[-seq_len:].copy()
    features = recent[feature_cols].to_numpy(dtype=np.float32)
    scaled = (features - mean) / np.where(scale == 0, 1.0, scale)
    batch = torch.from_numpy(scaled[None, ...]).to(device)

    with torch.no_grad():
        logits = model(batch).item()
        up_prob = float(1.0 / (1.0 + np.exp(-logits)))

    threshold = float(checkpoint.get("threshold", config.train.threshold))
    y_pred = 1 if up_prob >= threshold else 0
    return PredictResult(
        trade_date=str(recent.iloc[-1]["trade_date"]),
        up_prob=up_prob,
        y_pred=y_pred,
        threshold=threshold,
    )
