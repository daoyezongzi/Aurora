from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from aurora_ml.config import load_config
from aurora_ml.model import LSTMClassifier
from aurora_ml.train import TrainArtifacts, save_artifacts


def test_training_artifacts_are_created(tmp_path: Path) -> None:
    config = load_config("configs/default.yaml")
    output = replace(
        config.output,
        model_path=tmp_path / "model.pt",
        metrics_path=tmp_path / "metrics.json",
        predictions_path=tmp_path / "predictions.csv",
        scaler_path=tmp_path / "scaler.npz",
    )
    local_config = replace(config, output=output)

    model = LSTMClassifier(input_size=len(local_config.data.feature_cols))
    scaler = StandardScaler()
    scaler.fit(np.random.randn(32, len(local_config.data.feature_cols)).astype(np.float32))

    metrics = {
        "accuracy": 0.5,
        "f1": 0.4,
        "confusion_matrix": [[1, 1], [1, 1]],
    }
    predictions = pd.DataFrame(
        {
            "trade_date": ["2026-01-01", "2026-01-02"],
            "y_true": [1, 0],
            "up_prob": [0.6, 0.4],
            "y_pred": [1, 0],
        }
    )
    artifacts = TrainArtifacts(
        model_path=output.model_path,
        metrics_path=output.metrics_path,
        predictions_path=output.predictions_path,
        scaler_path=output.scaler_path,
    )

    save_artifacts(model, scaler, metrics, predictions, artifacts, local_config)

    assert output.model_path.exists()
    assert output.metrics_path.exists()
    assert output.predictions_path.exists()
    assert output.scaler_path.exists()
