from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    version: str
    seed: int
    device_policy: str


@dataclass(frozen=True)
class DataConfig:
    code: str
    start_date: str
    end_date: str
    raw_path: Path
    processed_path: Path
    expected_sha256: str
    refresh_by_default: bool
    label_col: str
    label_rule: str
    feature_cols: list[str]


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float
    sequence_length: int


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    num_layers: int
    dropout: float


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    learning_rate: float
    epochs: int
    early_stopping_patience: int
    threshold: float
    optimizer: str
    loss: str


@dataclass(frozen=True)
class OutputConfig:
    model_path: Path
    metrics_path: Path
    predictions_path: Path
    scaler_path: Path


@dataclass(frozen=True)
class AuroraConfig:
    project: ProjectConfig
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    train: TrainConfig
    output: OutputConfig


def load_config(path: str | Path) -> AuroraConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file_obj:
        raw = yaml.safe_load(file_obj) or {}

    project = _to_project(raw.get("project", {}))
    data = _to_data(raw.get("data", {}))
    split = _to_split(raw.get("split", {}))
    model = _to_model(raw.get("model", {}))
    train = _to_train(raw.get("train", {}))
    output = _to_output(raw.get("output", {}))

    _validate_ratios(split.train_ratio, split.val_ratio, split.test_ratio)

    return AuroraConfig(
        project=project,
        data=data,
        split=split,
        model=model,
        train=train,
        output=output,
    )


def _to_project(raw: dict[str, Any]) -> ProjectConfig:
    return ProjectConfig(
        name=str(raw.get("name", "Aurora")),
        version=str(raw.get("version", "v0.3")),
        seed=int(raw.get("seed", 42)),
        device_policy=str(raw.get("device_policy", "auto")),
    )


def _to_data(raw: dict[str, Any]) -> DataConfig:
    return DataConfig(
        code=str(raw.get("code", "002372.SZ")),
        start_date=str(raw.get("start_date", "2021-01-01")),
        end_date=str(raw.get("end_date", "latest")),
        raw_path=Path(str(raw.get("raw_path", "data/raw/002372.SZ_daily.csv"))),
        processed_path=Path(str(raw.get("processed_path", "data/processed/002372.SZ_features.csv"))),
        expected_sha256=str(raw.get("expected_sha256", "")).strip().lower(),
        refresh_by_default=bool(raw.get("refresh_by_default", False)),
        label_col=str(raw.get("label_col", "target_up")),
        label_rule=str(raw.get("label_rule", "close_t+1 > close_t")),
        feature_cols=[str(col) for col in raw.get("feature_cols", [])],
    )


def _to_split(raw: dict[str, Any]) -> SplitConfig:
    return SplitConfig(
        train_ratio=float(raw.get("train_ratio", 0.7)),
        val_ratio=float(raw.get("val_ratio", 0.15)),
        test_ratio=float(raw.get("test_ratio", 0.15)),
        sequence_length=int(raw.get("sequence_length", 20)),
    )


def _to_model(raw: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        hidden_size=int(raw.get("hidden_size", 64)),
        num_layers=int(raw.get("num_layers", 1)),
        dropout=float(raw.get("dropout", 0.0)),
    )


def _to_train(raw: dict[str, Any]) -> TrainConfig:
    return TrainConfig(
        batch_size=int(raw.get("batch_size", 32)),
        learning_rate=float(raw.get("learning_rate", 0.001)),
        epochs=int(raw.get("epochs", 30)),
        early_stopping_patience=int(raw.get("early_stopping_patience", 5)),
        threshold=float(raw.get("threshold", 0.5)),
        optimizer=str(raw.get("optimizer", "Adam")),
        loss=str(raw.get("loss", "BCEWithLogitsLoss")),
    )


def _to_output(raw: dict[str, Any]) -> OutputConfig:
    return OutputConfig(
        model_path=Path(str(raw.get("model_path", "outputs/model.pt"))),
        metrics_path=Path(str(raw.get("metrics_path", "outputs/metrics.json"))),
        predictions_path=Path(str(raw.get("predictions_path", "outputs/predictions.csv"))),
        scaler_path=Path(str(raw.get("scaler_path", "outputs/scaler.npz"))),
    )


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total:.6f}")
