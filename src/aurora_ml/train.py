from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aurora_ml.config import AuroraConfig
from aurora_ml.dataset import build_sequences, split_by_time
from aurora_ml.model import LSTMClassifier
from aurora_ml.utils import dump_json, ensure_parent, resolve_device, set_seed


@dataclass(frozen=True)
class TrainArtifacts:
    model_path: Path
    metrics_path: Path
    predictions_path: Path
    scaler_path: Path


def run_training(config: AuroraConfig, frame: pd.DataFrame) -> tuple[dict[str, Any], TrainArtifacts]:
    set_seed(config.project.seed)
    device = resolve_device(config.project.device_policy)

    train_df, val_df, test_df = split_by_time(
        frame=frame,
        train_ratio=config.split.train_ratio,
        val_ratio=config.split.val_ratio,
        test_ratio=config.split.test_ratio,
    )

    feature_cols = config.data.feature_cols
    label_col = config.data.label_col
    seq_len = config.split.sequence_length

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].to_numpy(dtype=np.float32))

    x_train, y_train, _ = _to_sequences(train_df, feature_cols, label_col, seq_len, scaler)
    x_val, y_val, _ = _to_sequences(val_df, feature_cols, label_col, seq_len, scaler)
    x_test, y_test, test_dates = _to_sequences(test_df, feature_cols, label_col, seq_len, scaler)

    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        raise ValueError(
            "sequence generation produced empty split. "
            "Try increasing data span or reducing sequence_length."
        )

    train_loader = _make_loader(x_train, y_train, batch_size=config.train.batch_size, shuffle=False)
    val_loader = _make_loader(x_val, y_val, batch_size=config.train.batch_size, shuffle=False)
    test_loader = _make_loader(x_test, y_test, batch_size=config.train.batch_size, shuffle=False)

    model = LSTMClassifier(
        input_size=len(feature_cols),
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience_left = config.train.early_stopping_patience
    train_losses: list[float] = []
    val_losses: list[float] = []

    for _epoch in range(1, config.train.epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_train_loss += float(loss.item()) * batch_x.size(0)
            train_count += batch_x.size(0)
        epoch_train_loss = running_train_loss / max(train_count, 1)
        train_losses.append(epoch_train_loss)

        epoch_val_loss = _evaluate_loss(model, val_loader, criterion, device)
        val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = config.train.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)

    test_logits, test_labels = _collect_logits_and_labels(model, test_loader, device)
    test_probs = _sigmoid_np(test_logits)
    test_preds = (test_probs >= config.train.threshold).astype(np.int64)
    test_labels_int = test_labels.astype(np.int64)

    metrics = {
        "project": config.project.name,
        "version": config.project.version,
        "seed": config.project.seed,
        "device": str(device),
        "data_code": config.data.code,
        "label_rule": config.data.label_rule,
        "sequence_length": seq_len,
        "feature_cols": feature_cols,
        "split_rows": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "sequence_samples": {
            "train": int(len(x_train)),
            "val": int(len(x_val)),
            "test": int(len(x_test)),
        },
        "train_loss_last": float(train_losses[-1]),
        "val_loss_best": float(best_val_loss),
        "accuracy": float(accuracy_score(test_labels_int, test_preds)),
        "f1": float(f1_score(test_labels_int, test_preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(test_labels_int, test_preds, labels=[0, 1]).tolist(),
        "threshold": float(config.train.threshold),
    }

    pred_df = pd.DataFrame(
        {
            "trade_date": test_dates,
            "y_true": test_labels_int,
            "up_prob": test_probs,
            "y_pred": test_preds,
        }
    )

    artifacts = TrainArtifacts(
        model_path=config.output.model_path,
        metrics_path=config.output.metrics_path,
        predictions_path=config.output.predictions_path,
        scaler_path=config.output.scaler_path,
    )
    save_artifacts(
        model=model,
        scaler=scaler,
        metrics=metrics,
        predictions=pred_df,
        artifacts=artifacts,
        config=config,
    )
    return metrics, artifacts


def save_artifacts(
    model: LSTMClassifier,
    scaler: StandardScaler,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    artifacts: TrainArtifacts,
    config: AuroraConfig,
) -> None:
    ensure_parent(artifacts.model_path)
    ensure_parent(artifacts.metrics_path)
    ensure_parent(artifacts.predictions_path)
    ensure_parent(artifacts.scaler_path)

    checkpoint = {
        "state_dict": model.state_dict(),
        "input_size": len(config.data.feature_cols),
        "hidden_size": config.model.hidden_size,
        "num_layers": config.model.num_layers,
        "dropout": config.model.dropout,
        "sequence_length": config.split.sequence_length,
        "feature_cols": config.data.feature_cols,
        "threshold": config.train.threshold,
    }
    torch.save(checkpoint, artifacts.model_path)
    dump_json(artifacts.metrics_path, metrics)
    predictions.to_csv(artifacts.predictions_path, index=False, encoding="utf-8")
    np.savez(
        artifacts.scaler_path,
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        feature_cols=np.array(config.data.feature_cols, dtype=object),
    )


def _to_sequences(
    split_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    sequence_length: int,
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    raw_features = split_df[feature_cols].to_numpy(dtype=np.float32)
    scaled_features = scaler.transform(raw_features).astype(np.float32)
    labels = split_df[label_col].to_numpy(dtype=np.float32)
    seq_x, seq_y = build_sequences(scaled_features, labels, sequence_length)

    dates = split_df["trade_date"].astype(str).tolist()
    seq_dates = dates[sequence_length - 1 :] if len(dates) >= sequence_length else []
    return seq_x, seq_y, seq_dates


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    running_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            running_loss += float(loss.item()) * batch_x.size(0)
            sample_count += batch_x.size(0)
    return running_loss / max(sample_count, 1)


def _collect_logits_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x).detach().cpu().numpy()
            logits_list.append(logits)
            label_list.append(batch_y.detach().cpu().numpy())
    return np.concatenate(logits_list), np.concatenate(label_list)


def _sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))
