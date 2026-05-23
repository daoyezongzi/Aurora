from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from aurora_ml.config import AuroraConfig, load_config
from aurora_ml.data_pipeline import prepare_data
from aurora_ml.dataset import split_by_time
from aurora_ml.train import run_training
from aurora_ml.utils import dump_json, ensure_dir


@dataclass(frozen=True)
class FlatData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    dates_test: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v0.2 model comparison and generate chart files")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to yaml config")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh raw data from Tushare before training",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()
    cfg = load_config(args.config)
    frame = load_or_prepare_frame(cfg, refresh=args.refresh)

    out_root = ROOT / "outputs" / "v2"
    chart_dir = out_root / "charts"
    ensure_dir(out_root)
    ensure_dir(chart_dir)

    lstm_row, lstm_pred = run_lstm_branch(cfg, frame, out_root)

    flat = build_flat_data(frame, cfg)
    lr_model, lr_note = fit_logreg(flat, seed=cfg.project.seed)
    mlp_model, mlp_note = fit_mlp(flat, seed=cfg.project.seed)

    lr_row, lr_pred = evaluate_and_save("logreg", lr_model, lr_note, flat, out_root)
    mlp_row, mlp_pred = evaluate_and_save("mlp", mlp_model, mlp_note, flat, out_root)

    rows = [lstm_row, lr_row, mlp_row]
    table_full = pd.DataFrame(rows)
    table_view = table_full[
        ["model", "accuracy", "f1", "precision", "recall", "roc_auc", "sample_count", "threshold"]
    ]
    table_view.to_csv(out_root / "compare_metrics.csv", index=False, encoding="utf-8")
    dump_json(out_root / "compare_metrics.json", {"rows": rows})

    pred_map = {"lstm": lstm_pred, "logreg": lr_pred, "mlp": mlp_pred}
    make_charts(table_full, pred_map, chart_dir)

    print("[Aurora] v2 done")
    print(f"[Aurora] compare table: {out_root / 'compare_metrics.csv'}")
    print(f"[Aurora] charts: {chart_dir}")
    print("[Aurora] model artifacts:")
    print(f"  - {out_root / 'lstm'}")
    print(f"  - {out_root / 'logreg'}")
    print(f"  - {out_root / 'mlp'}")
    return 0


def load_or_prepare_frame(cfg: AuroraConfig, refresh: bool) -> pd.DataFrame:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    need_prepare = refresh or (not cfg.data.processed_path.exists())
    if need_prepare:
        prepared = prepare_data(config=cfg, refresh=refresh, tushare_token=token)
        return prepared.train_df
    return pd.read_csv(cfg.data.processed_path)


def run_lstm_branch(cfg: AuroraConfig, frame: pd.DataFrame, out_root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    out_dir = out_root / "lstm"
    ensure_dir(out_dir)

    out_cfg = replace(
        cfg.output,
        model_path=out_dir / "model.pt",
        metrics_path=out_dir / "metrics.json",
        predictions_path=out_dir / "predictions.csv",
        scaler_path=out_dir / "scaler.npz",
    )
    project_cfg = replace(cfg.project, version="v0.2-lstm")
    local_cfg = replace(cfg, project=project_cfg, output=out_cfg)

    _, artifacts = run_training(config=local_cfg, frame=frame)
    pred = pd.read_csv(artifacts.predictions_path)

    y_true = pred["y_true"].to_numpy(dtype=np.int64)
    y_prob = pred["up_prob"].to_numpy(dtype=np.float64)
    row = build_metric_row(model_name="lstm", y_true=y_true, y_prob=y_prob, threshold=cfg.train.threshold)
    row["sample_count"] = int(len(pred))
    dump_json(out_dir / "metrics_v2.json", row)
    return row, pred


def build_flat_data(frame: pd.DataFrame, cfg: AuroraConfig) -> FlatData:
    train_raw, val_raw, test_raw = split_by_time(
        frame=frame,
        train_ratio=cfg.split.train_ratio,
        val_ratio=cfg.split.val_ratio,
        test_ratio=cfg.split.test_ratio,
    )

    seq_len = cfg.split.sequence_length
    feat_cols = cfg.data.feature_cols
    label_col = cfg.data.label_col

    train_df = train_raw.iloc[seq_len - 1 :].copy()
    val_df = val_raw.iloc[seq_len - 1 :].copy()
    test_df = test_raw.iloc[seq_len - 1 :].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("split is too short for v2 flat models after sequence warmup cut.")

    scaler = StandardScaler()
    scaler.fit(train_raw[feat_cols].to_numpy(dtype=np.float32))

    x_train = scaler.transform(train_df[feat_cols].to_numpy(dtype=np.float32))
    y_train = train_df[label_col].to_numpy(dtype=np.int64)
    x_val = scaler.transform(val_df[feat_cols].to_numpy(dtype=np.float32))
    y_val = val_df[label_col].to_numpy(dtype=np.int64)
    x_test = scaler.transform(test_df[feat_cols].to_numpy(dtype=np.float32))
    y_test = test_df[label_col].to_numpy(dtype=np.int64)
    dates_test = test_df["trade_date"].astype(str).tolist()

    return FlatData(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        dates_test=dates_test,
    )


def fit_logreg(data: FlatData, seed: int) -> tuple[LogisticRegression, dict[str, Any]]:
    best_note: dict[str, Any] = {}
    best_f1 = -1.0

    for c_value in [0.1, 0.5, 1.0, 3.0, 10.0]:
        model = LogisticRegression(C=c_value, max_iter=2000, solver="lbfgs", random_state=seed)
        model.fit(data.x_train, data.y_train)
        val_prob = model.predict_proba(data.x_val)[:, 1]
        val_pred = (val_prob >= 0.5).astype(np.int64)
        val_f1 = float(f1_score(data.y_val, val_pred, zero_division=0))
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_note = {"C": c_value, "val_f1": val_f1}

    if not best_note:
        raise RuntimeError("failed to fit logistic regression")

    x_fit = np.concatenate([data.x_train, data.x_val], axis=0)
    y_fit = np.concatenate([data.y_train, data.y_val], axis=0)
    final_model = LogisticRegression(C=float(best_note["C"]), max_iter=2000, solver="lbfgs", random_state=seed)
    final_model.fit(x_fit, y_fit)
    return final_model, best_note


def fit_mlp(data: FlatData, seed: int) -> tuple[MLPClassifier, dict[str, Any]]:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    best_note: dict[str, Any] = {}
    best_f1 = -1.0

    grid = [
        ((32,), 1e-4),
        ((64,), 1e-4),
        ((64, 32), 1e-4),
        ((64, 32), 1e-3),
    ]
    for hidden, alpha in grid:
        model = MLPClassifier(
            hidden_layer_sizes=hidden,
            alpha=alpha,
            learning_rate_init=1e-3,
            max_iter=500,
            random_state=seed,
        )
        model.fit(data.x_train, data.y_train)
        val_prob = model.predict_proba(data.x_val)[:, 1]
        val_pred = (val_prob >= 0.5).astype(np.int64)
        val_f1 = float(f1_score(data.y_val, val_pred, zero_division=0))
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_note = {"hidden": list(hidden), "alpha": alpha, "val_f1": val_f1}

    if not best_note:
        raise RuntimeError("failed to fit MLP")

    x_fit = np.concatenate([data.x_train, data.x_val], axis=0)
    y_fit = np.concatenate([data.y_train, data.y_val], axis=0)
    final_model = MLPClassifier(
        hidden_layer_sizes=tuple(best_note["hidden"]),
        alpha=float(best_note["alpha"]),
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=seed,
    )
    final_model.fit(x_fit, y_fit)
    return final_model, best_note


def evaluate_and_save(
    name: str,
    model: LogisticRegression | MLPClassifier,
    note: dict[str, Any],
    data: FlatData,
    out_root: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    out_dir = out_root / name
    ensure_dir(out_dir)

    y_prob = model.predict_proba(data.x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(np.int64)

    row = build_metric_row(model_name=name, y_true=data.y_test, y_prob=y_prob, threshold=0.5)
    row["sample_count"] = int(len(data.y_test))
    row["tune_note"] = note

    pred = pd.DataFrame(
        {
            "trade_date": data.dates_test,
            "y_true": data.y_test,
            "up_prob": y_prob,
            "y_pred": y_pred,
        }
    )
    dump_json(out_dir / "metrics.json", row)
    pred.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8")
    return row, pred


def build_metric_row(model_name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    row: dict[str, Any] = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": matrix,
    }
    try:
        row["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        row["roc_auc"] = float("nan")
    return row


def make_charts(table: pd.DataFrame, pred_map: dict[str, pd.DataFrame], chart_dir: Path) -> None:
    ensure_dir(chart_dir)
    make_metric_bar_svg(table, chart_dir / "metric_bars.svg")
    make_roc_svg(pred_map, chart_dir / "roc_curves.svg")
    make_confusion_svg(table, chart_dir / "confusion_matrices.svg")


def make_metric_bar_svg(table: pd.DataFrame, out_path: Path) -> None:
    models = table["model"].tolist()
    metrics = ["accuracy", "f1", "roc_auc"]
    colors = {"accuracy": "#4c78a8", "f1": "#59a14f", "roc_auc": "#f28e2b"}

    width = 1000
    height = 520
    left = 90
    right = 40
    top = 60
    bottom = 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    parts: list[str] = []
    parts.append(svg_text(30, 34, "v0.2 model comparison", 24, "#111"))
    parts.append(svg_line(left, top + plot_h, left + plot_w, top + plot_h, "#333", 2))
    parts.append(svg_line(left, top, left, top + plot_h, "#333", 2))

    for i in range(6):
        score = i * 0.2
        y = top + plot_h * (1.0 - score)
        parts.append(svg_line(left, y, left + plot_w, y, "#e4e4e4", 1))
        parts.append(svg_text(left - 42, y + 5, f"{score:.1f}", 13, "#444"))

    group_w = plot_w / max(len(models), 1)
    bar_w = group_w / 5.0
    for m_idx, model_name in enumerate(models):
        x0 = left + m_idx * group_w
        parts.append(svg_text(x0 + group_w * 0.42, top + plot_h + 28, model_name, 14, "#222"))
        for k_idx, metric_name in enumerate(metrics):
            value = float(table.iloc[m_idx][metric_name])
            if np.isnan(value):
                value = 0.0
            bar_h = plot_h * max(0.0, min(1.0, value))
            x = x0 + bar_w * (k_idx + 1)
            y = top + plot_h - bar_h
            parts.append(svg_rect(x, y, bar_w * 0.85, bar_h, colors[metric_name], "#ffffff"))
            parts.append(svg_text(x + 2, y - 6, f"{value:.3f}", 11, "#333"))

    legend_x = left + plot_w - 220
    legend_y = top + 12
    for idx, metric_name in enumerate(metrics):
        y = legend_y + idx * 22
        parts.append(svg_rect(legend_x, y - 10, 16, 12, colors[metric_name], "#ffffff"))
        parts.append(svg_text(legend_x + 24, y, metric_name, 13, "#333"))

    write_svg(out_path, width, height, "".join(parts))


def make_roc_svg(pred_map: dict[str, pd.DataFrame], out_path: Path) -> None:
    width = 760
    height = 620
    left = 80
    right = 40
    top = 60
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    colors = {"lstm": "#4c78a8", "logreg": "#59a14f", "mlp": "#f28e2b"}
    parts: list[str] = []
    parts.append(svg_text(30, 34, "ROC curves", 24, "#111"))
    parts.append(svg_line(left, top + plot_h, left + plot_w, top + plot_h, "#333", 2))
    parts.append(svg_line(left, top, left, top + plot_h, "#333", 2))

    for i in range(6):
        tick = i * 0.2
        x = left + plot_w * tick
        y = top + plot_h * (1.0 - tick)
        parts.append(svg_line(x, top, x, top + plot_h, "#efefef", 1))
        parts.append(svg_line(left, y, left + plot_w, y, "#efefef", 1))
        parts.append(svg_text(x - 10, top + plot_h + 25, f"{tick:.1f}", 12, "#444"))
        parts.append(svg_text(left - 38, y + 5, f"{tick:.1f}", 12, "#444"))

    parts.append(
        svg_polyline(
            [(left, top + plot_h), (left + plot_w, top)],
            stroke="#888",
            stroke_width=1.5,
            fill="none",
            dash="6 4",
        )
    )

    legend_x = left + plot_w - 170
    legend_y = top + 18
    for idx, (name, pred) in enumerate(pred_map.items()):
        y_true = pred["y_true"].to_numpy(dtype=np.int64)
        y_prob = pred["up_prob"].to_numpy(dtype=np.float64)
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        points = []
        for fp, tp in zip(fpr, tpr):
            px = left + plot_w * float(fp)
            py = top + plot_h * (1.0 - float(tp))
            points.append((px, py))
        parts.append(svg_polyline(points, stroke=colors.get(name, "#333"), stroke_width=2.5, fill="none"))

        y = legend_y + idx * 22
        parts.append(svg_line(legend_x, y - 5, legend_x + 18, y - 5, colors.get(name, "#333"), 3))
        parts.append(svg_text(legend_x + 24, y, name, 13, "#333"))

    parts.append(svg_text(width // 2 - 20, height - 20, "FPR", 14, "#333"))
    parts.append(svg_text(16, height // 2, "TPR", 14, "#333", rotate=-90))
    write_svg(out_path, width, height, "".join(parts))


def make_confusion_svg(table: pd.DataFrame, out_path: Path) -> None:
    models = table["model"].tolist()
    cell = 70
    panel_w = 250
    width = 40 + len(models) * panel_w
    height = 340
    parts: list[str] = []
    parts.append(svg_text(20, 34, "Confusion matrix by model", 22, "#111"))

    for idx, model_name in enumerate(models):
        row = table.loc[table["model"] == model_name].iloc[0]
        matrix = np.array(row["confusion_matrix"], dtype=np.int64)
        peak = max(int(matrix.max()), 1)

        x0 = 30 + idx * panel_w
        y0 = 90
        parts.append(svg_text(x0 + 60, 72, model_name, 16, "#222"))
        parts.append(svg_text(x0 + 34, y0 - 12, "pred 0", 11, "#444"))
        parts.append(svg_text(x0 + 108, y0 - 12, "pred 1", 11, "#444"))
        parts.append(svg_text(x0 - 2, y0 + 42, "true 0", 11, "#444", rotate=-90))
        parts.append(svg_text(x0 - 2, y0 + 112, "true 1", 11, "#444", rotate=-90))

        for i in range(2):
            for j in range(2):
                value = int(matrix[i, j])
                scale = value / peak
                shade = 245 - int(150 * scale)
                fill = f"rgb({shade},{shade},255)"
                x = x0 + j * cell
                y = y0 + i * cell
                parts.append(svg_rect(x, y, cell, cell, fill, "#6d8dc4"))
                parts.append(svg_text(x + 25, y + 42, str(value), 16, "#111"))

    write_svg(out_path, width, height, "".join(parts))


def write_svg(path: Path, width: int, height: int, body: str) -> None:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">{body}</svg>'
    )
    path.write_text(svg, encoding="utf-8")


def svg_line(x1: float, y1: float, x2: float, y2: float, color: str, width: float) -> str:
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{color}" stroke-width="{width:.2f}" />'
    )


def svg_rect(x: float, y: float, w: float, h: float, fill: str, stroke: str) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1" />'
    )


def svg_text(x: float, y: float, value: str, size: int, color: str, rotate: int = 0) -> str:
    if rotate == 0:
        return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" font-family="Arial">{value}</text>'
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" font-family="Arial" '
        f'transform="rotate({rotate} {x:.2f} {y:.2f})">{value}</text>'
    )


def svg_polyline(
    points: list[tuple[float, float]],
    stroke: str,
    stroke_width: float,
    fill: str,
    dash: str | None = None,
) -> str:
    raw = " ".join([f"{x:.2f},{y:.2f}" for x, y in points])
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{raw}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width:.2f}"{dash_attr} />'
    )


if __name__ == "__main__":
    raise SystemExit(main())
