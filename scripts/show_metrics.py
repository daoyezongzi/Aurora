from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aurora_ml.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show Aurora metrics summary")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to yaml config")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    metrics_path = Path(config.output.metrics_path)
    preds_path = Path(config.output.predictions_path)
    if not metrics_path.exists():
        print(f"[Aurora] metrics file not found: {metrics_path}")
        return 1

    with metrics_path.open("r", encoding="utf-8") as file_obj:
        metrics = json.load(file_obj)

    print("[Aurora] metrics summary")
    print(f"  accuracy: {metrics.get('accuracy')}")
    print(f"  f1: {metrics.get('f1')}")
    print(f"  threshold: {metrics.get('threshold')}")
    print(f"  confusion_matrix: {metrics.get('confusion_matrix')}")
    print(f"  device: {metrics.get('device')}")
    print(f"  split_rows: {metrics.get('split_rows')}")
    print(f"  sequence_samples: {metrics.get('sequence_samples')}")

    if preds_path.exists():
        pred_info = summarize_predictions(preds_path)
        print("[Aurora] prediction summary")
        print(f"  rows: {pred_info['rows']}")
        print(f"  y_true_counts: {pred_info['y_true_counts']}")
        print(f"  y_pred_counts: {pred_info['y_pred_counts']}")
        print(f"  up_prob_min: {pred_info['up_prob_min']}")
        print(f"  up_prob_max: {pred_info['up_prob_max']}")
        print(f"  up_prob_mean: {pred_info['up_prob_mean']}")
    else:
        print(f"[Aurora] predictions file not found: {preds_path}")
    return 0


def summarize_predictions(path: Path) -> dict[str, object]:
    rows = 0
    true_counts: dict[str, int] = {}
    pred_counts: dict[str, int] = {}
    probs: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            rows += 1
            y_true = (row.get("y_true") or "").strip()
            y_pred = (row.get("y_pred") or "").strip()
            true_counts[y_true] = true_counts.get(y_true, 0) + 1
            pred_counts[y_pred] = pred_counts.get(y_pred, 0) + 1
            try:
                probs.append(float(row.get("up_prob", "nan")))
            except ValueError:
                continue

    if probs:
        up_prob_min = min(probs)
        up_prob_max = max(probs)
        up_prob_mean = sum(probs) / len(probs)
    else:
        up_prob_min = None
        up_prob_max = None
        up_prob_mean = None

    return {
        "rows": rows,
        "y_true_counts": true_counts,
        "y_pred_counts": pred_counts,
        "up_prob_min": up_prob_min,
        "up_prob_max": up_prob_max,
        "up_prob_mean": up_prob_mean,
    }


if __name__ == "__main__":
    raise SystemExit(main())
