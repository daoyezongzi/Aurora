from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from aurora_ml.config import load_config
from aurora_ml.data_pipeline import prepare_data
from aurora_ml.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Aurora LSTM classifier")
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
    config = load_config(args.config)

    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if args.refresh or not config.data.processed_path.exists():
        prepared = prepare_data(config=config, refresh=args.refresh, tushare_token=token)
        frame = prepared.train_df
    else:
        frame = pd.read_csv(config.data.processed_path)

    metrics, artifacts = run_training(config=config, frame=frame)
    print("[Aurora] training done")
    print(f"[Aurora] accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")
    print(f"[Aurora] model: {artifacts.model_path}")
    print(f"[Aurora] metrics: {artifacts.metrics_path}")
    print(f"[Aurora] predictions: {artifacts.predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
