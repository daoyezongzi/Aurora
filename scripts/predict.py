from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aurora_ml.config import load_config
from aurora_ml.infer import predict_latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latest-window prediction with trained Aurora model")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to yaml config")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()
    config = load_config(args.config)

    result = predict_latest(config)
    print("[Aurora] latest prediction")
    print(f"[Aurora] trade_date={result.trade_date}")
    print(f"[Aurora] up_prob={result.up_prob:.6f}")
    print(f"[Aurora] threshold={result.threshold:.2f}")
    print(f"[Aurora] y_pred={result.y_pred}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
