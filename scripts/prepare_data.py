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

from aurora_ml.config import load_config
from aurora_ml.data_pipeline import prepare_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare raw and processed dataset for Aurora v0.3")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to yaml config")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh raw data from Tushare even if local snapshot exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()
    config = load_config(args.config)
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    prepared = prepare_data(config=config, refresh=args.refresh, tushare_token=token)

    print(f"[Aurora] raw rows: {len(prepared.raw_df)}")
    print(f"[Aurora] train rows: {len(prepared.train_df)}")
    print(f"[Aurora] raw path: {prepared.raw_path}")
    print(f"[Aurora] processed path: {prepared.processed_path}")
    print(f"[Aurora] raw sha256: {prepared.raw_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
