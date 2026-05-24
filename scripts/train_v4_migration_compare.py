from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aurora_ml.config import AuroraConfig, load_config
from aurora_ml.data_pipeline import build_training_frame, prepare_data
from aurora_ml.utils import dump_json, ensure_dir
import train_v3_robust as v3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train v0.4 migration compare: pandas features vs SQL time-series features."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to yaml config")
    parser.add_argument("--refresh", action="store_true", help="Refresh raw data from Tushare before compare")

    # Keep these aligned with v3 to make A/B results comparable.
    parser.add_argument("--thr-goal", default="f0.5", choices=["f0.5", "f1", "f2", "fbeta"])
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--thr-min", type=float, default=0.30)
    parser.add_argument("--thr-max", type=float, default=0.70)
    parser.add_argument("--thr-step", type=float, default=0.01)
    parser.add_argument("--thr-eps", type=float, default=0.01)
    parser.add_argument("--thr-anchor", type=float, default=0.50)
    parser.add_argument("--final-thr-mode", default="val", choices=["rolling_median", "val"])
    parser.add_argument("--cfg-eps", type=float, default=0.001)
    parser.add_argument("--min-precision", type=float, default=0.56)
    parser.add_argument("--min-recall", type=float, default=0.62)
    parser.add_argument("--goal-penalty", type=float, default=1.2)
    parser.add_argument("--mlp-max-iter", type=int, default=900)
    parser.add_argument("--calib", default="none", choices=["none", "sigmoid", "isotonic"])
    parser.add_argument("--calib-cv", type=int, default=3)
    parser.add_argument("--cfg-std-pen", type=float, default=0.20)
    parser.add_argument("--cfg-thr-pen", type=float, default=0.12)
    return parser.parse_args()


def evaluate_frame(
    frame: pd.DataFrame,
    cfg: AuroraConfig,
    opts: v3.TuneOpts,
    out_root: Path,
) -> dict[str, Any]:
    ensure_dir(out_root)
    ensure_dir(out_root / "final")
    ensure_dir(out_root / "charts")

    fold_rows, cfg_rows = v3.rolling_validate(frame, cfg, opts)
    fold_df = pd.DataFrame(fold_rows)
    cfg_df = pd.DataFrame(cfg_rows)
    fold_df.to_csv(out_root / "rolling_metrics.csv", index=False, encoding="utf-8")
    cfg_df.to_csv(out_root / "config_metrics.csv", index=False, encoding="utf-8")
    dump_json(out_root / "rolling_metrics.json", {"rows": fold_rows})
    dump_json(out_root / "config_metrics.json", {"rows": cfg_rows})

    best_cfg = v3.pick_best_config(cfg_rows, opts)
    rolling_thr = v3.pick_final_threshold(fold_rows, opts)
    final_row, pred_df = v3.run_final_eval(frame, cfg, best_cfg, opts, out_root / "final", rolling_thr)
    pred_df.to_csv(out_root / "final" / "predictions.csv", index=False, encoding="utf-8")
    dump_json(out_root / "final" / "metrics.json", final_row)

    err_df, err_summary = v3.build_error_analysis(pred_df, threshold=float(final_row["threshold"]))
    err_df.to_csv(out_root / "error_cases.csv", index=False, encoding="utf-8")
    dump_json(out_root / "error_summary.json", err_summary)
    v3.make_charts(fold_df, err_summary, out_root / "charts")

    roll_f1 = np.asarray(fold_df["test_f1"].to_list(), dtype=float)
    roll_thr = np.asarray(fold_df["best_threshold"].to_list(), dtype=float)
    return {
        "rows": int(len(frame)),
        "final_f1": float(final_row["f1"]),
        "final_precision": float(final_row["precision"]),
        "final_recall": float(final_row["recall"]),
        "final_accuracy": float(final_row["accuracy"]),
        "final_roc_auc": float(final_row["roc_auc"]),
        "final_threshold": float(final_row["threshold"]),
        "false_positive_count": int(err_summary["false_positive_count"]),
        "false_negative_count": int(err_summary["false_negative_count"]),
        "near_threshold_error_rate": float(err_summary["near_threshold_error_rate"]),
        "rolling_avg_f1": float(roll_f1.mean()),
        "rolling_min_f1": float(roll_f1.min()),
        "rolling_std_f1": float(roll_f1.std(ddof=0)),
        "rolling_std_threshold": float(roll_thr.std(ddof=0)),
        "best_hidden": str(best_cfg["hidden"]),
        "best_alpha": float(best_cfg["alpha"]),
    }


def compare_feature_frames(
    pandas_df: pd.DataFrame,
    sql_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    left = pandas_df[["trade_date", *feature_cols, label_col]].copy()
    right = sql_df[["trade_date", *feature_cols, label_col]].copy()
    merged = left.merge(right, on="trade_date", how="inner", suffixes=("_pandas", "_sql"))

    rows: list[dict[str, Any]] = []
    for col in feature_cols:
        p = pd.to_numeric(merged[f"{col}_pandas"], errors="coerce")
        s = pd.to_numeric(merged[f"{col}_sql"], errors="coerce")
        diff = (s - p).abs()
        rows.append(
            {
                "column": col,
                "count": int(diff.notna().sum()),
                "mean_abs_diff": float(diff.mean()),
                "max_abs_diff": float(diff.max()),
                "pandas_mean": float(p.mean()),
                "sql_mean": float(s.mean()),
                "mean_rel_diff": float(diff.mean() / (abs(float(p.mean())) + 1e-12)),
            }
        )

    lp = pd.to_numeric(merged[f"{label_col}_pandas"], errors="coerce").astype("Int64")
    ls = pd.to_numeric(merged[f"{label_col}_sql"], errors="coerce").astype("Int64")
    mismatch = int((lp != ls).fillna(True).sum())

    comp_df = pd.DataFrame(rows).sort_values("mean_abs_diff", ascending=False).reset_index(drop=True)
    summary = {
        "pandas_rows": int(len(pandas_df)),
        "sql_rows": int(len(sql_df)),
        "joined_rows": int(len(merged)),
        "label_mismatch_count": mismatch,
        "label_mismatch_rate": float(mismatch / max(len(merged), 1)),
        "feature_mean_abs_diff_avg": float(comp_df["mean_abs_diff"].mean()) if len(comp_df) > 0 else 0.0,
        "feature_max_abs_diff_max": float(comp_df["max_abs_diff"].max()) if len(comp_df) > 0 else 0.0,
    }
    return comp_df, summary


def main() -> int:
    args = parse_args()
    load_dotenv()
    cfg = load_config(args.config)
    opts = v3.build_tune_opts(args)

    token = os.getenv("TUSHARE_TOKEN", "").strip()
    prepared = prepare_data(
        config=cfg,
        refresh=args.refresh,
        tushare_token=token,
        feature_engine="pandas",
    )
    frame_pandas = prepared.train_df.copy()
    frame_sql = build_training_frame(prepared.raw_df, cfg, feature_engine="sql")

    out_root = ROOT / "outputs" / "v4"
    pandas_out = out_root / "pandas"
    sql_out = out_root / "sql"
    ensure_dir(out_root)
    ensure_dir(pandas_out)
    ensure_dir(sql_out)
    frame_pandas.to_csv(pandas_out / "processed_features.csv", index=False, encoding="utf-8")
    frame_sql.to_csv(sql_out / "processed_features.csv", index=False, encoding="utf-8")

    feat_comp, feat_summary = compare_feature_frames(
        pandas_df=frame_pandas,
        sql_df=frame_sql,
        feature_cols=cfg.data.feature_cols,
        label_col=cfg.data.label_col,
    )
    feat_comp.to_csv(out_root / "feature_consistency.csv", index=False, encoding="utf-8")
    dump_json(out_root / "feature_consistency_summary.json", feat_summary)

    frame_pandas_eval = v3.cut_warmup(frame_pandas, cfg.split.sequence_length)
    frame_sql_eval = v3.cut_warmup(frame_sql, cfg.split.sequence_length)
    pandas_metrics = evaluate_frame(frame_pandas_eval, cfg, opts, pandas_out)
    sql_metrics = evaluate_frame(frame_sql_eval, cfg, opts, sql_out)

    comp_rows = []
    for key in [
        "final_f1",
        "final_precision",
        "final_recall",
        "final_accuracy",
        "final_roc_auc",
        "final_threshold",
        "false_positive_count",
        "false_negative_count",
        "near_threshold_error_rate",
        "rolling_avg_f1",
        "rolling_min_f1",
        "rolling_std_f1",
        "rolling_std_threshold",
    ]:
        p = float(pandas_metrics[key])
        s = float(sql_metrics[key])
        comp_rows.append({"metric": key, "pandas": p, "sql": s, "sql_minus_pandas": s - p})
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(out_root / "compare_metrics.csv", index=False, encoding="utf-8")

    summary = {
        "options": {
            "thr_goal": opts.thr_goal,
            "final_thr_mode": opts.final_thr_mode,
            "mlp_max_iter": opts.mlp_max_iter,
            "min_precision": opts.min_precision,
            "min_recall": opts.min_recall,
            "goal_penalty": opts.goal_penalty,
            "calib": opts.calib,
        },
        "feature_consistency_summary": feat_summary,
        "pandas_metrics": pandas_metrics,
        "sql_metrics": sql_metrics,
    }
    dump_json(out_root / "compare_summary.json", summary)

    print("[Aurora] v4 migration compare done")
    print(f"[Aurora] feature consistency: {out_root / 'feature_consistency.csv'}")
    print(f"[Aurora] compare metrics: {out_root / 'compare_metrics.csv'}")
    print(f"[Aurora] summary: {out_root / 'compare_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
