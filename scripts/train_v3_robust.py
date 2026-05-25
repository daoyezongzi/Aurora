from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from aurora_ml.config import AuroraConfig, load_config
from aurora_ml.data_pipeline import prepare_data
from aurora_ml.dataset import split_by_time
from aurora_ml.utils import dump_json, ensure_dir


@dataclass(frozen=True)
class FoldCut:
    fold_id: int
    train_start: int
    train_end: int
    val_end: int
    test_end: int


@dataclass(frozen=True)
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    dates_test: list[str]


@dataclass(frozen=True)
class TuneOpts:
    thr_goal: str
    beta: float
    thr_min: float
    thr_max: float
    thr_step: float
    thr_eps: float
    thr_anchor: float
    final_thr_mode: str
    cfg_eps: float
    min_precision: float
    min_recall: float
    goal_penalty: float
    mlp_max_iter: int
    calib: str
    calib_cv: int
    cfg_std_pen: float
    cfg_thr_pen: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Aurora robust workflow: rolling + threshold + error analysis")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to yaml config")
    parser.add_argument("--refresh", action="store_true", help="Refresh raw data from Tushare before training")
    parser.add_argument(
        "--thr-goal",
        default="f0.5",
        choices=["f0.5", "f1", "f2", "fbeta"],
        help="Threshold objective; default f0.5 to reduce false positives",
    )
    parser.add_argument("--beta", type=float, default=0.5, help="Beta for fbeta when --thr-goal=fbeta")
    parser.add_argument("--thr-min", type=float, default=0.30, help="Min threshold in search grid")
    parser.add_argument("--thr-max", type=float, default=0.70, help="Max threshold in search grid")
    parser.add_argument("--thr-step", type=float, default=0.01, help="Step size in threshold search grid")
    parser.add_argument(
        "--thr-eps",
        type=float,
        default=0.01,
        help="Near-best tolerance in threshold score; pick stable threshold inside this band",
    )
    parser.add_argument(
        "--thr-anchor",
        type=float,
        default=0.50,
        help="Preferred threshold center when scores are close",
    )
    parser.add_argument(
        "--final-thr-mode",
        default="val",
        choices=["rolling_median", "val"],
        help="How to choose final threshold: rolling median or final-val best",
    )
    parser.add_argument(
        "--cfg-eps",
        type=float,
        default=0.001,
        help="Near-best tolerance for model config selection across a fold",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.56,
        help="Precision floor in threshold scoring; lower values are penalized",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.62,
        help="Recall floor in threshold scoring; lower values are penalized",
    )
    parser.add_argument(
        "--goal-penalty",
        type=float,
        default=1.2,
        help="Penalty weight when precision/recall falls below floors",
    )
    parser.add_argument(
        "--mlp-max-iter",
        type=int,
        default=900,
        help="Max iterations for MLP training",
    )
    parser.add_argument(
        "--calib",
        default="none",
        choices=["none", "sigmoid", "isotonic"],
        help="Probability calibration method",
    )
    parser.add_argument("--calib-cv", type=int, default=3, help="Cross-validation folds for calibration")
    parser.add_argument(
        "--cfg-std-pen",
        type=float,
        default=0.20,
        help="Penalty weight for cross-fold score std when picking final config",
    )
    parser.add_argument(
        "--cfg-thr-pen",
        type=float,
        default=0.12,
        help="Penalty weight for cross-fold threshold std when picking final config",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    cfg = load_config(args.config)
    opts = build_tune_opts(args)
    frame = load_or_prepare_frame(cfg, refresh=args.refresh)
    frame = cut_warmup(frame, cfg.split.sequence_length)

    out_root = ROOT / "outputs" / "v3"
    chart_dir = out_root / "charts"
    ensure_dir(out_root)
    ensure_dir(chart_dir)

    fold_rows, cfg_rows = rolling_validate(frame, cfg, opts)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_root / "rolling_metrics.csv", index=False, encoding="utf-8")
    dump_json(out_root / "rolling_metrics.json", {"rows": fold_rows})
    cfg_df = pd.DataFrame(cfg_rows)
    cfg_df.to_csv(out_root / "config_metrics.csv", index=False, encoding="utf-8")
    dump_json(out_root / "config_metrics.json", {"rows": cfg_rows})

    best_cfg = pick_best_config(cfg_rows, opts)
    rolling_thr = pick_final_threshold(fold_rows, opts)
    final_row, final_pred = run_final_eval(frame, cfg, best_cfg, opts, out_root / "final", rolling_thr)

    final_pred.to_csv(out_root / "final" / "predictions.csv", index=False, encoding="utf-8")
    dump_json(out_root / "final" / "metrics.json", final_row)

    err_df, err_summary = build_error_analysis(final_pred, threshold=float(final_row["threshold"]))
    err_df.to_csv(out_root / "error_cases.csv", index=False, encoding="utf-8")
    dump_json(out_root / "error_summary.json", err_summary)

    make_charts(fold_df, err_summary, chart_dir)

    print("[Aurora] v3 done")
    print(f"[Aurora] rolling metrics: {out_root / 'rolling_metrics.csv'}")
    print(f"[Aurora] config metrics: {out_root / 'config_metrics.csv'}")
    print(f"[Aurora] final metrics: {out_root / 'final' / 'metrics.json'}")
    print(f"[Aurora] error summary: {out_root / 'error_summary.json'}")
    print(f"[Aurora] charts: {chart_dir}")
    return 0


def build_tune_opts(args: argparse.Namespace) -> TuneOpts:
    thr_goal = str(args.thr_goal).strip().lower()
    beta = float(args.beta)
    if thr_goal != "fbeta":
        beta = 0.5 if thr_goal == "f0.5" else (1.0 if thr_goal == "f1" else 2.0)

    thr_min = float(args.thr_min)
    thr_max = float(args.thr_max)
    thr_step = float(args.thr_step)
    thr_eps = float(args.thr_eps)
    thr_anchor = float(args.thr_anchor)
    final_thr_mode = str(args.final_thr_mode).strip().lower()
    cfg_eps = float(args.cfg_eps)
    min_precision = float(args.min_precision)
    min_recall = float(args.min_recall)
    goal_penalty = float(args.goal_penalty)
    mlp_max_iter = int(args.mlp_max_iter)
    if not (0.0 < thr_min < thr_max < 1.0):
        raise ValueError("threshold range must satisfy 0 < thr_min < thr_max < 1")
    if thr_step <= 0.0:
        raise ValueError("thr_step must be positive")
    if thr_eps < 0.0:
        raise ValueError("thr_eps must be >= 0")
    if not (0.0 < thr_anchor < 1.0):
        raise ValueError("thr_anchor must be in (0, 1)")
    if thr_anchor < thr_min or thr_anchor > thr_max:
        raise ValueError("thr_anchor must stay inside [thr_min, thr_max]")
    if cfg_eps < 0.0:
        raise ValueError("cfg_eps must be >= 0")
    if final_thr_mode not in {"rolling_median", "val"}:
        raise ValueError("final_thr_mode must be one of: rolling_median, val")
    if not (0.0 <= min_precision <= 1.0):
        raise ValueError("min_precision must be in [0, 1]")
    if not (0.0 <= min_recall <= 1.0):
        raise ValueError("min_recall must be in [0, 1]")
    if goal_penalty < 0.0:
        raise ValueError("goal_penalty must be >= 0")
    if mlp_max_iter <= 0:
        raise ValueError("mlp_max_iter must be > 0")

    calib = str(args.calib).strip().lower()
    calib_cv = int(args.calib_cv)
    if calib != "none" and calib_cv < 2:
        raise ValueError("calib_cv must be >=2 when calibration is enabled")
    cfg_std_pen = float(args.cfg_std_pen)
    cfg_thr_pen = float(args.cfg_thr_pen)
    if cfg_std_pen < 0.0 or cfg_thr_pen < 0.0:
        raise ValueError("config penalties must be >= 0")

    return TuneOpts(
        thr_goal=thr_goal,
        beta=beta,
        thr_min=thr_min,
        thr_max=thr_max,
        thr_step=thr_step,
        thr_eps=thr_eps,
        thr_anchor=thr_anchor,
        final_thr_mode=final_thr_mode,
        cfg_eps=cfg_eps,
        min_precision=min_precision,
        min_recall=min_recall,
        goal_penalty=goal_penalty,
        mlp_max_iter=mlp_max_iter,
        calib=calib,
        calib_cv=calib_cv,
        cfg_std_pen=cfg_std_pen,
        cfg_thr_pen=cfg_thr_pen,
    )


def load_or_prepare_frame(cfg: AuroraConfig, refresh: bool) -> pd.DataFrame:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    need_prepare = refresh or (not cfg.data.processed_path.exists())
    if not need_prepare:
        frame = pd.read_csv(cfg.data.processed_path)
        required = {"trade_date", cfg.data.label_col, *cfg.data.feature_cols}
        if required.issubset(set(frame.columns)):
            return frame
        need_prepare = True
    if need_prepare:
        prepared = prepare_data(config=cfg, refresh=refresh, tushare_token=token)
        return prepared.train_df
    return pd.read_csv(cfg.data.processed_path)


def cut_warmup(frame: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    if len(frame) <= seq_len:
        raise ValueError("not enough rows after sequence warmup cut")
    return frame.iloc[seq_len - 1 :].reset_index(drop=True).copy()


def rolling_validate(frame: pd.DataFrame, cfg: AuroraConfig, opts: TuneOpts) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cuts = build_folds(len(frame), cfg)
    if len(cuts) < 2:
        raise ValueError("rolling validation needs at least 2 folds; try more data or smaller window.")

    rows: list[dict[str, Any]] = []
    cfg_rows: list[dict[str, Any]] = []
    for cut in cuts:
        train_df = frame.iloc[cut.train_start : cut.train_end].copy()
        val_df = frame.iloc[cut.train_end : cut.val_end].copy()
        test_df = frame.iloc[cut.val_end : cut.test_end].copy()

        split = to_split_data(train_df, val_df, test_df, cfg.data.feature_cols, cfg.data.label_col)
        model_cfg, val_best, fold_cfg_rows = tune_mlp(
            split.x_train,
            split.y_train,
            split.x_val,
            split.y_val,
            cfg.project.seed,
            opts,
        )
        for item in fold_cfg_rows:
            cfg_rows.append(
                {
                    "fold_id": cut.fold_id,
                    "hidden": item["hidden"],
                    "alpha": item["alpha"],
                    "best_threshold": item["best_threshold"],
                    "val_score_best": item["val_score_best"],
                    "val_f1_best": item["val_f1_best"],
                    "threshold_goal": opts.thr_goal,
                    "calibration": opts.calib,
                }
            )

        fit_x = np.concatenate([split.x_train, split.x_val], axis=0)
        fit_y = np.concatenate([split.y_train, split.y_val], axis=0)
        model = fit_prob_model(model_cfg["hidden"], model_cfg["alpha"], fit_x, fit_y, cfg.project.seed, opts)

        test_prob = model.predict_proba(split.x_test)[:, 1]
        test_row = build_metrics(split.y_test, test_prob, val_best["threshold"])

        row = {
            "fold_id": cut.fold_id,
            "train_start_date": str(train_df.iloc[0]["trade_date"]),
            "train_end_date": str(train_df.iloc[-1]["trade_date"]),
            "val_end_date": str(val_df.iloc[-1]["trade_date"]),
            "test_end_date": str(test_df.iloc[-1]["trade_date"]),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "hidden": model_cfg["hidden"],
            "alpha": model_cfg["alpha"],
            "threshold_goal": val_best["goal"],
            "calibration": opts.calib,
            "best_threshold": val_best["threshold"],
            "threshold_anchor": float(opts.thr_anchor),
            "threshold_eps": float(opts.thr_eps),
            "config_eps": float(opts.cfg_eps),
            "val_score_best": val_best["score"],
            "val_f1_best": val_best["f1"],
            "test_accuracy": test_row["accuracy"],
            "test_f1": test_row["f1"],
            "test_precision": test_row["precision"],
            "test_recall": test_row["recall"],
            "test_roc_auc": test_row["roc_auc"],
        }
        rows.append(row)
    return rows, cfg_rows


def build_folds(n_rows: int, cfg: AuroraConfig) -> list[FoldCut]:
    target_folds = 3
    min_gap = 20

    val_len = max(int(n_rows * cfg.split.val_ratio), 40)
    test_len = max(int(n_rows * cfg.split.test_ratio), 40)
    train_len = max(int(n_rows * cfg.split.train_ratio), 120)

    # If default split is too large for rolling, shrink train window first.
    if target_folds > 1:
        max_train = n_rows - val_len - test_len - (target_folds - 1) * min_gap
        train_len = min(train_len, max_train)
    train_len = max(train_len, 120)

    remain = n_rows - (train_len + val_len + test_len)
    if target_folds > 1:
        step = max(remain // (target_folds - 1), min_gap)
    else:
        step = max(test_len // 2, min_gap)
    step = max(step, min_gap)

    cuts: list[FoldCut] = []
    start = 0
    fold_id = 1
    while True:
        train_end = start + train_len
        val_end = train_end + val_len
        test_end = val_end + test_len
        if test_end > n_rows:
            break
        cuts.append(
            FoldCut(
                fold_id=fold_id,
                train_start=start,
                train_end=train_end,
                val_end=val_end,
                test_end=test_end,
            )
        )
        fold_id += 1
        start += step
    return cuts


def to_split_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
) -> SplitData:
    scaler = StandardScaler()
    scaler.fit(train_df[feat_cols].to_numpy(dtype=np.float32))

    x_train = scaler.transform(train_df[feat_cols].to_numpy(dtype=np.float32))
    y_train = train_df[label_col].to_numpy(dtype=np.int64)
    x_val = scaler.transform(val_df[feat_cols].to_numpy(dtype=np.float32))
    y_val = val_df[label_col].to_numpy(dtype=np.int64)
    x_test = scaler.transform(test_df[feat_cols].to_numpy(dtype=np.float32))
    y_test = test_df[label_col].to_numpy(dtype=np.int64)
    dates_test = test_df["trade_date"].astype(str).tolist()
    return SplitData(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        dates_test=dates_test,
    )


def tune_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    opts: TuneOpts,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    grid = [
        {"hidden": [64, 32], "alpha": 1e-3},
        {"hidden": [64, 32], "alpha": 1e-4},
        {"hidden": [64], "alpha": 1e-3},
        {"hidden": [32], "alpha": 1e-4},
    ]
    cand_rows: list[dict[str, Any]] = []
    for cfg_item in grid:
        model = fit_prob_model(cfg_item["hidden"], cfg_item["alpha"], x_train, y_train, seed, opts)
        val_prob = model.predict_proba(x_val)[:, 1]
        tuned = tune_threshold(y_val, val_prob, opts)
        cand_rows.append(
            {
                "hidden": cfg_item["hidden"],
                "alpha": cfg_item["alpha"],
                "best_threshold": float(tuned["threshold"]),
                "val_score_best": float(tuned["score"]),
                "val_f1_best": float(tuned["f1"]),
                "threshold_goal": str(tuned["goal"]),
            }
        )

    best_row = pick_stable_row(
        cand_rows,
        opts,
        score_key="val_score_best",
        thr_key="best_threshold",
        f1_key="val_f1_best",
        eps=opts.cfg_eps,
    )
    if best_row is None:
        raise RuntimeError("tune_mlp failed to find candidate")
    best_cfg = {"hidden": best_row["hidden"], "alpha": best_row["alpha"]}
    best_val = {
        "score": float(best_row["val_score_best"]),
        "f1": float(best_row["val_f1_best"]),
        "threshold": float(best_row["best_threshold"]),
        "goal": str(best_row["threshold_goal"]),
    }
    return best_cfg, best_val, cand_rows


def tune_threshold(y_true: np.ndarray, prob: np.ndarray, opts: TuneOpts) -> dict[str, float | str]:
    rows: list[dict[str, float | str]] = []
    for thr in np.arange(opts.thr_min, opts.thr_max + 1e-9, opts.thr_step):
        pred = (prob >= thr).astype(np.int64)
        f1 = float(f1_score(y_true, pred, zero_division=0))
        score = threshold_metric(y_true, pred, opts)
        rows.append({"score": float(score), "f1": f1, "threshold": float(round(thr, 2)), "goal": opts.thr_goal})
    best = pick_stable_row(rows, opts, score_key="score", thr_key="threshold", f1_key="f1")
    if best is None:
        raise RuntimeError("tune_threshold found no candidate")
    return {"score": float(best["score"]), "f1": float(best["f1"]), "threshold": float(best["threshold"]), "goal": str(best["goal"])}


def pick_stable_row(
    rows: list[dict[str, Any]],
    opts: TuneOpts,
    score_key: str,
    thr_key: str,
    f1_key: str,
    eps: float | None = None,
) -> dict[str, Any] | None:
    if not rows:
        return None
    top_score = max(float(row[score_key]) for row in rows)
    gap = float(opts.thr_eps) if eps is None else float(eps)
    kept = [row for row in rows if float(row[score_key]) >= (top_score - gap)]
    if not kept:
        kept = rows
    kept.sort(
        key=lambda row: (
            abs(float(row[thr_key]) - float(opts.thr_anchor)),
            -float(row[score_key]),
            -float(row[f1_key]),
        )
    )
    return kept[0]


def threshold_metric(y_true: np.ndarray, pred: np.ndarray, opts: TuneOpts) -> float:
    goal = opts.thr_goal
    if goal == "f1":
        base = float(f1_score(y_true, pred, zero_division=0))
    elif goal == "f2":
        base = float(fbeta_score(y_true, pred, beta=2.0, zero_division=0))
    elif goal == "fbeta":
        base = float(fbeta_score(y_true, pred, beta=float(opts.beta), zero_division=0))
    else:
        base = float(fbeta_score(y_true, pred, beta=0.5, zero_division=0))

    prec = float(precision_score(y_true, pred, zero_division=0))
    rec = float(recall_score(y_true, pred, zero_division=0))
    penalty = 0.0
    if prec < float(opts.min_precision):
        penalty += float(opts.goal_penalty) * (float(opts.min_precision) - prec)
    if rec < float(opts.min_recall):
        penalty += float(opts.goal_penalty) * (float(opts.min_recall) - rec)
    return float(base - penalty)


def snap_threshold(thr: float, opts: TuneOpts) -> float:
    clipped = min(max(float(thr), float(opts.thr_min)), float(opts.thr_max))
    offset = round((clipped - float(opts.thr_min)) / float(opts.thr_step))
    snapped = float(opts.thr_min) + float(offset) * float(opts.thr_step)
    snapped = min(max(snapped, float(opts.thr_min)), float(opts.thr_max))
    return float(round(snapped, 2))


def pick_final_threshold(rows: list[dict[str, Any]], opts: TuneOpts) -> float:
    vals = [float(row["best_threshold"]) for row in rows if "best_threshold" in row]
    if not vals:
        return snap_threshold(float(opts.thr_anchor), opts)
    med = float(np.median(np.asarray(vals, dtype=float)))
    return snap_threshold(med, opts)


def build_mlp_estimator(hidden: list[int], alpha: float, seed: int, max_iter: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=tuple(hidden),
        alpha=float(alpha),
        learning_rate_init=1e-3,
        max_iter=int(max_iter),
        random_state=seed,
    )


def fit_prob_model(
    hidden: list[int],
    alpha: float,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    seed: int,
    opts: TuneOpts,
):
    base = build_mlp_estimator(hidden, alpha, seed, opts.mlp_max_iter)
    if opts.calib == "none":
        base.fit(x_fit, y_fit)
        return base
    model = CalibratedClassifierCV(estimator=base, method=opts.calib, cv=opts.calib_cv)
    model.fit(x_fit, y_fit)
    return model


def pick_best_config(rows: list[dict[str, Any]], opts: TuneOpts) -> dict[str, Any]:
    bucket: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "vals": [], "thrs": []})
    key_map: dict[str, dict[str, Any]] = {}

    for row in rows:
        key = f"{row['hidden']}-{row['alpha']}"
        key_map[key] = {"hidden": row["hidden"], "alpha": row["alpha"]}
        bucket[key]["count"] += 1
        score = float(row.get("val_score_best", row.get("val_f1_best", 0.0)))
        bucket[key]["vals"].append(score)
        bucket[key]["thrs"].append(float(row.get("best_threshold", opts.thr_anchor)))

    best_key = ""
    best_score = -1.0
    for key, agg in bucket.items():
        vals = np.asarray(agg["vals"], dtype=float)
        thrs = np.asarray(agg["thrs"], dtype=float)
        mean_val = float(vals.mean()) if vals.size else 0.0
        std_val = float(vals.std(ddof=0)) if vals.size else 0.0
        std_thr = float(thrs.std(ddof=0)) if thrs.size else 0.0
        score = mean_val - float(opts.cfg_std_pen) * std_val - float(opts.cfg_thr_pen) * std_thr
        if score > best_score:
            best_score = score
            best_key = key

    if not best_key:
        raise RuntimeError("pick_best_config failed")
    return key_map[best_key]


def run_final_eval(
    frame: pd.DataFrame,
    cfg: AuroraConfig,
    model_cfg: dict[str, Any],
    opts: TuneOpts,
    out_dir: Path,
    rolling_thr: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    ensure_dir(out_dir)
    train_df, val_df, test_df = split_by_time(
        frame=frame,
        train_ratio=cfg.split.train_ratio,
        val_ratio=cfg.split.val_ratio,
        test_ratio=cfg.split.test_ratio,
    )
    split = to_split_data(train_df, val_df, test_df, cfg.data.feature_cols, cfg.data.label_col)

    base_model = fit_prob_model(model_cfg["hidden"], model_cfg["alpha"], split.x_train, split.y_train, cfg.project.seed, opts)
    val_prob = base_model.predict_proba(split.x_val)[:, 1]
    val_best = tune_threshold(split.y_val, val_prob, opts)
    val_thr = float(val_best["threshold"])
    if opts.final_thr_mode == "rolling_median":
        best_thr = snap_threshold(float(rolling_thr), opts)
        thr_source = "rolling_median"
    else:
        best_thr = snap_threshold(val_thr, opts)
        thr_source = "val_best"

    x_fit = np.concatenate([split.x_train, split.x_val], axis=0)
    y_fit = np.concatenate([split.y_train, split.y_val], axis=0)
    final_model = fit_prob_model(model_cfg["hidden"], model_cfg["alpha"], x_fit, y_fit, cfg.project.seed, opts)
    test_prob = final_model.predict_proba(split.x_test)[:, 1]
    metric_row = build_metrics(split.y_test, test_prob, best_thr)
    metric_row["model"] = "mlp_v3"
    metric_row["hidden"] = model_cfg["hidden"]
    metric_row["alpha"] = model_cfg["alpha"]
    metric_row["threshold_goal"] = opts.thr_goal
    metric_row["threshold_beta"] = float(opts.beta)
    metric_row["threshold_search"] = [float(opts.thr_min), float(opts.thr_max), float(opts.thr_step)]
    metric_row["threshold_eps"] = float(opts.thr_eps)
    metric_row["threshold_anchor"] = float(opts.thr_anchor)
    metric_row["final_threshold_mode"] = str(opts.final_thr_mode)
    metric_row["final_threshold_source"] = thr_source
    metric_row["rolling_threshold_median"] = float(rolling_thr)
    metric_row["val_best_threshold"] = val_thr
    metric_row["config_eps"] = float(opts.cfg_eps)
    metric_row["min_precision"] = float(opts.min_precision)
    metric_row["min_recall"] = float(opts.min_recall)
    metric_row["goal_penalty"] = float(opts.goal_penalty)
    metric_row["mlp_max_iter"] = int(opts.mlp_max_iter)
    metric_row["calibration"] = opts.calib
    metric_row["calibration_cv"] = int(opts.calib_cv)
    metric_row["config_std_pen"] = float(opts.cfg_std_pen)
    metric_row["config_thr_pen"] = float(opts.cfg_thr_pen)
    metric_row["val_score_best"] = float(val_best["score"])
    metric_row["val_f1_at_best"] = float(val_best["f1"])
    metric_row["brier_score"] = float(brier_score_loss(split.y_test, test_prob))
    metric_row["sample_count"] = int(len(split.y_test))

    pred = pd.DataFrame(
        {
            "trade_date": split.dates_test,
            "y_true": split.y_test,
            "up_prob": test_prob,
            "y_pred": (test_prob >= best_thr).astype(np.int64),
        }
    )
    return metric_row, pred


def build_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float) -> dict[str, Any]:
    pred = (prob >= thr).astype(np.int64)
    matrix = confusion_matrix(y_true, pred, labels=[0, 1]).tolist()
    row: dict[str, Any] = {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "confusion_matrix": matrix,
    }
    try:
        row["roc_auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        row["roc_auc"] = float("nan")
    return row


def build_error_analysis(
    pred_df: pd.DataFrame,
    threshold: float,
    near_band: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    err = pred_df.copy()
    err["error_type"] = "ok"
    err.loc[(err["y_true"] == 0) & (err["y_pred"] == 1), "error_type"] = "false_positive"
    err.loc[(err["y_true"] == 1) & (err["y_pred"] == 0), "error_type"] = "false_negative"
    err["abs_gap_from_threshold"] = (err["up_prob"] - float(threshold)).abs()
    err = err.sort_values(["error_type", "abs_gap_from_threshold"], ascending=[True, True]).reset_index(drop=True)

    fp = int((err["error_type"] == "false_positive").sum())
    fn = int((err["error_type"] == "false_negative").sum())
    ok = int((err["error_type"] == "ok").sum())

    near_mask = err["abs_gap_from_threshold"] <= float(near_band)
    near_err = int(((err["error_type"] != "ok") & near_mask).sum())
    all_err = int((err["error_type"] != "ok").sum())
    near_rate = float(near_err / all_err) if all_err > 0 else 0.0

    summary = {
        "sample_count": int(len(err)),
        "ok_count": ok,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "error_count": all_err,
        "threshold": float(threshold),
        "near_threshold_band": float(near_band),
        "near_threshold_error_count": near_err,
        "near_threshold_error_rate": near_rate,
    }
    return err, summary


def make_charts(fold_df: pd.DataFrame, err: dict[str, Any], chart_dir: Path) -> None:
    ensure_dir(chart_dir)
    make_fold_f1_svg(fold_df, chart_dir / "rolling_f1.svg")
    make_fold_thr_svg(fold_df, chart_dir / "rolling_threshold.svg")
    make_error_bar_svg(err, chart_dir / "error_distribution.svg")


def make_fold_f1_svg(fold_df: pd.DataFrame, out_path: Path) -> None:
    x_vals = fold_df["fold_id"].tolist()
    y_vals = fold_df["test_f1"].tolist()
    write_line_chart(out_path, "Rolling Test F1", x_vals, y_vals, "#59a14f", 0.0, 1.0)


def make_fold_thr_svg(fold_df: pd.DataFrame, out_path: Path) -> None:
    x_vals = fold_df["fold_id"].tolist()
    y_vals = fold_df["best_threshold"].tolist()
    write_line_chart(out_path, "Best Threshold By Fold", x_vals, y_vals, "#4c78a8", 0.25, 0.75)


def make_error_bar_svg(err: dict[str, Any], out_path: Path) -> None:
    labels = ["ok", "false_positive", "false_negative"]
    values = [int(err["ok_count"]), int(err["false_positive_count"]), int(err["false_negative_count"])]
    colors = ["#59a14f", "#f28e2b", "#e15759"]

    width = 700
    height = 460
    left = 90
    top = 60
    plot_h = 300
    bar_w = 120
    gap = 70
    max_v = max(max(values), 1)

    parts: list[str] = []
    parts.append(svg_text(24, 34, "Error Distribution", 24, "#111"))
    parts.append(svg_line(left, top + plot_h, width - 40, top + plot_h, "#333", 2))
    parts.append(svg_line(left, top, left, top + plot_h, "#333", 2))
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + 40 + i * (bar_w + gap)
        h = plot_h * (value / max_v)
        y = top + plot_h - h
        parts.append(svg_rect(x, y, bar_w, h, colors[i], "#ffffff"))
        parts.append(svg_text(x + 20, top + plot_h + 26, label, 13, "#333"))
        parts.append(svg_text(x + 36, y - 8, str(value), 12, "#333"))
    parts.append(svg_text(24, 410, f"near_threshold_error_rate={err['near_threshold_error_rate']:.3f}", 13, "#333"))
    write_svg(out_path, width, height, "".join(parts))


def write_line_chart(
    out_path: Path,
    title: str,
    x_vals: list[int],
    y_vals: list[float],
    color: str,
    y_min: float,
    y_max: float,
) -> None:
    width = 820
    height = 500
    left = 90
    right = 40
    top = 60
    bottom = 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    parts: list[str] = []
    parts.append(svg_text(24, 34, title, 24, "#111"))
    parts.append(svg_line(left, top + plot_h, left + plot_w, top + plot_h, "#333", 2))
    parts.append(svg_line(left, top, left, top + plot_h, "#333", 2))

    for i in range(6):
        v = y_min + (y_max - y_min) * (i / 5.0)
        y = top + plot_h * (1.0 - (v - y_min) / (y_max - y_min))
        parts.append(svg_line(left, y, left + plot_w, y, "#efefef", 1))
        parts.append(svg_text(left - 50, y + 4, f"{v:.2f}", 12, "#444"))

    if not x_vals:
        write_svg(out_path, width, height, "".join(parts))
        return

    x_min = min(x_vals)
    x_max = max(x_vals)
    x_span = max(x_max - x_min, 1)
    pts: list[tuple[float, float]] = []
    for x_raw, y_raw in zip(x_vals, y_vals):
        x = left + plot_w * ((x_raw - x_min) / x_span)
        y_clamped = min(max(y_raw, y_min), y_max)
        y = top + plot_h * (1.0 - (y_clamped - y_min) / (y_max - y_min))
        pts.append((x, y))
        parts.append(svg_circle(x, y, 3.2, color))
        parts.append(svg_text(x - 7, y - 10, f"{y_raw:.3f}", 10, "#333"))
        parts.append(svg_text(x - 4, top + plot_h + 24, str(x_raw), 12, "#444"))
    parts.append(svg_polyline(pts, stroke=color, stroke_width=2.2, fill="none"))
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


def svg_text(x: float, y: float, value: str, size: int, color: str) -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" font-family="Arial">{value}</text>'


def svg_polyline(
    points: list[tuple[float, float]],
    stroke: str,
    stroke_width: float,
    fill: str,
) -> str:
    raw = " ".join([f"{x:.2f},{y:.2f}" for x, y in points])
    return f'<polyline points="{raw}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" />'


def svg_circle(x: float, y: float, r: float, fill: str) -> str:
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" />'


if __name__ == "__main__":
    raise SystemExit(main())
