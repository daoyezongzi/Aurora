# Aurora

Aurora v0.1 是一个可复现的最小可行项目，用于完成 `002372.SZ` 的 T+1 涨跌二分类任务。

- 数据：Tushare 日线（支持本地冻结快照）
- 模型：PyTorch LSTM
- 标签：`y_t = 1{close_{t+1} > close_t}`
- 切分：时间顺序 `70% / 15% / 15%`
- 输出：`outputs/model.pt`、`outputs/metrics.json`、`outputs/predictions.csv`

## Quick Start

1. 创建虚拟环境并安装依赖（推荐 Python 3.11+）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

2. 配置 `.env`（至少包含 `TUSHARE_TOKEN`）：

```env
TUSHARE_TOKEN=your_token_here
```

3. 逐步运行：

```powershell
python scripts/prepare_data.py --config configs/default.yaml
python scripts/train_lstm.py --config configs/default.yaml
python scripts/predict.py --config configs/default.yaml
```

## One-Click (Windows BAT)

仓库根目录提供一键脚本：[run_aurora.bat](D:\Github_Storage\Py_finalwork\run_aurora.bat)

```powershell
run_aurora.bat
```

默认模式是 `full`，会按顺序执行：
1. `prepare_data`
2. `train_lstm`
3. `predict`

可选模式：

```powershell
run_aurora.bat full
run_aurora.bat full-refresh
run_aurora.bat prepare
run_aurora.bat prepare-refresh
run_aurora.bat train
run_aurora.bat predict
run_aurora.bat metrics
run_aurora.bat v2
run_aurora.bat v2-refresh
run_aurora.bat test
```

如果 `.venv\Scripts\python.exe` 存在，脚本会优先使用虚拟环境 Python；否则自动回退系统 `python`。

`full` / `full-refresh` / `train` 模式结束后，会自动打印指标摘要：
- accuracy
- f1
- threshold
- confusion_matrix
- 测试集预测分布

你也可以单独查看指标：

```powershell
run_aurora.bat metrics
```

## v0.2 多模型对比（LSTM + LogReg + MLP）

`v2` 模式会一次性训练并评估 3 个模型：
- `lstm`
- `logreg`（LogisticRegression）
- `mlp`（多层感知机）

运行命令：

```powershell
run_aurora.bat v2
```

如需先刷新 Tushare 原始数据：

```powershell
run_aurora.bat v2-refresh
```

### v2 输出文件

- 总表：`outputs/v2/compare_metrics.csv`
- 总表（JSON）：`outputs/v2/compare_metrics.json`
- 单模型指标：
  - `outputs/v2/lstm/metrics_v2.json`
  - `outputs/v2/logreg/metrics.json`
  - `outputs/v2/mlp/metrics.json`
- 单模型预测：
  - `outputs/v2/lstm/predictions.csv`
  - `outputs/v2/logreg/predictions.csv`
  - `outputs/v2/mlp/predictions.csv`
- 对比图表（SVG）：
  - `outputs/v2/charts/metric_bars.svg`
  - `outputs/v2/charts/roc_curves.svg`
  - `outputs/v2/charts/confusion_matrices.svg`

### 如何看 v2 结果

1. 先看 `compare_metrics.csv`：快速比较三模型 `accuracy/f1/precision/recall/roc_auc`。
2. 再看 `confusion_matrices.svg`：判断是“漏报多”还是“误报多”。
3. 再看 `roc_curves.svg`：比较排序能力（曲线越靠左上通常越好）。
4. 最后看各模型 `predictions.csv`：定位具体日期的预测偏差。

## Validate

```powershell
python -m pytest -q
```

## Repo Layout

- `src/aurora_ml/`：核心实现（配置、数据、模型、训练、推理）
- `scripts/`：命令行入口
- `configs/default.yaml`：统一配置
- `data/`：原始/处理后数据
- `outputs/`：训练输出（默认不提交）
- `tests/`：最小测试集
