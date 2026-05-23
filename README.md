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
run_aurora.bat test
```

如果 `.venv\Scripts\python.exe` 存在，脚本会优先使用虚拟环境 Python；否则自动回退系统 `python`。

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
