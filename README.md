# Aurora

Aurora 当前交付主线是 **v0.3 稳健性增强**，目标是：

1. Rolling validation（滚动验证）
2. Threshold tuning（阈值调优）
3. Error analysis（误差分析）

## Quick Start

1. 创建虚拟环境并安装依赖（推荐 Python 3.11+）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

2. 配置 `.env`（首次拉数或刷新数据时需要）：

```env
TUSHARE_TOKEN=your_token_here
```

## One-Click (Windows BAT)

仓库根目录脚本：[run_aurora.bat](D:\Github_Storage\Py_finalwork\run_aurora.bat)

默认直接运行 v0.3：

```powershell
run_aurora.bat
```

可选模式：

```powershell
run_aurora.bat prepare
run_aurora.bat prepare-refresh
run_aurora.bat v3
run_aurora.bat v3-refresh
run_aurora.bat test
```

模式说明：

1. `prepare`：准备数据（优先本地冻结数据）
2. `prepare-refresh`：强制从 Tushare 拉新数据
3. `v3`：执行 v0.3 全流程（rolling + threshold + error analysis）
4. `v3-refresh`：先刷新数据，再执行 v0.3
5. `test`：运行最小测试集

## v0.3 输出文件

1. 滚动验证：
- `outputs/v3/rolling_metrics.csv`
- `outputs/v3/rolling_metrics.json`

2. 最终评估：
- `outputs/v3/final/metrics.json`
- `outputs/v3/final/predictions.csv`

3. 误差分析：
- `outputs/v3/error_cases.csv`
- `outputs/v3/error_summary.json`

4. 图表（SVG）：
- `outputs/v3/charts/rolling_f1.svg`
- `outputs/v3/charts/rolling_threshold.svg`
- `outputs/v3/charts/error_distribution.svg`

## 如何读结果

1. `rolling_metrics.csv`：看每个 fold 的 `test_f1/test_roc_auc/best_threshold` 稳定性。
2. `final/metrics.json`：看最终主指标（accuracy/f1/precision/recall/roc_auc）。
3. `error_summary.json`：看误报与漏报结构，特别是近阈值错误占比。
4. `error_cases.csv`：逐条定位错判日期与概率。

## Validate

```powershell
python -m pytest -q
```

## Docs

详细过程与解释文档在 `docs/` 目录：

1. `docs/version_iterations.txt`
2. `docs/run_method_v0.1.txt`
3. `docs/grill_notes_v0.1.txt`
4. `docs/build_process_v0.1.txt`

## Repo Layout

1. `src/aurora_ml/`：核心实现（配置、数据、模型、训练、推理）
2. `scripts/`：命令行入口（当前主入口是 `train_v3_robust.py`）
3. `configs/default.yaml`：统一配置
4. `data/`：原始/处理后数据
5. `outputs/`：训练输出（默认不提交）
6. `tests/`：最小测试集
