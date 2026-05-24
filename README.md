# Aurora

Aurora 当前交付主线是 **v0.4 时序特征工程迁移与对比**。  
课程方向采用“方向1：自主选题（结构化时序二分类）”。

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

默认执行 v0.3 稳态训练：

```powershell
run_aurora.bat
```

可选模式：

```powershell
run_aurora.bat prepare
run_aurora.bat prepare-refresh
run_aurora.bat v3
run_aurora.bat v3-refresh
run_aurora.bat v4
run_aurora.bat v4-refresh
run_aurora.bat test
```

模式说明：

1. `prepare`：准备数据（优先本地冻结数据）
2. `prepare-refresh`：强制从 Tushare 拉新数据
3. `v3`：执行 v0.3 稳健流程（rolling + threshold + error analysis）
4. `v3-refresh`：先刷新数据，再执行 v0.3
5. `v4`：执行 v0.4 迁移对比（pandas vs sql）
6. `v4-refresh`：先刷新数据，再执行 v0.4
7. `test`：运行最小测试集

## v0.4 新增内容

1. 时序特征工程双引擎：
- `feature_engine=pandas`（迁移前）
- `feature_engine=sql`（OpenMLDB 风格窗口 SQL，本地可复现实现）

2. 新脚本：
- `scripts/train_v4_migration_compare.py`

3. 关键输出：
- `outputs/v4/feature_consistency.csv`
- `outputs/v4/feature_consistency_summary.json`
- `outputs/v4/compare_metrics.csv`
- `outputs/v4/compare_summary.json`
- `outputs/v4/pandas/final/metrics.json`
- `outputs/v4/sql/final/metrics.json`

## 课程提交材料对应关系

1. 代码包（可运行代码+注释+数据）：
- 代码：`src/` + `scripts/` + `configs/` + `run_aurora.bat`
- 数据：`data/raw/` + `data/processed/`

2. 报告 8 个部分建议对照：
- 问题定义：`docs/version_iterations.txt`（版本目标与任务定义）
- 数据处理：`scripts/prepare_data.py` + `src/aurora_ml/data_pipeline.py`
- 模型架构：`scripts/train_v3_robust.py`
- 训练配置：`configs/default.yaml` + `docs/run_method_v0.1.txt`
- 评估结果：`outputs/v3/*` 与 `outputs/v4/*`
- 改进：`docs/build_process_v0.1.txt`（v0.3.x 到 v0.4 过程）
- 比较：`outputs/v4/compare_metrics.csv`
- 总结：`docs/grill_notes_v0.1.txt`（可补最终结论）

3. 2 分钟答辩 PPT 关键素材：
- 数据来源与类型：Tushare 股票日线 + 派生时序特征
- 算法选择：MLP 主线 + rolling validation + threshold tuning
- 迁移亮点：pandas 特征工程迁移到 SQL 风格流水线并做一致性对比

## 如何读结果

1. `outputs/v3/rolling_metrics.csv`：跨时间窗稳定性（`test_f1/best_threshold`）
2. `outputs/v3/final/metrics.json`：最终主指标（accuracy/f1/precision/recall/roc_auc）
3. `outputs/v3/error_summary.json`：误报/漏报结构与 near-threshold 错误率
4. `outputs/v4/compare_metrics.csv`：迁移前后指标差值（`sql_minus_pandas`）

## Validate

```powershell
python -m pytest -q
```

## Docs

1. `docs/version_iterations.txt`
2. `docs/run_method_v0.1.txt`
3. `docs/grill_notes_v0.1.txt`
4. `docs/build_process_v0.1.txt`

## Repo Layout

1. `src/aurora_ml/`：核心实现（配置、数据、模型、训练、推理）
2. `scripts/`：训练与对比入口（`train_v3_robust.py` / `train_v4_migration_compare.py`）
3. `configs/default.yaml`：统一配置
4. `data/`：原始与处理后数据
5. `outputs/`：训练输出（默认不提交）
6. `tests/`：最小测试集

## Git 提交前最小检查

1. `run_aurora.bat v3` 成功，`outputs/v3/final/metrics.json` 更新
2. `run_aurora.bat v4` 成功，`outputs/v4/compare_metrics.csv` 更新
3. `python -m pytest -q` 通过
4. `git status` 仅包含你计划提交的文件
