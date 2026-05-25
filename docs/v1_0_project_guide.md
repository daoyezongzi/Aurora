# Aurora v1.0 项目总览（给第一次接触项目的人）

## 1. 这个项目是做什么的
Aurora 是一个课程机器学习项目：
- 输入：历史日线数据（以 `002372.SZ` 为样例）
- 任务：预测“下一交易日是涨还是跌”（二分类）
- 输出：概率、分类结果、评估指标、误差分析图

一句话：这是一个把“数据 -> 特征 -> 训练 -> 评估 -> 复盘”完整走通的股票方向预测实验工程。

## 2. 你只要记住这 3 个文件
1. `configs/default.yaml`：所有核心配置。
2. `scripts/train_v3_robust.py`：主训练流程（rolling + threshold + error analysis）。
3. `scripts/train_v4_migration_compare.py`：pandas 与 SQL 特征工程一致性对比流程。

## 3. 项目目录怎么读
- `src/aurora_ml/`: 核心能力（配置、数据、特征、训练、推理）。
- `scripts/`: 可直接执行的入口脚本。
- `data/`: 原始数据与处理后数据。
- `outputs/`: 每个版本实验产物（v1/v2/v3/v4）。
- `docs/`: 文档、报告、答辩素材。

## 4. 从 0 到 1 的运行流程
1. 准备环境：安装依赖，配置 `TUSHARE_TOKEN`。
2. 准备数据：运行 `prepare` 模式，得到处理后特征。
3. 训练稳健版本：运行 `v3` 模式，得到 rolling 与 final 指标。
4. 迁移对比：运行 `v4` 模式，对比 pandas 与 SQL 特征链路。
5. 看结果：优先查看 `outputs/v3/final/metrics.json` 和 `outputs/v4/compare_metrics.csv`。

## 5. 一键命令（Windows）
```powershell
./run_aurora.bat v3
./run_aurora.bat v4
./run_aurora.bat test
```

## 6. 结果怎么解释
- `f1`: 综合精度与召回，课程答辩主指标。
- `precision`: 模型说“会涨”时有多准。
- `recall`: 真涨样本被抓住了多少。
- `roc_auc`: 概率排序能力。
- `error_summary.json`: 错误结构（误报/漏报比例）。

## 7. 当前版本结论（v1.0）
- v3 主线可稳定复现完整产物。
- v4 证明 pandas 与 SQL 风格特征工程结果一致（核心指标差值 0）。
- 文档与结构已整理为可答辩、可交接、可扩展到 TradeEye 的状态。
