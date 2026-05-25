# Aurora (Py_finalwork)

Aurora 是一个课程机器学习项目，任务是基于股票日线数据做“次日涨跌二分类”预测。

当前交付版本：`v1.0`（2026-05-24 冻结）

## 1. 快速开始

### 1) 创建环境并安装依赖
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

### 2) 配置环境变量
创建 `.env`，至少包含：
```env
TUSHARE_TOKEN=your_tushare_token
```

### 3) 一键运行（Windows）
```powershell
./run_aurora.bat v3
./run_aurora.bat v4
./run_aurora.bat test
```

## 2. 常用模式
- `prepare`: 准备数据
- `prepare-refresh`: 强制刷新数据
- `v3`: 稳健训练主线（rolling + threshold + error analysis）
- `v3-refresh`: 刷新数据后跑 v3
- `v4`: pandas vs SQL 特征迁移对比
- `v4-refresh`: 刷新数据后跑 v4
- `test`: 运行测试

## 3. 结果查看
- v3 最终指标：`outputs/v3/final/metrics.json`
- v3 错误分析：`outputs/v3/error_summary.json`
- v4 迁移对比：`outputs/v4/compare_metrics.csv`
- v4 一致性摘要：`outputs/v4/feature_consistency_summary.json`

## 4. 文档入口（v1.0）
- 项目总览：`docs/v1_0_project_guide.md`
- 开发记忆：`docs/v1_0_dev_memory.md`
- 版本迭代：`docs/version_iterations.txt`
- 课程报告草稿：`docs/v1_0_course_report.md`
- PPT 思路：`docs/v1_0_ppt_outline.md`
- 答辩问答：`docs/v1_0_teacher_qa.md`
- TradeEye 对接方案：`docs/v1_0_tradeeye_integration.md`

## 5. 目录结构
- `src/aurora_ml/`: 核心实现
- `scripts/`: 训练/对比入口脚本
- `configs/`: 配置
- `data/`: 数据
- `outputs/`: 各版本实验产物
- `docs/`: 文档与答辩材料

## 6. 版本说明
- v0.1: 最小闭环
- v0.2: 多模型对比
- v0.3: 稳健性增强
- v0.4: SQL 风格特征迁移对比
- v1.0: 交付冻结（结构清理 + 文档闭环 + 对接方案）
