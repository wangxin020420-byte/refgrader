# RefGrader 项目现状

> 状态日期：2026-08-31  
> 说明：本文档描述当前主方法、已验证结果、已知限制和下一步，不替代按日期记录的 `CURRENT_PROGRESS.md`。

## 1. 项目目标

RefGrader 面向主观题自动评分，核心目标是将视觉/文本证据、细粒度评分准则、独立多次评分、三支决策和可审计校准组合为统一流水线。当前主方法不是单次调用大模型，而是：

```text
数据与题目契约
-> 初始/优化评分准则
-> Stage 1 事实与视觉证据提取
-> Stage 2 三次独立语义评分
-> U_E / U_S / U_R 风险建模
-> A3WA 路由（POS / BND / NEG）
-> BND 结构化证据仲裁
-> 3WD-Core
-> 可选 validation 残差校正
-> 最终分数、完整性检查、评估与归档
```

## 2. 当前主方法配置

| 项目 | 当前行为 |
| --- | --- |
| 默认文本模型 | `glm-5.2`，`thinking=disabled` |
| 默认视觉模型 | `glm-4.6v` |
| 可选文本模型 | 通过 `model_runtime.py` 和命令参数切换 DeepSeek/GLM，不修改默认值 |
| 私有数据活动评分准则 | 45 道题，受 `active_rubric_set.json` 的哈希约束 |
| 公共数据运行 | 必须显式传入独立的 `--prepared-dir` |
| 运行结果 | 本机 `results_runs/` 保存 checkpoint；可移植结果进入 `refgrader-artifacts` |

模型切换只对当前进程生效。GLM 与 DeepSeek 的模型、端点、温度和思考模式均进入运行契约；使用不同模型契约时不能复用旧 validation/calibration 配置。

## 3. 模块边界

| 边界 | 主要文件 | 责任 |
| --- | --- | --- |
| 主评分 | `main_pipeline.py`、`step4_vlm_grader.py` | 证据提取、三次评分、风险、路由和最终分数 |
| 准则 | `step3_rrd_generator.py`、`rubric_semantics.py` | 候选生成、语义契约、细粒度准则验证 |
| 校准 | `calibration_utils.py`、`scripts/calibrate_a3wa.py` | A3WA、BND、残差及门禁 |
| 模型契约 | `model_runtime.py` | GLM/DeepSeek 兼容配置和环境传递 |
| 私有数据编排 | `scripts/run_csbench.py` | 45 题优化、批改、校准、评估、归档 |
| 公共数据适配 | `benchmark_datasets/` | ASAP-SAS、Mohler、SAS-Bench 的只读规范化 |
| 公共实验编排 | `scripts/benchmarks/run_benchmark.py` | 使用独立 prepared 数据运行主方法 |
| 论文协议对齐 | `scripts/benchmarks/project_sasbench_official_protocol.py` 等 | 将既有整体分投影为 SAS-Bench 官方分步协议并计算论文指标 |

公共数据模块不修改 `data/csbench`。它只在指定的 prepared 目录、运行目录和协议输出目录写文件，因此不会覆盖私有数据评分准则或活动 A3WA 配置。

## 4. 已完成工作

### 4.1 私有计算机学科数据

- 已形成 45 道题的活动评分准则集合和严格哈希校验。
- 已完成私有 validation、A3WA/残差诊断、unsafe POS 与 BND 预算分析。
- 已实现教师标签疑似问题筛选和本地审核界面；原始标签只读，人工决定单独保存。
- 已实现文本、OCR、视觉占位符证据合并以及证据契约检查。

### 4.2 Mohler 公共数据

- 已完成 81 题、2,273 份答案的数据适配、完整性审计和全量外部测试。
- 全量 3WD 结果：MAE `0.7024`、RMSE `1.1238`、Pearson `0.8001`、SER2 `7.92%`、Bias `-0.5112`。
- 该结果属于跨数据集外部验证；若使用私有数据校准参数，应明确标注为迁移评估，不能表述为 Mohler 数据内调参结果。

### 4.3 SAS-Bench 论文复现与主方法测试

- 已审计原始 4,109 条记录，排除 16 条教师分超过题目满分的无效记录，正式测试为 4,093 条、1,111 个有效题目上下文。
- 已完成官方代码路线的 DeepSeek 复现，但模型为当前 `deepseek-chat`/DeepSeek V4 接口，并非论文 DeepSeek-V3 的完全同模型复现。
- 已使用 RefGrader 主流程完成 4,093 条整体评分。
- RefGrader 3WD：MAE `2.0794`、RMSE `3.2971`、Pearson `0.8265`；12 任务宏平均 QWK `76.71%`，答案数加权 QWK `77.02%`。
- 论文官方 CCS/ECS 对齐仍未完成：当前分步协议投影冒烟为 `11/12`，必须修复唯一失败后再进入 4,093 条全量投影。

详细状态见 `SAS_BENCH_REPRODUCTION_PROGRESS.md`。

## 5. 当前已知限制

1. 私有数据的 unsafe POS 部署门禁尚未通过，当前 A3WA 配置仍需按实验标签说明正式或实验性质。
2. SAS-Bench 运行中 `3WD-Core` 与最终 `3WD` 相同，说明该外部运行没有产生有效残差修正；这不是结果文件错误，但不能声称残差层带来提升。
3. SAS-Bench 官方 CCS/ECS 需要分步得分与错因标签；整体分跑完不等于论文协议评估完成。
4. 根目录仍保留若干早期一次性脚本和旧流水线副本，虽未进入当前入口，但增加了维护混淆，见 `REPOSITORY_AUDIT.md`。
5. `CURRENT_PROGRESS.md` 是历史日志且较长，不适合作为当前状态入口；后续优先维护本文件。

## 6. 下一步优先级

1. 读取 SAS-Bench 冒烟的 `projection_failures.jsonl`，使用同一输出标签只补跑 1 条。
2. 冒烟达到 `12/12` 后，执行 4,093 条标签盲分步协议投影。
3. 用官方脚本计算 RefGrader 的 CCS/ECS，并与论文 DeepSeek-V3、当前官方代码复现做同表比较。
4. 对表现差异最大的任务做错误分层，不使用测试标签回调模型或阈值。
5. 完成教师标签审核；冻结标签策略后再生成最终私有数据结论。
6. 将可证明无引用的旧脚本移入历史分支或删除，保留可复现提交标签。

