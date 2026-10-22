# RefGrader

主观题自动评分研究项目。日常入口更新：2026-10-22。

当前工作顺序：**冻结两轮基线 → 对齐数据与实际配置 → 验证集离线诊断 → 改进风险、阈值和BND门禁 → 独立确认实验**。

## 先看哪里

| 入口 | 内容 |
|---|---|
| [当前状态](PROJECT_STATUS.md) | 哪些实验已完成、哪些尚待归档；不把历史结论当作当前结果 |
| [目录与职责](docs/PROJECT_LAYOUT.md) | 代码、输入、冻结结果、审核记录和临时资料分别在哪里 |
| [冻结与多机同步](docs/guides/FROZEN_BASELINES_GUIDE.md) | PC2为何没有Git更改、归档命令、校验和同步顺序 |
| [下一阶段执行清单](experiments/ROADMAP.md) | 有序推进的任务、验收条件和暂停条件 |
| [实验登记表](experiments/run_registry/README.md) | PC1、PC2和历史对照的身份及状态 |
| [完整操作手册](docs/guides/COMMANDS_GUIDE.md) | 原有实验入口、续跑、监控及停止命令；结合具体run ID使用 |

## 两个日常仓库

```text
<workspace>/
├─ refgrader-main/                    代码、版本化输入、当前配置和研究索引
│  ├─ main_pipeline.py               当前评分入口
│  ├─ calibration_utils.py           风险、路由、动作及校正实现
│  ├─ scripts/                       实验、诊断、审核、归档工具
│  ├─ prompts/  canonicalizers/  ocr/
│  ├─ data/csbench/                  版本化数据、准则、划分、审核决定
│  ├─ experiments/                   冻结输入、实验索引和独立分析
│  ├─ docs/                          操作指南、研究记录、历史资料
│  ├─ results_runs/  logs/            本机运行结果；Git忽略
│  └─ .local-maintenance/            本机维护及临时构建；Git忽略
└─ refgrader-artifacts/               可移植实验历史；单独Git仓库
   └─ csbench/frozen_baselines/
      ├─ reviewed45_glm52_20261019_221414_317274/   PC1优化准则
      └─ initial85_glm52_20261020_pc2/             PC2初始准则，导出后出现
```

`CSBench_new` 是可选的同级上游数据仓库，不是当前实验的实时输入。
`refgrader-initial-ablation` 是PC2独立代码checkout；结果不会自动进入另一个checkout的Git视图。
VS Code可打开[多仓库工作区](RefGrader.code-workspace)；缺少上游目录时可从工作区移除该可选文件夹。

**`results_runs/`没有出现在源代码管理中是预期行为。**主仓库忽略运行结果和缓存；完成后需显式冻结到历史仓库。
初始准则工作流没有自动发布步骤。不要用`git add -f results_runs`解决结果同步。

## 当前方法与数据边界

评分链路：固定题目准则 → 作答证据提取与映射 → 三次重复语义评分与基础分选择 → 三支路由及BND证据仲裁 → 可选残差校正。

| 分数字段 | 含义 |
|---|---|
| `single_first_score` | 三次评分中的第一次，由`model_scores_history[0]`导出 |
| `model_avg_score` | 三次评分均值 |
| `selected_baseline_score` | 系统实际采用的基础分 |
| `three_way_core_score` | 路由与仲裁后的核心分，不含残差校正 |
| `final_calibrated_score` | 启用残差校正后的机器分；转人工状态仍须保留 |

本轮两臂计划使用同一份85条人工审核决定快照，calibration/validation/test有效人数为340/339/2675。
当前仍在增长的审核文件不等于这份冻结快照；不得用最新审核政策覆盖旧实验后继续解释同一结果。
同模型重复评分也不等同于三个独立教师。

GLM文本模型合同为`glm-5.2`、thinking disabled；视觉模型为`glm-4.6v`。
每条智谱调用均须走`https://open.bigmodel.cn/api/coding/paas/v4`；新调用前执行离线目的URL验证。
冻结工具只使用Python标准库，不调用模型，也不需要API Key。

## 冻结与验证的最短入口

从本机代码根目录运行：

```powershell
.\venv\Scripts\python.exe scripts\freeze_experiment_baseline.py freeze --arm pc1-optimized --artifacts-repo ..\refgrader-artifacts --dry-run
.\venv\Scripts\python.exe scripts\freeze_experiment_baseline.py freeze --arm pc1-optimized --artifacts-repo ..\refgrader-artifacts
```

PC2优先使用[单命令导出入口](docs/guides/FROZEN_BASELINES_GUIDE.md)，无需修改现有主实验Python路径。
每个快照包含原始工作流、数据与划分、准则、验证/测试检查点、原始评估、补算QWK及SHA-256清单。
重复执行会验证已有快照，不覆盖内容。旧的逐题归档和历史实验目录保留。

“completed”只代表执行完成；部署门禁失败的配置仍记为`experimental`。
代码仓库和实验历史仓库分别提交、推送；人工审核JSONL不与方法开发并发跨机写入。

## 其他指南

- [初始准则消融](docs/guides/INITIAL_RUBRIC_ABLATION_GUIDE.md)
- [审核版本主实验](docs/guides/REVIEWED_FULL_RUN_GUIDE.md)
- [教师标签审核](docs/guides/TEACHER_LABEL_REVIEW_GUIDE.md)
- [CSBench数据与实验](docs/guides/CSBENCH_GUIDE.md)
- [OCR环境](docs/guides/OCR_GUIDE.md)
- [Coding Plan离线检查](docs/guides/CODING_PLAN_VERIFICATION.md)
- [公开数据实验](docs/guides/PUBLIC_BENCHMARK_GUIDE.md)
- [SAS-Bench既有研究记录](docs/research/SAS_BENCH_REPRODUCTION_PROGRESS.md)
- [SCALE4固定构建指南](SCALE4_REFGRADER_GUIDE.md)（构建脚本依赖根目录路径，保留原位）

历史日志及过时交接说明已进入`docs/archive/`，只用于追溯。早期脚本以`.py.txt`保存，避免误执行。
评分模块、测试模块、冻结输入、运行结果及审核资料的原路径未迁移。
