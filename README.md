# RefGrader

> 跨设备项目入口。最后更新：2026-07-18。

RefGrader 是面向主观题自动评分的实验系统。当前研究主线是把混合视觉证据、细粒度评分准则、三次独立语义评分、三支决策（3WD/A3WA）、边界仲裁和可选残差校正组合为可审计的评分流水线。

当前评分准则语义契约为版本 4。分值大于等于 4 分的父项先区分结果充分、正交结果、组成部分、过程主导和严格原子五类：只有正交结果/组成部分使用等权拆分；复杂过程题要求过程至少占 80%、核心过程至少占 50%、最终结论不超过 20%，短过程题要求过程至少占 65%、结论不超过 35%。多字段结构使用逐字段规范化，不能再由一个局部数字触发整体匹配。优化候选只有通过父项分值守恒、角色权重和语义可追溯性检查后才会替换当前准则，正式批改前会再次执行同一门禁。

本文档只回答五个问题：项目目前是什么、三个仓库分别保存什么、多个设备如何保持一致、一次实验会自动复制什么、其他 Markdown 文档应该去哪里查。完整命令见 [COMMANDS_GUIDE.md](COMMANDS_GUIDE.md)，按日期的开发记录见 [CURRENT_PROGRESS.md](CURRENT_PROGRESS.md)。

## 1. 当前系统

正式 CSBench 流程使用仓库内嵌的 `data/csbench` 数据快照，通常不再依赖同级 `CSBench_new`：

```text
data/csbench 固定数据与 split
-> initial rubric
-> optimized rubric
-> csbench_hybrid 证据提取（raw_text / PaddleOCR / 视觉映射）
-> 三次独立语义评分
-> U_E / U_S / U_R 风险与可信隶属度
-> A3WA 阈值：POS / BND / NEG
-> BND 结构化证据仲裁
-> three_way_core_score
-> 可选 validation 残差校正
-> final_calibrated_score
-> 评估与 artifacts 归档
```

主要分数含义：

| 名称 | 含义 |
| --- | --- |
| `single` | 单次语义评分结果 |
| `avg` | 三次独立语义评分均值 |
| `selected` | 旧的选择式基线，用于历史对比 |
| `3wd-core` | 三支路由和 BND 证据仲裁后的分数，不含残差校正 |
| `3wd` | `3wd-core` 加上显式启用的 validation 残差校正 |

论文中应分别报告 `avg -> 3wd-core` 和 `3wd-core -> 3wd`，避免把残差校正的贡献错误归因于三支决策。

## 2. 三个仓库与工作区

建议在 VS Code 多根工作区中同时打开以下三个同级目录：

```text
RefGrader-CSBench.code-workspace
├── refgrader-main
├── CSBench_new
└── refgrader-artifacts
```

三个目录是三个独立 Git 仓库，必须分别拉取、检查、提交和推送。

| 仓库 | 定位 | 日常是否必需 | Git 中保存的主要内容 |
| --- | --- | --- | --- |
| `refgrader-main` | 代码与当前生效实验配置 | 必需 | 代码、内嵌数据快照、initial/optimized rubric、manifest、active A3WA |
| `CSBench_new` | 上游数据标注与协作仓库 | 仅更新数据时需要 | 原始题目、答案、图片和其他标注工作 |
| `refgrader-artifacts` | 跨设备实验结果仓库 | 正式实验必需 | rubric 优化记录、validation/calibration/test 结果、评估 CSV、日志和运行清单 |

### 2.1 `refgrader-main` 是代码、输入和当前配置真源

以下内容会进入 Git：

```text
代码、测试、prompts/*.md、项目文档
data/csbench/exam_database.json
data/csbench/teacher_scores.json
data/csbench/answer_metadata.jsonl
data/csbench/student_images/          # Git LFS
data/csbench/reference_images/        # Git LFS
data/csbench/rubrics/initial/
data/csbench/rubrics/optimized/       # 当前批准的优化准则
data/csbench/rubrics/manifests/       # 当前优化来源与哈希
data/csbench/rubrics/active_rubric_set.json
data/csbench/calibration/active_a3wa_config.json
data/csbench/splits/
```

以下内容由 `.gitignore` 排除，只保留在运行设备：

```text
venv/、.venv-ocr/                     # 每台设备独立安装
logs/                                 # 本机运行日志
results_runs/                         # checkpoint 和工作结果
ocr_cache/                            # 逐图片 OCR/事实缓存
```

optimized rubric、manifest 和 active A3WA 体积小且决定正式批改行为，因此必须随代码同步；checkpoint、日志和 OCR 缓存更新频繁且体积较大，仍只在本机运行目录保存，并把可移植历史结果复制到 `refgrader-artifacts`。

### 2.2 `CSBench_new` 是上游数据仓库

普通实验不要从它实时读取数据。只有确认上游数据更新并准备生成新的正式数据版本时，才执行显式导入和审计。导入完成后，以 `refgrader-main/data/csbench` 的提交版本作为该轮实验输入，避免另一项标注工作影响正在进行的实验。

数据结构、导入和审计规则见 [CSBENCH_GUIDE.md](CSBENCH_GUIDE.md)。

### 2.3 `refgrader-artifacts` 是历史实验真源

该仓库用于把一台设备产生的实验结果传递到服务器或其他本地电脑。典型结构为：

```text
csbench/<question_id>/
├── rubric_optimizations/<run_id>/
├── validation_runs/<run_id>/
├── calibration_runs/<run_id>/
└── grading_runs/<run_id>/
    ├── calibration/a3wa_config.json
    ├── rubrics/
    ├── rubric_optimization/
    ├── grading/
    ├── evaluation/
    ├── dataset/question_split.json
    ├── logs/experiment.log
    └── run_manifest.json
```

`run_manifest.json` 记录代码提交、数据提交、题目批次、split、模型/配置来源、文件哈希和结果数量，用于确认不同设备拿到的是同一轮实验。

两类真源不能混用：`refgrader-main/data/csbench` 表示下一次正式批改默认采用的**当前配置**；`refgrader-artifacts` 表示按 run ID 固化、不可覆盖解释的**历史证据**。重新优化或校准会修改前者，同时继续向后者新增历史快照。

## 3. 多设备约定

当前常用位置示例：

| 设备 | `refgrader-main` 示例路径 | 说明 |
| --- | --- | --- |
| 本地电脑 A | `C:\Users\wx\Desktop\refgrader-main` | 代码分析与本地评估 |
| 本地电脑 B | `D:\Users\王鑫020420\Desktop\refgrader-main` | 本地正式实验 |
| 实验室服务器 | `/home/E125221219/projects/refgrader` | 长时间后台实验 |

路径可以不同，但三个仓库应保持同级关系：

```text
<workspace>/refgrader-main
<workspace>/CSBench_new
<workspace>/refgrader-artifacts
```

不要同步或复制虚拟环境。每台设备分别建立 `venv` 和 `.venv-ocr`；PaddleX 模型缓存、OCR 缓存也属于设备本地状态，不由 Git 同步。

当前代码仍包含代码内 API 凭据配置，因此克隆后可能无需额外环境变量即可调用模型，但这种做法存在泄露风险。后续应轮换密钥并迁移到设备环境变量或不入库的配置文件；不得在文档、日志或 artifacts 中新增明文密钥。

## 4. 每次开始工作

先分别确认两个日常仓库没有未提交冲突，再拉取代码、LFS 数据和 artifacts。

在 `refgrader-main`：

```powershell
git status -sb
git pull --ff-only
git lfs pull
git lfs fsck
```

在 `refgrader-artifacts`：

```powershell
git status -sb
git pull --ff-only
```

如果本轮不修改上游数据，不需要操作 `CSBench_new`。如果需要导入上游新版本，先处理该仓库的协作冲突，再按 `CSBENCH_GUIDE.md` 生成和审计内嵌快照。

运行正式实验前建议执行：

```powershell
& ".\venv\Scripts\python.exe" scripts\audit_csbench_snapshot.py --prepared-dir data\csbench
& ".\venv\Scripts\python.exe" -m unittest test_canonicalizers.py test_rubric_semantics.py test_a3wa_theory.py test_csbench_artifact_sync.py -q
```

拉取后还应确认当前配置存在且无本地差异：

```powershell
Test-Path data\csbench\rubrics\active_rubric_set.json
git status --short
```

正式 `grade` 会重新核对数据快照、split、initial/optimized rubric、optimization manifest 和 active A3WA 的 SHA-256。任一文件与 active 清单不一致都会拒绝使用旧配置。test 未显式传入 `--a3wa-config` 时会自动采用覆盖当前题目的 `active_a3wa_config.json`；如果当前配置缺失、过期或不覆盖待测题目，test 会直接停止，只有显式传入 `--no-active-a3wa` 才允许执行无校准配置的消融实验。validation 始终不使用 A3WA 配置。

服务器对应使用当前 Conda 环境中的 `python`。完整的 Windows、Linux、后台、监控、停止、恢复和评估命令统一维护在 [COMMANDS_GUIDE.md](COMMANDS_GUIDE.md)。

## 5. 实验期间文件写到哪里

以 `CO_3 CO_4 --split test` 为例：

```text
results_runs/csbench_co3_co4_full/
├── active_run.json
└── runs/
    └── 20260716_210000/
        ├── run_state.json
        ├── completion_report.json
        ├── CO_3_grading_checkpoint.json
        ├── CO_3_graded_results.json
        ├── CO_3_rejected.json
        ├── CO_3_failed.json
        ├── CO_4_*.json
        ├── progress.json
        └── evaluation/
```

这些文件在 `refgrader-main` 中被 Git 忽略。因此批改正在运行时，VS Code 源代码管理没有出现更改是正常现象，不代表结果没有保存。

断点续跑依赖这些本机文件。使用相同题目集合、相同 split 和相同配置重新执行，并且不加 `--force`，程序会读取 `active_run.json`，继续同一 `run_id` 并跳过 checkpoint 中已有的成功 ID。`--force` 会创建新的时间戳运行目录，不再删除或覆盖历史实验；普通续跑不要使用 `--force`。需要精确选择历史运行时使用 `--run-id <ID>`。

## 6. 什么会自动复制

当前代码的实际行为如下：

| 操作 | 自动评估 | 自动复制到 artifacts | 自动 Git 提交/推送 |
| --- | --- | --- | --- |
| 完整 rubric 优化成功 | 不适用 | 是，进入 `rubric_optimizations` | 否 |
| validation/calibration 批改结束 | calibration 仅接受完整 validation | 是，进入独立阶段目录并标记完整性 | 否 |
| test 批改结束 | 是，按成功 checkpoint 评估 | 是，进入 `grading_runs` 并标记完整性 | 否 |
| `--limit` 调试运行 | 否 | 否 | 否 |
| 使用 `--no-artifacts` | 视命令而定 | 否 | 否 |
| 有失败/缺失但结构一致 | 是，标记 `partial` 和覆盖率 | 是，保留 failed/missing ID | 否 |
| 重复 ID、跨 split 或结果文件互相矛盾 | 否 | 否 | 否 |

完整 test 的自动收尾顺序是：

```text
检查 checkpoint、graded/rejected 与固定 test split 的结构一致性
-> 生成 completion_report.json（complete/partial、覆盖率、失败 ID）
-> 对成功 checkpoint 评估 single / avg / selected / 3wd-core / 3wd
-> 导出 compare.csv 和 summary.json
-> 复制运行到同级 refgrader-artifacts
```

默认只复制文件，不执行 `git add`、`commit` 或 `push`。这样用户可以先检查结果再决定是否共享。只有显式使用 `--push-artifacts` 才会自动提交和推送；日常建议保持手动推送。

### 部分结果与后续续跑

不完整 test 不再被丢在单机：只要批次正常收尾且结果结构一致，系统会评估已有成功样本，写入 `completion_report.json`，并以 `run_status=partial` 复制到 artifacts。报告同时保存成功数、预期数、覆盖率、missing ID 和 failed ID，不能把 partial 指标冒充全量指标。

后续不加 `--force` 续跑时，成功重试的 ID 会从 `failed.json` 删除；收尾阶段使用原 `run_id` 原子更新同一个 artifacts 目录和 index 记录。validation 也可按 partial 归档，但 `calibrate` 仍强制要求完整 validation，防止缺失模式造成校准偏差。

## 7. 实验结束后的 Git 操作

1. rubric 优化或 A3WA 校准后，先检查 `refgrader-main` 中 active 配置的 Git 更改。
2. 检查 `refgrader-artifacts` 是否出现预期目录和 `run_manifest.json`。
3. 确认 checkpoint 数量、failed 数量、评估样本数和题目 split。
4. 分别检查、提交并推送两个仓库；不要把两个仓库混成一次提交。
5. 另一台设备先拉取 `refgrader-main` 获取当前配置；需要评估历史结果时再拉取 artifacts。

常用检查：

```powershell
git -C "..\refgrader-artifacts" status --short
git status --short
```

不要把 `results_runs` 强行加入 `refgrader-main` Git；当前配置只保留准则、manifest、active 清单和 active A3WA，逐样本结果仍只归档到 artifacts。

## 8. 跨设备恢复规则

### 只评估已经发布的结果

在目标设备拉取 `refgrader-main` 和 `refgrader-artifacts` 后，使用 `scripts/evaluate_artifacts.py`。具体单项/多项分数组合和 `run_id` 选择见 `COMMANDS_GUIDE.md`。

### 继续一个已发布阶段

使用 `scripts/restore_csbench_artifacts.py` 恢复 rubric、validation、calibration 或 grading 文件，再用相同题目批次和配置续跑。恢复命令见 `COMMANDS_GUIDE.md`。

### 继续 partial 阶段

部分运行已包含 checkpoint、failed、completion report 和稳定 `run_id`。在目标设备拉取 artifacts 后，用 `scripts/restore_csbench_artifacts.py --run-id <ID>` 恢复；脚本会写入对应的版本化本地目录并设为 active。随后用相同题目、split 和 A3WA 配置执行不带 `--force` 的 grade 命令，即可继续同一实验。

## 9. 文档职责

项目中只有一个主 README。其他 `.md` 文件不是重复 README，而应各自承担单一职责：

| 文件 | 唯一职责 | 应该写什么 | 不应该写什么 |
| --- | --- | --- | --- |
| `README.md` | 项目入口和跨设备规则 | 当前架构、仓库边界、Git/复制行为、文档地图 | 大段历史日志、所有长命令 |
| `COMMANDS_GUIDE.md` | 操作手册 | 可复制命令、参数、监控、停止、恢复、评估、发布 | 重复研究背景和历史结果 |
| `CSBENCH_GUIDE.md` | 数据说明 | 数据结构、内嵌快照、split、导入、审计、rubric 数据契约 | 日常实验命令全集 |
| `OCR_GUIDE.md` | OCR 专项手册 | `.venv-ocr`、PaddleX 模型缓存、证据缓存、OCR 排错 | 三支决策理论和通用 Git 流程 |
| `CURRENT_PROGRESS.md` | 研发日志与交接记录 | 按日期追加的修改、验证结果、已知问题和下一步 | 作为日常命令的唯一来源 |
| `prompts/*.md` | 运行时模型提示模板 | 提取、评分、盲审清单和边界仲裁指令 | 项目说明；修改它们会改变实验行为 |

文档维护规则：

1. 项目行为发生变化时，先更新 README 中对应的“当前事实”。
2. 可复制命令只在 `COMMANDS_GUIDE.md` 保留完整版本，README 只给最短入口。
3. 每次代码修改和实验结论按日期追加到 `CURRENT_PROGRESS.md`，不要再复制整段到 README。
4. 数据和 OCR 的专项细节分别链接到对应指南。
5. 旧结论与当前行为冲突时，以代码、测试和 README 的“最后更新”日期为准，历史原因到 `CURRENT_PROGRESS.md` 查询。

## 10. 关键文件

| 路径 | 作用 |
| --- | --- |
| `scripts/run_csbench.py` | CSBench 统一入口、完整性检查、评估和 artifacts 发布 |
| `main_pipeline.py` | rubric 优化与正式批改调度、checkpoint/failed 保存 |
| `step4_vlm_grader.py` | 三次评分、三支路由、BND 仲裁和最终分数 |
| `calibration_utils.py` | A3WA 配置、残差校正及校准工具 |
| `evaluate.py` | 指标计算和比较结果导出 |
| `scripts/calibrate_a3wa.py` | 从 validation 生成 A3WA/可选残差配置 |
| `scripts/evaluate_artifacts.py` | 在任意设备评估已发布 artifacts |
| `scripts/restore_csbench_artifacts.py` | 从 artifacts 恢复中间阶段 |
| `ocr/backend.py` | 主环境到独立 PaddleOCR 环境的调用边界 |
| `data/csbench/manifest.json` | 内嵌数据快照来源和统计信息 |
| `data/csbench/rubrics/active_rubric_set.json` | 当前数据、准则、split 与 A3WA 哈希清单 |
| `data/csbench/calibration/active_a3wa_config.json` | test 默认使用的当前 A3WA/可选残差配置 |

## 11. 已知操作风险

- 智谱模型高峰期可能返回 429，失败样本会进入 `*_failed.json`。
- partial 指标只覆盖成功样本，论文比较必须同时报告覆盖率和失败样本处理方式。
- 同一题目批次换用不同 rubric、数据快照或 A3WA config 时，不应直接复用旧 checkpoint。
- 多设备同时向 `refgrader-artifacts` 写入前应先拉取并确认工作区干净。
- 多设备不得同时执行 rubric optimize 或 calibrate；这两个阶段会修改 `refgrader-main` 的当前配置。
- 不要在两个设备上同时续跑同一个尚未归档的结果目录。
- `CSBench_new` 的协作修改不会自动进入正式实验；必须显式导入并提交 `data/csbench`。

## 12. 新设备接手检查表

```text
[ ] 已克隆/拉取 refgrader-main
[ ] 已执行 git lfs pull 和 git lfs fsck
[ ] 已克隆/拉取 refgrader-artifacts
[ ] 三个仓库在同级目录，并在 VS Code 多根工作区中可见
[ ] 已建立主 venv 和独立 .venv-ocr
[ ] PaddleOCR 单图测试通过，模型缓存位置唯一
[ ] data/csbench 快照审计通过
[ ] 单元测试通过
[ ] artifacts 工作区在新实验开始前干净
[ ] 已确认本轮题目、split、rubric 批次和 A3WA config
[ ] active_rubric_set.json 校验通过，refgrader-main 无未拉取的配置提交
```
