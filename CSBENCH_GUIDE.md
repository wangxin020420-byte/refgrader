# CSBench 本地接入与测试

## 1. 目录

保持代码和数据集独立：

```text
C:\Users\wx\Desktop\
├── refgrader-main\
└── CSBench_new\
```

CSBench 原始目录只读。所有转换、准则优化、OCR 和评分结果都写入
RefGrader 当前项目，不修改原始数据集。

## 2. 证据策略

`csbench_hybrid` 使用统一混合证据，不采用 Text-only：

```text
已有 raw_text
+ cleaned 学生图片
+ 图表区域 PaddleOCR 标签/数字/坐标
+ GLM-4.6V 图形关系
→ 当前文本模型执行 rubric 事实映射（默认 GLM-5.2，无思考模式）
→ 原有 Stage2
→ 原有 3WD
```

文本模型、思考模式和视觉模型由 `model_runtime.py` 与统一 CLI 参数共同确定。事实映射、三次语义评分和 rubric 文本裁判共享同一文本模型契约，避免同一实验混用不同模型。切换文本模型后必须重新运行 validation 和 A3WA/残差校准。

普通文字题直接使用 `raw_text`，不会运行 PaddleOCR。含图答案或
`raw_text` 中出现“如图所示、见图、如下表”等占位语时，才运行
PaddleOCR 和图形关系解析。

## 3. 首次生成兼容数据

在 RefGrader 根目录执行：

```powershell
.\venv\Scripts\python.exe scripts\prepare_csbench.py `
  --dataset-root C:\Users\wx\Desktop\CSBench_new `
  --output-dir data\csbench `
  --link-mode copy `
  --exclude-questions OS_1 OS_2 `
  --force
```

生成：

```text
data/csbench/
├── exam_database.json
├── teacher_scores.json
├── answer_metadata.jsonl
├── manifest.json
├── rubrics/
│   ├── source/       # CSBench 原始准则副本
│   ├── initial/      # RefGrader 标准化初始准则
│   ├── optimized/    # 当前批准的 VARIANCE_OPT 输出（进入 Git）
│   ├── manifests/    # 当前优化来源、样本和哈希（进入 Git）
│   └── active_rubric_set.json # 当前配置总清单
├── calibration/
│   └── active_a3wa_config.json # test 默认配置
├── splits/
│   └── by_question/  # 每题 calibration/validation/test
└── student_images/
```

`copy` 会占用额外磁盘，但项目文件与 CSBench 原始文件完全隔离。
当前转换结果：

```text
43 道题
3326 条答案
OS_1、OS_2 因缺少 question 定义暂时排除
train/validation/test 按题目划分为 31/5/7 道题
```

## 4. 离线验证转换与路由

```powershell
.\venv\Scripts\python.exe scripts\check_csbench_integration.py `
  --prepared-dir data\csbench
```

预期全部为 `PASS`：

```text
complete_answer_id_lookup
text_transcription_route
visual_placeholder_route
diagram_merge
```

## 5. 统一实验入口

`scripts/run_csbench.py` 会根据题号自动推导题库、准则、缓存、结果和进度路径。
更换题目时只修改一个题号。

优化 CO_3 准则：

```bash
python scripts/run_csbench.py optimize CO_3
```

后台正式批改：

```bash
python scripts/run_csbench.py grade CO_3 --background --force
```

查看状态和日志：

```bash
python scripts/run_csbench.py status
```

```bash
python scripts/run_csbench.py tail
```

断点续跑：

```bash
python scripts/run_csbench.py grade CO_3 --background
```

评估并导出 CSV：

```bash
python scripts/run_csbench.py evaluate CO_3 --export
```

完整命令说明见 `COMMANDS_GUIDE.md`。

## 6. 数据安全

- `actual_score` 只在评估和结果记录中使用，不进入 Stage1 提示词。
- `answer1.jsonl` 不参与转换，避免重复。
- `delete/` 不参与正式数据。
- 每题的 calibration、validation、test 答案互不重叠。
- calibration 只用于评分准则优化，test 只用于最终实验。
- 外层 question-train/validation/test 与每题内部答案划分是两个正交层级：31 个
  question-train 用于方法开发，5 个 question-validation 用于跨题模型选择，7 个
  question-test 用于最终跨题泛化报告；每题内部 calibration/validation/test 仍分别
  服务于准则优化、A3WA/BND/残差校准和答案级最终评估。
- `FULL` 默认要求 optimized 准则存在，不会静默回退 initial。
- 如只做提取冒烟测试，可显式增加 `--allow-initial-rubric`。
- `data/csbench/` 是版本化的内嵌批改快照；当前 `rubrics/optimized`、`rubrics/manifests`、`active_rubric_set.json` 和 `calibration/active_a3wa_config.json` 必须提交，用于保证不同设备正式批改配置一致。`ocr_cache/`、`results_runs/` 和日志仍不提交 Git。

### 6.1 教师标签质量控制

教师标签去噪采用“原始数据不可变、候选自动筛查、人工确认、读取时应用”的四步流程。图片、`teacher_scores.json`、`answer_metadata.jsonl` 和 split 文件不移动、不改写；因此原始实验始终可以通过设置环境变量
`REFGRADER_SAMPLE_POLICY_MODE=raw` 复现。

1. 对全部 split 重新评分，生成可跨设备恢复的完整审计运行：

```powershell
$RunId = "teacher_audit_all7_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
.\venv\Scripts\python.exe scripts\run_csbench.py grade `
  CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 `
  --split all `
  --force `
  --run-id $RunId `
  --no-active-a3wa
```

不要添加 `--no-artifacts`。运行结束后，结果会自动复制到
`refgrader-artifacts/csbench/<题号>/audit_runs/<run_id>`。如果存在失败样本，
已完成 checkpoint 和失败清单仍会复制；使用同一 `run_id` 断点续传后会更新
同一 artifact 目录。另一台设备恢复命令：

```powershell
.\venv\Scripts\python.exe scripts\restore_csbench_artifacts.py `
  CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 `
  --stage audit `
  --run-id <run_id> `
  --force
```

2. 从一轮现有批改 checkpoint 生成候选：

```powershell
.\venv\Scripts\python.exe scripts\audit_teacher_labels.py `
  CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 `
  --results-dir "results_runs\<批次目录>\runs\<run_id>"
```

候选依据题内鲁棒差值阈值产生，同时输出 `U_E/U_S/U_R`、提取质量、路由和多个模型分数。候选只表示“需要复核”，不能直接判为教师噪声；存在提取风险、评分不稳定或 rubric 映射风险时会降为 `P2`。

3. 推荐启动本地审核界面，不再直接编辑多份候选 CSV：

```powershell
.\venv\Scripts\python.exe scripts\review_teacher_labels.py `
  --report-run <报告目录名>
```

界面会按 `question_id + answer_id` 合并模型均分与 3WD-Core
报告中的重复候选，同时展示学生原图、教师分、多个模型分数、路由、
`U_E/U_S/U_R`、提取文本和初筛提示。每次保存都会原子更新
`data/csbench/quality_control/reviews/<报告目录名>_decisions.jsonl`。
详细使用说明见 `TEACHER_LABEL_REVIEW_GUIDE.md`。

人工决定包括：

- `confirmed_noise`：确认教师标签不可用于实验；
- `corrected`：保留样本，并在 `corrected_score` 填写复核分；
- `retained_hard_case`：教师标签有效，只是模型难例；
- `ambiguous`：暂不处理，继续保留。

4. 在界面中检查决定后，点击“生成并启用策略”。也可以使用命令行对界面生成
的决定 JSONL 先生成候选策略，再显式启用：

```powershell
.\venv\Scripts\python.exe scripts\compile_sample_quality_policy.py `
  --decisions "data\csbench\quality_control\reviews\<报告目录>_decisions.jsonl" `
  --output "data\csbench\quality_control\policies\candidate_policy.json"

.\venv\Scripts\python.exe scripts\compile_sample_quality_policy.py `
  --decisions "data\csbench\quality_control\reviews\<报告目录>_decisions.jsonl" `
  --policy-id "co_teacher_review_v1" `
  --activate
```

5. 提交活动策略后，所有设备拉取同一版本。批改、rubric optimization、validation、A3WA/残差校准和正式评估都会统一排除 `confirmed_noise`，并使用 `corrected` 的复核分。策略描述和哈希进入 active 配置、A3WA 配置、checkpoint run signature、completion report 和 artifacts；策略发生变化时，旧运行不能被误续跑。

`ambiguous` 与 `retained_hard_case` 默认不排除，避免把模型错误或评分准则错误误判为教师噪声。正式论文应同时报告原始标签结果、清洗后主结果、排除数量与规则，并保留策略文件作为可审计证据。

## 7. 评分准则语义契约

评分准则优化遵循“官方父项不变、证据结构细化”的约束：

- 每个初始评分项具有稳定的 `parent_id`、`parent_points` 和 `split_policy`。
- `scoring_policy=strict_atomic` 表示不可拆分的单一结果项，只能补充等价归一化或诊断证据。
- `scoring_policy=additive_split` 表示多个独立且均为满分必要条件的加法项；无官方权重时只允许等权正交拆分。
- `scoring_policy=final_sufficient_partial_credit` 表示“正确最终答案足以取得父项满分，最终答案错误或缺失时仍可依据明确过程证据获得部分分”。同一父项必须恰好包含一个 `full_credit_trigger`，过程分不得超过 `fallback_cap`。
- `scoring_policy=role_weighted_additive` 表示过程主导题。复杂推导至少拆为 `support_process/core_process/final`，过程不少于 80%、核心过程不少于 50%、结论不超过 20%；短推导至少拆为 `core_process/final`，过程不少于 65%、结论不超过 35%。官方明确权重优先。
- 分值小于 4 分的普通项不因分值本身强制拆分；分值大于等于 4 分的项必须先按 `task_semantics` 分类为 `strict_atomic/result_sufficient/orthogonal_additive/component_additive/process_dominant`。正交/组成部分采用等权原子，过程与结论不得机械等权。
- 高分 `strict_atomic` 单一结果项不机械拆分，必须记录 `decomposition_exemption=strict_atomic_single_outcome`，从而区分“合理保持原子”与“遗漏拆分”。OCR 失败、学生错误和评分模型偶然误判不能改变该分类。
- 官方未给出子项权重时，正交或组成部分只允许等权；过程主导题使用上述角色比例约束。裸结论默认只获得低权重 `final` 分，不能反推过程；只有题干明确要求证明/理由时才允许 `dependency_mode=evidence_required`。
- 多字段地址、表项和带标签记录使用 `structured_fields` 逐字段比较；`bit_vector` 仅用于真正的位掩码或位集合。缺字段、错字段或字段次序错误只能得到部分匹配，不能触发确定性满分。
- 优化模型输出后会执行结构验收；未完成最小拆分时，验收错误会反馈给模型并最多重试 3 次。最终仍不合格时本轮优化失败，并保留当前有效 optimized rubric，不会先写入粗粒度草稿或用失败候选覆盖它。
- 正式优化不再把 `noninferiority_baseline_fallback` 当作成功结果。候选生成失败或语义契约失败仍会保留旧 active rubric 并以非零状态结束。若候选先通过结构契约、随后被完整的配对教师分非劣回放拒绝，则系统可将未改动的官方基线记录为 `calibrated_noninferior_baseline_selected` 并继续批处理；manifest 会保存回放覆盖率、MAE、严重退化数和延期拆分原因。该模式是经数据门禁选择出的正式结果，不等同于诊断 fallback。
- 优化结果除总分校验外，还必须通过父项分值守恒、父项可追溯、最小子项数、等权约束、唯一满分触发项、过程分上限和满分答案不变性校验。正式 `grade` 会再次运行同一结构校验，不能只依赖 manifest 中的成功标志。
- 版本 5 进一步把父项标准答案中的二进制、十六进制和最终判断作为不可变事实锚点。子项允许改变分组、空格和表示格式，但不得引入父项不存在的强事实字面量，也不得反转最终结论。
- 结构验收后，候选准则还会复用 calibration 的已提取事实重新评分，并与初始准则在同一批教师分上做配对非劣验收。覆盖率不足、平均绝对误差超过总分比例界限或出现单样本严重退化时，候选被拒绝并保留初始准则。该门禁不读取 validation/test 标签。

CO_1 使用层次评分父项 `step_1`：地址字段 `2.0` 分、有效地址 `1.5` 分、最终操作数 `1.5` 分，过程兜底上限为 `3.5`。最终操作数规范化后匹配 `37H` 时父项直接得 `5.0`，不要求学生额外写出过程；最终答案错误或缺失时，最终项计 `0`，只累计有书面证据的前两项。`37H`、`110111B`、`110111₂` 等显式进制表示会确定性归一化；无后缀数字只在对应条款明确声明 `implicit_bases` 时解释。

当至少两次有效语义探测中的严格多数均由规范化器确定性命中最终答案，且层次父项覆盖整道题的全部正分条款时，系统把满分视为评分准则的硬约束。三支路由仍按风险正常记录并可触发人工复核，但 BND Agent 和 validation 残差校准不得下调该确定性满分；语义模型自行判断的 MATCH、单次探测或只覆盖部分分值的层次条款不会触发此约束。

当多数语义探针确认层次父项由最终答案触发满分时，3WD 风险计算会把该最终项投影为父项全部分值，并从风险分母中移除非必需过程项。这样“答案正确但未展开过程”不会产生虚假的高留白风险；未触发满分时仍保留全部过程项，缺失证据继续影响 `U_E`、BND/NEG 路由和人工复核。

语义契约当前版本为 `6`。旧 optimization manifest、实际结构未通过版本 6 校验，或未记录 `semantic_policy_validated=true` 的 optimized rubric 不允许进入正式批改，必须执行：

```bash
python scripts/run_csbench.py optimize CO_1 --force
```

优化落盘后，统一入口会把 manifest 中的设备绝对路径转换为 `${REFGRADER_ROOT}` / `${PREPARED_CSBENCH_ROOT}` 占位符，并原子更新 `active_rubric_set.json`。正式 `grade` 会校验数据快照、split、initial/optimized rubric 和 manifest 的 SHA-256；文件存在但哈希不一致同样会拒绝运行。任意 active rubric 改变都会令旧 active A3WA 标记为 stale，必须重新完成 validation 和 `calibrate` 才能重新激活；test 不会静默回退默认参数，无校准消融必须显式传 `--no-active-a3wa`。

A3WA 校准会分别统计 BND 的结构化加分和降分动作在 validation 上相对 `avg` 的误差收益。每个方向只有在达到最小样本数且平均收益为正时才会写入 `boundary_policy.allow_raise/allow_lower=true`；否则运行时保持基线分。留一题交叉验证会在每个折内只用其余题目重新拟合动作开关，被留出的题目不参与阈值、不确定性或动作方向选择。跨题非劣、路由预算、总体 BND 正收益和已启用动作正收益共同组成 deployment gate。门禁失败的配置会保留为诊断输出，但不会自动激活、发布或进入正式 test。

候选准则还必须通过 calibration 教师分非劣门禁。若候选出现严重样本退化，
系统不会为了满足强制拆分而接受负优化；它会保留评分内容完全未改变的基线，
并在 optimization manifest 中记录
`semantic_validation_mode=calibrated_noninferior_baseline_selected`、
`fallback_reason` 和 `decomposition_deferred=true`。该例外只允许未改变的基线
延迟拆分，任何修改过分值、答案锚点或结构的候选仍必须完整通过版本 6 契约。诊断模式 `noninferiority_baseline_fallback` 仍禁止进入正式批改。
批量任务中断后使用 `optimize ... --resume`，已通过当前契约和哈希检查的题目会
跳过，其他题目复用既有 calibration checkpoint 继续执行。

CO_4 当前官方初始分值为 `2 + 2 + 2 + 2 + 2 + 5 + 5 = 20`。前五项分别考查地址字段参数，后两项分别考查两个地址的 Cache 命中推导及结论。优化后前五项保持父项分值不变，两个 5 分过程主导父项各至少形成三个角色子项，典型比例为 `1.5 + 2.5 + 1.0`，但实际条目必须由官方答案支持并通过比例门禁，而非按题号硬编码。

### 7.1 自动细粒度优化边界

`data/csbench/rubrics/source` 和 `data/csbench/rubrics/initial` 保存官方粗粒度输入，不预先写入人工设计的细粒度答案。优化阶段依据题目、官方答案和 rubric calibration 样本判断父项属于原子结果、多结果并列、过程与结论复合或图示/序列复合，再生成可审计的细粒度候选。候选必须保持父项分值和题目总分守恒，只能使用官方答案能够支持的独立证据，不得增加隐藏作答要求或跨父项移动分值。

CO_1 至 CO_7 用于审计自动优化行为，而不是在输入文件中硬编码拆分结果。重点检查包括：CO_1 的显式结果充分规则应保留；CO_2 的图示组成、CO_3/CO_5 的正交结果、CO_4 的复杂推导、CO_6 的组成计算和 CO_7 的短推导应按各自任务语义生成不同结构。当前实现使用 rubric calibration 样本的多次评分方差定位模糊项，并通过父项分值守恒、角色比例、最小子项数、事实锚点守恒和配对教师分非劣回放决定候选能否保存。教师分门禁只使用 calibration，不使用 validation 或 test 数据。

教师宽松给分在语义契约中具体表示：等价进制、等价单位、大小写、空格和箭头等纯形式差异不扣分；上游数值错误但下游公式或映射正确时保留对应过程分；不要求复现标准答案的全部算术展开。宽松不等于给精确数值设置通用 10% 容差，也不等于从裸最终答案反推未书写的过程。只有显式的 `final_sufficient_partial_credit` 父项允许最终答案触发父项满分。

## 7.2 当前三支决策校准契约

validation 教师分只用于拟合“可安全自动批改”的单调隶属度、conformal
不确定性区间和 A3WA 损失参数，不进入 test 推理，也不直接形成默认加分表。
POS 自动接受，NEG 拒判并进入人工复核，BND 先执行一次结构化二次审查：只有
明确条目 ID、可核验证据、合法 reason type、足够置信度且方向一致时才允许在条目
分值和全题 20% 双重上限内改分，否则转人工复核。

论文评估必须至少报告两组结果：`avg -> three_way_core_score` 衡量三支决策本身，
`three_way_core_score -> final_calibrated_score` 衡量可选残差层。默认残差层关闭，避免
把 validation 的整体偏差校正错误归因于 3WD。

启用残差层时，样本量达到 `min_cell_count` 的题目/路由单元才直接学习校正；较小的题目级单元仍保存为方向诊断。当运行时只能回退到跨题 route/global 校正，而本题至少 3 个 validation 样本显示相反且达到最小实质幅度的残差方向时，本次跨题校正被阻止。这样保留层次回退能力，同时避免其他题目的系统偏差反向修正当前题目。

## 8. 内嵌数据快照

正式批改默认使用仓库内的 `data/csbench`，不再依赖同级目录中的 `CSBench_new`：

- `exam_database.json`：题目文本、参考图和准则路径；
- `answer_metadata.jsonl`、`teacher_scores.json`：学生原文与教师分；
- `student_images/`：3,326张正式样本图片；
- `reference_images/`：题目图和标准答案图；
- `rubrics/source`、`rubrics/initial`：权威准则镜像与初始可执行准则；
- `splits/`：固定 calibration、validation、test 划分。

运行时结果按 split 隔离：test 为兼容旧评估命令继续使用
`results_runs/csbench_<题目集合>_full`，validation 和 calibration 分别使用
`..._validation`、`..._calibration`。完整中间结果发布到 artifacts 的
`validation_runs`/`calibration_runs`，可通过
`scripts/restore_csbench_artifacts.py` 在另一设备恢复；因此不再需要手工复制或修改
checkpoint 与 rubric manifest。

图片通过 Git LFS 管理。普通实验使用不带 `--dataset-root` 的命令。只有明确决定导入外部数据集新版本时，才运行 prepare；`run_csbench.py run --dataset-root ...` 会在 prepare 后自动调用 `embed_csbench_snapshot.py`，补齐参考图并消除绝对路径。导入后使用下列命令严格审计：

```bash
python scripts/audit_csbench_snapshot.py --source-root /path/to/CSBench_new
```

审计覆盖重复JSON键、题目总分、答案ID、教师分、图像引用、split覆盖以及外部源与内嵌快照的一致性。
