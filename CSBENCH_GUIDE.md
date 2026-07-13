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
→ GLM-5.1 rubric 事实映射
→ 原有 Stage2
→ 原有 3WD
```

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
│   ├── optimized/    # VARIANCE_OPT 输出
│   └── manifests/    # 优化来源、样本和哈希记录
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
- `FULL` 默认要求 optimized 准则存在，不会静默回退 initial。
- 如只做提取冒烟测试，可显式增加 `--allow-initial-rubric`。
- `data/csbench/` 是版本化的内嵌批改快照；其中运行期 `rubrics/optimized` 和 `rubrics/manifests` 不提交。`ocr_cache/`、`results_runs/` 仍不提交 Git。

## 7. 评分准则语义契约

评分准则优化遵循“官方父项不变、证据结构细化”的约束：

- 每个初始评分项具有稳定的 `parent_id`、`parent_points` 和 `split_policy`。
- `scoring_policy=strict_atomic` 表示不可拆分的单一结果项，只能补充等价归一化或诊断证据。
- `scoring_policy=additive_split` 表示多个独立且均为满分必要条件的加法项；无官方权重时只允许等权正交拆分。
- `scoring_policy=final_sufficient_partial_credit` 表示“正确最终答案足以取得父项满分，最终答案错误或缺失时仍可依据明确过程证据获得部分分”。同一父项必须恰好包含一个 `full_credit_trigger`，过程分不得超过 `fallback_cap`。
- 不再因为单项分值大于等于 4 分而强制拆分。OCR 失败、学生错误和评分模型偶然误判也不能改变评分语义。
- 官方未给出子项权重时，只允许使用等权、正交、均为满分必要条件的证据原子；无法满足时保持父项计分不变，仅增加诊断证据。
- 优化结果除总分校验外，还必须通过父项分值守恒、父项可追溯、唯一满分触发项、过程分上限和满分答案不变性校验。校验失败会回退 immutable initial rubric；再次失败则拒绝落盘。

CO_1 使用层次评分父项 `step_1`：地址字段 `2.0` 分、有效地址 `1.5` 分、最终操作数 `1.5` 分，过程兜底上限为 `3.5`。最终操作数规范化后匹配 `37H` 时父项直接得 `5.0`，不要求学生额外写出过程；最终答案错误或缺失时，最终项计 `0`，只累计有书面证据的前两项。`37H`、`110111B`、`110111₂` 等显式进制表示会确定性归一化；无后缀数字只在对应条款明确声明 `implicit_bases` 时解释。

当至少两次有效语义探测中的严格多数均由规范化器确定性命中最终答案，且层次父项覆盖整道题的全部正分条款时，系统把满分视为评分准则的硬约束。三支路由仍按风险正常记录并可触发人工复核，但 BND Agent 和 validation 残差校准不得下调该确定性满分；语义模型自行判断的 MATCH、单次探测或只覆盖部分分值的层次条款不会触发此约束。

当多数语义探针确认层次父项由最终答案触发满分时，3WD 风险计算会把该最终项投影为父项全部分值，并从风险分母中移除非必需过程项。这样“答案正确但未展开过程”不会产生虚假的高留白风险；未触发满分时仍保留全部过程项，缺失证据继续影响 `U_E`、BND/NEG 路由和人工复核。

语义契约当前版本为 `2`。旧 optimization manifest 或未记录 `semantic_policy_validated=true` 的 optimized rubric 不允许进入正式批改，必须执行：

```bash
python scripts/run_csbench.py optimize CO_1 --force
```

CO_4 当前官方初始分值为 `2 + 2 + 2 + 2 + 2 + 5 + 5 = 20`。前五项分别考查地址字段参数，后两项分别考查两个地址的 Cache 命中结论及理由。

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
