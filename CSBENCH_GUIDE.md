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

## 5. 优化评分准则

正式评分前，先使用每道题独立的 calibration 答案优化准则：

```powershell
.\venv\Scripts\python.exe main_pipeline.py --mode VARIANCE_OPT --questions CO_2 --sample-size 5 --database-path data\csbench\exam_database.json --answer-metadata data\csbench\answer_metadata.jsonl --initial-rubric-dir data\csbench\rubrics\initial --rubric-dir data\csbench\rubrics\optimized --results-dir results_runs\csbench_co2_rubric_opt --extraction-backend csbench_hybrid --ocr-cache-dir ocr_cache\csbench --progress-file results_runs\csbench_co2_rubric_opt\progress.json --force-rerun
```

该过程不读取教师真实分数，不覆盖 `source` 或 `initial`，结果写入：

```text
data/csbench/rubrics/optimized/CO/CO_2_rubric_standard.json
data/csbench/rubrics/manifests/CO/CO_2_optimization.json
```

## 6. 测试普通文字答案

```powershell
.\venv\Scripts\python.exe main_pipeline.py `
  --mode OCR_ONLY `
  --questions CO_1 `
  --student-ids ANS_CO_01 `
  --database-path data\csbench\exam_database.json `
  --teacher-db data\csbench\teacher_scores.json `
  --answer-metadata data\csbench\answer_metadata.jsonl `
  --initial-rubric-dir data\csbench\rubrics\initial `
  --rubric-dir data\csbench\rubrics\optimized `
  --results-dir results_runs\csbench_text_smoke `
  --extraction-backend csbench_hybrid `
  --ocr-cache-dir ocr_cache\csbench
```

普通文字题应直接使用 `raw_text`，不生成该样本的 PaddleOCR JSON。

## 7. 测试“如图所示”答案

```powershell
.\venv\Scripts\python.exe main_pipeline.py `
  --mode OCR_ONLY `
  --questions CO_2 `
  --student-ids ANS_CO_516 `
  --database-path data\csbench\exam_database.json `
  --teacher-db data\csbench\teacher_scores.json `
  --answer-metadata data\csbench\answer_metadata.jsonl `
  --initial-rubric-dir data\csbench\rubrics\initial `
  --rubric-dir data\csbench\rubrics\optimized `
  --results-dir results_runs\csbench_visual_smoke `
  --extraction-backend csbench_hybrid `
  --ocr-cache-dir ocr_cache\csbench `
  --force-rerun
```

结果位置：

```text
ocr_cache/csbench/CO_2/ANS_CO_516.json
ocr_cache/csbench/facts/CO_2/ANS_CO_516.json
results_runs/csbench_visual_smoke/CO_2_ocr_only.json
```

视觉模型只报告可见连接。离散标签不会被强行拼成执行路径。

## 8. 完整运行一份答案

```powershell
.\venv\Scripts\python.exe main_pipeline.py `
  --mode FULL `
  --questions CO_1 `
  --student-ids ANS_CO_01 `
  --database-path data\csbench\exam_database.json `
  --teacher-db data\csbench\teacher_scores.json `
  --answer-metadata data\csbench\answer_metadata.jsonl `
  --initial-rubric-dir data\csbench\rubrics\initial `
  --rubric-dir data\csbench\rubrics\optimized `
  --results-dir results_runs\csbench_full_smoke `
  --extraction-backend csbench_hybrid `
  --ocr-cache-dir ocr_cache\csbench `
  --force-rerun
```

## 9. 先运行少量样本，再运行整题

五份样本：

```powershell
.\venv\Scripts\python.exe main_pipeline.py `
  --mode FULL --questions CO_2 --answer-split test --img-limit 5 `
  --database-path data\csbench\exam_database.json `
  --teacher-db data\csbench\teacher_scores.json `
  --answer-metadata data\csbench\answer_metadata.jsonl `
  --initial-rubric-dir data\csbench\rubrics\initial `
  --rubric-dir data\csbench\rubrics\optimized `
  --results-dir results_runs\csbench_co2_5 `
  --extraction-backend csbench_hybrid `
  --ocr-cache-dir ocr_cache\csbench `
  --force-rerun
```

整道题：

```powershell
.\venv\Scripts\python.exe main_pipeline.py `
  --mode FULL --questions CO_2 --answer-split test `
  --database-path data\csbench\exam_database.json `
  --teacher-db data\csbench\teacher_scores.json `
  --answer-metadata data\csbench\answer_metadata.jsonl `
  --initial-rubric-dir data\csbench\rubrics\initial `
  --rubric-dir data\csbench\rubrics\optimized `
  --results-dir results_runs\csbench_co2_full `
  --extraction-backend csbench_hybrid `
  --ocr-cache-dir ocr_cache\csbench `
  --force-rerun
```

## 10. 评估

```powershell
.\venv\Scripts\python.exe evaluate.py `
  --questions CO_2 `
  --results-dir results_runs\csbench_co2_full `
  --result-source checkpoint `
  --teacher-db data\csbench\teacher_scores.json `
  --database-path data\csbench\exam_database.json `
  --compare `
  --compare-score-keys single avg selected 3wd
```

`evaluate.py` 会从 CSBench 兼容题库动态读取每道题满分。

## 11. 数据安全

- `actual_score` 只在评估和结果记录中使用，不进入 Stage1 提示词。
- `answer1.jsonl` 不参与转换，避免重复。
- `delete/` 不参与正式数据。
- 每题的 calibration、validation、test 答案互不重叠。
- calibration 只用于评分准则优化，test 只用于最终实验。
- `FULL` 默认要求 optimized 准则存在，不会静默回退 initial。
- 如只做提取冒烟测试，可显式增加 `--allow-initial-rubric`。
- `data/csbench/`、`ocr_cache/`、`results_runs/` 均不提交 Git。
