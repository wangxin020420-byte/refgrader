# RefGrader Latest Status

## 2026-06-25 Unified CSBench Execution Flow

RefGrader now supports a single CSBench command that can prepare data, optimize
rubrics, and launch formal grading in sequence. This is the recommended entry
point for CO_1 to CO_7 batch experiments.

Main idea:

```text
versioned embedded snapshot under data/csbench
-> rubric optimization from data/csbench/rubrics/initial
-> formal grading with data/csbench/rubrics/optimized
-> optional evaluation/export
-> portable artifacts copied to refgrader-artifacts
```

Important directories:

```text
data/csbench
  Versioned, portable RefGrader grading snapshot. It contains 43 questions,
  3,326 answers, student images, reference images, scores, and fixed splits.

data/csbench/rubrics/initial
  Initial rubrics converted from the source dataset.

data/csbench/rubrics/optimized
  Rubrics produced by the variance optimization stage.

results_runs
  Runtime checkpoints and local run outputs.

../refgrader-artifacts
  Portable experiment artifacts for syncing server results back to local.
```

Normal grading no longer requires a sibling `CSBench_new` checkout. The
separate dataset repository can continue annotation work without changing a
formal RefGrader experiment. Only an explicit `--dataset-root` import updates
the embedded snapshot. Images under the snapshot are managed with Git LFS;
runtime optimized rubrics, manifests, OCR caches, and experiment results stay
untracked.

Four common CO_1 to CO_7 commands:

```bash
python scripts/run_csbench.py run CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --force --background
```

Use this for normal experiments with the versioned embedded snapshot. It runs:

```text
background optimize rubrics -> formal grading
```

```bash
python scripts/run_csbench.py run CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --dataset-root /home/E125221219/CSBench_new --force --background
```

Use this only for an intentional external dataset import. It runs:

```text
background prepare -> finalize embedded snapshot -> optimize rubrics -> formal grading
```

```bash
python scripts/run_csbench.py optimize CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --force
```

Use this to regenerate optimized rubrics only.

```bash
python scripts/run_csbench.py grade CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --background --force
```

Use this to rerun formal grading only, assuming optimized rubrics already
exist and match their manifests.

Monitoring and evaluation:

```bash
python scripts/run_csbench.py status
python scripts/run_csbench.py tail
python scripts/run_csbench.py evaluate CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --export
```

For a complete `--split test` run, `grade` now performs the evaluation/export
and copies a complete portable run to the sibling `refgrader-artifacts`
repository automatically. The copy remains as local Git changes for review;
commit and push are still manual unless `--push-artifacts` is explicitly used.
Complete validation/calibration splits are published to their own
`validation_runs`/`calibration_runs` directories and are never mixed with
formal `grading_runs`. Limited debug runs and incomplete checkpoints are not
published. A complete validation grade is published immediately;
`run_csbench.py calibrate` then archives a second immutable validation run with
the derived A3WA config, allowing another device to restore and
continue the test stage. When `--a3wa-config` is provided for test, the exact
config and SHA-256 are archived with the formal run.

Notes:

```text
1. prepare_csbench.py is not a model step. It only converts CSBench_new into
   the data/csbench layout expected by RefGrader.
2. Re-run prepare only when the dataset changed, after pulling CSBench, or
   when data/csbench may be stale.
3. With --background, the new run subcommand starts the whole chained workflow
   in the background. The background process still runs stages sequentially, so
   grading cannot start before optimized rubrics exist.
4. Separate optimize and grade commands remain available for controlled tests.
```

## 2026-06-22 CSBench Hybrid Dataset Support

RefGrader can now use the external `CSBench_new` dataset without copying it
into this repository. See `CSBENCH_GUIDE.md`.

Implemented:

```text
scripts/prepare_csbench.py
  Converts 8 primary answer JSONL files and question JSON files.
  Excludes answer1.jsonl, delete/, and currently undefined OS_1/OS_2.
  Builds per-question copy/hard-link views and three-layer rubric directories.

scripts/run_csbench.py
  Provides one unified interface for rubric optimization, grading, monitoring,
  and evaluation. Only the question ID normally needs to change.

csbench_hybrid extraction backend
  Uses existing human raw_text for written content.
  Runs PaddleOCR + GLM-4.6V only for answers marked as visual or containing
  placeholders such as "如图所示".
  Never treats the placeholder itself as diagram evidence.

Dynamic runtime inputs
  --database-path
  --teacher-db
  --answer-metadata

Evaluation
  evaluate.py reads dynamic question max scores and complete answer IDs.
```

Local prepared-data validation:

```text
43 questions
3326 answers
31/5/7 question-level train/validation/test split
complete answer ID lookup: PASS
text transcription route: PASS
visual-placeholder route: PASS
diagram merge: PASS
```

The original Q1-Q7 defaults remain unchanged.

## 2026-06-18 PaddleOCR Optional Backend And Q2 Sequence Test

The legacy `glm_vlm` extractor remains the default. A new optional extraction
path is now available:

```text
PaddleOCR raw evidence
-> conservative blank-authenticity triage
-> GLM-5.1 blind fact mapping
-> conditional GLM-4.6V diagram-relation parsing
-> mapped-fact cache
-> existing Stage2 / 3WD grading
```

Implemented in order:

```text
1. Verified isolated .venv-ocr environment and single-image OCR.
2. Upgraded ocr/paddle_ocr_worker.py to schema_version=2.
3. Added SHA-256 raw OCR caching and conservative blank triage.
4. Added --extraction-backend {glm_vlm,paddle_glm5}.
5. Added OCR_ONLY and GRADE_ONLY modes.
6. Added GLM-5.1 OCR-to-checklist fact mapping.
7. Added conditional GLM-4.6V parsing only for diagram rubric items.
8. Kept glm_vlm unchanged for ablation comparison.
```

Cache layout:

```text
ocr_cache/Q2/E01914115_Q2.json
  Raw PaddleOCR tokens, confidence, coordinates, image hash, blank evidence.

ocr_cache/facts/Q2/E01914115_Q2.json
  GLM-5.1 mapped facts, conditional diagram facts, model metadata, image hash.
```

Focused Q2 second-subquestion test:

```text
Sample: E01914115_Q2
PaddleOCR: 23 tokens, mean confidence 0.891514
Blank authenticity: confirmed_nonblank
Conditional diagram parser: enabled
Observed sequence:
  用户程序→A→C→D→C→E→B→用户程序

Automated relation checks:
  first A              PASS
  C interrupted by D   PASS
  D returns to C       PASS
  E before B           PASS
  return to user       PASS
```

Run the focused check:

```powershell
.\venv\Scripts\python.exe scripts\check_q2_sequence_extraction.py
```

Actual one-sample `GRADE_ONLY` result:

```text
backend        model scores       avg    final   teacher
glm_vlm        15.8,18.0,17.0     16.9   18.0    18.0
paddle_glm5    12.0,11.4,10.0     11.1   12.5    18.0
```

Interpretation: the new conditional diagram path now captures the important
`C→D→C` middle nesting/return relation. PaddleOCR still misses several
handwritten five-bit mask words in the upper answer, so `paddle_glm5` must
remain optional and must not replace `glm_vlm` as the formal default yet.

Commands:

```powershell
# Extraction only: create/reuse raw OCR and mapped-fact caches.
.\venv\Scripts\python.exe main_pipeline.py `
  --mode OCR_ONLY --questions Q2 --student-ids E01914115 `
  --extraction-backend paddle_glm5 `
  --results-dir results_runs\q2_ocr_backend_test `
  --rubric-dir results_rrd_vlm

# Grade only: no PaddleOCR or diagram extraction is rerun.
.\venv\Scripts\python.exe main_pipeline.py `
  --mode GRADE_ONLY --questions Q2 --student-ids E01914115 `
  --extraction-backend paddle_glm5 `
  --results-dir results_runs\q2_grade_only_test `
  --rubric-dir results_rrd_vlm --force-rerun

# End-to-end optional backend.
.\venv\Scripts\python.exe main_pipeline.py `
  --mode FULL --questions Q2 `
  --extraction-backend paddle_glm5

# Legacy ablation baseline; this remains the default.
.\venv\Scripts\python.exe main_pipeline.py `
  --mode FULL --questions Q2 `
  --extraction-backend glm_vlm
```

Blank test result:

```text
Normal Q5 E01914115_Q5: confirmed_nonblank.
Known blank Q5 E02014181_Q5: uncertain, not falsely confirmed as blank/nonblank.
```

`uncertain` is intentional: printed question text and score marks prevent a
truthful blank conclusion from OCR alone.

### Local/server synchronization

Commit and synchronize source code, scripts, and dependency declarations. Do
not synchronize `venv/`, `.venv-ocr/`, `ocr_cache/`, or `results_runs/`.
Virtual environments contain platform-specific binaries and must be rebuilt on
each machine.

After pulling the same commit on a Linux server, run once:

```bash
chmod +x scripts/setup_paddle_ocr.sh
./scripts/setup_paddle_ocr.sh
```

Then run one question with the optional backend:

```bash
./run_experiment.sh run \
  --mode FULL \
  --questions Q2 \
  --extraction-backend paddle_glm5 \
  --force-rerun
```

The runtime automatically uses `.venv-ocr/Scripts/python.exe` on Windows and
`.venv-ocr/bin/python` on Linux.

## 2026-06-14 Generalized Rubric And 3WD Update

Current paper-facing chain:

```text
official coarse rubric -> structured JSON rubric -> high-variance item diagnosis
-> metadata-aware rubric refinement -> U_E/U_S/U_R primary risks -> A3WA route
```

Key implementation points:

```text
1. No teacher historical error signal is used to rewrite rubrics.
2. High-variance samples are diagnostic evidence, not direct answer keys.
3. Rubric refinement must classify the issue source first:
   rubric_ambiguity, rubric_granularity, extraction_failure,
   equivalent_representation_gap, or scoring_model_error.
4. Only rubric_ambiguity and rubric_granularity may change rubric wording or split points.
5. Extraction failure and equivalent-expression gaps can only add metadata:
   answer_type, canonicalization, evidence_source, dependency_group,
   source_text, parent_official_item.
6. 3WD routing exposes three primary variables:
   U_E = evidence quality risk,
   U_S = score stability risk,
   U_R = rubric adaptation risk.
7. Route risk is R(x) = (U_E + U_S + U_R) / 3,
   and confidence is mu(x) = 1 - R(x).
```

Updated files: `calibration_utils.py`, `main_pipeline.py`, `step3_rrd_generator.py`, `step4_vlm_grader.py`, and `evaluate.py`.

Validation commands:

```powershell
python -m py_compile calibration_utils.py step4_vlm_grader.py main_pipeline.py step3_rrd_generator.py evaluate.py
python evaluate.py --compare --questions Q2 Q3 --compare-score-keys single avg selected 3wd --compare-output outputs/q2_q3_primary_risk_compare.csv
python scripts\replay_calibration.py --results-dir results_rrd_vlm --files results_rrd_vlm\Q2_grading_checkpoint.json results_rrd_vlm\Q3_grading_checkpoint.json
```

## 运行命令速查

项目已经提供服务器后台运行脚本 `run_experiment.sh`。该脚本内部已经使用 `nohup` 启动 `main_pipeline.py`，因此在实验室服务器上不要再手动套一层 `nohup python main_pipeline.py ... &`。

本地 Windows 只跑 Q2、Q3：

```powershell
python main_pipeline.py --mode FULL --questions Q2 Q3 --progress-file results_rrd_vlm\progress_q2_q3_local.json
```

实验室 Linux 服务器只跑 Q4、Q5：

```bash
cd /home/E125221219/projects/refgrader
conda activate ref-grader
./run_experiment.sh run --mode FULL --questions Q4 Q5 --progress-file results_rrd_vlm/progress_q4_q5_server.json
```

服务器管理命令：

```bash
./run_experiment.sh status
./run_experiment.sh tail
./run_experiment.sh stop
python monitor.py --watch
```

题目选择优先级：

```text
1. 命令行 --questions 优先级最高，会覆盖 main_pipeline.py 中的 GRADING_CONFIG。
2. 不传 --questions 时，FULL 模式使用 GRADING_CONFIG。
3. 不传 --questions 时，VARIANCE_OPT 模式使用 VARIANCE_CONFIG。
4. 本地和服务器同时跑不同题目时，建议使用不同 progress 文件。
```

## 2026-06-06 Selected Baseline Update

Read this section first. It records the latest code change after the Q6/Q7
formal-result analysis.

Generality update:

```text
The selected-baseline logic no longer uses max_score >= 15.0 to guess whether a
question is a complex derivation/calculation task.

Instead, calibration_utils.infer_rubric_task_profile() infers the task structure
from rubric item metadata:
  answer_type / role / process-point ratio / result-point ratio /
  numeric-formula-point ratio / concept-judgement-point ratio.

The upper-consensus baseline is enabled only when the rubric has enough process
structure:
  complex_derivation_task
  and process_points_ratio >= 0.60
  and numeric_formula_points_ratio >= 0.45
  and concept_judgement_points_ratio < 0.60
```

This is more paper-friendly than a full-score threshold because the route policy
is now tied to the scoring evidence structure defined by the rubric, not to a
specific question id or score value.

The active pipeline is now:

```text
exam image preprocessing -> VLM fact extraction -> three LLM scores
-> model_avg_score -> selected_baseline_score
-> A3WA risk/confidence route -> POS/BND/NEG
-> evidence-gated BND score correction -> final_calibrated_score
```

The project now compares four score layers:

```text
single_first_score        first score from model_scores_history
model_avg_score           ordinary average of three model scores
selected_baseline_score   risk-aware 3WD baseline before route/BND action
final_calibrated_score    final 3WD score after POS/BND/NEG handling
```

Implementation files changed in this update:

```text
calibration_utils.py
  adds select_baseline_score() and selected-baseline-aware BND policy.

step4_vlm_grader.py
  formal scoring pipeline now stores selected_baseline_score, baseline_policy,
  baseline_score_source, and baseline_selection_signals.

scripts/replay_calibration.py
  offline replay now mirrors the formal selected-baseline pipeline.

evaluate.py
  --compare now reports single / avg / selected / 3WD.
  --score-key selected_baseline_score is supported.
  Aliases are supported: single / avg / selected / 3wd.
  --compare-score-keys can select a subset, for example:
    python evaluate.py --compare --questions Q6 Q7 --compare-score-keys avg selected 3wd
  --compare-output exports selected-related columns.

prompts/stage2_logic_grading.md
  adds Strict Equivalence Guards for formulas, mappings, dimensions, and target
  quantities. Lenient grading no longer means wrong relations can be MATCH.
```

Validation already run:

```bash
python -m py_compile calibration_utils.py scripts\replay_calibration.py step4_vlm_grader.py evaluate.py
python scripts\replay_calibration.py --results-dir results_rrd_vlm --files results_rrd_vlm\Q6_grading_checkpoint.json results_rrd_vlm\Q7_grading_checkpoint.json
python evaluate.py --compare --questions Q6 Q7 --compare-output outputs\q6_q7_selected_compare_check.csv
```

Replay result on current Q6/Q7 checkpoints:

```text
Q6:    MAE 3.076 -> 2.954, RMSE 3.853 -> 3.695, QWK 0.720 -> 0.746
Q7:    unchanged, MAE remains 1.013
GLOBAL MAE 2.053 -> 1.991, RMSE 3.074 -> 2.974, Pearson 0.760 -> 0.776
```

Important compatibility note:

```text
Old result JSON files do not contain selected_baseline_score. evaluate.py falls
back to model_avg_score for old files. The selected column will show real
differences only after rerunning the formal experiment with the updated code.
```

## 2026-06-06 Q6/Q7 Formal Run

This top section is intentionally ASCII-only because older README content has encoding corruption. For the current project state, read this section first, then read CURRENT_PROGRESS.md.

Current pipeline:

```text
exam image preprocessing -> VLM fact extraction -> three LLM scores -> model average
-> A3WA risk/confidence route -> POS/BND/NEG -> evidence-gated BND score correction
```

## 2026-07-09 Validation-Calibrated 3WD Update

Current 3WD runtime now supports a validation-learned additive score
calibration layer:

```text
selected baseline
-> A3WA route POS/BND/NEG
-> BND evidence-gated action policy
-> validation route/score-band correction
-> final_calibrated_score
```

The calibration config is produced by `scripts/calibrate_a3wa.py` from
validation checkpoints. It still selects `loss_params` and `risk_weights`, and
now also writes `score_calibration`, an interpretable residual table grouped by:

```text
question_id + route + score_band
question_id + route
question_id
route
global
```

Runtime application is guarded in `calibration_utils.apply_route_score_calibration`:

- positive corrections are blocked when core over-credit or core contradiction
  evidence is explicit;
- negative corrections require core contradiction, confirmed core over-score, or
  allowed agent over-evidence;
- NEG samples are not automatically score-calibrated.

The BND lower gate is stricter than earlier versions. Auxiliary evidence,
`not_comparable`, unsupported high-score risk, and bare-answer risk cannot drive
lowering by themselves. This targets the recent failure mode where already
under-scored answers were lowered again.

Stage-1 fact mapping also has a degradation path. If GLM-5.1 structured fact
mapping fails because of throttling or unparsable output, the pipeline keeps the
raw human transcription or OCR text as conservative facts and records
`fact_mapping_degraded=true` in `extraction_evidence`, instead of failing the
whole answer immediately.

Evaluation now reports `SER(>2)` by default and exported comparison CSV files
include full `student_id`, absolute errors for single/avg/selected/final, and
`score_calibration_*` audit fields.

Latest server run:

```text
run_id = 20260605_225725
mode = FULL
questions = Q6 Q7
force_rerun = true
completed_at = 2026-06-06
```

Important evaluation note:

```text
Q6_graded_results.json has 63 normal records.
Q6_grading_checkpoint.json has all 68 completed records.
The 5 extra Q6 records are NEG/rejected records in Q6_rejected.json.
Use Q6_grading_checkpoint.json and Q7_grading_checkpoint.json for full-run analysis.
```

Latest generated analysis files:

```text
outputs/q6_q7_20260606_single_vs_avg_vs_3wd.csv
outputs/q6_q7_20260606_checkpoint_analysis.csv
```

Full-checkpoint metrics:

```text
Q6, N=68
single: MAE=3.309 RMSE=4.199 QWK=0.673 Pearson=0.827 TAR2=51.5% Bias=-2.779 Over>2=2 Under>2=31
avg:    MAE=3.126 RMSE=3.890 QWK=0.714 Pearson=0.856 TAR2=47.1% Bias=-2.562 Over>2=3 Under>2=33
3WD:    MAE=3.076 RMSE=3.853 QWK=0.720 Pearson=0.855 TAR2=48.5% Bias=-2.512 Over>2=3 Under>2=32

Q7, N=67
single: MAE=1.151 RMSE=2.138 QWK=0.647 Pearson=0.701 TAR2=85.1% Bias=+0.393 Over>2=6 Under>2=4
avg:    MAE=1.037 RMSE=2.033 QWK=0.685 Pearson=0.739 TAR2=86.6% Bias=+0.419 Over>2=7 Under>2=2
3WD:    MAE=1.013 RMSE=1.993 QWK=0.697 Pearson=0.749 TAR2=88.1% Bias=+0.375 Over>2=6 Under>2=2

GLOBAL Q6+Q7, N=135
single: MAE=2.238 RMSE=3.339 Pearson=0.714 TAR2=68.1% Bias=-1.205 Over>2=8 Under>2=35
avg:    MAE=2.090 RMSE=3.110 Pearson=0.753 TAR2=66.7% Bias=-1.082 Over>2=10 Under>2=35
3WD:    MAE=2.053 RMSE=3.074 Pearson=0.760 TAR2=68.1% Bias=-1.079 Over>2=9 Under>2=34
```

Interpretation:

```text
1. In this run, 3WD is still better than the current model_avg baseline.
   Global MAE improves 2.090 -> 2.053, RMSE improves 3.110 -> 3.074,
   Pearson improves 0.753 -> 0.760, and TAR2 improves 66.7% -> 68.1%.

2. The gain is too small. Out of 135 samples, 3WD improves 5, keeps 129 unchanged,
   and worsens 1 compared with model_avg.

3. Compared with the 2026-06-05 CSV result, the latest run is worse overall:
   previous global 3WD MAE=1.915, current global 3WD MAE=2.053.
   The base model_avg also worsened: previous avg MAE=2.005, current avg MAE=2.090.

4. Q6 remains the main bottleneck: final Bias=-2.512 and Under>2=32.
   Many teacher-high/model-low answers enter BND but are still kept or reject_raise.

5. Q7 improves over avg, but several low-teacher-score answers are still released as POS high-confidence overestimates.
```

Largest current errors:

```text
Q6:
  E12314033_Q6 teacher=18 final=8.3 diff=-9.7 route=BND gate=reject_raise
  E12314037_Q6 teacher=20 final=11.0 diff=-9.0 route=BND gate=reject_raise
  E12214023_Q6 teacher=12 final=4.0 diff=-8.0 route=BND gate=keep_minor_change
  E12214091_Q6 teacher=11 final=3.0 diff=-8.0 route=NEG

Q7:
  E12314133_Q7 teacher=0 final=8.9 diff=+8.9 route=POS
  E12214212_Q7 teacher=0 final=8.2 diff=+8.2 route=POS
  E12314129_Q7 teacher=2 final=6.7 diff=+4.7 route=BND gate=keep_minor_change
  E12314065_Q7 teacher=7 final=3.3 diff=-3.7 route=POS
```

Current conclusion:

```text
The framework is not invalid: 3WD still improves over the current model_avg baseline.
But the effect is not strong enough for the final paper result.
The next optimization should focus on Q6 conservative BND-UP and Q7 POS high-over misses.
```

---
# RefGrader 项目说明

## 最新项目状态（2026-06-06）

RefGrader 当前是一个面向主观题/计算题自动阅卷的实验项目。核心流程是：

```text
试卷图像裁剪与去红笔 -> VLM 提取学生作答事实 -> LLM 三次评分 -> 模型均分
-> A3WA 风险/可信度三支路由 -> POS/BND/NEG -> BND 有证据才有限改分
```

当前研究重点不是简单调 prompt，而是把三支决策从经验触发规则改成可解释的风险建模：

```text
多源风险信号 -> 综合风险 R(x) -> 可信度 mu(x)=1-R(x)
-> A3WA 非对称损失参数 alpha/beta -> POS/BND/NEG
```

当前关键文件：

```text
CURRENT_PROGRESS.md               最新进展与新对话接入上下文，优先阅读。
calibration_utils.py              A3WA 风险、阈值、BND action policy、no-harm gate。
step4_vlm_grader.py               正式评分主流程。
prompts/stage2_logic_grading.md   Stage2 评分提示词模板。
prompts/boundary_arbitration.md   BND 仲裁提示词模板。
evaluate.py                       评估脚本，支持 single / avg / 3WD 三种形式对比。
outputs/q6_q7_20260606_single_vs_avg_vs_3wd.csv
                                  最新 Q6/Q7 每条样本对比分析。
outputs/q6_q7_20260606_checkpoint_analysis.csv
                                  最新 Q6/Q7 checkpoint 口径完整分析。
```

最新正式实验为 Q6/Q7，服务器记录如下：

```text
run_id = 20260605_225725
mode = FULL
questions = Q6 Q7
force_rerun = true
completed_at = 2026-06-06
```

注意：本次 Q6 有 68 个完成样本，但 `Q6_graded_results.json` 只有 63 条，另外 5 条 NEG/人工复核样本在 `Q6_rejected.json`。因此完整评估应使用 `Q6_grading_checkpoint.json` 与 `Q7_grading_checkpoint.json`，不要只看 `graded_results`。

最新 Q6/Q7 完整 checkpoint 指标：

```text
Q6, N=68
single: MAE=3.309 RMSE=4.199 QWK=0.673 Pearson=0.827 TAR2=51.5% Bias=-2.779 Over>2=2 Under>2=31
avg:    MAE=3.126 RMSE=3.890 QWK=0.714 Pearson=0.856 TAR2=47.1% Bias=-2.562 Over>2=3 Under>2=33
3WD:    MAE=3.076 RMSE=3.853 QWK=0.720 Pearson=0.855 TAR2=48.5% Bias=-2.512 Over>2=3 Under>2=32

Q7, N=67
single: MAE=1.151 RMSE=2.138 QWK=0.647 Pearson=0.701 TAR2=85.1% Bias=+0.393 Over>2=6 Under>2=4
avg:    MAE=1.037 RMSE=2.033 QWK=0.685 Pearson=0.739 TAR2=86.6% Bias=+0.419 Over>2=7 Under>2=2
3WD:    MAE=1.013 RMSE=1.993 QWK=0.697 Pearson=0.749 TAR2=88.1% Bias=+0.375 Over>2=6 Under>2=2

GLOBAL Q6+Q7, N=135
single: MAE=2.238 RMSE=3.339 Pearson=0.714 TAR2=68.1% Bias=-1.205 Over>2=8 Under>2=35
avg:    MAE=2.090 RMSE=3.110 Pearson=0.753 TAR2=66.7% Bias=-1.082 Over>2=10 Under>2=35
3WD:    MAE=2.053 RMSE=3.074 Pearson=0.760 TAR2=68.1% Bias=-1.079 Over>2=9 Under>2=34
```

本次结论：

```text
1. 当前 3WD 相对本次 model_avg 仍有正向提升：
   global MAE 2.090 -> 2.053，RMSE 3.110 -> 3.074，Pearson 0.753 -> 0.760。

2. 但提升幅度偏小：135 条样本中，3WD 相对 avg 只改善 5 条、保持 129 条、变差 1 条。
   BND gate 目前偏保守，很多 BND 样本被拦截后没有实际改分。

3. 与 2026-06-05 的上一轮 CSV 结果相比，本轮整体退化：
   previous global 3WD MAE=1.915，current global 3WD MAE=2.053。
   同时 base model_avg 也退化：previous avg MAE=2.005，current avg MAE=2.090。

4. Q6 是主要问题：系统性低估仍然严重，final Bias=-2.512，Under>2=32。
   很多教师高分样本进入 BND 后仍被 reject_raise 或 keep_minor_change。

5. Q7 相对 avg 有改善，但仍存在低教师分样本被 POS 高置信放行的问题。
```

最新样本级问题：

```text
Q6 最大误差样本：
  E12314033_Q6 teacher=18 final=8.3 diff=-9.7 route=BND gate=reject_raise
  E12314037_Q6 teacher=20 final=11.0 diff=-9.0 route=BND gate=reject_raise
  E12214023_Q6 teacher=12 final=4.0 diff=-8.0 route=BND gate=keep_minor_change
  E12214091_Q6 teacher=11 final=3.0 diff=-8.0 route=NEG

Q7 最大误差样本：
  E12314133_Q7 teacher=0 final=8.9 diff=+8.9 route=POS
  E12214212_Q7 teacher=0 final=8.2 diff=+8.2 route=POS
  E12314129_Q7 teacher=2 final=6.7 diff=+4.7 route=BND gate=keep_minor_change
  E12314065_Q7 teacher=7 final=3.3 diff=-3.7 route=POS
```

当前判断：最新系统不是无效，3WD 对本次 avg baseline 仍有提升；但效果还不足以作为最终论文结果。下一步应重点处理两类问题：

```text
Q6: BND-UP 对教师宽松给分样本仍然过于保守，导致高分样本大量低估。
Q7: POS 高置信样本中仍有低教师分严重高估，需要更强的 unsupported-high-score 复查触发。
```

后续新对话请优先阅读 `CURRENT_PROGRESS.md`，它比 README 下方历史内容更接近当前项目状态。

---

以下为历史 README 内容，部分段落存在旧编码乱码，仅作为历史记录保留。
# RefGrader 项目说明

RefGrader 是一个面向计算机组成原理类主观题/计算题的自动阅卷实验项目。项目以整页试卷图片为输入，先裁剪出单题作答区域并去除教师红笔批注，再用 VLM 提取学生作答事实，最后由文本模型按结构化评分细则打分，并通过三支决策机制对低置信度样本进行复查或拒判。

## 项目结构

| 路径 | 作用 |
| --- | --- |
| `raw_exams/` | 原始整页试卷扫描图，共 68 份。 |
| `cropped_with_scores/` | 带教师红笔分数的题目切片，用于抽取教师真实分。当前 Q1/Q2/Q3/Q5 为 68 张，Q4 为 41 张，Q6 为 67 张，Q7 为 68 张。 |
| `cleaned_patches/` | 去红笔后的题目切片，用于模型盲评。当前 Q1/Q2/Q3/Q5/Q6 为 68 张，Q4 为 41 张，Q7 为 67 张。 |
| `database/exam_database.json` | 题库配置，包含题号、满分、题干、参考答案、官方评分标准、题图/答案图路径和学生图片目录。 |
| `database/teacher_scores.json` | 教师真实评分表，按学生 ID 和题号保存分数，是评估指标的 ground truth。 |
| `database/images/` | 题目或参考答案附图。 |
| `results_rrd_vlm/` | 评分细则、方差检查点、批改检查点、最终评分结果、拒判结果和历史实验结果。 |
| `auto_process.py` | 批量裁剪整卷并去红笔脱敏。 |
| `clean_dataset_initial.py` | 单张试卷的高分辨率脱敏与交互式 ROI 裁剪脚本。 |
| `step0_extract_ground_truth.py` | 从带红笔切片中抽取教师分数，生成 `teacher_scores.json`。 |
| `step3_rrd_generator.py` | 生成 RRD 结构化评分细则，并基于高方差样本细化评分标准。 |
| `step4_vlm_grader.py` | 核心 3WD 阅卷流水线：盲提取、逻辑评分、自一致性采样、三支决策、边界域复查。 |
| `calibration_utils.py` | 通用后校准与 A3WA 三支决策工具：风险可信度、非对称阈值、动态 BND 限幅、离线 replay 共用逻辑。 |
| `main_pipeline.py` | 主入口，控制方差优化模式和正式批改模式。 |
| `evaluate.py` | 评估脚本，计算 MAE、RMSE、QWK、Pearson r、±2 命中率等指标。 |
| `scripts/replay_calibration.py` | 离线 replay 脚本，不调用 API，读取已有 checkpoint 模拟新三支决策对路由和最终分的影响。 |
| `run_experiment.sh` | 服务器后台实验管理脚本，使用 `nohup` 启动 `main_pipeline.py`，支持 `run/status/tail/stop/restart`。 |
| `test_air.py` | 简单接口测试脚本。 |

## 最新进展：A3WA 三支决策改造（2026-06-01）

### 改造背景

此前 3WD 路由主要依赖工程经验阈值，例如标准差、低分、空白率、严重错误比例等分别触发 POS/BND/NEG。该做法可以运行，但阈值来源难以理论解释，且多个风险信号分散触发，不利于后续论文论证。

当前改造参考论文 `2025_An_Asymmetric_Approach_to_Three-Way_Approximation_of_Fuzzy_Sets.pdf` 的 A3WA 思想，将三支决策重构为：

```text
多源风险信号 -> 综合风险 R(x) -> 自动评分可信度 μ(x) -> 非对称阈值 α/β -> POS/BND/NEG
```

对应关系：

| A3WA 概念 | RefGrader 中的含义 |
| --- | --- |
| membership value / `f(x)` | 自动评分可信度 `μ(x)` |
| `α` | POS 自动采信阈值 |
| `β` | NEG 人工复核阈值 |
| `m` | BND 边界域中间状态 |
| `1` | POS，直接采信 |
| `m` | BND，Agent 仲裁 |
| `0` | NEG，人工复核/拒判 |

### 当前正式实现

当前 `step4_vlm_grader.py` 的三支决策流程为：

```text
Stage 1: VLM 提取学生 facts
Stage 2: LLM 三次独立评分，得到 model_scores_history / model_avg_score / std_dev / strict_cots_all
Post calibration: 计算通用校准信号与 A3WA 可信度
3WD route:
  - 硬 NEG 兜底：提取失败、分数分歧过大、严重语义风险等直接进入 NEG
  - 否则计算 R(x)、μ(x)、α、β
  - μ >= α 进入 POS
  - β < μ < α 进入 BND
  - μ <= β 进入 NEG
```

综合风险当前定义为：

```text
U_extract = 0.5 * low_quality_rate + 0.5 * perception_failure_rate
U_score   = 0.5 * (std_dev / MAX_SCORE) + 0.5 * (score_spread / MAX_SCORE)
U_semantic = fatal_points_ratio，并吸收 unsupported MATCH / core anchor 风险
U_blank = blank_rate

R(x) = 0.35 * U_extract + 0.30 * U_score + 0.20 * U_semantic + 0.15 * U_blank
μ(x) = 1 - R(x)
```

A3WA 非对称阈值当前参数为：

```text
λ1 = 5      # 错误自动采信风险
λ2 = 1      # 不必要进入 BND 成本
μ1 = 3      # 不必要人工复核成本
μ2 = 7      # 高风险样本未送人工风险
m  = 0.5

α = (λ1 + λ2 * m) / (λ1 + λ2) = 0.917
β = (μ2 * m) / (μ1 + μ2) = 0.35
```

注意：`calibration_utils.py` 中已经实现 `optimize_a3wa_m()`，可以在离线/批次实验中按当前题目或批次的 `μ` 分布搜索最小信息损失的 `m*`；但正式在线批改是逐份试卷处理，当前正式 pipeline 默认使用 `m=0.5`，便于今晚实验稳定运行。

### BND Agent 动态限幅

原 BND 仲裁使用较固定的修正范围，容易解释为工程调参。当前改为可信度相关的动态限幅：

```text
review_strength = (α - μ) / (α - β)
delta = max(0.10 * MAX_SCORE, 0.30 * review_strength * MAX_SCORE)
upper_bound = avg_model_score + delta
lower_bound = 0
```

设计意图：

- `μ` 越接近 `α`，越接近 POS，Agent 修正空间较小。
- `μ` 越接近 `β`，越接近 NEG，Agent 可以有更大修正空间。
- 下界保持宽松，不强制抬高 Agent 的谨慎下调结果。
- 若存在明确高估风险（如高 fatal 比例、high_blank_high_score、unsupported MATCH、core anchor failed），则 `upper_bound <= avg_model_score`，避免 BND 过度加分。

### 新增结果字段

正式批改结果中会新增：

```json
"post_calibration": {
  "unsupported_match_points_ratio": ...,
  "method_final_verified_ratio": ...,
  "metadata_coverage": ...,
  "core_anchor_failed": false,
  "visual_blank_review": false,
  "rule_hits": []
},
"a3wa_decision": {
  "route": "POS/BND/NEG",
  "risk": ...,
  "mu": ...,
  "alpha": ...,
  "beta": ...,
  "m": ...,
  "reason": "...",
  "risk_components": {
    "U_extract": ...,
    "U_score": ...,
    "U_semantic": ...,
    "U_blank": ...
  }
}
```

这些字段是后续论文分析的核心审计信息，可用于统计：

- 不同路由下的平均 `R(x)` 和 `μ(x)`；
- 是否满足 `μ_POS > μ_BND > μ_NEG`；
- BND 样本中 Agent 修正幅度与 `μ(x)` 的关系；
- 硬 NEG 与可信度型 NEG 的来源差异。

### 离线 replay 验证结果

命令：

```bash
python scripts/replay_calibration.py --results-dir results_rrd_vlm --files results_rrd_vlm/Q5_grading_checkpoint.json results_rrd_vlm/Q6_grading_checkpoint.json results_rrd_vlm/Q7_grading_checkpoint.json
```

作用：不重新调用 VLM/LLM，不修改 checkpoint，只读取已有 `Q5/Q6/Q7_grading_checkpoint.json`，模拟新 A3WA 路由和 BND 动态限幅是否会破坏已有结果。

2026-06-01 在本机和实验室服务器上均已验证通过：

```text
GLOBAL
current N=203 MAE=2.492 RMSE=3.622 QWK=0.697 Pearson=0.744 TAR2=60.1% Bias=-1.605 Over>2=16 Under>2=65
replay  N=203 MAE=2.491 RMSE=3.621 QWK=0.697 Pearson=0.745 TAR2=60.1% Bias=-1.611 Over>2=16 Under>2=65
```

逐题 replay 结论：

| 题号 | replay 结论 |
| --- | --- |
| Q5 | 指标不变，说明 A3WA 路由不会额外破坏 Q5；Q5 主要问题仍是 OCR/提取失败导致低估。 |
| Q6 | 指标不变，说明当前动态限幅未在旧 checkpoint 上误伤；Q6 高估问题仍需结合更可靠的结构化元数据或正式新跑结果观察。 |
| Q7 | 小幅改善，MAE 1.172 -> 1.169，Pearson 0.724 -> 0.726。 |
| 全局 | 基本持平，说明新 3WD 逻辑可进入正式实验。 |

服务器上已执行并通过：

```bash
python -m py_compile calibration_utils.py scripts/replay_calibration.py step4_vlm_grader.py
python scripts/replay_calibration.py --results-dir results_rrd_vlm --files results_rrd_vlm/Q5_grading_checkpoint.json results_rrd_vlm/Q6_grading_checkpoint.json results_rrd_vlm/Q7_grading_checkpoint.json
```

### 今晚/后续服务器实验方式

推荐使用项目脚本后台运行：

```bash
conda activate ref-grader
cd /home/E125221219/projects/refgrader
./run_experiment.sh run
```

管理命令：

```bash
./run_experiment.sh status
./run_experiment.sh tail
./run_experiment.sh stop
python monitor.py --watch
```

`run_experiment.sh run` 内部使用 `nohup python3 main_pipeline.py ... &`，因此 SSH 断开或本地电脑关闭通常不影响服务器继续运行。实验日志写入 `logs/experiment_*.log`，PID 写入 `logs/refgrader.pid`。

正式跑完后评估：

```bash
python evaluate.py
```

## 核心工作流程

1. 原始试卷预处理

   运行 `auto_process.py` 后，系统按固定坐标从 `raw_exams/` 裁剪各题区域，保存到 `cropped_with_scores/`；再用 HSV 红色掩码擦除教师红笔痕迹，保存到 `cleaned_patches/`。

2. 教师真实分提取

   `step0_extract_ground_truth.py` 读取 `cropped_with_scores/` 中带红笔分数的切片，调用 VLM 识别教师分数，写入 `database/teacher_scores.json`。

3. 评分标准生成与优化

   `step3_rrd_generator.py` 将官方评分标准转成 JSON 细则。`main_pipeline.py` 的 `VARIANCE_OPT` 模式会抽取少量样本，使用同一份盲提取事实多次打分，依据分数方差或允许细化的复合高分条目触发语义检查。原子结果项不会再因为分值较高而被强制拆分。

   CSBench 使用三层准则目录：`rubrics/source` 保存原始准则副本，
   `rubrics/initial` 保存标准化初始准则，`rubrics/optimized` 保存
   `VARIANCE_OPT` 的输出。优化阶段只使用每题的 calibration 答案，
   不覆盖原始数据，也不使用 test 答案或教师真实分数。

4. 正式批改

   `main_pipeline.py` 的 `FULL` 模式读取题库配置与已生成评分细则，对 `cleaned_patches/Qx/` 下的学生作答逐张处理，并写入 `results_rrd_vlm/Qx_graded_results.json`。

5. 3WD 阅卷流水线

   `step4_vlm_grader.py` 的主要流程如下：

   - Stage 1: 根据评分细则生成脱敏提取清单，VLM 只做客观事实/OCR 提取。
   - Stage 1.5: 当空白率较高时，对未提取条目做二次聚焦提取。
   - Stage 2: 文本模型对同一份事实独立评分 3 次，得到 `model_scores_history`、`model_avg_score` 和 `std_dev`。
   - 三支决策:
     - POS: 模型自一致且无异常，直接采用均分。
     - BND: 分数波动、低分异常或边界样本，调用宽容复查 Agent 得到 `final_calibrated_score`。
     - NEG: 提取质量失败或分歧极端，写入 `Qx_rejected.json` 交人工复核。

6. 结果评估

   `evaluate.py` 默认评估 Q4-Q7，并支持比较 `model_avg_score` 与 `final_calibrated_score`。评估指标包括 MAE、RMSE、QWK、Pearson r、±2 命中率和系统偏差。

## 历史测试数据

| 阶段 | 文件/数据 | 说明 |
| --- | --- | --- |
| 数据集 | `raw_exams/` | 68 份原始试卷。 |
| 教师分 | `database/teacher_scores.json` | 68 名学生，覆盖 Q1-Q7；无效分以 -1 表示。 |
| 早期 Q2 实验 | `Q2_初始单轮_results.json`、`Q2_results多轮迭代.json`、`Q2_迎合参考答案_results.json`、`Q2_防止迎合失败版_results.json` | 用于验证单轮、多轮、参考答案迎合与防迎合策略。 |
| Q2 正式小样本 | `Q2_graded_results.json` | 10 份样本，MAE 4.900，RMSE 6.745，QWK 0.2888，Pearson r 0.7649，±2 命中率 50.0%。 |
| Q4 策略对比 | `Q4_graded_results_最高分上限.json`、`Q4_graded_results去除二次评分最高分上限.json`、`Q4_graded_results_depseek-v4-flash.json`、`Q4_graded_results.json` | 对比最高分上限、去上限、DeepSeek 与当前 GLM 方案。 |
| Q5/Q6/Q7 旧版结果 | `Q5_graded_results_5.20.json`、`Q6_graded_results_5.20.json`、`Q7_graded_results_5.20.json` | 2026-05-20 左右的阶段性结果。 |
| 最近正式结果 | `Q4_graded_results.json`、`Q5_graded_results.json`、`Q6_graded_results.json`、`Q7_graded_results.json` | 当前 README 中的主要评估对象；Q4 最新时间为 2026-05-20，Q5/Q6/Q7 最新时间为 2026-05-22。 |

## 最近测试结果

以下结果基于 `results_rrd_vlm/Q4-Q7_graded_results.json` 中的正常批改样本，不含 `*_rejected.json` 中的拒判样本。±2 命中率表示最终评分与教师评分差在正负 2 分以内，包含边界值。

### 3WD 后最终分

| 题号 | N | MAE | RMSE | QWK | Pearson r | ±2 命中率 | 平均偏差 | 高估>2 | 低估>2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4 | 41 | 1.854 | 2.164 | 0.9170 | 0.9195 | 73.2% | -0.24 | 4 | 7 |
| Q5 | 67 | 3.388 | 4.678 | 0.6510 | 0.8258 | 52.2% | -2.93 | 3 | 29 |
| Q6 | 68 | 2.162 | 2.970 | 0.8619 | 0.8626 | 63.2% | +0.38 | 15 | 10 |
| Q7 | 66 | 0.886 | 1.947 | 0.7189 | 0.7752 | 90.9% | +0.55 | 6 | 0 |
| 全局 | 242 | 2.101 | 3.220 | - | 0.8070 | 69.4% | -0.59 | - | - |

### 3WD 前后对比

| 题号 | 模型均分 MAE | 3WD 后 MAE | 模型均分 ±2 | 3WD 后 ±2 | 主要变化 |
| --- | ---: | ---: | ---: | ---: | --- |
| Q4 | 2.000 | 1.854 | 70.7% | 73.2% | 小幅改善，QWK 与 Pearson 同步提升。 |
| Q5 | 3.552 | 3.388 | 50.7% | 52.2% | 小幅改善，但系统性低估仍严重。 |
| Q6 | 2.250 | 2.162 | 57.4% | 63.2% | ±2 命中率改善较明显，但高估样本增加。 |
| Q7 | 1.076 | 0.886 | 86.4% | 90.9% | MAE 和 ±2 改善，QWK 略降。 |
| 全局 | 2.248 | 2.101 | 65.7% | 69.4% | 总体只小幅提升，MAE 降低 0.147，±2 提升 3.7 个百分点。 |

### 路由与质量概况

| 题号 | POS | BND | 正常样本提取质量 | 平均 std | 平均空白率 | 拒判文件 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| Q4 | 15 | 26 | 41 个 high | 0.537 | 25.05% | `Q4_rejected.json` 1 条 |
| Q5 | 35 | 32 | 67 个 high | 0.304 | 26.16% | `Q5_rejected.json` 1 条 |
| Q6 | 12 | 56 | 68 个 high | 0.361 | 23.44% | `Q6_rejected.json` 1 条 |
| Q7 | 37 | 29 | 66 个 high | 0.186 | 12.88% | `Q7_rejected.json` 1 条 |

## 当前主要问题

1. 3WD 的边界域复查主要是“宽容加分”，当前实现会将最终分限制为不低于 `model_avg_score`，因此可以修正低估，却基本不能纠正高估。
2. BND 触发条件覆盖面大，但触发后复查 Agent 只读取一份 `strict_cot`，没有系统利用 3 次独立评分的分歧信息。
3. 提取质量判定过宽，最新 Q4-Q7 的正常样本几乎全部被判为 `high`，但高误差样本中仍存在明显 OCR/事实提取问题。
4. `model_avg_score` 使用向上取整，会引入额外高估倾向；Q6/Q7 的高估样本在 3WD 后更突出。
5. 评分细则仍有粗粒度与教师口径不一致问题，Q5 低估尤其明显，说明现有细则不能覆盖教师实际给过程分的方式。
6. 方差优化只看模型自一致性，不直接看教师误差；低方差错误样本不会被识别为难例。
7. 评估与配置存在不一致风险，例如 `evaluate.py` 中 Q1 满分写为 10，而题库中 Q1 满分为 5。
8. 代码中存在明文 API Key，且部分中文注释/字符串显示为乱码，影响维护和安全。

## 为什么三支决策提升不明显

三支决策目前更像一个“低分复查器”，不是完整的误差校准器。它能把部分被模型低估的样本往上拉，但不能把模型高估的样本往下压；同时，NEG 几乎只在极端分歧时触发，无法拦截低方差但事实提取错误的样本。最终表现就是 MAE 和 ±2 命中率有改善，但改善幅度有限，并且某些题目会出现偏差方向从低估转向高估。

## 优化方案

1. 将 BND 复查改为双向校准：允许 `final_calibrated_score` 低于 `model_avg_score`，同时给复查 Agent 明确任务，即“确认加分、确认扣分、维持原分”三选一，而不是默认宽容加分。

2. 用教师误差构造难例池：从历史结果中抽取 `abs(final - teacher) > 2` 的样本，按题号、误差方向、路由、空白率和评分条目聚类，优先用这些样本反向优化评分细则。

3. 将方差优化从“模型内部分歧”升级为“误差驱动优化”：细则修订触发条件增加 MAE、±2 未命中、系统偏差和条目级错判率，避免低方差但稳定错判的样本漏检。

4. 建立条目级校准表：按题目和评分条目统计 `score_given` 与教师分差的关系，识别长期低估/高估的条目，并为每个条目设置保守补偿、扣分上限或人工复核阈值。

5. 重写 Q5 评分细则：将“扩展编码、一地址/零地址数量、让出编码位”拆成更细的过程分节点，显式允许教师常给的中间过程分，优先解决 Q5 的系统性低估。

6. 强化事实提取质检：新增事实提取二审，不只检查空白率和废话值，还检查数值量级、单位、题干参数误抄、关键结论缺失和提取项之间的链式一致性。

7. 调整 3WD 阈值：BND 不应只由 `avg <= 0.8 * MAX_SCORE` 触发，应结合题目历史误差、分数段、空白率、条目致命错误比例和模型分歧；NEG 应增加“低方差但高风险事实”的拒判条件。

8. 取消向上取整均分：将 `ceil(mean)` 改为保留小数或四舍五入，并在最终输出前按题目满分和教师评分粒度做离散化，减少系统性高估。

9. 聚合 3 次评分的完整理由：复查 Agent 输入不应只用第一份 `strict_cot`，应提供三份评分的条目差异、共同扣分项和争议项，让仲裁针对真实分歧工作。

10. 建立验证集和回归评测脚本：固定 Q4-Q7 当前结果作为 baseline，每次修改细则或阈值后自动输出 MAE、RMSE、QWK、Pearson r、±2 命中率、偏差方向和显著退化样本。

11. 分题设置策略：Q4/Q6 可重点做边界域仲裁和高估校正；Q5 优先做细则重构和过程分恢复；Q7 优先处理低分学生被高估的问题，增加零分/低分保护规则。

12. 安全与工程化：将 API Key 移入环境变量，修复乱码注释和字符串，统一题目满分配置来源，避免评估脚本和题库配置不一致。

## 历史三支决策实现说明（已被 A3WA 版本替代）

以下内容描述的是 A3WA 改造前的风险驱动 3WD 版本，保留用于理解演进过程。2026-06-01 之后的当前实现以本文前部“最新进展：A3WA 三支决策改造”为准。

旧版代码曾将三支决策从单一阈值触发改为风险驱动：

- POS：低风险样本直接接受模型均分。
- BND：中风险样本进入边界仲裁，遵循教师宽松口径，优先恢复过程分；出现明确高估风险时谨慎下调。
- NEG：高风险样本拒判，交人工复核。

旧版核心风险字段包括 `perception_risk`、`uncertainty_index`、`fatal_points_ratio`、`high_blank_high_score`、`lenient_review_signal` 和 `risk_features`。当前 A3WA 版本仍保留这些字段作为风险分量和审计信息，但最终路由由 `a3wa_decision.route` 主导。
