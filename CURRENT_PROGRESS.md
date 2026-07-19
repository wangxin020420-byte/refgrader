# Current Progress

Last updated: 2026-07-19

## 2026-07-19 Rubric Candidate Safety Gate And Residual Transfer Guard

Rubric semantic contract version 5 closes the failure exposed by the latest
CO_4 run. Split children may no longer invent binary or hexadecimal answer
literals or reverse the parent's final judgement. After deterministic
validation, each refined candidate is replayed on the isolated rubric-
calibration facts and compared with the original rubric against teacher scores.
Insufficient replay coverage, MAE beyond the score-relative non-inferiority
margin, or a severe paired regression rejects the candidate without replacing
the active rubric. `optimize` now explicitly uses the embedded CSBench teacher
database rather than the legacy default database.

Validation residual calibration remains a separate optional layer. It now
retains question-level diagnostics even when a cell is too small to deploy a
question-specific correction. A route/global fallback is blocked when at least
three local validation examples show an opposite material residual direction.
This prevents cross-question transfer from masking or amplifying a question-
specific rubric bias.

## 2026-07-17 Tracked Active Experiment Configuration

The current formal configuration is now versioned in `refgrader-main` instead
of being discoverable only through local ignored directories or historical
artifacts. `data/csbench/rubrics/optimized`, `rubrics/manifests`,
`rubrics/active_rubric_set.json`, and
`calibration/active_a3wa_config.json` are Git-visible. Optimization manifests
use portable root placeholders rather than device-specific absolute paths.

`optimize` atomically refreshes the active rubric set; `calibrate` copies the
derived configuration to the tracked active A3WA path and binds it to dataset,
split, and optimized-rubric hashes. Formal grading validates those hashes and
test grading automatically selects a valid active A3WA config. Any changed
rubric makes the previous A3WA metadata stale until validation/calibration is
repeated. `restore_csbench_artifacts.py` also refreshes this active state, while
`refgrader-artifacts` remains the immutable, run-ID-based historical archive.

Tests cover portable manifests/configs, active hash validation, stale A3WA
invalidation, and artifact restoration. Optimize/calibrate remain single-writer
operations across devices; Git commit/push stays manual.

## 2026-07-17 Automatic Rubric Optimization Boundary

The CO_1 to CO_7 audit found that several compound high-value items can cause
all-or-nothing grading, but manually decomposing and locking those items in the
source rubric would hide the contribution of the rubric optimization module.
The manually authored CO_2 to CO_6 decompositions were therefore removed.

The source and immutable initial rubrics again contain the official coarse
criteria. Fine-grained criteria must be proposed during rubric optimization
from the question, official answer, and isolated rubric-calibration samples.
Generated candidates must conserve each parent score, use only independently
verifiable official evidence, avoid hidden requirements, and remain coarse when
the task is genuinely result-only. CO_1 to CO_7 now serve as behavioral audit
cases rather than hard-coded decomposition templates.

The current acceptance check covers score conservation, equal weighting,
semantic traceability, immutable answer facts, and paired teacher-score
non-inferiority on rubric-calibration data. Validation and test labels remain
outside rubric candidate selection.

The general instructor-aligned semantics remain: equivalent representations
and pure formatting differences do not lose points, concise valid methods and
propagated-error process evidence retain their declared credit, exact numeric
answers do not receive a generic 10% tolerance, and a bare final answer cannot
be used to infer unwritten process unless the parent explicitly declares a
final-answer-sufficient policy.

## 2026-07-16 Versioned Grading Runs And Partial Artifact Resume

The grading lifecycle now uses one stable run identity from local execution to
cross-device artifact restore:

1. New grading runs are written under
   `results_runs/csbench_<batch>_<split>/runs/<run_id>/`; `active_run.json`
   selects the default resume target.
2. `grade --force` creates a new timestamped run instead of deleting the prior
   checkpoint. The same command without `--force` resumes the active run.
3. `run_state.json` binds a run to its question set, split, rubric hashes,
   split hashes, and A3WA config hash. A mismatched resume is rejected.
4. Result validation distinguishes structural corruption from incomplete
   coverage. Structurally valid partial test runs are evaluated and archived
   with `completion_report.json`, coverage, missing IDs, and failed IDs.
5. Complete validation remains mandatory for A3WA calibration. Partial
   validation may be archived for recovery but cannot fit thresholds or
   residual correction.
6. Re-publishing the same run atomically updates the same artifact directory;
   the CSBench index is upserted rather than duplicated.
7. Restoring an artifact run recreates the versioned local directory and marks
   it active, enabling cross-device continuation with the same run ID.
8. Failed records are deduplicated by student ID and removed automatically
   when a later retry succeeds. JSON checkpoint writes are atomic.

Regression coverage was added for fresh/run-resume separation, partial result
inspection, same-ID artifact updates, and versioned artifact restoration.

## 2026-07-14 Evidence-Calibrated A3WA And Sequential Review

The earlier runtime mixed A3WA thresholds with route-rewriting heuristics and
enabled validation residual correction by default. That made a positive final
MAE result difficult to attribute to 3WD. The current implementation uses:

1. `U_E`, `U_S`, and `U_R` as the only membership inputs.
2. A validation-fitted monotonic logistic membership in the safe-auto-grading
   fuzzy set; increasing any risk cannot increase membership.
3. A3WA alpha/beta derived only from asymmetric losses. Review signals are
   diagnostics and no longer rewrite the mathematical route.
4. A split-conformal score interval as an uncertainty audit, not a score bonus.
5. A sequential BND action that accepts changes only from structured,
   item-capped, direction-consistent evidence; otherwise it defers to a human.
6. Residual correction disabled by default. `three_way_core_score` isolates the
   3WD contribution from optional `final_calibrated_score` residual correction.
7. Leave-one-question-out diagnostics, paired bootstrap normalized-MAE delta,
   route-budget checks, and a deployment gate.
8. Evaluation coverage, review rate, selective MAE, unsafe acceptance, and AURC.

Offline smoke replay on legacy checkpoints correctly produced zero BND gain
because those records do not contain the new structured evidence. The generated
config was marked experimental rather than inventing a positive gain. A fresh
validation/test run is required to measure the new mechanism.

## 2026-07-13 Force-Rerun OCR Isolation And Failure Propagation

Rubric/grading `--force` now reuses raw PaddleOCR JSON when its recorded image
SHA-256 still matches. It continues to invalidate rubric, mapped-fact, and
grading outputs, but no longer turns every experiment rerun into an unnecessary
OCR model rerun. The dedicated OCR-only path retains explicit force behavior.

PaddleOCR subprocess failures now include captured stdout/stderr, and the main
pipeline re-raises fatal exceptions after recording error progress. The command
wrapper therefore receives a nonzero exit code and cannot publish stale rubric
artifacts or continue into validation after a failed optimization stage.

## 2026-07-18 Rubric Semantic Contract V4

Version 4 replaces the old atomic-versus-equal-split rule with five parent
semantics: strict atomic, result sufficient, orthogonal additive, component
additive, and process dominant. High-value complex derivations reserve at least
80% for written process (at least 50% for the core inference) and at most 20%
for the final conclusion; short derivations use a 65%/35% boundary. These are
structural constraints, not question-specific point assignments. Labeled
multi-field records now use `structured_fields`, so a matching offset or group
number cannot deterministically validate an otherwise incorrect address layout.
The semantic contract version bump invalidates old optimized manifests and
requires a fresh all-question optimization before validation and test grading.

## 2026-07-13 CO_1 Hierarchical Rubric And Semantic Gate

The CO_1 regression was traced to a semantic mismatch: a single atomic 5-point
final-answer item removed the partial process credit present in teacher labels.
The implemented contract now supports three explicit policies:

```text
strict_atomic
additive_split
final_sufficient_partial_credit
```

CO_1 uses one 5-point hierarchical parent with `2.0` points for address-field
evidence, `1.5` for effective-address evidence, and `1.5` for the final operand.
A canonical `37H` final answer grants the complete parent score; otherwise only
supported process evidence contributes, capped at `3.5`. The base-number
canonicalizer handles explicit hexadecimal/binary equivalence and only accepts
bare numbers under an explicit rubric `implicit_bases` policy.

The 3WD risk view follows the same semantics. Once a strict majority of grading
probes activates final-answer full credit, optional process children are removed
from the extraction-risk denominator and the final trigger carries the complete
parent weight. Missing optional work therefore cannot create a false BND/NEG
route, while non-triggered answers retain the complete process-risk structure.

When every positive-point item belongs to such a hierarchical parent and at
least two successful probes have a canonicalizer-backed strict majority for
full credit, the score is a deterministic rubric constraint. Boundary agents
and validation residuals cannot lower it, while the 3WD route and review fields
remain available for risk auditing.

Rubric semantic contract version 3 adds a minimum-decomposition gate for
high-value criteria. Composite `additive_split` parents worth at least 4 points
must contain at least two equal-weight, independently verifiable scoring
children. High-value `strict_atomic` single-outcome criteria remain intact but
carry an explicit audited exemption. Failed refinements are retried with the
validator feedback and never overwrite the currently active optimized rubric.
The same structural validator runs again before formal grading, so a successful
manifest cannot hide an unsplit or otherwise invalid rubric. Targeted unit tests
and the embedded CO_1--CO_7 structural audit pass. A new API experiment is still
required to measure final MAE/QWK changes; offline validation is supporting
evidence, not a replacement for the held-out test.

Variance-optimization probes now apply the same canonical and hierarchical
scoring rules as formal grading before variance and hard-sample selection are
computed, so rubric refinement and test grading no longer use different score
semantics.

## 2026-07-10 Safe Automatic Test Finalization

`scripts/run_csbench.py grade --split test` now finalizes a complete formal run
without a separate command: it validates the checkpoint against the immutable
test split, evaluates `single / avg / selected / 3WD`, exports the comparison
CSV, and copies one complete run into the sibling `refgrader-artifacts` Git
repository. It does not commit or push unless `--push-artifacts` is explicitly
provided.

Publication is blocked for validation/calibration splits, limited debug runs,
missing or duplicate test IDs, split contamination, checkpoint/result
mismatches, or unresolved failed samples. When `--a3wa-config` is supplied, the
exact configuration is copied to `calibration/a3wa_config.json` and its SHA-256
is recorded in `run_manifest.json`.

For resumed mixed batches, questions whose complete test checkpoints existed
before the command are marked `preexisting_completed_checkpoint`; the newly
supplied A3WA config is archived only for questions actually graded by that
command. This prevents an old CO_1 checkpoint from being mislabeled when a new
config is used only for CO_2.

The default portable artifact run contains the initial and optimized rubrics,
optimization manifest/checkpoint, grading checkpoint and routed result files,
evaluation CSV, progress, runtime log, A3WA configuration, and run manifest.
Raw OCR and per-answer fact caches remain opt-in via `--include-raw-ocr` and
`--include-facts`.

## 2026-07-09 Validation-Calibrated 3WD And Safer Visual/Fallback Path

Current implementation status:

```text
1. scripts/calibrate_a3wa.py still searches A3WA loss parameters and risk
   weights, and now also writes score_calibration.

2. score_calibration is a validation residual table grouped by:
   question+route+score_band -> question+route -> question -> route -> global.
   It is intentionally interpretable and additive rather than a black-box model.

3. step4_vlm_grader.py loads the same A3WA config through
   A3WA_CALIBRATION_CONFIG / --a3wa-config. After POS/BND/NEG routing and BND
   action policy, non-NEG samples pass through apply_route_score_calibration().

4. calibration_utils.py now has stricter BND lower permission:
   lower needs confirmed core over-score, core contradiction, allowed agent
   over-evidence, or direct-only high-score with weak core support. Auxiliary
   evidence and not_comparable cannot drive lower by themselves.

5. Stage-1 fact mapping is degraded instead of hard failing when GLM-5.1 fact
   mapping is unavailable. The pipeline preserves raw transcription/OCR text as
   conservative facts and records fact_mapping_degraded in extraction_evidence.

6. evaluate.py now prints SER(>2) by default and exports score_calibration
   audit fields in comparison CSVs.
```

Recommended full recalibration workflow:

```bash
python scripts/run_csbench.py grade CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --split validation --force --no-artifacts

python scripts/calibrate_a3wa.py \
  --files results_runs/csbench_co1_co2_co3_co4_co5_co6_co7_full/CO_*_grading_checkpoint.json \
  --teacher-db data/csbench/teacher_scores.json \
  --database-path data/csbench/exam_database.json \
  --output results_runs/csbench_a3wa_route_score_calibrated.json

python scripts/run_csbench.py grade CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 \
  --split test --force --a3wa-config results_runs/csbench_a3wa_route_score_calibrated.json --no-artifacts

python scripts/run_csbench.py evaluate CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --export --push-artifacts
```

## 2026-06-25 CSBench Unified Run Command And CO_1-CO_7 Batch Plan

Current implementation status:

```text
1. CSBench_new remains an external dataset repository.
   It is not copied into RefGrader.

2. RefGrader uses scripts/prepare_csbench.py to generate a compatible
   data/csbench view from CSBench_new.

3. scripts/run_csbench.py now has a unified `run` subcommand.
   It can execute:
     prepare compatible view -> optimize rubrics -> formal grading
   from one command.

4. The separate commands still exist:
     optimize = only rubric optimization
     grade    = only formal grading
     evaluate = metric evaluation/export

5. For `run --background`, the whole chained workflow is started in the
   background. Inside that background process, stages still run sequentially:
     prepare -> optimize -> grade
   This allows the local window to be closed while still preventing grading
   from starting before optimized rubrics exist.
```

Recommended CO_1 to CO_7 full rerun command on the lab server:

```bash
cd /home/E125221219/projects/refgrader
conda activate ref-grader
python scripts/run_csbench.py run CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --dataset-root /home/E125221219/CSBench_new --force --background
```

Meaning:

```text
1. Regenerate data/csbench from /home/E125221219/CSBench_new.
2. Re-optimize CO_1 to CO_7 rubrics.
3. Run formal CO_1 to CO_7 grading after optimization succeeds.
4. The whole chain runs in the background, so the terminal can be closed after
   the command returns.
5. --force overwrites old optimized rubrics and old grading checkpoints.
```

Use this shorter command only when `data/csbench` is already current:

```bash
python scripts/run_csbench.py run CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --force --background
```

Separate-stage commands remain available:

```bash
python scripts/run_csbench.py optimize CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --force
python scripts/run_csbench.py grade CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --background --force
python scripts/run_csbench.py evaluate CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --export
```

Monitoring commands:

```bash
python scripts/run_csbench.py status
python scripts/run_csbench.py tail
python scripts/run_csbench.py outputs CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7
```

CSBench answer-source correction status:

```text
CO_2:
  The interrupt mask words were corrected to match the original Q2 answer:
    A: 11111
    B: 01000
    C: 01101
    D: 01111
    E: 01001

CO_4:
  The final cache-hit judgement was corrected to match the original Q4 answer:
    address 7F0057H hits Cache block 1.
```

Important synchronization rule:

```text
RefGrader code changes are committed to the RefGrader repository.
CSBench question/answer corrections are committed to the CSBench_new repository.
Portable experiment results are committed to refgrader-artifacts.

After pulling CSBench_new on the server, rerun prepare_csbench or use the
unified run command with --dataset-root so data/csbench is regenerated.
```

Current command documentation:

```text
COMMANDS_GUIDE.md now lists four CO_1-CO_7 execution modes:
  1. prepare -> optimize -> grade
  2. optimize -> grade
  3. optimize only
  4. grade only
```

## 2026-06-16 Q2/Q3 Extraction-Lock Fix And Checkpoint Evaluation

This update addresses the Q2/Q3 failure mode where visual extraction failure
caused empty facts, positive evidence collapsed to zero, BND raises were blocked,
and some samples were locked into NEG/model-average fallback.

Implemented generic changes:

```text
1. step4_vlm_grader.py
   - Added preprocess_student_image_to_base64() with contrast/upscale/sharpen
     image views for second-pass extraction.
   - Added a clean override of stage1_targeted_reextraction() before Stage2.
   - The retry is triggered by blank, perception-failure, low-quality, or
     structure-missing facts. It does not use question ids.
   - Recovered facts are audited through _extraction_recovered_by.

2. calibration_utils.py
   - Extraction failure is no longer treated as a hard NEG reason by itself.
     It becomes extraction_retry_review and routes to BND.
   - fatal_points_ratio no longer counts missing method/intermediate items as
     semantic fatal when a strong final/result item is already matched.
   - Added result_anchored_undercredit_review and a guarded BND raise channel
     for result-concentrated tasks with low baseline, strong result evidence,
     low bare-answer risk, and low unsupported-high-score risk.
   - Large score spread now triggers hard NEG only when the average score is
     not already very low, avoiding "consistent low under-credit" being treated
     as rejection evidence.

3. evaluate.py
   - Added --result-source graded|checkpoint.
   - Default remains graded, preserving older commands.
   - --result-source checkpoint reads *_grading_checkpoint.json and includes
     NEG/rejected samples for full formal analysis.
```

Validation run:

```powershell
python -m py_compile calibration_utils.py step4_vlm_grader.py evaluate.py main_pipeline.py
git diff --check -- calibration_utils.py step4_vlm_grader.py evaluate.py
python scripts\replay_calibration.py --results-dir results_rrd_vlm --files results_rrd_vlm\Q2_grading_checkpoint.json results_rrd_vlm\Q3_grading_checkpoint.json results_rrd_vlm\Q4_grading_checkpoint.json
python evaluate.py --result-source checkpoint --compare --questions Q2 Q3 Q4 --compare-score-keys single avg selected 3wd
```

Replay on existing checkpoints:

```text
Q2 current MAE=4.084 -> replay MAE=4.053; QWK 0.326 -> 0.331; Bias -3.207 -> -3.176
Q3 current MAE=3.301 -> replay MAE=3.113; QWK 0.336 -> 0.375; Bias -3.301 -> -3.113
Q4 unchanged: MAE=2.089; QWK=0.852
GLOBAL current MAE=3.364 -> replay MAE=3.277; QWK 0.489 -> 0.505
```

Interpretation:

```text
The replay confirms that the route/BND logic moves in the correct direction on
Q2/Q3 without damaging Q4 in existing checkpoints. The new image-enhanced
second-pass extraction cannot be validated by replay; it requires a fresh FULL
run because it changes Stage1 VLM extraction inputs.
```

Follow-up audit fix on 2026-06-16:

```text
The previous replay was not sufficient for NEG -> BND samples because old NEG
records do not contain boundary_gate / boundary-agent evidence. Therefore replay
can show route changes but cannot show the real score effect for those samples.
Fresh FULL rerun is required.

Additional fixes:
  1. extraction_retry_review now also triggers on blank_rate >= 0.40 and on
     very-low-score/no-positive-evidence extraction failure patterns. This is
     generic and does not depend on visual item tags.
  2. low_score_nonblank_review was added to A3WA routing. Very low baseline
     samples with nonblank extracted facts are routed to BND instead of hard NEG.
     This prevents nonblank but semantically-disputed answers from being locked
     at model_avg=0 without boundary arbitration.
  3. The old duplicate stage1_targeted_reextraction definition was removed.
  4. Several historically mojibake logging/prompt fallback strings in
     step4_vlm_grader.py were replaced with ASCII/English strings so the file
     compiles reliably.
```

Validation after the follow-up fix:

```text
python -m py_compile calibration_utils.py step4_vlm_grader.py evaluate.py main_pipeline.py scripts\replay_calibration.py
git diff --check -- calibration_utils.py step4_vlm_grader.py evaluate.py CURRENT_PROGRESS.md

Q2/Q3/Q4 replay:
  Q2: old NEG -> new BND = 7 samples; MAE 4.084 -> 4.053
  Q3: old NEG -> new BND = 6 samples; MAE 3.301 -> 3.113
  Q4: unchanged; MAE 2.089

Q6/Q7 safety replay:
  Q6: Over>2 remains 2; MAE 2.666 -> 2.657
  Q7: Over>2 remains 6; MAE 0.961 -> 0.951
  GLOBAL Q6+Q7: MAE 1.820 -> 1.810; Over>2 remains 8
```

Important interpretation:

```text
The route-level objective is now satisfied: Q2 and Q3 severe old NEG samples are
sent to BND for fresh arbitration. Replay still cannot estimate their final
score improvement because old NEG samples lack saved boundary-agent outputs.
The next valid test is a --force-rerun FULL experiment.
```

## 2026-06-16 Q2/Q3 Audit Closure And Force-Rerun Checklist

This section closes the audit of the 2026-06-16 Q2/Q3 fix and records the
concrete next step: the only valid validation of the NEG->BND rescue and the
image-enhanced extraction is a --force-rerun FULL experiment. The diagnostic
below was run on existing checkpoints, so it validates routing only, not
Stage1 extraction or boundary-agent raises.

Audit closure findings (verified by diagnostic replay on checkpoints):

```text
1. extraction_retry_review fix works.
   - Q2 NEG 8 -> 1 (7 rerouted to BND; the remaining one is a true
     large_score_spread disagreement).
   - Q3 NEG 6 -> 0 (all 6 rerouted to BND).
   - The Q3 rescue is driven by the new blank_rate >= 0.40 branch of
     extraction_retry_review, NOT by low_score_nonblank_review.

2. low_score_nonblank_review triggers very rarely in practice
   (Q2=1, Q3=0, Q6=3). Its avg_ratio <= 0.30 condition is too strict to be
   the main driver. It is correct and harmless (every triggered sample has a
   non-low teacher score, so no truly-zero paper is wrongly promoted), but it
   should NOT be credited as the Q3 fix.

3. Q6 also has 4 ex-NEG samples rerouted to BND (E12314013/29/161/173,
   teacher 4-8, model_avg 3-4) via extraction_retry_review. This was not
   noted in the previous summary and is an additional force-rerun risk point.

4. Replay blindness is now explicit. scripts/replay_calibration.py prints a
   WARNING for BND samples that have no saved boundary_gate (route changed
   but score not raised in replay). Diagnostic counts:
     Q2 = 8, Q3 = 12, GLOBAL(Q2+Q3) = 20.
   Q3=12 is higher than the 6 ex-NEG because 6 original-BND Q3 samples also
   lack boundary_gate. None of these samples' real effect is visible in
   replay, by construction.

5. Watch sample: E12314101_Q2 (teacher=0, blank paper) is also rerouted to
   BND. force-rerun must confirm it stays at ~0 (not falsely raised).
```

Tooling cleanup done this round:

```text
- scripts/replay_calibration.py now reports how many BND samples are
  unevaluable in replay (no saved boundary_gate), so route-vs-score
  confusion is no longer silent.
- Deleted outputs/q2_q3_20260613_full_compare.csv (byte-identical duplicate
  of q2_q3_20260613_compare.csv).
```

Status:

```text
Implementation: correct across three rounds of changes; compiles clean; no
side-effect misfires.
Validation: still insufficient. 17 rerouted ex-NEG samples (Q2x7, Q3x6, Q6x4)
plus additional original-BND samples are invisible in replay. The reported
replay MAE improvements do NOT include any effect of the NEG->BND rescue.
Do not conclude the rescue works until force-rerun.
```

Force-rerun command:

```powershell
python main_pipeline.py --mode FULL --questions Q2 Q3 Q6 Q7 --force-rerun --progress-file results_rrd_vlm\progress_q2_q3_q6_q7_fix.json
python evaluate.py --result-source checkpoint --compare --questions Q2 Q3 Q6 Q7 --compare-score-keys single avg selected 3wd --compare-output outputs\q2_q3_q6_q7_fix_compare.csv
```

Force-rerun acceptance checklist (priority order):

```text
1. The 17+ rerouted ex-NEG samples: final vs teacher.
   Expect Q2/Q3 high-teacher samples (teacher 6-20) to be raised. Also count
   _extraction_recovered_by non-empty across all samples; if it is mostly
   empty, image enhancement is not working and the extraction root cause
   persists.

2. Q6/Q7 Over>2 must NOT increase after force-rerun.
   The replay "Over>2 unchanged" result is NOT trustworthy because replay did
   not raise the rerouted samples. Only force-rerun can confirm the loosened
   raise channels do not introduce over-credit on the main paper questions.

3. E12314101_Q2 (blank paper) final must stay ~0.
   Guards the NEG-rescue + boundary-agent path against falsely raising
   truly-empty answers.

4. Use a single --result-source checkpoint scope for all reported tables.
   Abandon the old compare CSV (graded_results, 64 rows) to avoid mixing
   run/scope with the 68-row checkpoint numbers (checkpoint final MAE 4.084
   vs old graded 5.43 differ by >1 point).
```

## 2026-06-06 Selected Baseline / Evaluation Update

This section records the latest code update made after the 2026-06-06 Q6/Q7
formal-result analysis. It should be treated as the newest active implementation
state.

Follow-up generality update:

```text
The old selected-baseline prototype used max_score >= 15.0 as a weak proxy for
"complex calculation/derivation task". This has been removed.

The current implementation infers task structure from rubric metadata and item
types instead:
  answer_type: formula / direct_numeric / derived_numeric / sequence /
               table_entry / judgement / concept_keyword
  role: parameter / method / intermediate / final / unknown

calibration_utils.infer_rubric_task_profile() summarizes the rubric into:
  task_type
  complex_derivation_task
  upper_consensus_eligible
  process_points_ratio
  result_points_ratio
  numeric_formula_points_ratio
  visual_sequence_points_ratio
  concept_judgement_points_ratio
```

The selected upper-consensus baseline is now allowed only when the rubric shows
enough process/evidence structure:

```text
upper_consensus_eligible =
  complex_derivation_task
  and process_points_ratio >= 0.60
  and numeric_formula_points_ratio >= 0.45
  and concept_judgement_points_ratio < 0.60
```

This keeps the method generic: it no longer depends on question id or full-score
thresholds. For example, Q7 is still recognized as a calculation task, but its
process_points_ratio is only 0.50, so it is not eligible for max-of-three
selected-baseline promotion. Q6 has process_points_ratio=1.00 and remains
eligible.

Main change:

```text
The initial three-run model average is no longer the only score baseline used
by 3WD. The pipeline now computes a risk-aware selected_baseline_score before
A3WA routing and BND arbitration.
```

Score layers now used for analysis:

```text
single_first_score        = the first score in model_scores_history
model_avg_score           = ordinary average of the three model scores
selected_baseline_score   = 3WD-selected baseline before POS/BND/NEG action
final_calibrated_score    = final 3WD score after route and possible BND action
```

`selected_baseline_score` is produced by `calibration_utils.select_baseline_score()`:

```text
Default:
  selected_baseline_score = model_avg_score

Guarded upper-consensus case:
  if the task looks like a complex calculation/derivation task,
  and result correctness evidence is present,
  and method/process evidence is present,
  and lenient-undercredit evidence is present,
  and over-credit risk is low,
  then selected_baseline_score = max(model_scores_history)
```

The selected baseline is then used consistently by:

```text
step4_vlm_grader.py:
  A3WA routing uses selected_baseline_score.
  POS directly accepts selected_baseline_score.
  BND arbitration uses selected_baseline_score as the baseline.
  Result JSON now stores selected_baseline_score, baseline_policy,
  baseline_score_source, and baseline_selection_signals.

scripts/replay_calibration.py:
  Offline replay now mirrors the formal pipeline and uses selected_baseline_score
  for POS, NEG fallback, and BND action policy.

evaluate.py:
  --compare now reports four score types:
  single / avg / selected / 3WD.
  --score-key selected_baseline_score is supported.
  --score-key also supports aliases: single / avg / selected / 3wd.
  --compare-score-keys can select any subset, for example:
    python evaluate.py --compare --questions Q6 Q7 --compare-score-keys avg selected 3wd
  --compare-output CSV now contains selected_baseline_score, selected_diff,
  selected_gain_vs_avg, final_gain_vs_selected, baseline_policy, and
  baseline_score_source.
```

Stage2 prompt update:

```text
prompts/stage2_logic_grading.md now has Strict Equivalence Guards.
The lenient grading policy applies to missing expansion and minor arithmetic
details, but it does not allow wrong formulas, wrong mappings, wrong target
quantities, or dimension/unit contradictions to be marked as MATCH.
```

Validation commands already run locally:

```bash
python -m py_compile calibration_utils.py scripts\replay_calibration.py step4_vlm_grader.py evaluate.py
python scripts\replay_calibration.py --results-dir results_rrd_vlm --files results_rrd_vlm\Q6_grading_checkpoint.json results_rrd_vlm\Q7_grading_checkpoint.json
python evaluate.py --compare --questions Q6 Q7 --compare-output outputs\q6_q7_selected_compare_check.csv
```

Replay result on the current Q6/Q7 checkpoint:

```text
Q6 current -> replay:
  MAE 3.076 -> 2.954
  RMSE 3.853 -> 3.695
  QWK 0.720 -> 0.746
  Pearson 0.855 -> 0.865
  TAR2 48.5% -> 50.0%
  Under>2 32 -> 31

Q7 current -> replay:
  unchanged, MAE remains 1.013

GLOBAL current -> replay:
  MAE 2.053 -> 1.991
  RMSE 3.074 -> 2.974
  QWK 0.711 -> 0.733
  Pearson 0.760 -> 0.776
  TAR2 68.1% -> 68.9%
  Under>2 34 -> 33
```

Important interpretation:

```text
Existing result JSON files generated before this update do not contain a real
selected_baseline_score field. In evaluate.py, old files fallback selected to
model_avg_score for compatibility. The selected column will show real differences
only after the next formal experiment is rerun with the updated step4_vlm_grader.py.
```

## 2026-06-06 Latest Formal Q6/Q7 Run

The latest lab-server formal experiment has completed:

```text
run_id = 20260605_225725
mode = FULL
questions = Q6 Q7
force_rerun = true
model_provider = glm5
vlm_model = glm-4.6v
text_model = glm-5.1
completed_at = 2026-06-06
```

Important evaluation note:

```text
Q6_graded_results.json contains 63 normal records, while Q6_grading_checkpoint.json contains all 68 completed records.
The missing 5 Q6 records are NEG/rejected records in Q6_rejected.json.
For full-experiment analysis, use Q6_grading_checkpoint.json and Q7_grading_checkpoint.json.
```

New per-sample analysis files were generated:

```text
outputs/q6_q7_20260606_single_vs_avg_vs_3wd.csv
outputs/q6_q7_20260606_checkpoint_analysis.csv
```

These CSV files contain every Q6/Q7 sample with teacher score, first single score, model average score, final 3WD score, error, gain/loss, route, boundary action, A3WA confidence/risk, post-calibration signals, and diagnosis tags.

Full-checkpoint metrics from the latest run:

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

Interpretation of this run:

```text
1. Within the latest run, 3WD is still positive compared with model_avg:
   global MAE improves 2.090 -> 2.053, RMSE improves 3.110 -> 3.074,
   Pearson improves 0.753 -> 0.760, TAR2 improves 66.7% -> 68.1%.

2. The improvement is too small. Only 5 of 135 samples improve over model_avg,
   129 are unchanged, and 1 worsens. The BND gate is currently very conservative.

3. Compared with the previous 2026-06-05 single/avg/3WD CSV, the latest formal
   run is worse overall:
   previous global 3WD MAE=1.915, current global 3WD MAE=2.053.
   This is partly because the base model_avg also worsened:
   previous global avg MAE=2.005, current global avg MAE=2.090.

4. Q6 is the main problem. The final 3WD score still has strong underestimation:
   Q6 final Bias=-2.512 and Under>2=32. Many teacher-high/model-low samples
   remain unchanged because BND rejects raises or keeps minor changes.

5. Q7 is directionally better than model_avg in the latest run, but it still has
   severe overestimation on several low-teacher-score samples. Some of these are
   POS high-confidence cases, so BND never gets a chance to correct them.
```

Sample-level diagnosis from the latest run:

```text
Q6 route counts: BND=46, POS=17, NEG=5
Q6 boundary actions: keep_minor_change=27, reject_raise=15, medium_raise=1, small_raise=2, reject_lower=1, NO_GATE=22
Q6 3WD changes: improved=3, unchanged=65, worsened=0
Q6 main issue: BND catches many risky samples but still does not raise enough for teacher-lenient high-score answers.

Largest Q6 final errors:
  E12314033_Q6 teacher=18 final=8.3 diff=-9.7 route=BND gate=reject_raise
  E12314037_Q6 teacher=20 final=11.0 diff=-9.0 route=BND gate=reject_raise
  E12214023_Q6 teacher=12 final=4.0 diff=-8.0 route=BND gate=keep_minor_change
  E12214091_Q6 teacher=11 final=3.0 diff=-8.0 route=NEG
  E12314117_Q6 teacher=8 final=2.0 diff=-6.0 route=BND gate=keep_minor_change

Q7 route counts: POS=47, BND=20
Q7 boundary actions: keep_minor_change=13, reject_raise=4, large_lower=3, NO_GATE=47
Q7 3WD changes: improved=2, unchanged=64, worsened=1
Q7 main issue: several severe overestimations are POS high-confidence samples, not BND samples.

Largest Q7 final errors:
  E12314133_Q7 teacher=0 final=8.9 diff=+8.9 route=POS
  E12214212_Q7 teacher=0 final=8.2 diff=+8.2 route=POS
  E12314129_Q7 teacher=2 final=6.7 diff=+4.7 route=BND gate=keep_minor_change
  E12314065_Q7 teacher=7 final=3.3 diff=-3.7 route=POS
  E02014181_Q7 teacher=2 final=5.3 diff=+3.3 route=BND gate=reject_raise
```

Current conclusion:

```text
The latest system is not broken: 3WD still improves over the current model_avg baseline.
However, the improvement is not strong enough, and the latest formal run is weaker than
the previous run. The main bottleneck is no longer only A3WA routing; it is the combination
of unstable base model scoring, conservative BND-UP, and missed POS high-over cases.
```

## Current Goal

RefGrader is being optimized for a paper-oriented three-way decision (3WD) framework. The immediate goal is to replace the earlier engineering-style route triggers with a more theoretically explainable A3WA-inspired decision process:

```text
multi-source risk signals -> R(x) -> confidence mu(x) -> asymmetric alpha/beta -> POS/BND/NEG
```

The current code has moved beyond the first A3WA implementation: A3WA loss parameters and risk weights can now be calibrated offline, and BND arbitration uses a validation-aligned action policy rather than freely accepting the boundary agent's total score.

## Latest Implementation

The latest implementation introduces an A3WA-inspired 3WD layer plus offline cost-sensitive calibration and a BND action policy.

2026-06-03 update: the latest Q6/Q7 formal run showed that the previous
conservative BND action policy caused `final_calibrated_score == model_avg_score`
for every Q6/Q7 sample. The route/audit layer worked, but it produced no score
gain. Based on the confirmed grading practice that the instructor is lenient on
complex calculation problems, the active scoring design has been revised from
"strict standard-solution matching" to "instructor-aligned lenient process-credit
grading".

Extraction quality was also made rubric-aware. Short answers such as "yes",
"correct", "hit", or "miss" are no longer treated as low-quality extraction when
the rubric item is a judgement/conclusion item. The same generic words remain
low-quality for formula, numeric, mapping, and process items where concrete
content is required.

Changed or added files:

```text
calibration_utils.py
scripts/replay_calibration.py
scripts/calibrate_a3wa.py
prompts/stage2_logic_grading.md
prompts/boundary_arbitration.md
step4_vlm_grader.py
evaluate.py
README.md
CURRENT_PROGRESS.md
```

Main logic now used by `step4_vlm_grader.py`:

```text
Stage 1: VLM extracts objective facts from the answer image.
Stage 2: LLM scores the same facts three times.
  Stage2 now uses prompts/stage2_logic_grading.md instead of the old inline mojibake prompt.
Post calibration: generic calibration and A3WA confidence are computed.
3WD route:
  hard NEG guard
  else compute R(x), mu(x), alpha, beta
  if mu >= alpha -> POS
  elif mu <= beta -> NEG
  else -> BND
BND action:
  call boundary arbitration agent for a candidate score
  BND now uses prompts/boundary_arbitration.md instead of the old inline mojibake prompt.
  apply validation-aligned action policy
  keep model_avg when evidence is weak
  allow BND-UP when the answer/result evidence and minimum process evidence indicate instructor-style under-credit
  allow BND-DOWN only for unsupported high-score evidence, not merely because the score is high
```

Current risk function:

```text
U_extract = 0.5 * low_quality_rate + 0.5 * perception_failure_rate
U_score   = 0.5 * (std_dev / MAX_SCORE) + 0.5 * (score_spread / MAX_SCORE)
U_semantic = fatal_points_ratio, with optional unsupported MATCH / core anchor risk
U_blank = blank_rate

R(x) = 0.35 * U_extract + 0.30 * U_score + 0.20 * U_semantic + 0.15 * U_blank
mu(x) = 1 - R(x)
```

The latest lenient-scoring signals added to `post_calibration` are:

```text
result_correctness_signal:
  whether final answer / key conclusion evidence is correct or near-correct.

method_evidence_signal:
  whether the answer contains formula, relation, conversion, mapping, or computation trace evidence.

bare_answer_risk:
  correct-looking final answer without enough process evidence.

lenient_undercredit_signal:
  model may be too strict under instructor-style process-credit grading.

unsupported_high_score_risk:
  high score is unsupported by answer correctness or process evidence.
```

Updated route rule:

```text
if hard NEG:
  NEG
elif mu >= alpha and lenient_undercredit_signal >= 0.08 and score is below high band:
  BND
elif mu >= alpha and unsupported_high_score_risk >= 0.25:
  BND
elif mu >= alpha:
  POS
elif mu <= beta:
  NEG
else:
  BND
```

Updated BND action policy:

```text
BND-UP:
  requires lenient_undercredit_signal >= 0.08,
  result_correctness_signal >= 0.65,
  method_evidence_signal or partial evidence,
  low bare-answer risk,
  and low unsupported-high-score risk.

auto_small_raise:
  allowed when lenient-undercredit evidence is strong even if the boundary agent
  returns nearly the same score.

BND-DOWN:
  allowed only for unsupported high-score evidence.
```

Current A3WA parameters:

```text
lambda1 = 5
lambda2 = 1
mu1 = 3
mu2 = 7
m = 0.5

alpha = (lambda1 + lambda2 * m) / (lambda1 + lambda2) = 0.917
beta  = (mu2 * m) / (mu1 + mu2) = 0.35
```

`calibration_utils.py` also includes `optimize_a3wa_m()` for offline/batch analysis. The formal online pipeline currently uses `m=0.5` because grading runs sample-by-sample and does not have full batch confidence distribution during routing.

## A3WA Calibration

The fixed prior parameters are no longer treated as the final setting. A new offline calibration script searches loss parameters and risk weights:

```bash
python scripts/calibrate_a3wa.py --files results_rrd_vlm/Q5_graded_results.json results_rrd_vlm/Q6_graded_results.json results_rrd_vlm/Q7_graded_results.json --output results_rrd_vlm/a3wa_calibration_config.json
```

The formal pipeline automatically loads:

```text
results_rrd_vlm/a3wa_calibration_config.json
```

or a custom path from:

```text
A3WA_CALIBRATION_CONFIG
```

Current generated config:

```text
lambda1 = 5.0
lambda2 = 2.0
mu1 = 2.0
mu2 = 5.0
m = 0.4

alpha = 0.828571
beta = 0.285714

risk_weights:
  extract = 0.35
  score = 0.30
  semantic = 0.20
  blank = 0.15
  overcredit = 0.00
```

The calibration objective is a cost-sensitive validation objective, not manual threshold tuning:

```text
MAE(final)
+ penalty if final worse than model_avg
+ penalty if TAR2 falls below model_avg
+ penalty if BND ratio is too high
+ penalty if POS is not easier than BND
+ penalty if BND gain is negative
```

Latest calibration summary on current Q5/Q6/Q7 result files:

```text
baseline model_avg MAE = 2.1948
calibrated simulated MAE = 2.1766
TAR2 = 63.5%
routes = POS 151, BND 41
BND gain = +0.0854
```

## BND Arbitration And Action Policy

BND samples still call the boundary arbitration agent, but the agent is no longer allowed to freely determine the final score. The current logic is:

```text
baseline_score = model_avg_score
candidate_score = boundary_agent_score

candidate_score can come from:
  structured missed_credit_items - over_credit_items
  or legacy calibrated_score fallback

action in:
  keep_baseline
  keep_minor_change
  small_raise
  small_lower
  large_lower
  reject_raise
  reject_lower
```

Direction evidence is generic and not tied to any question ID:

```text
over_score_risk examples:
  high_blank_high_score
  high fatal-points ratio
  unsupported MATCH
  core_anchor_failed
  post-calibration upper cap

under_score_risk examples:
  extraction uncertainty on nonblank answers
  large score disagreement
  substantial FORMAT_MINOR mass
```

`a3wa_dynamic_bounds()` is now direction-aware:

```text
small_margin = max(0.05 * MAX_SCORE, 0.5)
large_margin = max(delta_from_mu, 0.20 * MAX_SCORE, 1.5)

if over_score_risk:
  lower_bound = model_avg_score - large_margin
else:
  lower_bound = model_avg_score - small_margin

if under_score_risk:
  upper_bound = model_avg_score + large_margin
else:
  upper_bound = model_avg_score + small_margin

if strong_over_score_risk:
  upper_bound <= model_avg_score
```

The important design point is:

```text
BND only means "needs review"; it does not mean "must change score".
The default fallback is model_avg_score when no reliable profitable-action evidence exists.
```

## Latest Validation

The earlier A3WA-only run was weaker than the model-average baseline:

```text
For Q5/Q6/Q7 formal results:
model_avg global MAE = 2.136
old 3WD final global MAE = 2.168

Root cause:
POS samples were unchanged.
All benefit and damage came from BND.
Q5/Q6 BND had negative net gain, while Q7 BND had positive net gain.
```

The latest no-harm-gate implementation passed syntax validation:

```bash
python -m py_compile calibration_utils.py scripts/replay_calibration.py step4_vlm_grader.py evaluate.py
git diff --check
```

`git diff --check` only reports CRLF line-ending warnings on Windows; no whitespace errors were found.

Latest replay after adding calibration config and the aligned action policy:

```text
Q5 current MAE=3.127 RMSE=4.266 QWK=0.673 Pearson=0.783 TAR2=46.9%
Q5 replay  MAE=3.131 RMSE=4.268 QWK=0.673 Pearson=0.783 TAR2=46.9%

Q6 current MAE=2.455 RMSE=3.273 QWK=0.787 Pearson=0.875 TAR2=57.8%
Q6 replay  MAE=2.444 RMSE=3.268 QWK=0.788 Pearson=0.875 TAR2=57.8%

Q7 current MAE=1.019 RMSE=2.041 QWK=0.706 Pearson=0.719 TAR2=82.8%
Q7 replay  MAE=0.988 RMSE=2.011 QWK=0.710 Pearson=0.727 TAR2=84.4%

GLOBAL current N=192 MAE=2.200 RMSE=3.321 QWK=0.730 Pearson=0.759 TAR2=62.5% Bias=-1.205
GLOBAL replay  N=192 MAE=2.188 RMSE=3.313 QWK=0.730 Pearson=0.759 TAR2=63.0% Bias=-1.193
```

Interpretation:

```text
Calibration reduces invalid BND usage and restores the Q7 bad lower case to model_avg.
It is still a validation/calibration result and must be confirmed by a fresh formal experiment.
```

Latest replay after instructor-aligned lenient process-credit update on the
fresh Q6/Q7 formal result files:

```text
Q6 current MAE=2.987 RMSE=3.797 QWK=0.719 Pearson=0.893 TAR2=48.3% Bias=-2.520
Q6 replay  MAE=2.837 RMSE=3.707 QWK=0.735 Pearson=0.881 TAR2=53.3% Bias=-2.357

Q7 current MAE=1.421 RMSE=2.183 QWK=0.667 Pearson=0.688 TAR2=77.6% Bias=-0.296
Q7 replay  MAE=1.390 RMSE=2.148 QWK=0.665 Pearson=0.694 TAR2=79.1% Bias=-0.222

GLOBAL Q6+Q7 current N=127 MAE=2.161 RMSE=3.054 QWK=0.701 Pearson=0.790 TAR2=63.8% Bias=-1.346
GLOBAL Q6+Q7 replay  N=127 MAE=2.073 RMSE=2.988 QWK=0.714 Pearson=0.792 TAR2=66.9% Bias=-1.231
```

Interpretation:

```text
The new logic no longer degenerates to model_avg for every sample.
High-confidence but likely under-credited samples can be routed to BND and receive
auto_small_raise when result evidence is strong enough.
Replay is positive on Q6/Q7, but a fresh formal run is still required because
Stage2 and boundary prompts have also changed.
```

## Server Experiment Workflow

Preferred server workflow:

```bash
conda activate ref-grader
cd /home/E125221219/projects/refgrader
./run_experiment.sh run
```

Useful management commands:

```bash
./run_experiment.sh status
./run_experiment.sh tail
./run_experiment.sh stop
python monitor.py --watch
```

`run_experiment.sh run` uses `nohup python3 main_pipeline.py ... &`, writes logs to `logs/experiment_*.log`, and records PID in `logs/refgrader.pid`. SSH disconnect or local computer shutdown should not stop the server job.

After completion:

```bash
python evaluate.py --compare --questions Q5 Q6 Q7
```

## New Output Fields

Formal grading results now include:

```json
"post_calibration": {
  "unsupported_match_points_ratio": 0.0,
  "method_final_verified_ratio": 0.0,
  "metadata_coverage": 0.0,
  "core_anchor_failed": false,
  "visual_blank_review": false,
  "rule_hits": []
},
"a3wa_decision": {
  "route": "POS/BND/NEG",
  "risk": 0.0,
  "mu": 1.0,
  "alpha": 0.916667,
  "beta": 0.35,
  "m": 0.5,
  "reason": "...",
  "risk_components": {
    "U_extract": 0.0,
    "U_score": 0.0,
    "U_semantic": 0.0,
    "U_blank": 0.0
  }
},
"boundary_gate": {
  "final_score": 0.0,
  "baseline_score": 0.0,
  "raw_candidate_score": 0.0,
  "bounded_candidate_score": 0.0,
  "delta_from_baseline": 0.0,
  "accepted": false,
  "action": "keep_baseline / keep_minor_change / small_raise / small_lower / large_lower / reject_lower / reject_raise",
  "gate_reason": "...",
  "lower_bound": 0.0,
  "upper_bound": 0.0,
  "direction_signals": {
    "over_score_risk": false,
    "under_score_risk": false,
    "strong_over_score_risk": false,
    "over_reasons": [],
    "under_reasons": [],
    "strong_over_reasons": []
  }
}
```

These fields are intended for paper analysis and route audit.

`evaluate.py --compare` now prints an additional audit section:

```text
3WD gain audit | final_calibrated_score vs model_avg_score

For each question and route:
  improved count
  worsened count
  unchanged count
  mean gain
  mean final-minus-average delta

It also prints the top samples worsened by 3WD correction.
```

## Prompt Template Refactor

The active Stage2 and BND prompts have been moved to UTF-8 template files:

```text
prompts/stage2_logic_grading.md
prompts/boundary_arbitration.md
```

`step4_vlm_grader.py` still keeps the old inline prompt strings as fallback only. The actual prompt sent to the model is loaded through `render_prompt_template()`. This avoids editing the historical mojibake prompt blocks directly while making the active prompts readable, maintainable, and suitable for paper appendix/reproducibility.

## 2026-06-03 Pre-Run Check For Q6/Q7

Before the next lab-server run on Q6/Q7, the BND raise gate was tightened in a
targeted, question-agnostic way. The issue found during replay was that some
parameter-heavy Q7 samples had partial final-result matches and weak parameter
support, but still triggered `auto_small_raise`. The fix keeps lenient process
credit, but separates:

```text
result_correctness_signal:
  MATCH / FORMAT_MINOR / PARTIAL_MATCH on result-like items.

result_strong_signal:
  only MATCH / FORMAT_MINOR on result-like items.

direct_points_ratio:
  how much of the rubric is direct parameter/value identification.
```

`apply_boundary_action_policy()` now suppresses automatic BND-UP when all three
conditions hold:

```text
direct_points_ratio >= 0.30
direct_awarded_ratio < 0.70
result_strong_signal < 0.65
```

This keeps the teacher-lenient behavior for answers with strong final-result
evidence, while avoiding extra upward correction for samples whose final answer
is only partially matched and whose key parameters are weak.

Validation run on Q6/Q7 checkpoints:

```text
Q6 current MAE=2.974 TAR2=48.5% Under>2=33
Q6 replay  MAE=2.841 TAR2=52.9% Under>2=30

Q7 current MAE=1.421 TAR2=77.6% Under>2=8
Q7 replay  MAE=1.379 TAR2=79.1% Under>2=7

GLOBAL current MAE=2.203 TAR2=63.0% Bias=-1.407 Over>2=9 Under>2=41
GLOBAL replay  MAE=2.116 TAR2=65.9% Bias=-1.314 Over>2=9 Under>2=37
```

AST syntax validation passed for:

```text
calibration_utils.py
scripts/replay_calibration.py
step4_vlm_grader.py
evaluate.py
```

Validation already run:

```text
python -m py_compile step4_vlm_grader.py calibration_utils.py scripts/calibrate_a3wa.py scripts/replay_calibration.py evaluate.py
template placeholder replacement check: passed
replay_calibration with A3WA config: passed
```

## 2026-06-04 Q6/Q7 Post-Run Gate Tightening

The latest formal Q6/Q7 run showed that the A3WA/BND framework is not useless,
but the BND raise policy was still too permissive in several Q6 boundary cases.
Q7 was generally positive; Q6 was mixed because a few low-teacher-score samples
were raised without enough answer/process evidence.

Latest full-checkpoint formal result before this patch:

```text
Q6 checkpoint, N=68:
model_avg  MAE=3.319 RMSE=4.144 QWK=0.665 Pearson=0.859 TAR2=48.5% Bias=-2.837 Over>2=2 Under>2=33
final      MAE=3.272 RMSE=4.061 QWK=0.680 Pearson=0.848 TAR2=45.6% Bias=-2.687 Over>2=4 Under>2=33

Q7 checkpoint, N=67:
model_avg  MAE=1.378 RMSE=2.314 QWK=0.590 Pearson=0.629 TAR2=80.6% Bias=-0.118 Over>2=6 Under>2=7
final      MAE=1.325 RMSE=2.258 QWK=0.606 Pearson=0.647 TAR2=82.1% Bias=-0.051 Over>2=6 Under>2=6

GLOBAL checkpoint, N=135:
model_avg  MAE=2.356 RMSE=3.363 Pearson=0.732 TAR2=64.4% Bias=-1.487 Over>2=8 Under>2=40
final      MAE=2.306 RMSE=3.292 Pearson=0.738 TAR2=63.7% Bias=-1.379 Over>2=10 Under>2=39
```

Per-sample diagnosis from `outputs/q6_q7_latest_full_checkpoint_analysis.csv`:

```text
Q6 improved: 4 samples
Q6 worsened: 3 samples
Q6 unchanged: 61 samples

Q6 worsened examples:
  E12314125_Q6 teacher=2.0 avg=2.7 final=4.1
  E12314113_Q6 teacher=3.0 avg=4.7 final=6.0
  E12314029_Q6 teacher=4.0 avg=4.3 final=5.0

Common cause:
  These samples had weak lenient_undercredit_signal and weak result evidence,
  but the old BND-UP fallback still allowed a small raise because the baseline
  score was low and the score/extraction risk was nonzero.
```

Code change after this diagnosis is limited to `calibration_utils.py`.

1. Added `weak_result_high_score_review`:

```text
avg_ratio >= 0.65
result_strong_signal <= 0.50
unsupported_high_score_risk >= 0.10
```

This does not directly lower a score. It only routes high-score/weak-result
cases from POS into BND so the boundary agent must provide evidence before any
correction is accepted.

2. Softened `semantic_risk_too_high` hard NEG:

```text
u_semantic >= 0.75
and lenient_undercredit < 0.10
and result_strong < 0.50
and method_evidence < 0.50
```

Previously, high semantic risk could send a sample directly to NEG even when
there was meaningful result or process evidence. This conflicted with the
confirmed instructor-lenient grading style for complex calculation problems.

3. Tightened BND-UP:

The old generic fallback was removed:

```text
avg_ratio <= 0.70
and (score_risk >= 0.12 or extract >= 0.20)
and unsupported_high_score < 0.25
```

The current BND-UP now requires `lenient_raise_ready`, meaning the raise must be
supported by answer correctness and process/partial evidence, not merely by low
baseline score plus uncertainty.

4. Strong lenient auto-raise now requires stronger evidence:

```text
lenient_undercredit >= 0.12
result_strong >= 0.50
method_evidence >= 0.50 or partial_or_format_evidence >= 0.15
avg_ratio <= 0.70
```

Replay validation after this patch:

```text
Q6 current MAE=3.272 RMSE=4.061 QWK=0.680 Pearson=0.848 TAR2=45.6% Bias=-2.687 Over>2=4 Under>2=33
Q6 replay  MAE=3.222 RMSE=4.050 QWK=0.685 Pearson=0.855 TAR2=50.0% Bias=-2.737 Over>2=2 Under>2=32

Q7 current MAE=1.325 RMSE=2.258 QWK=0.606 Pearson=0.647 TAR2=82.1% Bias=-0.051 Over>2=6 Under>2=6
Q7 replay  MAE=1.328 RMSE=2.256 QWK=0.603 Pearson=0.647 TAR2=82.1% Bias=-0.048 Over>2=6 Under>2=6

GLOBAL current MAE=2.306 RMSE=3.292 QWK=0.661 Pearson=0.738 TAR2=63.7% Bias=-1.379 Over>2=10 Under>2=39
GLOBAL replay  MAE=2.282 RMSE=3.285 QWK=0.664 Pearson=0.742 TAR2=65.9% Bias=-1.402 Over>2=8 Under>2=38
```

Interpretation:

```text
Q6: positive replay direction. The main improvement is fewer unjustified high-over corrections.
Q7: essentially stable. The new weak-result review routes more samples to BND but does not introduce large side effects.
GLOBAL: MAE, Pearson, TAR2, Over>2, and Under>2 all improve in replay.
```

Important limitation:

```text
This is replay on the latest checkpoint, not a fresh formal run.
The lab-server formal run must use --force-rerun; otherwise old checkpoints may
be reused and the new code will not be applied to every sample.
```

## Known Issues

1. Q5 remains mainly limited by VLM/OCR extraction failures on visually complex handwritten encoding diagrams. The A3WA route does not directly solve this.
2. Q6 high-estimation cases may require reliable structured rubric metadata or stronger formula/dependency validation before hard caps can safely become more aggressive.
3. Automatically inferred rubric metadata currently defaults to safe/audit behavior. It does not enable strong hard-cap scoring unless metadata is explicit or trusted.
4. `m=0.5` is used in online grading for stability. For the paper, batch/offline experiments should evaluate adaptive `m*` via `optimize_a3wa_m()`.
5. The action policy is still rule-based and calibrated on current Q5/Q6/Q7 result files. For the paper, parameters must be selected on a validation split and reported on a held-out test split.
6. Stage2 and BND prompts are now UTF-8 templates. Full visual retry with image enhancement/cropping is still not implemented. Q5 remains dominated by visual extraction failures.
7. README still contains older historical result tables. Treat the "Latest Progress: A3WA" section and this file as the current handoff state.

## Next Steps

1. Push the latest local changes, especially `calibration_utils.py`.
2. Pull on the lab server.
3. Run syntax validation:

```bash
python -m py_compile calibration_utils.py scripts/replay_calibration.py step4_vlm_grader.py evaluate.py
```

4. Optionally run replay on the server to confirm the same direction:

```bash
python scripts/replay_calibration.py --results-dir results_rrd_vlm --files results_rrd_vlm/Q6_grading_checkpoint.json results_rrd_vlm/Q7_grading_checkpoint.json
```

5. Run the fresh formal Q6/Q7 experiment with checkpoint reset:

```bash
./run_experiment.sh run --mode FULL --questions Q6 Q7 --force-rerun
```

6. Monitor:

```bash
./run_experiment.sh status
./run_experiment.sh tail
python monitor.py --watch
```

7. After completion, run:

```bash
python evaluate.py --compare --questions Q6 Q7
```

8. Inspect `a3wa_decision`, `boundary_gate`, and `risk_features.boundary_gate_*` in the new result JSON files.
9. Compare the new formal result against:

```text
model_avg baseline
previous formal final before this patch
latest replay after this patch
route-level BND gain
```

10. For paper analysis, compute route-level audits:

```text
mean(mu_POS) > mean(mu_BND) > mean(mu_NEG)
mean(R_POS) < mean(R_BND) < mean(R_NEG)
BND agent correction magnitude vs mu
BND no-harm accepted/rejected counts
mean gain by boundary_gate.action
hard NEG reasons vs confidence-based NEG reasons
```

## Key Files

```text
README.md                         Long-term project overview and A3WA progress section.
CURRENT_PROGRESS.md               Short handoff file for new conversations.
calibration_utils.py              A3WA risk/confidence, thresholds, direction-aware bounds, no-harm gate.
step4_vlm_grader.py               Formal grading pipeline using A3WA route decisions and BND no-harm gate.
scripts/calibrate_a3wa.py         Offline cost-sensitive search for A3WA loss parameters and risk weights.
scripts/replay_calibration.py     Offline replay validation using calibration config and the same BND action policy.
prompts/stage2_logic_grading.md   Active UTF-8 Stage2 semantic grading prompt.
prompts/boundary_arbitration.md   Active UTF-8 BND structured arbitration prompt.
run_experiment.sh                 Server background experiment runner.
evaluate.py                       Final metrics plus 3WD gain audit.
results_rrd_vlm/*_grading_checkpoint.json  Existing checkpoint inputs for replay.
results_rrd_vlm/*_graded_results.json      Formal experiment outputs.
```

## Handoff Instruction

In a new conversation or after context loss, use:

```text
Please first read CURRENT_PROGRESS.md, then continue from the current project state.
```

## 2026-06-18 Local PaddleOCR Evaluation Setup

Purpose: evaluate PaddleOCR as an isolated visual-transcription backend before
changing the formal Stage1 grading pipeline.

Installed locally:

```text
Environment: .venv-ocr
Python: 3.11
PaddlePaddle: 3.3.1 CPU
PaddleOCR: 3.7.0
```

Added files:

```text
ocr/paddle_ocr_worker.py       Single-image or directory OCR to auditable JSON.
ocr/requirements.txt           Reproducible local OCR dependency versions.
scripts/setup_paddle_ocr.ps1   Rebuild the isolated OCR environment.
scripts/run_paddle_ocr.ps1     One-command OCR test entry point.
OCR_GUIDE.md                   Chinese usage and current capability boundary.
```

Validated:

```text
1. Normal Q5 answer E01914115_Q5: 7 OCR lines, mean confidence 0.944906.
2. Confirmed blank Q5 answer E02014181_Q5: PaddleOCR still detected printed
   question text and low-confidence noise. At confidence >= 0.5, 3 tokens
   remained. Therefore OCR output alone cannot verify a blank answer.
3. OCR JSON cache skips unchanged images by SHA-256.
4. PaddlePaddle 3.3.1 Windows CPU requires oneDNN/MKL-DNN disabled in the
   worker; this is now the default.
5. `pip check` and worker compilation passed.
```

Boundary at the end of the isolated-worker step:

```text
PaddleOCR was installed only for extraction evaluation. It was not yet connected
to step4_vlm_grader.py and does not affect formal scoring results. Before formal
integration, add printed-region suppression / handwriting detection and measure
OCR accuracy on blank, numeric, formula, bit-vector, table, and diagram samples.
```

## 2026-06-18 Optional PaddleOCR Formal Backend

The isolated OCR worker is now connected as an optional backend without
changing the default legacy extractor.

Implemented:

```text
--extraction-backend glm_vlm|paddle_glm5
--mode OCR_ONLY
--mode GRADE_ONLY
raw OCR cache: ocr_cache/<Qx>/<student>.json
mapped fact cache: ocr_cache/facts/<Qx>/<student>.json
GLM-5.1 blind fact mapping
conditional GLM-4.6V diagram parsing
conservative blank authenticity: confirmed_blank/nonblank/uncertain
```

Focused Q2 test:

```text
sample = E01914115_Q2
PaddleOCR tokens = 23
mean confidence = 0.891514
diagram path = 用户程序→A→C→D→C→E→B→用户程序
Q2 relation check = 5/5 PASS
```

One-sample ablation:

```text
legacy glm_vlm: model_avg=16.9, final=18.0, teacher=18.0
paddle_glm5:   model_avg=11.1, final=12.5, teacher=18.0
```

Conclusion: the conditional diagram parser fixes the important Q2 `C→D→C`
middle relation, but PaddleOCR still misses handwritten five-bit masks. Keep
`glm_vlm` as default and use `paddle_glm5` for staged experiments/ablation.

## Fixed Run Commands And Operational Notes

The project already has a server-side background runner: `run_experiment.sh`.
It wraps `main_pipeline.py` with `nohup`, writes a PID file, and provides
`status`, `tail`, and `stop` commands. Do not manually add another `nohup`
layer unless the script is unavailable.

Local Windows run for Q2/Q3:

```powershell
python main_pipeline.py --mode FULL --questions Q2 Q3 --progress-file results_rrd_vlm\progress_q2_q3_local.json
```

Lab Linux server run for Q4/Q5:

```bash
cd /home/E125221219/projects/refgrader
conda activate ref-grader
./run_experiment.sh run --mode FULL --questions Q4 Q5 --progress-file results_rrd_vlm/progress_q4_q5_server.json
```

Server management:

```bash
./run_experiment.sh status
./run_experiment.sh tail
./run_experiment.sh stop
python monitor.py --watch
```

Question selection priority:

```text
1. Command-line --questions has the highest priority and overrides GRADING_CONFIG.
2. If --questions is not provided, main_pipeline.py uses GRADING_CONFIG in FULL mode.
3. If --questions is not provided, main_pipeline.py uses VARIANCE_CONFIG in VARIANCE_OPT mode.
4. Use separate progress files when local and server jobs run different question sets.
```

## 2026-06-14 Rubric And 3WD Simplification Update

Goal: reduce over-engineered risk inputs and make the method easier to defend in a paper. The current implementation no longer relies on teacher historical error signals for rubric optimization. The pipeline keeps the official coarse rubric as the source, converts it to structured JSON, then uses high-variance samples only to identify rubric granularity or ambiguity problems.

Implemented changes:

```text
1. calibration_utils.py
   - Added structured item metadata inference: answer_type, role, canonicalization, evidence_source, source_text, parent_official_item.
   - Added generic support for base_number, bit_vector, sequence, set, relation, table_entry, and diagram_ocr item types.
   - Added three primary routing risks:
     U_E = evidence quality risk,
     U_S = score stability risk,
     U_R = rubric adaptation risk.
   - A3WA now prefers R(x) = (U_E + U_S + U_R) / 3 and mu(x) = 1 - R(x).
   - Older detailed risk fields are retained only as diagnostics and backward-compatible signals.

2. main_pipeline.py
   - VARIANCE_OPT now records strict_cots, item_scores_history, item_category_history, item_variance, max_item_variance, avg_item_variance.
   - Hard samples for rubric refinement are now ranked by item-level variance first, then total-score variance.
   - This prevents a stable-but-wrong total score from hiding unstable or ambiguous rubric items.

3. step3_rrd_generator.py
   - Generated/refined rubrics are normalized through prepare_rubrics_for_calibration().
   - High-variance sample prompts now include item-level variance and item-level judgment history.
   - Added generalization constraints: only rubric_ambiguity and rubric_granularity can rewrite rubric content; extraction_failure, equivalent_representation_gap, and scoring_model_error can only add metadata, not change scoring meaning.

4. step4_vlm_grader.py
   - Formal grading output now records U_E, U_S, U_R, primary_risk, and primary_mu in risk_features.

5. evaluate.py
   - CSV comparison export now includes U_E, U_S, U_R, primary_risk, and primary_mu for route auditing.
```

Validation performed:

```powershell
python -m py_compile calibration_utils.py step4_vlm_grader.py main_pipeline.py step3_rrd_generator.py evaluate.py
python evaluate.py --compare --questions Q2 Q3 --compare-score-keys single avg selected 3wd --compare-output outputs/q2_q3_primary_risk_compare.csv
python scripts\replay_calibration.py --results-dir results_rrd_vlm --files results_rrd_vlm\Q2_grading_checkpoint.json results_rrd_vlm\Q3_grading_checkpoint.json
```

Replay result on old Q2/Q3 checkpoints: the code runs successfully, but improvement is small because old checkpoints were generated before this rubric optimization update. The expected effect should be evaluated by rerunning the pipeline so that new rubric metadata and item-level variance selection can take effect.

## 2026-06-14 Step3 Rubric Generation Hardening

Purpose: fix the rubric-generation stage after Q2-Q5 rubric regeneration exposed schema drift and unstable prompt behavior.

Implemented in `step3_rrd_generator.py`:

```text
1. Switched Step3 to the same coding-plan API style used by Step4:
   VLM model = glm-4.6v, text/rubric model = glm-5.1.

2. Added generated-rubric schema normalization:
   - Unsupported answer_type values are mapped to supported generic types.
   - Examples: string -> concept_keyword, hex_string/numeric_or_hex -> base_number,
     boolean_string -> judgement, graph_node/graph_edge -> relation.
   - evidence_source is limited to text/formula/table/diagram.
   - Every generated item is passed through prepare_rubrics_for_calibration().

3. Replaced the effective Step3 prompts with clean UTF-8 Chinese prompts:
   - Cold-start rubric generation now requires stable fields, traceability, supported answer_type,
     supported evidence_source, and total-score conservation.
   - Trial conflict detection only reports rubric granularity/ambiguity conflicts; OCR failure and
     student errors are not treated as rubric problems.
   - Variance-based refinement now classifies issues before editing. Only rubric_ambiguity and
     rubric_granularity can rewrite or split scoring items. extraction_failure, equivalent_representation_gap,
     and scoring_model_error can only add metadata and cannot change scoring meaning.

4. Refined and fallback rubrics are normalized before being returned, so regenerated Q2-Q5 rubrics should not
   introduce unsupported schema fields into Step4 scoring or calibration.
```

Recommended regeneration command after this update:

```powershell
python main_pipeline.py --mode VARIANCE_OPT --questions Q2 Q3 Q4 Q5 --sample-size 5 --progress-file results_rrd_vlm\progress_rubric_q2_q5_clean.json --force-rerun
```

Then rerun scoring/evaluation with a separate progress file:

```powershell
python main_pipeline.py --mode FULL --questions Q2 Q3 Q4 Q5 --progress-file results_rrd_vlm\progress_q2_q5_clean.json --force-rerun
python evaluate.py --compare --questions Q2 Q3 Q4 Q5 --compare-score-keys single avg selected 3wd --compare-output outputs/q2_q5_clean_compare.csv
```
