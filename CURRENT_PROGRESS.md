# Current Progress

Last updated: 2026-06-03

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
