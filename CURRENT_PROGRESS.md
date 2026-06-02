# Current Progress

Last updated: 2026-06-02

## Current Goal

RefGrader is being optimized for a paper-oriented three-way decision (3WD) framework. The immediate goal is to replace the earlier engineering-style route triggers with a more theoretically explainable A3WA-inspired decision process:

```text
multi-source risk signals -> R(x) -> confidence mu(x) -> asymmetric alpha/beta -> POS/BND/NEG
```

The current code has moved beyond the first A3WA implementation: A3WA loss parameters and risk weights can now be calibrated offline, and BND arbitration uses a validation-aligned action policy rather than freely accepting the boundary agent's total score.

## Latest Implementation

The latest implementation introduces an A3WA-inspired 3WD layer plus offline cost-sensitive calibration and a BND action policy.

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
  accept the candidate only when the correction direction has supporting risk evidence
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

Validation already run:

```text
python -m py_compile step4_vlm_grader.py calibration_utils.py scripts/calibrate_a3wa.py scripts/replay_calibration.py evaluate.py
template placeholder replacement check: passed
replay_calibration with A3WA config: passed
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

1. Push the latest local changes, including `scripts/calibrate_a3wa.py` and `results_rrd_vlm/a3wa_calibration_config.json`.
2. Pull on the lab server.
3. Run syntax validation: `python -m py_compile calibration_utils.py scripts/calibrate_a3wa.py scripts/replay_calibration.py step4_vlm_grader.py evaluate.py`.
4. Run replay: `python scripts/replay_calibration.py --results-dir results_rrd_vlm --files results_rrd_vlm/Q5_graded_results.json results_rrd_vlm/Q6_graded_results.json results_rrd_vlm/Q7_graded_results.json`.
5. Run the formal experiment on the lab server with `./run_experiment.sh run`.
6. After completion, run `python evaluate.py --compare --questions Q5 Q6 Q7`.
7. Inspect `a3wa_decision`, `boundary_gate`, and `risk_features.boundary_gate_*` in the new result JSON files.
8. Compare new formal results against model_avg, old 3WD final, replay, and the calibrated validation simulation.
9. For paper analysis, compute route-level audits:

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
