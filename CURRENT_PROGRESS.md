# Current Progress

Last updated: 2026-06-01

## Current Goal

RefGrader is being optimized for a paper-oriented three-way decision (3WD) framework. The immediate goal is to replace the earlier engineering-style route triggers with a more theoretically explainable A3WA-inspired decision process:

```text
multi-source risk signals -> R(x) -> confidence mu(x) -> asymmetric alpha/beta -> POS/BND/NEG
```

The current code has moved one step beyond the first A3WA implementation: BND arbitration now uses a direction-aware no-harm gate so that boundary-agent scores cannot freely damage a strong model-average baseline.

## Latest Implementation

The latest implementation introduces an A3WA-inspired 3WD layer plus a direction-aware BND correction gate.

Changed or added files:

```text
calibration_utils.py
scripts/replay_calibration.py
step4_vlm_grader.py
evaluate.py
README.md
CURRENT_PROGRESS.md
```

Main logic now used by `step4_vlm_grader.py`:

```text
Stage 1: VLM extracts objective facts from the answer image.
Stage 2: LLM scores the same facts three times.
Post calibration: generic calibration and A3WA confidence are computed.
3WD route:
  hard NEG guard
  else compute R(x), mu(x), alpha, beta
  if mu >= alpha -> POS
  elif mu <= beta -> NEG
  else -> BND
BND action:
  call boundary arbitration agent for a candidate score
  apply direction-aware bounds and no-harm gate
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

## BND Arbitration And No-Harm Gate

BND samples still call the boundary arbitration agent, but the agent is no longer allowed to freely determine the final score. The current logic is:

```text
baseline_score = model_avg_score
candidate_score = boundary_agent_score

if candidate_score < baseline_score:
  accept lowering only when over_score_risk has evidence
elif candidate_score > baseline_score:
  accept raising only when under_score_risk has evidence and no strong over_score_risk exists
else:
  keep baseline_score
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
The default fallback is model_avg_score when no reliable directional evidence exists.
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

Latest checkpoint replay after adding the no-harm gate:

```text
Q5 current MAE=3.510 RMSE=4.692 QWK=0.653 Pearson=0.860 TAR2=48.5%
Q5 replay  MAE=3.401 RMSE=4.594 QWK=0.658 Pearson=0.847 TAR2=52.9%

Q6 current MAE=2.501 RMSE=3.308 QWK=0.787 Pearson=0.838 TAR2=55.9%
Q6 replay  MAE=2.407 RMSE=3.044 QWK=0.819 Pearson=0.866 TAR2=55.9%

Q7 current MAE=0.979 RMSE=1.815 QWK=0.798 Pearson=0.799 TAR2=86.4%
Q7 replay  MAE=1.048 RMSE=1.898 QWK=0.775 Pearson=0.774 TAR2=81.8%

GLOBAL current N=202 MAE=2.344 RMSE=3.489 QWK=0.720 Pearson=0.766 TAR2=63.4% Bias=-1.595
GLOBAL replay  N=202 MAE=2.298 RMSE=3.377 QWK=0.734 Pearson=0.773 TAR2=63.4% Bias=-1.452
```

Interpretation:

```text
The gate improves Q5/Q6 checkpoint replay, but suppresses some earlier Q7 BND gains.
The formal experiment should be rerun because newly generated boundary-agent outputs may differ from replayed old scores.
```

Offline simulation on the latest formal Q5/Q6/Q7 result files showed the intended effect:

```text
GLOBAL model_avg MAE = 2.136
GLOBAL old final MAE = 2.168
GLOBAL new-gate simulated MAE = 2.120

Q5 model_avg 3.103 -> new gate 3.069
Q6 model_avg 2.384 -> new gate 2.373
Q7 model_avg 0.977 -> new gate 0.972
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
  "action": "keep_baseline / accept_lower / accept_raise / reject_lower / reject_raise",
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

## Known Issues

1. Q5 remains mainly limited by VLM/OCR extraction failures on visually complex handwritten encoding diagrams. The A3WA route does not directly solve this.
2. Q6 high-estimation cases may require reliable structured rubric metadata or stronger formula/dependency validation before hard caps can safely become more aggressive.
3. Automatically inferred rubric metadata currently defaults to safe/audit behavior. It does not enable strong hard-cap scoring unless metadata is explicit or trusted.
4. `m=0.5` is used in online grading for stability. For the paper, batch/offline experiments should evaluate adaptive `m*` via `optimize_a3wa_m()`.
5. The no-harm gate is intentionally conservative. It protects Q5/Q6 from harmful BND lowering, but may reduce some Q7 gains when the old boundary agent happened to be correct.
6. README still contains older historical result tables. Treat the "Latest Progress: A3WA" section and this file as the current handoff state.

## Next Steps

1. Push the latest local changes to the remote repository and pull them on the lab server.
2. Run the formal experiment on the lab server with `./run_experiment.sh run`.
3. After completion, run `python evaluate.py --compare --questions Q5 Q6 Q7`.
4. Inspect `a3wa_decision`, `boundary_gate`, and `risk_features.boundary_gate_*` in the new result JSON files.
5. Compare new formal results against model_avg, old 3WD final, and replay.
6. For paper analysis, compute route-level audits:

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
scripts/replay_calibration.py     Offline replay validation using the same A3WA/BND gate logic.
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
