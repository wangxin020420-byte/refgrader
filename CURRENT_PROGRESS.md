# Current Progress

Last updated: 2026-06-01

## Current Goal

RefGrader is being optimized for a paper-oriented three-way decision (3WD) framework. The immediate goal is to replace the earlier engineering-style route triggers with a more theoretically explainable A3WA-inspired decision process:

```text
multi-source risk signals -> R(x) -> confidence mu(x) -> asymmetric alpha/beta -> POS/BND/NEG
```

The current code is ready for an overnight server experiment.

## Latest Implementation

The latest implementation introduces an A3WA-inspired 3WD layer.

Changed or added files:

```text
calibration_utils.py
scripts/replay_calibration.py
step4_vlm_grader.py
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

## BND Arbitration

BND samples still call the boundary arbitration agent, but final score is constrained by A3WA dynamic bounds:

```text
review_strength = (alpha - mu) / (alpha - beta)
delta = max(0.10 * MAX_SCORE, 0.30 * review_strength * MAX_SCORE)
upper_bound = avg_model_score + delta
lower_bound = 0
```

Important safety behavior:

```text
If strong over-score risk exists, upper_bound <= avg_model_score.
The lower bound is permissive, so cautious lowering by the agent is not erased.
```

## Latest Validation

Both local machine and lab server passed syntax and replay validation.

Server commands already run successfully:

```bash
python -m py_compile calibration_utils.py scripts/replay_calibration.py step4_vlm_grader.py

python scripts/replay_calibration.py --results-dir results_rrd_vlm --files results_rrd_vlm/Q5_grading_checkpoint.json results_rrd_vlm/Q6_grading_checkpoint.json results_rrd_vlm/Q7_grading_checkpoint.json
```

Latest replay output on server:

```text
Q5 current MAE=3.721 RMSE=4.823 QWK=0.637 Pearson=0.844 TAR2=47.1%
Q5 replay  MAE=3.721 RMSE=4.823 QWK=0.637 Pearson=0.844 TAR2=47.1%

Q6 current MAE=2.563 RMSE=3.384 QWK=0.775 Pearson=0.832 TAR2=57.4%
Q6 replay  MAE=2.563 RMSE=3.384 QWK=0.775 Pearson=0.832 TAR2=57.4%

Q7 current MAE=1.172 RMSE=2.126 QWK=0.716 Pearson=0.724 TAR2=76.1%
Q7 replay  MAE=1.169 RMSE=2.120 QWK=0.717 Pearson=0.726 TAR2=76.1%

GLOBAL current N=203 MAE=2.492 RMSE=3.622 QWK=0.697 Pearson=0.744 TAR2=60.1% Bias=-1.605 Over>2=16 Under>2=65
GLOBAL replay  N=203 MAE=2.491 RMSE=3.621 QWK=0.697 Pearson=0.745 TAR2=60.1% Bias=-1.611 Over>2=16 Under>2=65
```

Interpretation:

```text
The new A3WA logic does not obviously damage existing checkpoint results.
It is safe to run the formal overnight experiment.
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
python evaluate.py
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
}
```

These fields are intended for paper analysis and route audit.

## Known Issues

1. Q5 remains mainly limited by VLM/OCR extraction failures on visually complex handwritten encoding diagrams. The A3WA route does not directly solve this.
2. Q6 high-estimation cases may require reliable structured rubric metadata or stronger formula/dependency validation before hard caps can safely become more aggressive.
3. Automatically inferred rubric metadata currently defaults to safe/audit behavior. It does not enable strong hard-cap scoring unless metadata is explicit or trusted.
4. `m=0.5` is used in online grading for stability. For the paper, batch/offline experiments should evaluate adaptive `m*` via `optimize_a3wa_m()`.
5. README still contains older historical result tables. Treat the "Latest Progress: A3WA" section and this file as the current handoff state.

## Next Steps

1. Run the overnight formal experiment on the lab server with `./run_experiment.sh run`.
2. After completion, run `python evaluate.py`.
3. Inspect `a3wa_decision` and `risk_features.a3wa_*` in the new result JSON files.
4. Compare new formal results against replay and previous Q5/Q6/Q7 results.
5. For paper analysis, compute route-level audits:

```text
mean(mu_POS) > mean(mu_BND) > mean(mu_NEG)
mean(R_POS) < mean(R_BND) < mean(R_NEG)
BND agent correction magnitude vs mu
hard NEG reasons vs confidence-based NEG reasons
```

## Key Files

```text
README.md                         Long-term project overview and A3WA progress section.
CURRENT_PROGRESS.md               Short handoff file for new conversations.
calibration_utils.py              A3WA risk/confidence, thresholds, dynamic bounds, calibration utilities.
step4_vlm_grader.py               Formal grading pipeline using A3WA route decisions.
scripts/replay_calibration.py     Offline replay validation without API calls.
run_experiment.sh                 Server background experiment runner.
evaluate.py                       Final metrics.
results_rrd_vlm/*_grading_checkpoint.json  Existing checkpoint inputs for replay.
results_rrd_vlm/*_graded_results.json      Formal experiment outputs.
```

## Handoff Instruction

In a new conversation or after context loss, use:

```text
Please first read CURRENT_PROGRESS.md, then continue from the current project state.
```

