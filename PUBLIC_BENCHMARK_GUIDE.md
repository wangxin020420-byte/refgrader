# Public Benchmark Guide

This module runs public numeric-scoring benchmarks without changing the
existing CSBench workflow.

## Directory ownership

```text
refgrader-main/
  benchmark_datasets/              adapter and prepared-data contract
  benchmark_datasets/specs/        adapter specification templates
  scripts/benchmarks/              prepare, audit, run, calibrate, evaluate
  data/public_benchmarks/          prepared portable snapshots
  results_runs/public_benchmarks/  local runtime results (Git-ignored)
  ocr_cache/public_benchmarks/     text fact caches (Git-ignored)

refgrader-public-datasets/         sibling repository recommended for raw data
  asap_sas/raw/                    original downloaded files
  asap_sas/spec/                   completed prompt/rubric specification
  mohler/raw/                      extracted official Mohler archive
  mohler/spec/                     deterministic split specification

refgrader-artifacts/
  public_benchmarks/               portable experiment history
```

Raw public data stays outside `refgrader-main`. Only the normalized prepared
snapshot, its source hashes, question metadata, split files, rubrics, and gold
labels enter `data/public_benchmarks`.

Prepared files contain portable relative paths. The adapter assigns exact
per-question quotas to train, calibration, validation, and test while
interleaving score strata. Preparation fails if any split with a positive
configured ratio would be empty.

## ASAP-SAS preparation

1. Copy `benchmark_datasets/specs/asap_sas.example.json` to the raw-data
   repository.
2. Fill every answer set in `questions` with the official prompt, reference
   answer, maximum score, and scoring rubric.
3. Confirm the source column mapping. The adapter prefers a resolved score; if
   no resolved score exists, it uses the mean of the available human-rater
   scores and records that policy in `manifest.json`.
4. Prepare the immutable snapshot:

```powershell
.\venv\Scripts\python.exe scripts\benchmarks\prepare_dataset.py asap_sas `
  --source "..\refgrader-public-datasets\asap_sas\raw\train.tsv" `
  --spec "..\refgrader-public-datasets\asap_sas\spec\asap_sas.json" `
  --output-dir "data\public_benchmarks\asap_sas_v1"
```

To replace an existing prepared snapshot intentionally, add `--force`. Do not
use `--force` after an experiment has started unless the source hash or adapter
spec is deliberately changing.

## Mohler preparation

The official Mohler archive already contains questions, instructor reference
answers, student responses, two human scores, and their normalized average.
The adapter excludes question IDs commented out in `data/docs/files`, uses the
official normalized `ave` score as the authoritative gold label, and retains
both original and normalized individual human scores for reliability analysis.
It does not recompute or replace `ave` from those auxiliary files. Exam rater
scores from assignments 11 and 12 are additionally normalized from 0-10 to the
dataset's common 0-5 scale.

The archive does not contain an official machine-readable fine-grained rubric.
The prepared snapshot therefore records a reference-based reconstructed
holistic rubric explicitly; it must not be described as an official rubric.

```powershell
Expand-Archive `
  -LiteralPath "..\refgrader-public-datasets\mohler\raw\ShortAnswerGrading_v2.0.zip" `
  -DestinationPath "..\refgrader-public-datasets\mohler\raw\ShortAnswerGrading_v2.0" `
  -Force

Copy-Item `
  ".\benchmark_datasets\specs\mohler.example.json" `
  "..\refgrader-public-datasets\mohler\spec\mohler.json" `
  -Force

.\venv\Scripts\python.exe scripts\benchmarks\prepare_dataset.py mohler `
  --source "..\refgrader-public-datasets\mohler\raw\ShortAnswerGrading_v2.0" `
  --spec "..\refgrader-public-datasets\mohler\spec\mohler.json" `
  --output-dir "data\public_benchmarks\mohler_v1"
```

The official archive is expected to prepare 81 included questions and 2,273
student responses. Treat any different count as a source or parser mismatch
and stop before running an experiment.

## Audit and dry run

```powershell
.\venv\Scripts\python.exe scripts\benchmarks\audit_dataset.py `
  --prepared-dir "data\public_benchmarks\asap_sas_v1"

.\venv\Scripts\python.exe scripts\benchmarks\run_benchmark.py `
  --prepared-dir "data\public_benchmarks\asap_sas_v1" `
  workflow `
  --tag "asap_sas_smoke" `
  --dry-run
```

The audit rejects duplicate answers, missing labels, out-of-range scores,
overlapping splits, incomplete split coverage, and rubric totals that do not
match the question score. Its content hash also covers the adapter
specification, every split file, and both initial and optimized rubrics.
Prepared datasets also contain one validated rubric-provenance manifest per
question under `rubrics/manifests/<group>/`; grading refuses stale contracts or
rubric files whose hashes do not match these manifests.

Before a full run, execute one answer set with a small sample limit:

```powershell
.\venv\Scripts\python.exe scripts\benchmarks\run_benchmark.py `
  --prepared-dir "data\public_benchmarks\asap_sas_v1" `
  workflow ASAP_SAS_1 `
  --tag "asap_sas_smoke_001" `
  --score-calibration `
  --limit 5
```

`--limit` is only for pipeline smoke tests. Do not report or publish those
metrics as a formal benchmark result.

## Formal workflow

This one command grades validation, calibrates A3WA, grades test, and evaluates
`single`, `avg`, `selected`, `3WD-Core`, and final `3WD`:

```powershell
.\venv\Scripts\python.exe scripts\benchmarks\run_benchmark.py `
  --prepared-dir "data\public_benchmarks\asap_sas_v1" `
  workflow `
  --tag "asap_sas_glm52_off_001" `
  --score-calibration
```

Use a new tag for a new formal experiment. Reusing the same tag resumes the
same result directories unless `--force` is explicitly supplied.

## Separate stages

Validation:

```powershell
.\venv\Scripts\python.exe scripts\benchmarks\run_benchmark.py `
  --prepared-dir "data\public_benchmarks\asap_sas_v1" `
  grade --split validation --run-id "asap_sas_001_validation"
```

Calibration:

```powershell
.\venv\Scripts\python.exe scripts\benchmarks\run_benchmark.py `
  --prepared-dir "data\public_benchmarks\asap_sas_v1" `
  calibrate --validation-run-id "asap_sas_001_validation" `
  --score-calibration
```

Test with the generated configuration:

```powershell
.\venv\Scripts\python.exe scripts\benchmarks\run_benchmark.py `
  --prepared-dir "data\public_benchmarks\asap_sas_v1" `
  grade --split test --run-id "asap_sas_001_test" `
  --a3wa-config "results_runs\public_benchmarks\asap_sas_v1\calibration\asap_sas_001_validation_a3wa.json"
```

## Portable result publishing

```powershell
.\venv\Scripts\python.exe scripts\benchmarks\run_benchmark.py `
  --prepared-dir "data\public_benchmarks\asap_sas_v1" `
  publish --run-id "asap_sas_001_test" `
  --artifacts-repo "..\refgrader-artifacts"
```

The command copies, but does not commit or push, the exact run under
`refgrader-artifacts/public_benchmarks/<dataset_id>/runs/<run_id>`.
Each test run contains its exact A3WA configuration and a portable snapshot of
the question, split, and rubric inputs used by that run.

To archive the validation evidence from a full workflow as well, publish its
run ID separately:

```powershell
.\venv\Scripts\python.exe scripts\benchmarks\run_benchmark.py `
  --prepared-dir "data\public_benchmarks\asap_sas_v1" `
  publish --run-id "asap_sas_glm52_off_001_validation" `
  --artifacts-repo "..\refgrader-artifacts"
```

## Experimental controls

- Never use test labels for rubric generation, A3WA calibration, threshold
  selection, residual correction, or model selection.
- Keep `3WD-Core` separate from final `3WD`; the latter may include validation
  residual calibration.
- Compare models on the same prepared manifest hash and the same split files.
- Record a new dataset ID when changing source rows, label policy, prompt
  metadata, rubrics, or split seed.
- Public text datasets use `text_only`; they must not invoke PaddleOCR or the
  visual model.
