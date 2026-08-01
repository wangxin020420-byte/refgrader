# CSBench Sample Quality Control

This directory stores review metadata only. Raw answer records, teacher
scores, split files, and student images remain immutable in their existing
locations.

## Directory layout

- `reports/`: generated teacher-label review candidates. A disagreement is
  only a review signal, not proof that the teacher label is wrong.
- `reviews/`: human decisions in JSONL form.
- `policies/active_sample_policy.json`: optional active policy. When this file
  is absent, the full pipeline preserves the original raw-data behavior.

## Human decision values

- `confirmed_noise`: exclude the sample from optimization, validation,
  calibration, formal test evaluation, and resumed runs.
- `corrected`: retain the sample and overlay the reviewed teacher score.
- `retained_hard_case`: retain the original label as a valid difficult sample.
- `ambiguous`: retain the sample until a later review resolves it.

Physical files are never moved. The active policy is applied centrally at
read time, recorded in run signatures, copied into artifacts, and checked
when an experiment is resumed or restored on another device.

Complete `--split all` audit checkpoints are versioned separately in
`refgrader-artifacts/csbench/<question_id>/audit_runs/<run_id>`. They can be
restored on another device with `restore_csbench_artifacts.py --stage audit`.
Generated candidate reports and human decisions remain in this directory and
are synchronized through the `refgrader-main` repository.

## Preferred review interface

Use the local review interface instead of editing candidate CSV files by hand:

```powershell
.\venv\Scripts\python.exe scripts\review_teacher_labels.py `
  --report-run teacher_label_audit_all43_20260727_160210
```

The interface displays the student image, teacher score, model-average score,
3WD-Core score, risk fields, and the preliminary screening result together.
Review decisions are saved to `reviews/<report_run>_decisions.jsonl`. Saving a
decision does not move or delete any raw file and does not affect experiments
until the reviewer explicitly selects **Activate review policy**.

See `TEACHER_LABEL_REVIEW_GUIDE.md` at the repository root for the complete
Chinese beginner guide, filtering workflow, decision meanings, resume
behavior, and multi-device synchronization steps.
