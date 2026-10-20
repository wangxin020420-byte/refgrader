# RefGrader Agent Execution Rules

These rules are persistent project instructions. Apply them at the start of
every task, including after context compaction or a resumed conversation.

## 0. Quality Is The Constraint

- Correctness, completeness, data integrity, and reproducibility take priority
  over speed. Never omit necessary evidence, tests, or affected modules merely
  to stay within an investigation budget.
- The limits below govern the initial diagnostic pass, not the total work. They
  are intended to prevent aimless exploration while preserving full analysis
  when the evidence shows that broader scope is necessary.
- Expand the investigation when a change crosses module contracts, when the
  initial evidence is contradictory or insufficient, when data loss or
  experiment validity is at risk, or when the user explicitly requests a
  whole-project review.
- Before expanding, name the unresolved question and the specific additional
  files or tests needed. Expansion must be evidence-driven, not exploratory.
- Stop only when the requested outcome is supported by sufficient evidence,
  not when an arbitrary file or tool-call count is reached.

## 1. Default To The Fast Path

- Answer direct questions directly. Do not inspect the repository when the
  answer is already established by the supplied command output or file.
- For diagnostic tasks, identify the exact run ID, error, metric, or module
  first. Inspect only the files needed to test that hypothesis.
- Do not broaden a task into a whole-project review, documentation update,
  refactor, or historical comparison unless the user requests it or it is
  necessary to complete the task correctly.
- Do not browse the web for repository-local facts.

## 2. Initial Investigation Budget

- Initial diagnostic pass: target at most 3 targeted searches and 5 relevant
  files. Expand when required by the quality rules above.
- Prefer `rg` with exact symbols, error text, run IDs, or filenames.
- Do not recursively read `results_runs`, `refgrader-artifacts`, logs, OCR
  caches, images, or all Markdown files. Select the exact run and artifact
  first.
- Reuse facts already established in the conversation. Do not re-read the same
  files unless the working tree or relevant run changed.
- If the initial budget is insufficient, state the evidence gap and the exact
  additional scope before expanding.

## 3. Code Change Discipline

- Define one primary objective and its acceptance criteria before editing.
- Keep the first patch limited to the smallest ownership boundary that can fix
  the issue.
- Do not combine model configuration, rubric semantics, A3WA logic, artifact
  synchronization, workflow orchestration, and documentation in one change
  unless the requested behavior crosses those boundaries.
- Run targeted tests first. Run one relevant integration or dry-run check only
  after targeted tests pass.
- Do not run a full experiment to detect argument, path, encoding, import, or
  state-machine errors that a dry run or unit test can detect.
- Never claim a metric improvement from code inspection. Distinguish code
  correctness from empirical effectiveness.

## 4. Experiment Workflow Rules

- Never rerun a completed rubric, validation, calibration, or test stage unless
  its inputs changed or the user explicitly requests a rerun.
- Preserve run IDs for resume operations. Do not create a new result directory
  when continuing the same logical run.
- For unattended experiments, analytical deployment gates should label results
  as formal or experimental and record warnings. They should not terminate the
  remaining overnight stages unless continuing would corrupt data, mix
  incompatible inputs, or violate an explicit strict-mode request.
- Treat `refgrader-main` as code and active configuration, and
  `refgrader-artifacts` as portable experiment history. Do not mix unrelated
  runs in one commit.
- Before giving a long execution command, check parser compatibility, mutually
  exclusive flags, paths, run IDs, model contract, and resume behavior.

## 5. Communication

- Put the conclusion first, then the minimum evidence needed to support it.
- Match the user's requested scope and language. Avoid unrelated background,
  speculative branches, and repeated explanations.
- When an error is visible in supplied output, explain that error before
  proposing additional checks.
- Clearly label facts, inferences, and unverified hypotheses.
- If work is expanding beyond the initial scope, tell the user immediately.

## 6. Final Self-Check

Before every final response, verify:

1. Did I answer the newest request directly?
2. Did I inspect only relevant files and runs?
3. Did I avoid rerunning completed work?
4. Are commands minimal, parser-compatible, and tied to the correct run ID?
5. Did I separate verified facts from inference?
6. Did I avoid promising empirical improvement without experiment results?

## 7. Zhipu Endpoint Requirement

- User requires Coding Plan only for every Zhipu model call, including vision,
  rubric generation, scoring, extraction, and benchmark adapters. For OpenAI
  Chat Completions use exactly
  `https://open.bigmodel.cn/api/coding/paas/v4` (a final slash is equivalent).
- Never use the SDK default endpoint, the ordinary `/api/paas/v4` endpoint,
  a third-party proxy, or a fallback to balance billing for a Zhipu call.
- Before a new live Zhipu test or unattended experiment, verify the final
  request URL offline for both text and vision paths. Never treat key presence
  or a successful model response as evidence of the request destination.
- On 2026-10-19 the user confirmed that the reported balance deductions were
  caused by another service, not this experiment. The user explicitly removed
  the billing-investigation hold and requires correct endpoint routing only.
  Do not require official support confirmation, a billing receipt, or another
  billing approval before otherwise authorized experiments. Do not launch a
  paid test or resume an experiment merely to verify the URL.
- Preserve run IDs and saved outputs. Do not rewrite input fingerprints or
  silently change models to bypass resume validation.
