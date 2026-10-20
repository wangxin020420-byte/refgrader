"""One-shot, resumable local GLM-5.2 experiment using a frozen review snapshot.

--check is offline. --background launches only when the user runs this command.
No experiment code, raw labels, or live review decisions are rewritten here.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sample_quality import SampleQualityPolicy
from scripts.compile_sample_quality_policy import (
    build_policy, load_max_scores, load_valid_answers, normalized_decision,
)
from scripts.run_csbench import batch_slug, build_contexts, build_parser as csbench_parser

PREPARED = ROOT / "data/csbench"
DEFAULT_DECISIONS = PREPARED / "quality_control/reviews/teacher_label_audit_all43_20260727_160210_decisions.jsonl"
BASE = ROOT / "results_runs/reviewed_csbench45_glm52"
LATEST = BASE / "latest.json"
MODEL_ARGS = ["--text-provider", "glm5", "--thinking-mode", "disabled", "--vlm-provider", "glm4v"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_policy(prepared, raw, policy_id, source):
    valid, maxima = load_valid_answers(prepared / "answer_metadata.jsonl"), load_max_scores(prepared / "exam_database.json")
    rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    if not rows:
        raise ValueError("The review file has no saved decisions.")
    normalized = [normalized_decision(row, line_number=i, valid_answers=valid, max_scores=maxima) for i, row in enumerate(rows, 1)]
    policy = build_policy(normalized, policy_id=policy_id, source=source)
    policy["dataset_sha256"] = {name: sha(prepared / name) for name in ("teacher_scores.json", "answer_metadata.jsonl")}
    return policy, normalized


def selection_report(prepared, payload, normalized, expected_questions=45):
    exam = read(prepared / "exam_database.json")
    questions = sorted(q["question_id"] for q in exam)
    if len(questions) != expected_questions or len(set(questions)) != len(questions):
        raise ValueError(f"Expected {expected_questions} distinct questions, found {len(questions)}")
    policy_ids = {q: set(v) for q, v in payload["excluded"].items()}
    metadata = [json.loads(line) for line in (prepared / "answer_metadata.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    all_ids = {(r["question_id"], r["answer_id"]) for r in metadata}
    if len(all_ids) != len(metadata) or len({r["answer_id"] for r in metadata}) != len(metadata):
        raise ValueError("Duplicate answer identities in answer_metadata.jsonl")
    if {qid for qid, _ in all_ids} != set(questions):
        raise ValueError("Metadata question coverage differs from exam_database.json")
    totals = {s: {"raw": 0, "excluded": 0, "effective": 0, "corrected": 0} for s in ("calibration", "validation", "test")}
    per_question, warnings = {}, []
    corrected = {(r["question_id"], r["answer_id"]) for r in normalized if r["decision"] == "corrected"}
    for qid in questions:
        split = read(prepared / "splits/by_question" / f"{qid}.json")
        used = set()
        per_question[qid] = {}
        for name in totals:
            values = split[name]
            ids = set(values)
            if len(ids) != len(values) or used & ids:
                raise ValueError(f"Duplicate or overlapping answer split: {qid}/{name}")
            used |= ids
            excluded = ids & policy_ids.get(qid, set())
            n = len(ids - excluded)
            if n == 0:
                raise ValueError(f"No usable {name} answers for {qid}; refusing cross-split fallback.")
            if name == "calibration" and n < 5:
                warnings.append(f"{qid}: only {n} rubric-calibration answers remain; use available answers, never borrow test answers.")
            counts = {"raw": len(ids), "excluded": len(excluded), "effective": n, "corrected": sum((qid, aid) in corrected for aid in ids)}
            per_question[qid][name] = counts
            for key in counts:
                totals[name][key] += counts[key]
        if {(qid, aid) for aid in used} != {p for p in all_ids if p[0] == qid}:
            raise ValueError(f"Split coverage mismatch for {qid}")
    missing_reason = sum(r["decision"] in {"confirmed_noise", "corrected"} and not r["reason_code"] and not r["review_note"] for r in normalized)
    if missing_reason:
        warnings.append(f"{missing_reason} corrections/exclusions lack a written reason; retain this limitation in reporting.")
    return {"questions": questions, "raw_answers": len(metadata), "reviewed_answers": len(normalized), "not_reviewed_answers": len(metadata)-len(normalized), "all_answers_reviewed": len(metadata)==len(normalized), "splits": totals, "per_question": per_question, "warnings": warnings, "selection_rule": payload["selection_rule"]}


def fingerprints(prepared):
    files = [prepared / n for n in ("exam_database.json", "teacher_scores.json", "answer_metadata.jsonl", "manifest.json")]
    for folder in ("splits", "rubrics/initial", "rubrics/source", "student_images", "reference_images"):
        files.extend(p for p in (prepared / folder).rglob("*") if p.is_file())
    for name in ("main_pipeline.py", "step3_rrd_generator.py", "step4_vlm_grader.py", "calibration_utils.py", "model_runtime.py", "rubric_semantics.py", "sample_quality.py", "portable_hash.py", "requirements-main.lock.txt", "scripts/run_csbench.py", "scripts/calibrate_a3wa.py", "scripts/run_reviewed_full_workflow.py", "scripts/compile_sample_quality_policy.py"):
        files.append(ROOT / name)
    for folder in ("prompts", "canonicalizers", "ocr"):
        files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return {str(p.resolve()): sha(p) for p in sorted(set(files)) if p.is_file()}


def frozen_environment(policy_path):
    env = os.environ.copy()
    for name in ("REFGRADER_TEXT_TEMPERATURE_OVERRIDE", "REFGRADER_POST_SUCCESS_CMD", "REFGRADER_ARTIFACT_RUN_ID"):
        env.pop(name, None)
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "REFGRADER_SAMPLE_POLICY": str(policy_path), "REFGRADER_SAMPLE_POLICY_MODE": "active", "REFGRADER_TEXT_MODEL_PROVIDER": "glm5", "REFGRADER_TEXT_THINKING": "disabled", "REFGRADER_VLM_MODEL_PROVIDER": "glm4v", "REFGRADER_OCR_DEVICE": "cpu", "A3WA_CALIBRATION_CONFIG": ""})
    return env


def build_stages(state):
    prepared, job, artifacts = map(Path, (state["prepared"], state["job"], state["artifacts"]))
    contexts = build_contexts(str(prepared), state["questions"])
    slug, qids, rid = batch_slug(contexts), state["questions"], state["run_id"]
    validation = ROOT / f"results_runs/csbench_{slug}_validation/runs/{rid}_validation"
    test = ROOT / f"results_runs/csbench_{slug}_full/runs/{rid}_test"
    config = job / "a3wa_config.json"
    py = [sys.executable, "-u"]
    core = [*py, str(ROOT / "scripts/run_csbench.py"), "--prepared-dir", str(prepared)]
    def stage(name, command, attempts=1, proof=()):
        return {"name": name, "command": command, "attempts": attempts, "proof": [str(p) for p in proof]}
    stages = [
        stage("audit_snapshot", [*py, str(ROOT / "scripts/audit_csbench_snapshot.py"), "--prepared-dir", str(prepared)]),
        stage("audit_splits", [*py, str(ROOT / "scripts/audit_question_splits.py"), "--prepared-dir", str(prepared)]),
        stage("api_smoke", [*py, "-c", "from step4_vlm_grader import call_text_model; print(call_text_model([{'role':'user','content':'Reply with exactly OK.'}],temperature=0,timeout=60))"], 3),
    ]
    for ctx in contexts:
        work = job / "rubric_work" / ctx.question_id
        # Private progress directory: never let --resume mistake an old active
        # manifest for evidence that this new reviewed-data optimization ran.
        command = [*py, str(ROOT / "main_pipeline.py"), "--mode", "VARIANCE_OPT", "--questions", ctx.question_id,
                   "--database-path", str(ctx.database), "--answer-metadata", str(ctx.answer_metadata),
                   "--initial-rubric-dir", str(ctx.initial_dir), "--rubric-dir", str(ctx.optimized_dir),
                   "--teacher-db", str(ctx.teacher_db), "--sample-size", "5", "--resume-optimization",
                   "--extraction-backend", "csbench_hybrid", "--ocr-cache-dir", str(ROOT / "ocr_cache/csbench"),
                   "--results-dir", str(work), "--progress-file", str(work / "progress.json")]
        stages.append(stage("rubric_"+ctx.question_id, command, 4, [ctx.optimized_rubric, ctx.optimization_manifest]))
    stages += [stage("activate_rubrics", [*core, "activate", *qids])]
    common_publish = ["--artifacts-repo", str(artifacts), "--include-facts"]
    stages += [stage("publish_rubrics", [*core, "publish", *qids, "--stage", "rubric", "--run-id", rid+"_rubric", *common_publish], 3)]
    stages += [stage("grade_validation", [*core, "grade", *qids, "--split", "validation", "--run-id", rid+"_validation", "--require-complete", "--no-artifacts", "--device", "cpu", *MODEL_ARGS], state["grade_attempts"], [validation / f"{q}_grading_checkpoint.json" for q in qids])]
    stages += [stage("publish_validation", [*core, "publish", *qids, "--stage", "validation", "--run-id", rid+"_validation", "--results-dir", str(validation), *common_publish], 3)]
    stages += [stage("calibrate", [*core, "calibrate", *qids, "--source-run-id", rid+"_validation", "--output", str(config), "--score-calibration", "--diagnostics-only", *MODEL_ARGS], 1, [config])]
    stages += [stage("activate_calibration", [*core, "activate", *qids, "--a3wa-config", str(config), "--source-validation-run-id", rid+"_validation", "--allow-experimental-a3wa"])]
    applied = [part for q in qids for part in ("--a3wa-config-question", q)]
    stages += [stage("publish_calibration", [*core, "publish", *qids, "--stage", "validation", "--run-id", rid+"_calibrated", "--results-dir", str(validation), "--a3wa-config", str(config), *applied, *common_publish], 3)]
    stages += [stage("grade_test", [*core, "grade", *qids, "--split", "test", "--run-id", rid+"_test", "--a3wa-config", str(config), "--allow-experimental-a3wa", "--require-complete", "--no-artifacts", "--device", "cpu", *MODEL_ARGS], state["grade_attempts"], [test / f"{q}_grading_checkpoint.json" for q in qids])]
    stages += [stage("evaluate_test", [*core, "evaluate", *qids, "--run-id", rid+"_test", "--export", "--require-complete", "--no-artifacts"], 2)]
    stages += [stage("publish_test", [*core, "publish", *qids, "--stage", "full", "--run-id", rid+"_test", "--results-dir", str(test), "--a3wa-config", str(config), *applied, *common_publish], 3)]
    return stages, {"validation": str(validation), "test": str(test), "config": str(config)}


def verify_proof(proof):
    for filename, expected in proof.items():
        if not Path(filename).is_file() or sha(filename) != expected:
            raise RuntimeError(f"Completed stage output changed: {filename}. Do not mix runs.")


def verify_stage_proof(state, name, proof):
    """Recognize publish's lossless manifest path normalization, not content edits."""
    prepared = Path(state["prepared"]).resolve()

    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            for marker, root in (("${PREPARED_CSBENCH_ROOT}", prepared), ("${REFGRADER_ROOT}", ROOT)):
                if value == marker or value.startswith(marker + "/"):
                    return str(root / value[len(marker):].lstrip("/"))
        return value

    for filename, expected in proof.items():
        path = Path(filename)
        if path.is_file() and sha(path) == expected:
            continue
        eligible = (name.startswith("rubric_") and path.is_file()
                    and path.resolve().is_relative_to(prepared / "rubrics/manifests")
                    and path.name == name.removeprefix("rubric_") + "_optimization.json")
        if eligible:
            payload = read(path)
            if (payload.get("path_format") == "refgrader_placeholders_v1"
                    and payload.get("question_id") == name.removeprefix("rubric_")):
                payload.pop("path_format")
                rendered = json.dumps(expand(payload), ensure_ascii=False, indent=4)
                # main_pipeline's original JSON writer uses native newlines.
                if any(hashlib.sha256(rendered.replace("\n", newline).encode("utf-8")).hexdigest() == expected
                       for newline in ("\n", "\r\n")):
                    continue
        verify_proof({filename: expected})


def validate_plan(stages):
    parser = csbench_parser()
    for item in stages:
        command = item["command"]
        if str(ROOT / "scripts/run_csbench.py") in command:
            parser.parse_args(command[command.index(str(ROOT / "scripts/run_csbench.py")) + 1:])


def backup_products(prepared, job, questions):
    dest = job / "previous_products"
    paths = [prepared / "rubrics/active_rubric_set.json", prepared / "calibration/active_a3wa_config.json", prepared / "quality_control/policies/active_sample_policy.json"]
    for ctx in build_contexts(str(prepared), questions):
        paths += [ctx.optimized_rubric, ctx.optimization_manifest]
    manifest = {}
    for path in paths:
        if path.is_file():
            relative = path.relative_to(prepared)
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            manifest[relative.as_posix()] = sha(target)
            if sha(path) != manifest[relative.as_posix()]:
                raise RuntimeError("A product changed during backup: " + str(path))
    write(dest / "manifest.json", manifest)


def prepare_job(args):
    if args.new_run and args.run_id:
        raise ValueError("Use --new-run or --run-id, not both.")
    if args.run_id:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id):
            raise ValueError("Invalid run ID")
        job = BASE / args.run_id
        if not (job / "workflow.json").is_file():
            raise FileNotFoundError("Cannot resume unknown run ID: " + args.run_id)
        return read(job / "workflow.json")
    if LATEST.is_file() and not args.new_run:
        return read(Path(read(LATEST)["job"]) / "workflow.json")
    rid = "reviewed45_glm52_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    job = BASE / rid
    raw = args.decisions.read_bytes()
    payload, normalized = make_policy(PREPARED, raw, rid, "review_decisions.jsonl")
    report = selection_report(PREPARED, payload, normalized)
    job.mkdir(parents=True, exist_ok=False)
    (job / "review_decisions.jsonl").write_bytes(raw)
    write(job / "sample_policy.json", payload)
    write(job / "selection.json", report)
    inputs = fingerprints(PREPARED)
    write(job / "input_sha256.json", inputs)
    backup_products(PREPARED, job, report["questions"])
    state = {"schema_version": 1, "run_id": rid, "job": str(job), "prepared": str(PREPARED), "artifacts": str(args.artifacts_repo), "decisions_source": str(args.decisions), "review_sha256": hashlib.sha256(raw).hexdigest(), "policy_sha256": sha(job / "sample_policy.json"), "selection_sha256": sha(job / "selection.json"), "input_manifest_sha256": sha(job / "input_sha256.json"), "created_at": datetime.now().isoformat(timespec="seconds"), "questions": report["questions"], "grade_attempts": args.grade_attempts, "status": "prepared", "stage": "prepared", "completed": {}, "attempts": {}, "warnings": report["warnings"], "model": {"text_provider": "glm5", "text_model": "glm-5.2", "thinking": "disabled", "vlm_model": "glm-4.6v"}}
    _, state["paths"] = build_stages(state)
    write(job / "workflow.json", state)
    write(LATEST, {"job": str(job), "run_id": rid})
    return state


def verified_code_repairs(state, original_inputs):
    """Accept only explicitly recorded code-only repairs; keep original hashes."""
    job = Path(state["job"])
    path = job / "resume_code_repair.json"
    expected = state.get("resume_code_repair_sha256")
    if not path.exists() and not expected:
        return {}
    if not path.is_file() or not expected or sha(path) != expected:
        raise RuntimeError("Resume code-repair record is missing or changed")
    repair = read(path)
    if (repair.get("schema_version") != 1 or repair.get("run_id") != state["run_id"]
            or repair.get("original_input_manifest_sha256") != state["input_manifest_sha256"]):
        raise RuntimeError("Resume code-repair record belongs to different frozen inputs")
    allowed = {str((ROOT / name).resolve()) for name in
               ("scripts/run_csbench.py", "scripts/run_reviewed_full_workflow.py")}
    overrides = {}
    for item in repair.get("files", []):
        source = str(Path(item["path"]).resolve())
        backup = (job / item["before_copy"]).resolve()
        if (source not in allowed or source in overrides or source not in original_inputs
                or not backup.is_relative_to((job / "code_repairs").resolve())
                or item["before_sha256"] != original_inputs[source]
                or not backup.is_file() or sha(backup) != item["before_sha256"]
                or not Path(source).is_file() or sha(source) != item["after_sha256"]):
            raise RuntimeError("Resume code repair is invalid or outside its code-only scope")
        overrides[source] = item["after_sha256"]
    if not overrides:
        raise RuntimeError("Resume code repair has no verified changes")
    return overrides


def assert_inputs(state):
    job = Path(state["job"])
    for filename, expected in [("review_decisions.jsonl", state["review_sha256"]), ("sample_policy.json", state["policy_sha256"]), ("selection.json", state["selection_sha256"]), ("input_sha256.json", state["input_manifest_sha256"])]:
        if sha(job / filename) != expected:
            raise RuntimeError("Frozen workflow input changed: " + filename)
    original_inputs = read(job / "input_sha256.json")
    overrides = verified_code_repairs(state, original_inputs)
    verify_proof({name: overrides.get(name, expected) for name, expected in original_inputs.items()})


def preflight(args, state=None):
    artifacts = Path(state["artifacts"]) if state else args.artifacts_repo
    if not (artifacts / ".git").exists():
        raise FileNotFoundError("Sibling artifact repository missing: " + str(artifacts))
    from ocr.backend import default_ocr_python
    ocr_python = Path(os.getenv("REFGRADER_OCR_PYTHON", str(default_ocr_python())))
    if not ocr_python.is_file():
        raise FileNotFoundError("OCR environment missing: " + str(ocr_python))
    check = subprocess.run([str(ocr_python), "-c", "import importlib.util; assert all(importlib.util.find_spec(x) for x in ['paddle','paddleocr']); print('OCR_MODULES_FOUND')"], capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError("OCR dependencies are not installed in " + str(ocr_python))
    if state:
        assert_inputs(state)
        payload, report = read(Path(state["job"]) / "sample_policy.json"), read(Path(state["job"]) / "selection.json")
        live = Path(state["decisions_source"])
        if not live.is_file() or sha(live) != state["review_sha256"]:
            print("NOTE: live reviews have changed; resuming the saved snapshot. Use --new-run for a new data version.", flush=True)
        plan_state = state
    else:
        raw = args.decisions.read_bytes()
        payload, normalized = make_policy(PREPARED, raw, "offline_preview", "offline_preview")
        report = selection_report(PREPARED, payload, normalized)
        plan_state = {"prepared": str(PREPARED), "job": str(BASE / "PREVIEW"), "artifacts": str(artifacts), "questions": report["questions"], "run_id": "PREVIEW", "grade_attempts": args.grade_attempts}
    for ctx in build_contexts(str(PREPARED), report["questions"]):
        ctx.validate_initial(5)
    with tempfile.TemporaryDirectory(prefix="refgrader_policy_check_") as tmp:
        path = Path(tmp) / "policy.json"
        write(path, payload)
        policy = SampleQualityPolicy.load(PREPARED, explicit_path=path, force_raw=False)
        if policy.mode != "active":
            raise RuntimeError("Reviewed policy was not loaded")
    stages, _ = build_stages(plan_state)
    validate_plan(stages)
    print(json.dumps({"offline_preflight": "passed", "model": "GLM-5.2; thinking disabled; VLM GLM-4.6v", "questions": len(report["questions"]), "reviewed_answers": report["reviewed_answers"], "not_reviewed_answers": report["not_reviewed_answers"], "splits": report["splits"], "warnings": report["warnings"], "api_key_in_environment": bool(os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY")), "api_calls_made": 0}, ensure_ascii=False, indent=2), flush=True)
    return stages


@contextlib.contextmanager
def workflow_lock():
    BASE.mkdir(parents=True, exist_ok=True)
    path = BASE / "workflow.lock"
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("A reviewed full workflow is already running.") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


@contextlib.contextmanager
def keep_awake():
    fn = None
    if os.name == "nt":
        fn = ctypes.windll.kernel32.SetThreadExecutionState
        fn.argtypes, fn.restype = [ctypes.c_uint], ctypes.c_uint
        if not fn(0x80000001):
            print("WARNING: Windows did not accept the keep-awake request.", flush=True)
    try:
        yield
    finally:
        if fn:
            fn(0x80000000)


def save(state, **changes):
    state.update(changes, updated_at=datetime.now().isoformat(timespec="seconds"))
    write(Path(state["job"]) / "workflow.json", state)


def execute_stage(state, item, env, sleep=time.sleep, run=subprocess.run):
    name = item["name"]
    if name in state["completed"]:
        verify_stage_proof(state, name, state["completed"][name]["proof"])
        print("SKIP completed stage: " + name, flush=True)
        return
    for attempt in range(1, item["attempts"] + 1):
        assert_inputs(state)
        state["attempts"][name] = state["attempts"].get(name, 0) + 1
        save(state, status="running", stage=name, retry_seconds=0)
        print(f"\n===== {name} | attempt {attempt}/{item['attempts']} =====", flush=True)
        print(subprocess.list2cmdline(item["command"]), flush=True)
        result = run(item["command"], cwd=ROOT, env=env)
        if result.returncode == 0:
            proof = {p: sha(p) for p in item["proof"]}
            if name.startswith("rubric_"):
                build_contexts(state["prepared"], [name.removeprefix("rubric_")])[0].validate_optimized()
            if name == "calibrate":
                config = read(Path(state["job"]) / "a3wa_config.json")
                passed = (config.get("deployment_gate") or {}).get("passed") is True
                state["deployment_gate_passed"] = passed
                state["result_class"] = "passed_current_gates" if passed else "experimental"
                if not passed:
                    print("WARNING: deployment gate failed; continue as an experimental diagnostic, preserving gate details.", flush=True)
                if not (config.get("score_calibration") or {}).get("enabled"):
                    print("NOTE: residual calibration is disabled by its data checks; preserve core scoring.", flush=True)
            state["completed"][name] = {"at": datetime.now().isoformat(timespec="seconds"), "proof": proof}
            save(state)
            return
        if attempt < item["attempts"]:
            wait = min(600, 60 * 2 ** min(attempt - 1, 4))
            save(state, status="retry_wait", retry_seconds=wait, last_exit_code=result.returncode)
            print(f"Retry the same stage/run after {wait}s; completed work is preserved.", flush=True)
            sleep(wait)
    raise RuntimeError(f"Stage failed after {item['attempts']} attempts: {name}; resume this same workflow.")


def run_job(state, stages):
    env = frozen_environment(Path(state["job"]) / "sample_policy.json")
    with workflow_lock(), keep_awake():
        save(state, pid=os.getpid(), status="running")
        try:
            # Activate once for ordinary later commands; children always use the
            # immutable explicit policy even if review UI activates another one.
            if not state.get("policy_installed"):
                active = PREPARED / "quality_control/policies/active_sample_policy.json"
                active.parent.mkdir(parents=True, exist_ok=True)
                temporary = active.with_name(f".{active.name}.tmp-{os.getpid()}")
                shutil.copy2(Path(state["job"]) / "sample_policy.json", temporary)
                temporary.replace(active)
                save(state, policy_installed=True)
            for item in stages:
                execute_stage(state, item, env)
            assert_inputs(state)
            save(state, status="completed", stage="completed")
            destination = Path(state["artifacts"]) / "csbench/workflows" / state["run_id"]
            destination.mkdir(parents=True, exist_ok=True)
            for name in ("workflow.json", "selection.json", "review_decisions.jsonl", "sample_policy.json", "input_sha256.json", "a3wa_config.json"):
                shutil.copy2(Path(state["job"]) / name, destination / name)
            if state.get("resume_code_repair_sha256"):
                shutil.copy2(Path(state["job"]) / "resume_code_repair.json", destination / "resume_code_repair.json")
                shutil.copytree(Path(state["job"]) / "code_repairs", destination / "code_repairs", dirs_exist_ok=True)
            print("\nWORKFLOW COMPLETED: " + state["run_id"], flush=True)
            print("Result class: " + state.get("result_class", "unknown"), flush=True)
            print("Job: " + state["job"], flush=True)
            print("Test outputs: " + state["paths"]["test"], flush=True)
            print("Artifacts copied locally; review and commit later. No automatic Git push.", flush=True)
        except BaseException as exc:
            save(state, status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
            raise


def show_status():
    if not LATEST.is_file():
        print("No reviewed full workflow has been started.")
        return
    state = read(Path(read(LATEST)["job"]) / "workflow.json")
    try:
        with workflow_lock():
            running = False
    except RuntimeError:
        running = True
    print("Workflow process running: " + str(running))
    print(json.dumps({k: state.get(k) for k in ("run_id", "status", "stage", "updated_at", "pid", "console_log", "error", "result_class", "deployment_gate_passed")}, ensure_ascii=False, indent=2))
    print(f"Completed stages: {len(state['completed'])}")
    for split in ("validation", "test"):
        progress = Path(state["paths"][split]) / "progress.json"
        if progress.is_file():
            data = read(progress)
            print(split + " progress: " + str(progress))
            for qid, row in (data.get("questions") or {}).items():
                print(f"  {qid}: {row.get('completed', 0)}/{row.get('total_students', '?')}, failed={row.get('failed', 0)}, remaining={row.get('remaining', '?')}")
    log = state.get("console_log")
    if log and Path(log).is_file():
        print("\nLatest log lines:")
        with Path(log).open("rb") as handle:
            handle.seek(max(0, Path(log).stat().st_size - 65536))
            print("\n".join(handle.read().decode("utf-8", errors="replace").splitlines()[-25:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--background", action="store_true")
    actions.add_argument("--check", action="store_true", help="Offline only; never starts an experiment")
    actions.add_argument("--status", action="store_true")
    parser.add_argument("--new-run", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--artifacts-repo", type=Path, default=ROOT.parent / "refgrader-artifacts")
    parser.add_argument("--grade-attempts", type=int, default=36)
    args = parser.parse_args()
    if not 1 <= args.grade_attempts <= 100:
        parser.error("--grade-attempts must be between 1 and 100")
    if args.status:
        show_status()
        return 0
    if args.check:
        preflight(args)
        return 0
    # Resume validates its own frozen inputs, not the evolving live review file.
    existing = bool(args.run_id or (LATEST.is_file() and not args.new_run))
    if not existing:
        preflight(args)
    with workflow_lock():
        state = prepare_job(args)
        stages = preflight(args, state)
        if state["status"] == "completed":
            print("Already completed. Use --new-run only for an intentional new experiment: " + state["run_id"])
            return 0
    if not (os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY")):
        if not sys.stdin.isatty():
            raise RuntimeError("Set ZHIPUAI_API_KEY in the launching terminal.")
        key = getpass.getpass("GLM API key (hidden, used in memory only): ").strip()
        if not key:
            raise RuntimeError("API key was empty")
        os.environ["ZHIPUAI_API_KEY"] = key
    if args.background:
        logs = ROOT / "logs"
        logs.mkdir(exist_ok=True)
        log = logs / (state["run_id"] + ".log")
        save(state, console_log=str(log))
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--run-id", state["run_id"]]
        flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
        with log.open("ab") as handle:
            child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, creationflags=flags, start_new_session=os.name != "nt")
        print(f"Background workflow launched. PID={child.pid}\nRun ID: {state['run_id']}\nLog: {log}\nState: {Path(state['job'])/'workflow.json'}")
        return 0
    run_job(state, stages)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
