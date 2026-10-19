"""Isolated RefGrader experiment on the four SCALE public datasets.

No SCALE augmentation/SAE implementation is used. Never run this in the SCALE
reproduction directory. Repetitions are independent API experiments, not seeded
PLM training. Commands preserve successful stages and question checkpoints.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark_datasets.contract import load_json, write_json, sha256_file, audit_prepared_benchmark
from scripts.benchmarks.scale4_data import prepare, DATASETS, COMMIT
from scripts.benchmarks import run_benchmark as benchmark


def freeze(path, payload):
    if path.exists():
        if load_json(path) != payload:
            raise ValueError("Resume inputs changed; retain the old run and choose a new tag: " + str(path))
    else:
        write_json(path, payload)


def run_command(command, dry=False):
    print(subprocess.list2cmdline(command), flush=True)
    if not dry:
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            raise RuntimeError(f"Stage failed ({result.returncode}); saved work retained")


def selected_questions(context, split):
    return [q["question_id"] for q in context["questions"]
            if load_json(context["root"] / q["rubric_split_path"])[split]]


def cold_rubrics(context, state, dry):
    if dry:
        print(f"PLAN cold-start GLM-5.3 rubrics: {len(context['questions'])} question/reference contexts")
        return
    from step3_rrd_generator import generate_rrd_rubrics
    from rubric_semantics import prepare_rubric_semantic_contract, validate_refined_rubric
    for q in context["questions"]:
        marker = state / "cold" / (q["question_id"] + ".json")
        initial = context["root"] / q["initial_rubric_path"]
        if marker.exists():
            if load_json(marker)["sha256"] != sha256_file(initial):
                raise ValueError("Cold rubric changed")
            continue
        rubric = generate_rrd_rubrics(q["question_text"], q["ref_text"], q["official_rubric"], q["total_score"])
        if not rubric:
            raise RuntimeError("Cold rubric generation failed: " + q["question_id"])
        rubric = prepare_rubric_semantic_contract(rubric)
        valid, errors = validate_refined_rubric(rubric, rubric, q["total_score"], allow_unchanged_baseline=True)
        if not valid:
            raise ValueError("Cold rubric invalid: " + str(errors))
        write_json(initial, rubric)
        optimized = context["root"] / q["optimized_rubric_path"]
        write_json(optimized, rubric)
        manifest = context["root"] / "rubrics/manifests" / q["rubric_group"] / (q["question_id"] + "_optimization.json")
        payload = load_json(manifest)
        payload.update(initial_sha256=sha256_file(initial), optimized_sha256=sha256_file(optimized),
                       method="glm53_label_blind_reference_reconstruction", reconstructed_not_official=True)
        write_json(manifest, payload)
        write_json(marker, dict(sha256=sha256_file(initial)))
        print("COLD_RUBRIC_SAVED", q["question_id"], flush=True)


def optimize(context, state, dry):
    runtime_db = benchmark._runtime_database(context, state)
    for q in context["questions"]:
        marker = state / "optimization" / (q["question_id"] + ".json")
        optimized = context["root"] / q["optimized_rubric_path"]
        manifest = context["root"] / "rubrics/manifests" / q["rubric_group"] / (q["question_id"] + "_optimization.json")
        if marker.exists():
            if load_json(marker)["sha256"] != sha256_file(optimized):
                raise ValueError("Optimized rubric changed")
            continue
        if not q["rubric_calibration_ids"]:
            if not dry:
                write_json(marker, dict(sha256=sha256_file(optimized), status="NO_TRAIN_CONTEXT_LABEL_BLIND_ONLY"))
            continue
        command = [sys.executable, str(ROOT / "main_pipeline.py"), "--mode", "VARIANCE_OPT",
                   "--questions", q["question_id"], "--sample-size", "5", "--resume-optimization",
                   "--database-path", str(runtime_db), "--teacher-db", str(context["root"] / "teacher_scores.json"),
                   "--answer-metadata", str(context["root"] / "answer_metadata.jsonl"),
                   "--initial-rubric-dir", str(context["root"] / "rubrics/initial"),
                   "--rubric-dir", str(context["root"] / "rubrics/optimized"),
                   "--results-dir", str(state / "rubric_work"), "--progress-file", str(state / "rubric_progress.json"),
                   "--extraction-backend", "text_only", "--ocr-cache-dir", str(state / "facts"),
                   "--log-dir", str(state / "logs"), "--run-id", "rubric_" + q["question_id"]]
        run_command(command, dry)
        if not dry:
            info = load_json(manifest)
            if not info.get("semantic_policy_validated"):
                raise RuntimeError("Rubric did not pass semantic contract")
            write_json(marker, dict(sha256=sha256_file(optimized), status=info.get("semantic_validation_mode")))


def grade_stage(context, state, split, run_id, config, dry):
    marker = state / (split + "_complete.json")
    questions = selected_questions(context, split)
    directory = benchmark._run_dir(context["manifest"]["dataset_id"], run_id)
    if marker.exists():
        for name, digest in load_json(marker)["files"].items():
            if sha256_file(directory / name) != digest:
                raise ValueError("Completed grading output changed: " + name)
        return directory
    code = benchmark.grade(context, questions=questions, split=split, run_id=run_id,
        force=False, a3wa_config=str(config) if config else None, dry_run=dry,
        evaluate_after=False, allow_experimental_a3wa=True)
    if code:
        raise RuntimeError(f"Grading exit code {code}")
    if not dry:
        missing = benchmark._run_completeness(context, run_dir=directory, questions=questions, split=split, limit=None)
        if missing:
            raise RuntimeError("Incomplete grading: " + json.dumps(missing))
        hashes = {}
        for q in context["questions"]:
            if q["question_id"] not in questions:
                continue
            name = q["question_id"] + "_grading_checkpoint.json"
            rows = load_json(directory / name)
            expected = set(load_json(context["root"] / q["rubric_split_path"])[split])
            if len(rows) != len(expected) or {r["student_id"] for r in rows} != expected:
                raise ValueError("Prediction identity mismatch")
            hashes[name] = sha256_file(directory / name)
        write_json(marker, dict(files=hashes))
    return directory


def metric_report(context, directory):
    import numpy as np
    from sklearn.metrics import cohen_kappa_score
    gold, predicted = [], []
    labels = load_json(context["root"] / "teacher_scores.json")
    for q in context["questions"]:
        ids = load_json(context["root"] / q["rubric_split_path"])["test"]
        if not ids:
            continue
        rows = load_json(directory / (q["question_id"] + "_grading_checkpoint.json"))
        for r in rows:
            truth = float(labels[r["student_id"]][q["question_id"]])
            value = float(r["final_calibrated_score"])
            if not math.isfinite(value):
                raise ValueError("Non-finite prediction")
            gold.append(truth)
            predicted.append(value)
    y, p = np.array(gold), np.array(predicted)
    rmse = float(np.sqrt(np.mean((p-y)**2)))
    pearson = float(np.corrcoef(y,p)[0,1]) if np.std(p) and np.std(y) else None
    qwk = float(cohen_kappa_score(np.floor(np.clip(y,0,10)+0.5).astype(int),
        np.floor(np.clip(p,0,10)+0.5).astype(int), labels=list(range(11)), weights="quadratic"))
    return dict(records=len(y), method="RefGrader final_calibrated_score", pearson=pearson,
                rmse_raw_0_10=rmse, rmse_normalized=rmse/10, qwk=qwk if math.isfinite(qwk) else None,
                qwk_policy="fixed 11 levels; clip 0-10 then half-up; author policy unverified",
                out_of_range=int(np.sum((p<0)|(p>10))))


def experiment(dataset, repetition, tag, prepared, dry):
    work = ROOT / "results_runs/scale4_refgrader" / tag / dataset / f"repeat_{repetition}"
    work.mkdir(parents=True, exist_ok=True)
    run_data = work / "prepared"
    if not run_data.exists():
        shutil.copytree(prepared, run_data)
    context = benchmark._prepared_context(run_data)
    # Each repeat has an independent facts cache and result namespace.
    context["manifest"]["dataset_id"] = f"scale4_{dataset.lower()}_{tag}_r{repetition}"
    cold_rubrics(context, work, dry)
    optimize(context, work, dry)
    if not dry:
        write_json(run_data / "audit.json", audit_prepared_benchmark(run_data))
    context["audit"] = audit_prepared_benchmark(run_data)
    validation_id = tag + f"_r{repetition}_validation"
    grade_stage(context, work, "validation", validation_id, None, dry)
    config = work / "a3wa.json"
    calibrated = work / "calibration_complete.json"
    if calibrated.exists():
        if load_json(calibrated)["sha256"] != sha256_file(config):
            raise ValueError("Calibration changed")
    else:
        code, _ = benchmark.calibrate(context, questions=selected_questions(context,"validation"),
            validation_run_id=validation_id, output=str(config), score_calibration=True, dry_run=dry)
        if code:
            raise RuntimeError("Calibration failed")
        if not dry:
            write_json(calibrated, dict(sha256=sha256_file(config)))
    directory = grade_stage(context, work, "test", tag + f"_r{repetition}_test", config, dry)
    if not dry:
        report = metric_report(context, directory)
        report["a3wa_deployment_class"] = benchmark._a3wa_deployment_class(config)
        report["rubric_modes"] = dict(__import__('collections').Counter(
            load_json(f).get("status", "unknown") for f in (work / "optimization").glob("*.json")))
        write_json(work / "metrics.json", report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "plan", "run", "summarize"])
    parser.add_argument("--source", type=Path, default=ROOT / "sources/scale4_public")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--tag", default="refgrader_glm53_scale4_v1")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--stage-attempts", type=int, default=3)
    args = parser.parse_args()
    if not args.tag.replace('_','').replace('-','').isalnum() or not 1 <= args.repeats <= 5 or args.stage_attempts < 1:
        parser.error("Invalid tag, repeats or attempts")
    from filelock import FileLock
    lock_path = ROOT / "results_runs/scale4_refgrader" / (args.tag + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # Lifetime of this local variable covers main(); the OS lock is released on exit/crash.
    run_lock = FileLock(str(lock_path))
    run_lock.acquire(timeout=0)
    os.environ["REFGRADER_TEXT_MODEL_PROVIDER"] = "glm53"
    os.environ["REFGRADER_TEXT_THINKING"] = "disabled"
    os.environ["REFGRADER_VLM_MODEL_PROVIDER"] = "glm4v"
    os.environ.pop("REFGRADER_TEXT_TEMPERATURE_OVERRIDE", None)
    os.environ.pop("A3WA_CALIBRATION_CONFIG", None)
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if args.command == "run" and not (os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY")):
        parser.error("Set ZHIPUAI_API_KEY in the terminal, never in a config file")
    base = ROOT / "data/public_benchmarks/scale4"
    if args.command == "prepare":
        for ds in args.datasets:
            print(json.dumps(prepare(args.source, ds, base/ds), ensure_ascii=False), flush=True)
        print("FOUR_DATASETS_PREPARED")
        return
    work = ROOT / "results_runs/scale4_refgrader" / args.tag
    code_files = [ROOT/p for p in ("main_pipeline.py", "step3_rrd_generator.py", "step4_vlm_grader.py", "rubric_semantics.py", "calibration_utils.py", "model_runtime.py", "portable_hash.py", "sample_quality.py", "scripts/calibrate_a3wa.py", "scripts/benchmarks/run_scale4.py", "scripts/benchmarks/scale4_data.py", "scripts/benchmarks/run_benchmark.py", "benchmark_datasets/contract.py")]
    code_files += sorted(p for p in (ROOT/"prompts").iterdir() if p.is_file()) + sorted((ROOT/"canonicalizers").glob("*.py"))
    protocol = dict(method="RefGrader", model="glm-5.3", thinking="disabled", commit=COMMIT,
        repeats=args.repeats, datasets=args.datasets, api_seed_supported=False,
        use_scale_augmentation=False, score_calibration=True,
        code_hashes={p.relative_to(ROOT).as_posix():sha256_file(p) for p in code_files},
        data_hashes={ds: audit_prepared_benchmark(base/ds)["prepared_content_sha256"] for ds in args.datasets})
    freeze(work / "protocol.json", protocol)
    failures = []
    if args.command in ("run", "plan"):
        for ds in args.datasets:
            for repeat in range(1, args.repeats+1):
                for attempt in range(args.stage_attempts if args.command == "run" else 1):
                    try:
                        experiment(ds,repeat,args.tag,base/ds,args.command=="plan")
                        break
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as exc:
                        print(f"FAILED {ds} repeat={repeat} attempt={attempt+1}: {exc}", flush=True)
                        if attempt+1 < args.stage_attempts and args.command == "run":
                            time.sleep(30)
                        else:
                            failures.append(dict(dataset=ds, repeat=repeat, error=str(exc)))
        if args.command == "plan":
            if failures:
                raise RuntimeError(str(failures))
            print("FOUR_DATASET_PLAN_OK_NO_API_CALLS")
            return
    summary = dict(method="RefGrader", model="glm-5.3", datasets={}, failures=failures,
                   protocol="fixed public Chinese files; main experiment only; no SCALE synthetic data",
                   limitation="not exact author split/QWK protocol; ASAG 629 rows; SR lacks original questions")
    for ds in args.datasets:
        reports = []
        for i in range(1,args.repeats+1):
            f = work / ds / f"repeat_{i}/metrics.json"
            if f.exists():
                reports.append(load_json(f))
        values = {}
        for metric in ("pearson", "qwk", "rmse_raw_0_10", "rmse_normalized"):
            numbers = [r[metric] for r in reports if r[metric] is not None]
            values[metric] = dict(mean=statistics.mean(numbers) if numbers else None,
                                 sample_std=statistics.stdev(numbers) if len(numbers)>1 else None,
                                 valid_runs=len(numbers))
        summary["datasets"][ds] = dict(completed_runs=len(reports), expected_runs=args.repeats, metrics=values, runs=reports)
    summary["complete"] = not failures and all(v["completed_runs"]==args.repeats for v in summary["datasets"].values())
    write_json(work / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
