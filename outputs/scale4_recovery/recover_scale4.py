"""Audited, run-local rubric recovery; original bundle files stay immutable."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

TAG = "refgrader_glm53_scale4_single_v1"


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".recovery.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def checked_path(root, path):
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError("Path outside recovery workspace: " + str(path))
    return path


def activate(state, prepared, q, candidate, audit):
    """Journal before mutation; replay the same transaction after interruption."""
    qid = q["question_id"]
    if (state / "optimization" / (qid + ".json")).exists():
        raise RuntimeError("Cannot replace a completed question: " + qid)
    for name in ("validation_complete.json", "calibration_complete.json", "test_complete.json", "metrics.json"):
        if (state / name).exists():
            raise RuntimeError("Downstream work exists; refusing rubric mutation: " + name)
    initial = checked_path(state, prepared / q["initial_rubric_path"])
    optimized = checked_path(state, prepared / q["optimized_rubric_path"])
    manifest = checked_path(state, prepared / "rubrics/manifests" / q["rubric_group"] / (qid + "_optimization.json"))
    marker = state / "cold" / (qid + ".json")
    journal = audit / "transaction.json"
    if not journal.exists():
        for path in (initial, optimized, manifest, marker):
            if path.exists():
                backup = audit / "before" / path.relative_to(state)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup)
        payload = load(manifest)
        payload.update(method="glm53_label_blind_reference_reconstruction",
                       reconstructed_not_official=True,
                       recovery="strict_contract_generation_retry_v1")
        save(journal, {"candidate": candidate, "manifest": payload})
    transaction = load(journal)
    # Only this question's rubric-dependent caches are archived, never deleted.
    for path in (state / "rubric_work" / (qid + "_variance_checkpoint.json"),
                 state / "facts" / "variance_facts" / qid):
        path = checked_path(state, path)
        if path.exists():
            destination = audit / "invalidated" / path.relative_to(state)
            if destination.exists():
                raise RuntimeError("Unexpected recreated cache during transaction: " + str(path))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(destination))
    save(initial, transaction["candidate"])
    save(optimized, transaction["candidate"])
    payload = transaction["manifest"]
    payload.update(initial_sha256=sha(initial), optimized_sha256=sha(optimized))
    save(manifest, payload)
    save(marker, {"sha256": sha(initial)})
    save(audit / "activated.json", {"question_id": qid, "sha256": sha(initial),
                                   "variance_cache_archived": True})


def validate_candidate(candidate, original, score, prepare, validate):
    if not isinstance(candidate, list) or not candidate or not all(isinstance(x, dict) for x in candidate):
        return None, ["Expected a nonempty JSON array of rubric objects"]
    candidate = prepare(candidate)
    valid, errors = validate(original if original is not None else candidate, candidate, score)
    if valid:
        valid, errors = validate(candidate, candidate, score)
    return candidate, errors if not valid else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ["REFGRADER_TEXT_MODEL_PROVIDER"] = "glm53"
    os.environ["REFGRADER_TEXT_THINKING"] = "disabled"
    os.environ.pop("REFGRADER_TEXT_TEMPERATURE_OVERRIDE", None)
    from scripts.benchmarks import run_scale4 as runner
    from rubric_semantics import prepare_rubric_semantic_contract as prepare
    from rubric_semantics import validate_refined_rubric as validate
    from filelock import FileLock
    work = root / "results_runs/scale4_refgrader" / TAG
    lock = FileLock(str(work.parent / (TAG + ".lock")))
    with lock.acquire(timeout=0):
        # Confirm this is the unchanged deployed bundle, not a mixed local checkout.
        for name, expected in load(root / "BUNDLE_SHA256.json").items():
            if sha(checked_path(root, root / name)) != expected:
                raise RuntimeError("Bundle changed: " + name)
        protocol = load(work / "protocol.json")
        if protocol["repeats"] != 1 or protocol["datasets"] != ["ADS", "LE", "ASAG", "SR"]:
            raise RuntimeError("Unexpected run protocol")
        for name, expected in protocol["code_hashes"].items():
            if sha(checked_path(root, root / name)) != expected:
                raise RuntimeError("Frozen run code changed: " + name)
        audit_root = work / "rubric_recovery_v1"
        descriptor = {"wrapper_sha256": sha(Path(__file__).resolve()),
                      "original_protocol_sha256": sha(work / "protocol.json"),
                      "policy": "Retry invalid or unfinished cold rubrics; strict existing validator; archive dependent caches; preserve completed stages",
                      "scoring_code_changed": False, "repeats": 1}
        if (audit_root / "policy.json").exists() and load(audit_root / "policy.json") != descriptor:
            raise RuntimeError("Recovery script or protocol changed after recovery started")
        if args.run:
            save(audit_root / "policy.json", descriptor)
        print("RECOVERY_PREFLIGHT_OK; original scoring code unchanged", flush=True)
    if not args.run:
        return

    def cold(context, state, dry):
        if dry:
            raise RuntimeError("Recovery is not a plan runner")
        from step3_rrd_generator import generate_rrd_rubrics, normalize_generated_rubric
        from step4_vlm_grader import call_text_model, extract_and_parse_json
        for q in context["questions"]:
            qid = q["question_id"]
            initial = context["root"] / q["initial_rubric_path"]
            marker = state / "cold" / (qid + ".json")
            completed = state / "optimization" / (qid + ".json")
            audit = audit_root / state.parent.name / qid
            if completed.exists():
                if not marker.exists() or load(marker)["sha256"] != sha(initial):
                    raise RuntimeError("Completed initial rubric changed: " + qid)
                continue
            if (audit / "transaction.json").exists() and not (audit / "activated.json").exists():
                activate(state, context["root"], q, None, audit)
            original = None
            if marker.exists():
                if load(marker)["sha256"] != sha(initial):
                    raise RuntimeError("Cold rubric changed: " + qid)
                original = load(initial)
                _, errors = validate_candidate(original, original, q["total_score"], prepare, validate)
                if not errors:
                    continue
            else:
                errors = ["Cold generation not yet accepted"]
            for name in ("validation_complete.json", "test_complete.json", "metrics.json"):
                if (state / name).exists():
                    raise RuntimeError("Cannot repair a rubric after downstream evaluation")
            candidate = original
            accepted = False
            for attempt in range(1, 11):
                stamp = str(time.time_ns())
                print("RUBRIC_RECOVERY", qid, "attempt", attempt, flush=True)
                try:
                    if candidate is None:
                        candidate = generate_rrd_rubrics(q["question_text"], q["ref_text"], q["official_rubric"], q["total_score"])
                    else:
                        prompt = (
                            "Repair this generated RefGrader rubric. Return ONLY a JSON array. "
                            "Use only the supplied question and reference; no student answers or labels. "
                            "Preserve all reference facts, direction, numbers and coverage. "
                            "Total points must equal the question total exactly. "
                            "For an immutable original, preserve each parent's points and scoring_policy; "
                            "split into uniquely identified children with parent_id and parent_points. "
                            "Each role_weighted_additive parent needs exactly one final child and at least one core_process child; "
                            "complex parents also need support_process. Respect minimum_process_ratio, "
                            "minimum_core_process_ratio and maximum_final_ratio from each original parent. "
                            "All children need objective item and standard_answer_text. Do not duplicate credit or invent reference content. "
                            "Do not downgrade policies to evade validation.\n" + json.dumps({
                                "question": q["question_text"], "reference": q["ref_text"],
                                "official_rubric": q["official_rubric"], "total_score": q["total_score"],
                                "immutable_original": original, "last_candidate": candidate,
                                "validation_errors": errors}, ensure_ascii=False))
                        raw = call_text_model([{"role": "user", "content": prompt}], temperature=0.2, timeout=900)
                        save(audit / (stamp + "_response.json"), {"content": raw})
                        candidate = extract_and_parse_json(raw)
                    candidate = normalize_generated_rubric(candidate)
                    candidate, errors = validate_candidate(candidate, original, q["total_score"], prepare, validate)
                    save(audit / (stamp + "_validation.json"), {"candidate": candidate, "errors": errors})
                    if not errors:
                        activate(state, context["root"], q, candidate, audit)
                        accepted = True
                        print("STRICT_RUBRIC_ACCEPTED", qid, flush=True)
                        break
                except Exception as exc:
                    # Activation failures must be recovered transactionally, not regenerated.
                    if (audit / "transaction.json").exists():
                        raise
                    errors = [type(exc).__name__ + ": " + str(exc)]
                    save(audit / (stamp + "_error.json"), {"error_type": type(exc).__name__})
                time.sleep(3)
            if not accepted:
                raise RuntimeError("Strict rubric repair exhausted: " + qid + "; " + str(errors))

    old_experiment = runner.experiment

    def experiment(dataset, repetition, tag, prepared, dry):
        state = work / dataset / ("repeat_" + str(repetition))
        if (state / "metrics.json").exists():
            context = runner.benchmark._prepared_context(state / "prepared")
            context["manifest"]["dataset_id"] = f"scale4_{dataset.lower()}_{tag}_r{repetition}"
            if not (state / "test_complete.json").exists():
                raise RuntimeError("Metrics exist without completed test marker")
            directory = runner.benchmark._run_dir(context["manifest"]["dataset_id"], tag + f"_r{repetition}_test")
            for name, expected in load(state / "test_complete.json")["files"].items():
                if sha(directory / name) != expected:
                    raise RuntimeError("Completed test output changed")
            print("PRESERVED_COMPLETED_DATASET", dataset, flush=True)
            return
        return old_experiment(dataset, repetition, tag, prepared, dry)

    runner.cold_rubrics = cold
    runner.experiment = experiment
    sys.argv = [str(root / "scripts/benchmarks/run_scale4.py"), "run", "--tag", TAG,
                "--repeats", "1", "--stage-attempts", "3"]
    runner.main()


if __name__ == "__main__":
    main()
