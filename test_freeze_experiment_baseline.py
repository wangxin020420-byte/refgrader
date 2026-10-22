"""Integrity tests for the offline baseline freezer; no model dependencies."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import freeze_experiment_baseline as module


class FreezeBaselineTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parent / ".local-maintenance" / "test_tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifacts = self.root / "artifacts"
        (self.artifacts / ".git").mkdir(parents=True)

    def fixture(self, arm):
        repo = self.root / arm
        run_id = module.ARMS[arm][1]
        job = repo / "results_runs" / module.ARMS[arm][0] / run_id
        prepared = repo / ("data/csbench" if arm == "pc1-optimized" else "experiments/initial_rubric_ablation/reviewed85_20261019")

        def put(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            module.write(path, value)
            return path

        put(prepared / "teacher_scores.json", {sid: {"Q_1": score} for sid, score in
                                               [("v1", 0), ("v2", 2), ("t1", 0), ("t2", 1), ("t3", 2)]})
        put(prepared / "answer_metadata.jsonl", [])
        put(prepared / "exam_database.json", [{"question_id": "Q_1", "total_score": 2}])
        put(prepared / "splits/by_question/Q_1.json", {"calibration": [], "validation": ["v1", "v2"], "test": ["t1", "t2", "t3"]})
        put(prepared / "rubrics/initial/Q/Q_1_rubric_standard.json", [{"points": 2}])
        inputs = {str(p.resolve()): module.digest(p) for p in prepared.rglob("*") if p.is_file()}
        if arm == "pc2-initial":
            put(prepared / "snapshot_contract.json", {"files": {}})
            inputs[str((prepared / "snapshot_contract.json").resolve())] = module.digest(prepared / "snapshot_contract.json")
            put(job / "exam_database.json", [])
            inputs[str((job / "exam_database.json").resolve())] = module.digest(job / "exam_database.json")
        selection = {"questions": ["Q_1"], "splits": {"validation": {"effective": 2}, "test": {"effective": 3}}}
        put(job / "selection.json", selection)
        put(job / "sample_policy.json", {"excluded": {}, "corrected_scores": {}})
        (job / "review_decisions.jsonl").write_text("", encoding="utf-8")
        put(job / "input_sha256.json", inputs)
        model = {"text_model": "glm-5.2", "text_thinking": "disabled", "vlm_model": "glm-4.6v"}
        put(job / "a3wa_config.json", {"model_config": model,
                                     "sample_quality_policy": {"sha256": module.digest(job / "sample_policy.json")},
                                     "deployment_gate": {"passed": False}})
        completed, paths = {"calibrate": {"proof": {
            str((job / "a3wa_config.json").resolve()): module.digest(job / "a3wa_config.json")}}}, {}
        records = None
        for split, pairs in (("validation", [("v1", 0), ("v2", 2)]), ("test", [("t1", 0), ("t2", 1), ("t3", 2)])):
            directory = job / split if arm == "pc2-initial" else repo / "results_runs" / split / run_id
            paths[split] = str(directory.resolve())
            records = [{"student_id": sid, "model_scores_history": [score, score, score],
                        **{key: score for key in module.SCORES.values() if key != "single_first_score"}}
                       for sid, score in pairs]
            p = put(directory / "Q_1_grading_checkpoint.json", records)
            completed["grade_" + split] = {"proof": {str(p.resolve()): module.digest(p)}}
        if arm == "pc1-optimized":
            p = put(prepared / "rubrics/optimized/Q/Q_1_rubric_standard.json", [{"points": 2}])
            completed["rubric_Q_1"] = {"proof": {str(p.resolve()): module.digest(p)}}
        evaluation = job / "reports" if arm == "pc2-initial" else Path(paths["test"]) / "evaluation"
        evaluation.mkdir(parents=True, exist_ok=True)
        filename = "score_comparison.csv" if arm == "pc2-initial" else "compare.csv"
        with (evaluation / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["question", "student_id", "teacher", *module.SCORES.values()])
            writer.writeheader()
            for r in records:
                score = r["model_avg_score"]
                writer.writerow({"question": "Q_1", "student_id": r["student_id"], "teacher": score,
                                 **{key: score for key in module.SCORES.values()}})
        put(evaluation / "summary.json", {
            "per_question": [{"score_type": stage, "q_id": "Q_1", "n": 3, "total": 2, "QWK": 1}
                             for stage in module.SCORES],
            "global": [{"score_type": stage, "n": 3, "MAE": 0, "RMSE": 0} for stage in module.SCORES]})
        state = {"run_id": run_id, "status": "completed", "questions": ["Q_1"], "prepared": str(prepared.resolve()),
                 "paths": paths, "completed": completed,
                 "model": {"text_model": "glm-5.2", "thinking": "disabled", "vlm_model": "glm-4.6v"}}
        for file, key in (("selection.json", "selection_sha256"), ("sample_policy.json", "policy_sha256"),
                          ("review_decisions.jsonl", "review_sha256"), ("input_sha256.json", "input_manifest_sha256")):
            state[key] = module.digest(job / file)
        put(job / "workflow.json", state)
        return repo, job, evaluation

    def test_both_arms_are_frozen_verified_and_pairable(self):
        paths = []
        for arm in module.ARMS:
            repo, _, _ = self.fixture(arm)
            result = module.freeze(repo, arm, module.ARMS[arm][1], self.artifacts)
            manifest = module.verify(result["path"])
            self.assertEqual(manifest["result_class"], "experimental")
            self.assertEqual(module.read(Path(result["path"]) / "metrics.json")["3WD"]["pooled_raw_score_qwk"], 1)
            paths.append(result["path"])
        self.assertTrue(module.compare(*paths)["paired_data_and_model_contract_passed"])

    def test_dry_run_does_not_create_archive(self):
        repo, _, _ = self.fixture("pc2-initial")
        result = module.freeze(repo, "pc2-initial", module.ARMS["pc2-initial"][1], self.artifacts, True)
        self.assertEqual(result["status"], "dry_run_passed")
        self.assertFalse((self.artifacts / "csbench").exists())

    def test_repeat_verifies_without_overwriting(self):
        repo, _, _ = self.fixture("pc1-optimized")
        result = module.freeze(repo, "pc1-optimized", module.ARMS["pc1-optimized"][1], self.artifacts)
        before = module.digest(Path(result["path"]) / "baseline_manifest.json")
        repeated = module.freeze(repo, "pc1-optimized", module.ARMS["pc1-optimized"][1], self.artifacts)
        self.assertEqual(repeated["status"], "already_frozen_verified")
        self.assertEqual(before, module.digest(Path(result["path"]) / "baseline_manifest.json"))

    def test_incomplete_workflow_is_rejected(self):
        repo, job, _ = self.fixture("pc1-optimized")
        state = module.read(job / "workflow.json")
        state["status"] = "running"
        module.write(job / "workflow.json", state)
        with self.assertRaisesRegex(ValueError, "completed"):
            module.make_plan(repo, "pc1-optimized", state["run_id"])

    def test_changed_frozen_input_is_rejected(self):
        repo, job, _ = self.fixture("pc2-initial")
        prepared = Path(module.read(job / "workflow.json")["prepared"])
        (prepared / "teacher_scores.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fingerprint differs"):
            module.make_plan(repo, "pc2-initial", module.ARMS["pc2-initial"][1])

    def test_missing_or_edited_checkpoint_is_rejected(self):
        repo, job, _ = self.fixture("pc1-optimized")
        p = Path(module.read(job / "workflow.json")["paths"]["test"]) / "Q_1_grading_checkpoint.json"
        rows = module.read(p)
        rows.pop()
        module.write(p, rows)
        with self.assertRaisesRegex(ValueError, "checkpoint IDs"):
            module.make_plan(repo, "pc1-optimized", module.ARMS["pc1-optimized"][1])

    def test_edited_evaluation_is_rejected(self):
        repo, _, evaluation = self.fixture("pc2-initial")
        p = evaluation / "score_comparison.csv"
        p.write_text(p.read_text(encoding="utf-8").replace("Q_1,t2,1,", "Q_1,t2,0,"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "teacher label"):
            module.make_plan(repo, "pc2-initial", module.ARMS["pc2-initial"][1])

    def test_archive_tampering_and_extra_file_are_detected(self):
        repo, _, _ = self.fixture("pc1-optimized")
        path = Path(module.freeze(repo, "pc1-optimized", module.ARMS["pc1-optimized"][1], self.artifacts)["path"])
        (path / "evaluation/compare.csv").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            module.verify(path)

    def test_extra_file_in_frozen_directory_is_rejected(self):
        repo, _, _ = self.fixture("pc2-initial")
        path = Path(module.freeze(repo, "pc2-initial", module.ARMS["pc2-initial"][1], self.artifacts)["path"])
        (path / "new_analysis.txt").write_text("must be outside snapshot", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unrecorded files"):
            module.verify(path)

    def test_changed_calibration_proof_is_rejected(self):
        repo, job, _ = self.fixture("pc1-optimized")
        config = module.read(job / "a3wa_config.json")
        config["deployment_gate"]["passed"] = True
        module.write(job / "a3wa_config.json", config)
        with self.assertRaisesRegex(ValueError, "calibration proof"):
            module.make_plan(repo, "pc1-optimized", module.ARMS["pc1-optimized"][1])

    def test_change_during_copy_keeps_incomplete_archive(self):
        repo, _, _ = self.fixture("pc2-initial")
        original = module.shutil.copyfile

        def broken(source, destination):
            original(source, destination)
            Path(destination).write_bytes(b"changed while copying")

        with patch.object(module.shutil, "copyfile", side_effect=broken):
            with self.assertRaisesRegex(ValueError, "changed while copying"):
                module.freeze(repo, "pc2-initial", module.ARMS["pc2-initial"][1], self.artifacts)
        path = self.artifacts / "csbench/frozen_baselines" / module.ARMS["pc2-initial"][1]
        self.assertTrue(path.is_dir())
        self.assertFalse((path / "baseline_manifest.json").exists())

    def test_qwk_matches_quadratic_kappa_for_gapped_labels(self):
        self.assertAlmostEqual(module.qwk([0, 0, 4, 4], [0, 4, 4, 4]), 0.5)


if __name__ == "__main__":
    unittest.main()
