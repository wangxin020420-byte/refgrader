from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_csbench import (
    RUBRIC_SEMANTIC_CONTRACT_VERSION,
    calibrate,
    evaluate,
    grading_results_dir,
    inspect_results,
    optimization_evidence_paths,
    publish,
    select_grading_run,
    sha256_file,
    validate_complete_results,
)


class CSBenchArtifactSyncTests(unittest.TestCase):
    def test_formal_evaluation_includes_core_and_residual_scores(self):
        context = SimpleNamespace(
            question_id="CO_1",
            teacher_db=Path("teacher_scores.json"),
            database=Path("exam_database.json"),
        )
        args = SimpleNamespace(
            prepared_dir="data/csbench",
            questions=["CO_1"],
            dry_run=True,
            export=True,
            detail=False,
            no_artifacts=True,
        )
        with (
            patch("scripts.run_csbench.build_contexts", return_value=[context]),
            patch(
                "scripts.run_csbench.grading_results_dir",
                return_value=Path("results_runs/csbench_co1_full"),
            ),
            patch("scripts.run_csbench.execute", return_value=0) as execute_mock,
        ):
            self.assertEqual(evaluate(args), 0)
        command = execute_mock.call_args.args[0]
        compare_index = command.index("--compare-score-keys")
        self.assertEqual(
            command[compare_index + 1:compare_index + 6],
            ["single", "avg", "selected", "3wd-core", "3wd"],
        )

    def test_grading_directories_are_split_safe(self):
        contexts = [SimpleNamespace(question_id="CO_1")]
        with patch("scripts.run_csbench.PROJECT_ROOT", Path("/tmp/refgrader")):
            self.assertEqual(
                grading_results_dir(contexts, "test").name,
                "csbench_co1_full",
            )
            self.assertEqual(
                grading_results_dir(contexts, "validation").name,
                "csbench_co1_validation",
            )
            self.assertEqual(
                grading_results_dir(contexts, "calibration").name,
                "csbench_co1_calibration",
            )

    def test_force_creates_versioned_run_and_default_resumes_active(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "split.json"
            rubric = root / "rubric.json"
            split.write_text('{"test":["A"]}', encoding="utf-8")
            rubric.write_text('[{"id":"s1","points":1}]', encoding="utf-8")
            context = SimpleNamespace(
                question_id="CO_1",
                split_file=split,
                optimized_rubric=rubric,
            )
            with patch("scripts.run_csbench.PROJECT_ROOT", root):
                first, first_id = select_grading_run(
                    [context], "test", force_new=True, create=True
                )
                resumed, resumed_id = select_grading_run(
                    [context], "test", create=True
                )
                second, second_id = select_grading_run(
                    [context], "test", force_new=True, create=True
                )
            self.assertEqual((first, first_id), (resumed, resumed_id))
            self.assertNotEqual(first_id, second_id)
            self.assertNotEqual(first, second)

    def test_partial_results_are_reported_without_structural_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_file = root / "CO_1.json"
            split_file.write_text(
                json.dumps({"test": ["A", "B"]}), encoding="utf-8"
            )
            context = SimpleNamespace(
                question_id="CO_1", split_file=split_file
            )
            record = [{"student_id": "A"}]
            (root / "CO_1_grading_checkpoint.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
            (root / "CO_1_graded_results.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
            report = inspect_results([context], root, split_name="test")
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["checkpoint_total"], 1)
            self.assertEqual(report["expected_total"], 2)
            self.assertEqual(report["structural_errors"], [])

    def test_validation_requires_exact_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_file = root / "CO_1.json"
            split_file.write_text(
                json.dumps({"validation": ["A", "B"]}), encoding="utf-8"
            )
            context = SimpleNamespace(
                question_id="CO_1", split_file=split_file
            )
            checkpoint = root / "CO_1_grading_checkpoint.json"
            checkpoint.write_text(
                json.dumps([{"student_id": "A"}, {"student_id": "C"}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "missing validation"):
                validate_complete_results(
                    [context], root, split_name="validation"
                )

            checkpoint.write_text(
                json.dumps([{"student_id": "A"}, {"student_id": "B"}]),
                encoding="utf-8",
            )
            (root / "CO_1_graded_results.json").write_text(
                checkpoint.read_text(encoding="utf-8"), encoding="utf-8"
            )
            validate_complete_results(
                [context], root, split_name="validation"
            )

    def test_manifest_selects_exact_optimization_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = root / "combined" / "CO_1_variance_checkpoint.json"
            stale = root / "single" / "CO_1_variance_checkpoint.json"
            current.parent.mkdir()
            stale.parent.mkdir()
            current.write_text('[{"student_id":"A"}]', encoding="utf-8")
            stale.write_text('[{"student_id":"OLD"}]', encoding="utf-8")
            import hashlib

            manifest = root / "optimization_manifest.json"
            manifest.write_text(
                json.dumps({
                    "variance_checkpoint": str(current),
                    "variance_checkpoint_sha256": hashlib.sha256(
                        current.read_bytes()
                    ).hexdigest(),
                }),
                encoding="utf-8",
            )
            context = SimpleNamespace(
                question_id="CO_1", optimization_manifest=manifest
            )
            selected, _ = optimization_evidence_paths(context, stale.parent)
            self.assertEqual(selected, current)

    def test_validation_publish_uses_separate_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "refgrader"
            artifacts = Path(temporary) / "refgrader-artifacts"
            prepared = root / "data" / "csbench"
            root.mkdir()
            artifacts.mkdir()
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "init", "-q", str(artifacts)], check=True)
            (root / "tracked.txt").write_text("test", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(
                [
                    "git", "-C", str(root), "-c", "user.name=Test",
                    "-c", "user.email=test@example.com", "commit", "-qm", "test",
                ],
                check=True,
            )

            prepared.mkdir(parents=True)
            (prepared / "exam_database.json").write_text(
                json.dumps([{
                    "question_id": "CO_1",
                    "rubric_group": "CO",
                    "total_score": 5,
                }]),
                encoding="utf-8",
            )
            (prepared / "teacher_scores.json").write_text("{}", encoding="utf-8")
            (prepared / "answer_metadata.jsonl").write_text("", encoding="utf-8")
            split = prepared / "splits" / "by_question" / "CO_1.json"
            split.parent.mkdir(parents=True)
            split.write_text(
                json.dumps({"validation": ["A"], "test": ["T"]}),
                encoding="utf-8",
            )
            initial = prepared / "rubrics" / "initial" / "CO" / "CO_1_rubric_standard.json"
            optimized = prepared / "rubrics" / "optimized" / "CO" / "CO_1_rubric_standard.json"
            manifest = prepared / "rubrics" / "manifests" / "CO" / "CO_1_optimization.json"
            for path in (initial, optimized, manifest):
                path.parent.mkdir(parents=True, exist_ok=True)
            initial.write_text('[{"id":"s1","points":5}]', encoding="utf-8")
            optimized.write_text('[{"id":"s1","points":5}]', encoding="utf-8")
            manifest.write_text(
                json.dumps({
                    "rubric_semantic_contract_version": RUBRIC_SEMANTIC_CONTRACT_VERSION,
                    "semantic_policy_validated": True,
                    "initial_sha256": sha256_file(initial),
                    "optimized_sha256": sha256_file(optimized),
                }),
                encoding="utf-8",
            )
            validation = root / "results_runs" / "csbench_co1_validation"
            validation.mkdir(parents=True)
            record = json.dumps([{"student_id": "A", "3wd_route": "POS"}])
            (validation / "CO_1_grading_checkpoint.json").write_text(record, encoding="utf-8")
            (validation / "CO_1_graded_results.json").write_text(record, encoding="utf-8")

            calibration_args = SimpleNamespace(
                prepared_dir=str(prepared), questions=["CO_1"], output=None,
                bnd_max=0.60, top_k=8, min_cell_count=5, shrinkage_k=8.0,
                max_correction_ratio=0.12, max_correction_points=2.0,
                no_score_calibration=False, dry_run=True, no_artifacts=True,
                artifacts_repo=str(artifacts), run_id=None,
                include_raw_ocr=False, include_facts=False,
                push_artifacts=False,
            )
            with patch("scripts.run_csbench.PROJECT_ROOT", root):
                self.assertEqual(calibrate(calibration_args), 0)

            args = SimpleNamespace(
                prepared_dir=str(prepared), questions=["CO_1"],
                artifacts_repo=str(artifacts), stage="validation", run_id="run1",
                include_raw_ocr=False, include_facts=False, push=False,
                a3wa_config=None, a3wa_config_questions=[],
            )
            with patch("scripts.run_csbench.PROJECT_ROOT", root):
                self.assertEqual(publish(args), 0)
                self.assertEqual(publish(args), 0)
            destination = artifacts / "csbench" / "CO_1" / "validation_runs" / "run1"
            self.assertTrue((destination / "grading" / "grading_checkpoint.json").is_file())
            run_manifest = json.loads(
                (destination / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_manifest["answer_split"], "validation")
            self.assertEqual(run_manifest["published_stage"], "validation")
            index = json.loads(
                (artifacts / "csbench" / "index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(index), 1)

            optimized.unlink()
            manifest.unlink()
            for result_file in validation.glob("*.json"):
                result_file.unlink()

            from scripts import restore_csbench_artifacts as restore_module

            argv = [
                "restore_csbench_artifacts.py",
                "CO_1",
                "--stage", "validation",
                "--run-id", "run1",
                "--artifacts-repo", str(artifacts),
                "--prepared-dir", str(prepared),
                "--force",
            ]
            with (
                patch("scripts.run_csbench.PROJECT_ROOT", root),
                patch.object(restore_module, "PROJECT_ROOT", root),
                patch.object(sys, "argv", argv),
            ):
                self.assertEqual(restore_module.main(), 0)
            self.assertTrue(optimized.is_file())
            self.assertTrue(manifest.is_file())
            self.assertTrue(
                (
                    validation
                    / "runs"
                    / "run1"
                    / "CO_1_grading_checkpoint.json"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
