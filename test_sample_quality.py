import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evaluate
from sample_quality import SampleQualityPolicy, default_policy_path
from scripts.calibrate_a3wa import load_records
from scripts.evaluate_artifacts import stage_artifact_results
from scripts.run_csbench import effective_split_ids


class SampleQualityPolicyTests(unittest.TestCase):
    def setUp(self):
        policy_mode = patch.dict(
            os.environ,
            {"REFGRADER_SAMPLE_POLICY_MODE": "active"},
        )
        policy_mode.start()
        self.addCleanup(policy_mode.stop)

    def make_root(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        return temporary, root

    def write_policy(self, root, payload):
        path = default_policy_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def test_absent_policy_preserves_raw_behavior(self):
        temporary, root = self.make_root()
        self.addCleanup(temporary.cleanup)
        policy = SampleQualityPolicy.load(root)
        self.assertEqual(policy.mode, "raw")
        self.assertEqual(
            policy.filter_ids("CO_1", ["A", "B"]),
            {"A", "B"},
        )
        self.assertEqual(
            policy.effective_teacher_score("CO_1", "A", 4),
            4.0,
        )

    def test_active_policy_filters_and_overlays_without_mutating_source(self):
        temporary, root = self.make_root()
        self.addCleanup(temporary.cleanup)
        source_scores = {"A": 4.0, "B": 3.0, "C": 2.0}
        self.write_policy(
            root,
            {
                "schema_version": 1,
                "policy_id": "review-v1",
                "excluded": {"CO_1": {"A": {"reason": "noise"}}},
                "corrected_scores": {"CO_1": {"B": 4.5}},
            },
        )
        policy = SampleQualityPolicy.load(root)
        self.assertEqual(policy.filter_ids("CO_1", source_scores), {"B", "C"})
        self.assertIsNone(
            policy.effective_teacher_score("CO_1", "A", source_scores["A"])
        )
        self.assertEqual(
            policy.effective_teacher_score("CO_1", "B", source_scores["B"]),
            4.5,
        )
        self.assertEqual(source_scores["B"], 3.0)

    def test_policy_rejects_excluded_corrected_overlap(self):
        temporary, root = self.make_root()
        self.addCleanup(temporary.cleanup)
        self.write_policy(
            root,
            {
                "schema_version": 1,
                "policy_id": "invalid",
                "excluded": {"CO_1": ["A"]},
                "corrected_scores": {"CO_1": {"A": 4}},
            },
        )
        with self.assertRaisesRegex(ValueError, "both excluded and corrected"):
            SampleQualityPolicy.load(root)

    def test_run_split_selection_uses_active_policy(self):
        temporary, root = self.make_root()
        self.addCleanup(temporary.cleanup)
        split_file = root / "split.json"
        split_file.write_text(
            json.dumps({"test": ["A", "B", "C"]}),
            encoding="utf-8",
        )
        self.write_policy(
            root,
            {
                "schema_version": 1,
                "policy_id": "review-v1",
                "excluded": {"CO_1": ["B"]},
                "corrected_scores": {},
            },
        )
        ctx = SimpleNamespace(
            root=root,
            question_id="CO_1",
            split_file=split_file,
        )
        self.assertEqual(effective_split_ids(ctx, "test"), {"A", "C"})

    def test_calibration_skips_excluded_and_uses_corrected_score(self):
        temporary, root = self.make_root()
        self.addCleanup(temporary.cleanup)
        checkpoint = root / "CO_1_grading_checkpoint.json"
        checkpoint.write_text(
            json.dumps(
                [
                    {
                        "student_id": "A",
                        "model_avg_score": 2,
                        "final_calibrated_score": 2,
                    },
                    {
                        "student_id": "B",
                        "model_avg_score": 3,
                        "final_calibrated_score": 3,
                    },
                ]
            ),
            encoding="utf-8",
        )
        self.write_policy(
            root,
            {
                "schema_version": 1,
                "policy_id": "review-v1",
                "excluded": {"CO_1": ["A"]},
                "corrected_scores": {"CO_1": {"B": 4}},
            },
        )
        policy = SampleQualityPolicy.load(root)
        records = load_records(
            [str(checkpoint)],
            {"A": {"CO_1": 1}, "B": {"CO_1": 2}},
            {"CO_1": 5},
            policy,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["teacher"], 4.0)

    def test_evaluation_uses_same_policy(self):
        temporary, root = self.make_root()
        self.addCleanup(temporary.cleanup)
        self.write_policy(
            root,
            {
                "schema_version": 1,
                "policy_id": "review-v1",
                "excluded": {"CO_1": ["A"]},
                "corrected_scores": {"CO_1": {"B": 4}},
            },
        )
        previous = evaluate.SAMPLE_QUALITY_POLICY
        evaluate.SAMPLE_QUALITY_POLICY = SampleQualityPolicy.load(root)
        self.addCleanup(
            setattr, evaluate, "SAMPLE_QUALITY_POLICY", previous
        )
        scores = {"A": {"CO_1": 1}, "B": {"CO_1": 2}}
        self.assertIsNone(evaluate.get_teacher(scores, "A", "CO_1"))
        self.assertEqual(evaluate.get_teacher(scores, "B", "CO_1"), 4.0)

    def test_artifact_evaluation_stages_the_run_policy(self):
        temporary, root = self.make_root()
        self.addCleanup(temporary.cleanup)
        artifacts = root / "artifacts"
        results = root / "results"
        run = (
            artifacts
            / "csbench"
            / "CO_1"
            / "grading_runs"
            / "run1"
        )
        (run / "grading").mkdir(parents=True)
        (run / "dataset").mkdir()
        (run / "grading" / "grading_checkpoint.json").write_text(
            "[]", encoding="utf-8"
        )
        policy_payload = {
            "schema_version": 1,
            "policy_id": "review-v1",
            "excluded": {"CO_1": ["A"]},
            "corrected_scores": {},
        }
        policy_file = run / "dataset" / "sample_quality_policy.json"
        policy_file.write_text(json.dumps(policy_payload), encoding="utf-8")
        descriptor = SampleQualityPolicy.load(
            root, explicit_path=policy_file
        ).descriptor()
        (run / "run_manifest.json").write_text(
            json.dumps(
                {
                    "question_id": "CO_1",
                    "sample_quality_policy": descriptor,
                    "completion": {},
                }
            ),
            encoding="utf-8",
        )
        _, staged_policy = stage_artifact_results(
            artifacts_repo=artifacts,
            questions=["CO_1"],
            run_id="run1",
            results_dir=results,
        )
        self.assertIsNotNone(staged_policy)
        self.assertEqual(
            json.loads(staged_policy.read_text(encoding="utf-8")),
            policy_payload,
        )


if __name__ == "__main__":
    unittest.main()
