import json
import tempfile
import unittest
from pathlib import Path

from calibration_utils import apply_structured_boundary_action_policy
from main_pipeline import (
    persist_validated_rubric,
    restore_activation_files,
    snapshot_activation_files,
    validate_activated_optimized_rubric,
    validate_retained_optimized_rubric,
)
from unittest.mock import patch
from scripts.audit_question_splits import audit
from scripts.calibrate_a3wa import calibrate_boundary_action_gate
from scripts.run_csbench import build_parser, grade, validate_a3wa_deployment_gate


class ExperimentGateTests(unittest.TestCase):
    def test_grade_can_require_complete_coverage_for_unattended_retry(self):
        args = build_parser().parse_args([
            "grade",
            "CO_4",
            "--split",
            "test",
            "--require-complete",
        ])
        self.assertTrue(args.require_complete)

    def test_baseline_rubric_fallback_is_restricted_to_all_audit(self):
        args = build_parser().parse_args([
            "grade",
            "CO_4",
            "--split",
            "test",
            "--allow-baseline-rubric-fallback",
        ])
        with patch("scripts.run_csbench.build_contexts", return_value=[]):
            with self.assertRaisesRegex(ValueError, "restricted to --split all"):
                grade(args)

    def test_question_split_audit_covers_portable_snapshot(self):
        report = audit(Path("data/csbench"))
        self.assertEqual(report["status"], "passed", report["errors"])
        self.assertEqual(report["question_count"], 43)
        self.assertEqual(
            report["outer_question_splits"]["train"]["question_count"], 31
        )
        self.assertEqual(report["answer_count"], 3326)
        self.assertEqual(report["answer_metadata_count"], 3326)

    def test_failed_a3wa_gate_is_blocked_unless_explicitly_experimental(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "deployment_gate": {
                    "passed": False,
                    "requirements": {"validation_bnd_gain_positive": False},
                }
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "deployment gate did not pass"):
                validate_a3wa_deployment_gate(path)
            payload = validate_a3wa_deployment_gate(
                path, allow_experimental=True
            )
            self.assertFalse(payload["deployment_gate"]["passed"])

    def test_boundary_direction_requires_enough_positive_validation_gain(self):
        trial = [
            {
                "trial_route": "BND",
                "boundary_action": "accept_structured_raise",
                "avg": 4.0,
                "trial_score": 5.0,
                "teacher": 5.0,
            }
            for _ in range(3)
        ]
        trial.extend([
            {
                "trial_route": "BND",
                "boundary_action": "accept_structured_lower",
                "avg": 5.0,
                "trial_score": 4.0,
                "teacher": 6.0,
            }
            for _ in range(3)
        ])
        gate = calibrate_boundary_action_gate(trial, min_count=3)
        self.assertTrue(gate["allow_raise"])
        self.assertFalse(gate["allow_lower"])

    def test_disabled_lower_action_keeps_baseline(self):
        evidence = {
            "confidence": 0.9,
            "allowed_missed_points": 0.0,
            "allowed_over_points": 2.0,
            "missed_reason_types": [],
            "over_reason_types": ["contradiction"],
        }
        result = apply_structured_boundary_action_policy(
            avg_model_score=5.0,
            candidate_score=3.0,
            max_score=10.0,
            post_calibration={"rubric_item_points": {"step_1": 2.0}},
            agent_evidence=evidence,
            config={"allow_lower": False},
        )
        self.assertEqual(result["action"], "keep_baseline")
        self.assertEqual(result["final_score"], 5.0)
        self.assertEqual(
            result["gate_reason"], "lower_action_disabled_by_validation_gate"
        )

    def test_activation_transaction_restores_existing_and_created_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing.json"
            created = root / "created.json"
            existing.write_bytes(b"before")
            snapshot = snapshot_activation_files([existing, created])

            existing.write_bytes(b"after")
            created.write_bytes(b"new")
            restore_activation_files(snapshot)

            self.assertEqual(existing.read_bytes(), b"before")
            self.assertFalse(created.exists())

    def test_persisted_candidate_uses_baseline_fallback_for_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial.json"
            optimized = root / "optimized.json"
            baseline = [{
                "id": "step_1",
                "item": "derive result",
                "standard_answer_text": "answer",
                "points": 5.0,
            }]
            candidate = [{
                "id": "step_1_child",
                "parent_id": "step_1",
                "item": "candidate",
                "standard_answer_text": "answer",
                "points": 5.0,
            }]
            initial.write_text(json.dumps(baseline), encoding="utf-8")

            with patch(
                "main_pipeline.validate_refined_rubric",
                side_effect=[
                    (False, ["additive parent step_1 has one child"]),
                    (True, []),
                ],
            ):
                result = persist_validated_rubric(
                    initial,
                    optimized,
                    candidate,
                    5.0,
                    allow_baseline_fallback=True,
                )

            self.assertTrue(result["used_baseline_fallback"])
            self.assertEqual(
                result["candidate_errors"],
                ["additive parent step_1 has one child"],
            )
            self.assertEqual(
                json.loads(optimized.read_text(encoding="utf-8")),
                result["rubric"],
            )

    def test_persisted_candidate_remains_strict_outside_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial.json"
            optimized = root / "optimized.json"
            initial.write_text("[]", encoding="utf-8")
            with patch(
                "main_pipeline.validate_refined_rubric",
                return_value=(False, ["invalid persisted candidate"]),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Persisted optimized rubric failed",
                ):
                    persist_validated_rubric(
                        initial,
                        optimized,
                        [],
                        5.0,
                    )

    def test_rejected_candidate_can_retain_only_a_valid_incumbent(self):
        question = {"question_id": "CO_1", "total_score": 5.0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial.json"
            optimized = root / "optimized.json"
            manifest = root / "manifest.json"
            rubric = [{
                "id": "step_1",
                "description": "final answer",
                "standard_answer_text": "37H",
                "points": 5.0,
            }]
            initial.write_text(json.dumps(rubric), encoding="utf-8")
            optimized.write_text(json.dumps(rubric), encoding="utf-8")
            manifest.write_text(json.dumps({
                "semantic_validation_mode": "strict_candidate_contract",
            }), encoding="utf-8")

            with (
                patch("main_pipeline.optimized_rubric_output_path", return_value=str(optimized)),
                patch("main_pipeline.optimization_manifest_path", return_value=str(manifest)),
                patch("main_pipeline.initial_rubric_path_for", return_value=str(initial)),
                patch("main_pipeline.validate_optimized_rubric_provenance"),
                patch("main_pipeline.validate_refined_rubric", return_value=(True, [])),
            ):
                retained = validate_retained_optimized_rubric(question)
            self.assertEqual(retained["question_id"], "CO_1")
            self.assertTrue(retained["optimized_sha256"])

    def test_diagnostic_fallback_cannot_be_retained(self):
        question = {"question_id": "CO_1", "total_score": 5.0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial.json"
            optimized = root / "optimized.json"
            manifest = root / "manifest.json"
            initial.write_text("[]", encoding="utf-8")
            optimized.write_text("[]", encoding="utf-8")
            manifest.write_text(json.dumps({
                "semantic_validation_mode": "noninferiority_baseline_fallback",
            }), encoding="utf-8")
            with (
                patch("main_pipeline.optimized_rubric_output_path", return_value=str(optimized)),
                patch("main_pipeline.optimization_manifest_path", return_value=str(manifest)),
                patch("main_pipeline.initial_rubric_path_for", return_value=str(initial)),
                patch("main_pipeline.validate_optimized_rubric_provenance"),
            ):
                with self.assertRaisesRegex(ValueError, "diagnostic baseline fallback"):
                    validate_retained_optimized_rubric(question)

    def test_diagnostic_fallback_passes_persisted_postcondition_only_when_allowed(self):
        question = {"question_id": "CO_1", "total_score": 5.0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial.json"
            optimized = root / "optimized.json"
            manifest = root / "manifest.json"
            rubric = [{
                "id": "step_1",
                "item": "final answer",
                "standard_answer_text": "37H",
                "points": 5.0,
            }]
            initial.write_text(json.dumps(rubric), encoding="utf-8")
            optimized.write_text(json.dumps(rubric), encoding="utf-8")
            manifest.write_text(json.dumps({
                "semantic_validation_mode": "noninferiority_baseline_fallback",
            }), encoding="utf-8")

            with (
                patch("main_pipeline.optimized_rubric_output_path", return_value=str(optimized)),
                patch("main_pipeline.optimization_manifest_path", return_value=str(manifest)),
                patch("main_pipeline.initial_rubric_path_for", return_value=str(initial)),
                patch("main_pipeline.validate_optimized_rubric_provenance"),
                patch("main_pipeline.validate_refined_rubric", return_value=(True, [])),
            ):
                with self.assertRaisesRegex(ValueError, "diagnostic baseline fallback"):
                    validate_activated_optimized_rubric(question)
                activated = validate_activated_optimized_rubric(
                    question,
                    allow_baseline_fallback=True,
                )

            self.assertEqual(
                activated["semantic_validation_mode"],
                "noninferiority_baseline_fallback",
            )
            self.assertTrue(activated["optimized_sha256"])

    def test_persisted_postcondition_rejects_invalid_active_rubric(self):
        question = {"question_id": "POC_6", "total_score": 10.0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial.json"
            optimized = root / "optimized.json"
            manifest = root / "manifest.json"
            initial.write_text("[]", encoding="utf-8")
            optimized.write_text("[]", encoding="utf-8")
            manifest.write_text(json.dumps({
                "semantic_validation_mode": "strict_candidate_contract",
            }), encoding="utf-8")

            with (
                patch("main_pipeline.optimized_rubric_output_path", return_value=str(optimized)),
                patch("main_pipeline.optimization_manifest_path", return_value=str(manifest)),
                patch("main_pipeline.initial_rubric_path_for", return_value=str(initial)),
                patch("main_pipeline.validate_optimized_rubric_provenance"),
                patch(
                    "main_pipeline.validate_refined_rubric",
                    return_value=(False, ["role-weighted parent step_1 is invalid"]),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "persisted rubric"):
                    validate_activated_optimized_rubric(question)

    def test_calibrated_noninferior_baseline_can_be_retained(self):
        question = {"question_id": "CO_3", "total_score": 10.0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial.json"
            optimized = root / "optimized.json"
            manifest = root / "manifest.json"
            rubric = [{
                "id": "step_1",
                "item": "derive the result",
                "points": 10.0,
                "standard_answer_text": "result with process",
            }]
            initial.write_text(json.dumps(rubric), encoding="utf-8")
            optimized.write_text(json.dumps(rubric), encoding="utf-8")
            manifest.write_text(json.dumps({
                "semantic_validation_mode": "calibrated_noninferior_baseline_selected",
                "selected_variant": "baseline",
                "decomposition_deferred": True,
                "candidate_replay": {
                    "method": "paired_teacher_score_noninferiority",
                    "expected": 5,
                    "paired": 5,
                    "required": 4,
                    "accepted": False,
                    "reason": "candidate_mae_exceeds_margin",
                    "baseline_mae": 1.6,
                    "candidate_mae": 2.0,
                    "mae_margin": 0.2,
                    "severe_regressions": 0,
                },
            }), encoding="utf-8")

            with (
                patch("main_pipeline.optimized_rubric_output_path", return_value=str(optimized)),
                patch("main_pipeline.optimization_manifest_path", return_value=str(manifest)),
                patch("main_pipeline.initial_rubric_path_for", return_value=str(initial)),
                patch("main_pipeline.validate_optimized_rubric_provenance"),
            ):
                retained = validate_retained_optimized_rubric(question)
            self.assertEqual(retained["question_id"], "CO_3")


if __name__ == "__main__":
    unittest.main()
