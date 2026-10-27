import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from step4_vlm_grader import grade_student_3wd_pipeline
from label_free_scoring import score_question_with_grader
import test_label_free_scoring


class LabelFreeProbeTests(unittest.TestCase):
    def setUp(self):
        self.rubric = [{"id":"s1", "points":2, "answer_type":"base_number", "standard_answer_text":"42H"}]
        self.reply = json.dumps({"total_score":2, "details":[{"id":"s1", "score_given":2, "error_category":"MATCH"}]})

    def test_three_probes_return_before_legacy_config_and_boundary(self):
        with (
            patch("step4_vlm_grader.stage1_extract_with_backend", return_value=('{"s1":"42H"}', {})),
            patch("step4_vlm_grader.stage2_logic_grading", return_value=self.reply),
            patch("step4_vlm_grader.load_a3wa_runtime_config", side_effect=AssertionError("legacy config read")),
            patch("step4_vlm_grader.boundary_arbitration_agent", side_effect=AssertionError("legacy arbitration called")),
            patch("step4_vlm_grader.apply_route_score_calibration", side_effect=AssertionError("legacy residual applied")),
        ):
            result = grade_student_3wd_pipeline(
                "A1.png", "question", json.dumps(self.rubric),
                blind_checklist="checklist", extraction_backend="text_only",
                student_transcription="42H", label_free_probe_only=True,
            )
        self.assertEqual(set(result), {"baseline_score", "risks", "valid_probe_count"})
        self.assertEqual(result["valid_probe_count"], 3)
        self.assertEqual(result["baseline_score"], 2)
        self.assertEqual(set(result["risks"]), {"U_E", "U_S", "U_R"})

    def test_incomplete_probes_fail_closed_before_routing(self):
        with (
            patch("step4_vlm_grader.stage1_extract_with_backend", return_value=('{"s1":"42H"}', {})),
            patch("step4_vlm_grader.stage2_logic_grading", side_effect=[self.reply, self.reply, None]),
            patch("step4_vlm_grader.load_a3wa_runtime_config", side_effect=AssertionError("legacy config read")),
        ):
            result = grade_student_3wd_pipeline(
                "A1.png", "question", json.dumps(self.rubric),
                blind_checklist="checklist", extraction_backend="text_only",
                student_transcription="42H", label_free_probe_only=True,
            )
        self.assertTrue(result["_pipeline_failed"])
        self.assertEqual(result["error_type"], "label_free_incomplete_probes")

    def test_raw_probes_bypass_legacy_score_and_risk_postprocessing(self):
        reply = json.dumps({"total_score": 1, "details": [
            {"id": "s1", "score_given": 1, "error_category": "MATCH"}
        ]})
        with (
            patch("step4_vlm_grader.stage1_extract_with_backend", return_value=('{"s1":"42H"}', {})),
            patch("step4_vlm_grader.stage2_logic_grading", return_value=reply),
            patch("step4_vlm_grader.apply_canonical_score_floor", side_effect=AssertionError("legacy floor")),
            patch("step4_vlm_grader.apply_hierarchical_scoring_policy", side_effect=AssertionError("legacy hierarchy")),
            patch("step4_vlm_grader.apply_role_weighted_scoring_policy", side_effect=AssertionError("legacy role policy")),
            patch("step4_vlm_grader.build_post_grading_calibration", side_effect=AssertionError("legacy risk fusion")),
            patch("step4_vlm_grader.build_risk_profile", side_effect=AssertionError("legacy risk profile")),
        ):
            result = grade_student_3wd_pipeline(
                "A1.png", "question", json.dumps(self.rubric),
                blind_checklist="checklist", extraction_backend="text_only",
                student_transcription="42H", label_free_probe_only=True,
            )
        self.assertEqual(result["baseline_score"], 1)
        self.assertEqual(result["valid_probe_count"], 3)
        self.assertEqual(set(result["risks"]), {"U_E", "U_S", "U_R"})

    def test_mocked_extraction_and_probes_reach_separate_three_way_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            test_label_free_scoring.LabelFreeScoringTests().fixture(root)
            (root / "rubrics/initial/Q/Q_1_rubric_standard.json").write_text(
                json.dumps([{"id":"s1", "points":10, "answer_type":"base_number", "standard_answer_text":"42H"}]), encoding="utf-8")
            teacher = root / "teacher_scores.json"
            teacher.write_text("forbidden", encoding="utf-8")
            original = Path.open
            def guard(path, *args, **kwargs):
                if path.name == "teacher_scores.json":
                    raise AssertionError("teacher score file read")
                return original(path, *args, **kwargs)
            def grader(context, answer):
                return grade_student_3wd_pipeline(
                    answer["answer_id"] + ".png", context["question_text"],
                    json.dumps(context["initial_rubric"]), blind_checklist="checklist",
                    extraction_backend="text_only", student_transcription=answer["raw_text"],
                    answer_metadata=answer, label_free_probe_only=True,
                )
            with (
                patch.object(Path, "open", guard),
                patch("step4_vlm_grader.stage1_extract_with_backend", return_value=('{"s1":"42H"}', {})),
                patch("step4_vlm_grader.stage2_logic_grading", return_value=self.reply),
                patch("step4_vlm_grader.load_a3wa_runtime_config", side_effect=AssertionError("legacy config read")),
                patch("step4_vlm_grader.boundary_arbitration_agent", side_effect=AssertionError("legacy arbitration called")),
            ):
                result = score_question_with_grader(root, "Q_1", "test", grader, m=0.5,
                    loss_parameters={"lambda1":5,"lambda2":1,"mu1":3,"mu2":7})
            self.assertEqual(len(result["results"]), 3)
            self.assertFalse(result["ready_for_live_scoring"])
            self.assertFalse(result["grader_provenance_audited"])


if __name__ == "__main__":
    unittest.main()
