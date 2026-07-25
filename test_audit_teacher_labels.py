import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_teacher_labels import analyze_question


def checkpoint_row(answer_id: str, score: float) -> dict:
    return {
        "student_id": answer_id,
        "model_avg_score": score,
        "three_way_core_score": score,
        "selected_baseline_score": score,
        "final_calibrated_score": score,
        "extraction_quality": "high",
        "extraction_risk": 0.0,
        "std_dev": 0.0,
        "risk_features": {"U_E": 0.0, "U_S": 0.0, "U_R": 0.0},
    }


class TeacherLabelAuditTests(unittest.TestCase):
    def analyze(self, teacher_values: list[float], model_values: list[float]):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "CO_1_grading_checkpoint.json"
            rows = [
                checkpoint_row(f"ANS_CO_{index}", score)
                for index, score in enumerate(model_values, start=1)
            ]
            path.write_text(json.dumps(rows), encoding="utf-8")
            teacher_scores = {
                f"ANS_CO_{index}": {"CO_1": teacher}
                for index, teacher in enumerate(teacher_values, start=1)
            }
            return analyze_question(
                "CO_1",
                path,
                teacher_scores=teacher_scores,
                metadata={},
                max_score=20.0,
                score_key="model_avg_score",
                sigma=3.0,
                minimum_ratio=0.15,
                minimum_points=1.0,
                severe_ratio=0.25,
                severe_points=2.0,
            )

    def test_systematic_bias_cannot_raise_threshold_above_severe_bound(self):
        candidates, summary = self.analyze(
            teacher_values=[15.0] * 6,
            model_values=[7.0] * 6,
        )

        self.assertEqual(summary["raw_robust_threshold"], 8.0)
        self.assertEqual(summary["threshold"], 5.0)
        self.assertTrue(summary["systematic_bias"])
        self.assertEqual(summary["median_residual"], 8.0)
        self.assertEqual(len(candidates), 6)
        self.assertTrue(
            all(row["question_systematic_bias"] for row in candidates)
        )

    def test_within_question_outlier_is_reported(self):
        candidates, summary = self.analyze(
            teacher_values=[10.0, 10.0, 10.0, 10.0, 15.0],
            model_values=[10.0] * 5,
        )

        self.assertFalse(summary["systematic_bias"])
        self.assertEqual([row["answer_id"] for row in candidates], ["ANS_CO_5"])
        self.assertIn(
            "within_question_outlier",
            candidates[0]["candidate_reasons"],
        )


if __name__ == "__main__":
    unittest.main()
