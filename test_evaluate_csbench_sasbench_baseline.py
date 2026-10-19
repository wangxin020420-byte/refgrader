import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmarks.evaluate_csbench_sasbench_baseline import evaluate


class EvaluateCSBenchSASBenchBaselineTests(unittest.TestCase):
    def test_preserves_half_points_and_rejects_missing_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prediction = root / "0_Q_prediction.jsonl"
            rows = [
                {"id": "A1", "manual_label": 2.5, "pred_label": 2.5, "total": 5},
                {"id": "A2", "manual_label": 5.0, "pred_label": 4.0, "total": 5},
            ]
            prediction.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            summary = evaluate(root, root / "evaluation", 2)
            self.assertEqual(summary["record_count"], 2)
            self.assertAlmostEqual(summary["global_holistic_metrics"]["MAE"], 0.5)
            self.assertIsNone(summary["CCS"])
            with self.assertRaisesRegex(ValueError, "Prediction count mismatch"):
                evaluate(root, root / "other", 3)

    def test_reports_metrics_by_response_type(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prediction = root / "0_Q_prediction.jsonl"
            rows = [
                {
                    "id": "A1",
                    "manual_label": 2,
                    "pred_label": 2,
                    "total": 5,
                    "response_type": "pure_text",
                    "steps": [{"response": "text"}],
                },
                {
                    "id": "A2",
                    "manual_label": 1,
                    "pred_label": 0,
                    "total": 5,
                    "response_type": "visual_dependent",
                    "steps": [{"response": "如图所示"}],
                },
                {
                    "id": "A3",
                    "manual_label": 0,
                    "pred_label": 0,
                    "total": 5,
                    "response_type": "blank",
                    "steps": [{"response": ""}],
                },
            ]
            prediction.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            summary = evaluate(root, root / "evaluation", 3)

            by_type = summary["response_type_metrics"]
            self.assertEqual(by_type["pure_text"]["record_count"], 1)
            self.assertEqual(by_type["visual_dependent"]["record_count"], 1)
            self.assertEqual(by_type["blank"]["record_count"], 1)
            self.assertTrue(
                (root / "evaluation" / "metrics_by_question_and_response_type.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
