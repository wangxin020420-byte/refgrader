import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmarks.evaluate_csbench_refgrader_stratified import (
    materialize_predictions,
)
from scripts.benchmarks.evaluate_csbench_sasbench_baseline import evaluate


class RefGraderStratifiedEvaluationTests(unittest.TestCase):
    def test_materializes_compare_scores_without_changing_source_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "evaluation"
            source.mkdir()
            records = [
                {
                    "id": "A1",
                    "manual_label": 1.0,
                    "total": 2.0,
                    "response_type": "pure_text",
                    "steps": [{"response": "answer"}],
                },
                {
                    "id": "A2",
                    "manual_label": 0.0,
                    "total": 2.0,
                    "response_type": "blank",
                    "steps": [{"response": ""}],
                },
            ]
            with (source / "0_Q1_ShortAns.jsonl").open(
                "w", encoding="utf-8"
            ) as handle:
                for row in records:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            compare = root / "compare.csv"
            with compare.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["student_id", "teacher", "final_calibrated_score"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "student_id": "A1",
                            "teacher": 1.0,
                            "final_calibrated_score": 1.5,
                        },
                        {
                            "student_id": "A2",
                            "teacher": 0.0,
                            "final_calibrated_score": 0.0,
                        },
                    ]
                )
            prediction_dir = materialize_predictions(
                compare, source, output, "final_calibrated_score", 2
            )
            summary = evaluate(prediction_dir, output, 2)
            self.assertEqual(summary["record_count"], 2)
            self.assertEqual(
                summary["response_type_metrics"]["pure_text"]["record_count"], 1
            )
            self.assertEqual(
                summary["response_type_metrics"]["blank"]["record_count"], 1
            )
            rows = [
                json.loads(line)
                for line in (
                    prediction_dir / "0_Q1_ShortAns_prediction.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[0]["manual_label"], 1.0)
            self.assertEqual(rows[0]["pred_label"], 1.5)


if __name__ == "__main__":
    unittest.main()
