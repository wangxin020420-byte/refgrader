import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmarks.evaluate_sasbench_holistic_qwk import (
    evaluate_holistic_qwk,
)
from scripts.benchmarks.filter_sasbench_official_common_subset import (
    filter_predictions,
)
from scripts.benchmarks.project_sasbench_official_protocol import (
    CORRECT_NAME,
    ProjectionFailure,
    build_projection_prompt,
    materialize_predictions,
    parse_projection_response,
    safe_projection_input,
)


class SASBenchProtocolProjectionTests(unittest.TestCase):
    def test_projection_failure_preserves_last_invalid_response(self):
        failure = ProjectionFailure(
            "invalid response",
            raw_response='{"unexpected":true}',
        )
        self.assertEqual(failure.raw_response, '{"unexpected":true}')

    def test_prompt_uses_only_label_blind_source_fields(self):
        source = {
            "id": "A1",
            "question": "Q",
            "reference": "R",
            "analysis": "A",
            "total": 5,
            "manual_label": 1,
            "steps": [
                {
                    "response": "student response",
                    "label": 1,
                    "errors": ["teacher-only-error"],
                }
            ],
        }
        safe = safe_projection_input(source)
        prompt = build_projection_prompt(
            safe,
            {
                "guideline": "G",
                "errors": [{"name": "allowed", "description": "D"}],
            },
        )
        self.assertNotIn("manual_label", json.dumps(safe))
        self.assertNotIn("teacher-only-error", prompt)
        self.assertNotIn('"label"', prompt)
        self.assertIn("student response", prompt)

    def test_projection_contract_requires_exact_steps_and_known_errors(self):
        valid = parse_projection_response(
            '{"steps":[{"step_score":2,"errors":["allowed"]},'
            '{"step_score":1,"errors":["步骤正确"]}]}',
            expected_steps=2,
            total=5,
            allowed_errors={"allowed", CORRECT_NAME},
        )
        self.assertEqual(valid[0]["step_score"], 2)
        with self.assertRaisesRegex(ValueError, "Step count mismatch"):
            parse_projection_response(
                '{"steps":[{"step_score":2,"errors":["allowed"]}]}',
                expected_steps=2,
                total=5,
                allowed_errors={"allowed", CORRECT_NAME},
            )
        with self.assertRaisesRegex(ValueError, "Unknown error"):
            parse_projection_response(
                '{"steps":[{"step_score":2,"errors":["invented"]}]}',
                expected_steps=1,
                total=5,
                allowed_errors={"allowed", CORRECT_NAME},
            )

    def test_materialized_prediction_preserves_source_gold_for_official_evaluator(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            records = {
                "0_Task": [
                    {
                        "id": "A1",
                        "total": 5,
                        "manual_label": 4,
                        "steps": [{"response": "x", "label": 4, "errors": []}],
                    }
                ]
            }
            written = materialize_predictions(
                records,
                {"A1": {"pred_steps": [{"step_score": 3, "errors": [CORRECT_NAME]}]}},
                {"A1": 3.9},
                output,
            )
            self.assertEqual(written, 1)
            row = json.loads(
                (output / "0_Task_prediction.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(row["manual_label"], 4)
            self.assertEqual(row["pred_label"], 3)
            self.assertEqual(len(row["pred_steps"]), len(row["steps"]))

    def test_holistic_qwk_groups_by_source_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "runtime_exam_database.json"
            database.write_text(
                json.dumps(
                    [
                        {
                            "question_id": "Q1",
                            "source_task": "0_Task",
                            "total_score": 5,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            compare = root / "compare.csv"
            fields = ["question", "student_id", "teacher", *(
                "single_first_score model_avg_score selected_baseline_score "
                "three_way_core_score final_calibrated_score"
            ).split()]
            with compare.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    dict.fromkeys(fields, "0")
                    | {"question": "Q1", "student_id": "A1", "teacher": "1"}
                )
                writer.writerow(
                    dict.fromkeys(fields, "1")
                    | {"question": "Q1", "student_id": "A2", "teacher": "1"}
                )
            rows, summary = evaluate_holistic_qwk(compare, database)
            self.assertEqual(summary["record_count"], 2)
            self.assertEqual(summary["task_count"], 1)
            self.assertEqual(len(rows), 5)

    def test_filter_predictions_keeps_only_common_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            path = source / "0_Task_prediction.jsonl"
            path.write_text(
                json.dumps({"id": "A1"}) + "\n" + json.dumps({"id": "A2"}) + "\n",
                encoding="utf-8",
            )
            tasks, records = filter_predictions(source, output, {"A2"})
            self.assertEqual((tasks, records), (1, 1))
            self.assertEqual(
                json.loads((output / path.name).read_text(encoding="utf-8"))["id"],
                "A2",
            )


if __name__ == "__main__":
    unittest.main()
