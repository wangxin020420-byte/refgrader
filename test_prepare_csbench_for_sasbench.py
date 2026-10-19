import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmarks.prepare_csbench_for_sasbench import prepare


class PrepareCSBenchForSASBenchTests(unittest.TestCase):
    def test_prepares_only_nonempty_split_answers_without_optimized_rubric(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "prepared"
            output = Path(temp) / "output"
            (root / "splits").mkdir(parents=True)
            (root / "exam_database.json").write_text(
                json.dumps(
                    [
                        {
                            "question_id": "Q_1",
                            "subject": "CS",
                            "question_text": "question",
                            "ref_text": "reference",
                            "official_rubric": "official guideline",
                            "optimized_rubric_path": "must/not/be/read.json",
                            "total_score": 5,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (root / "teacher_scores.json").write_text(
                json.dumps({"A1": {"Q_1": 3.5}, "A2": {"Q_1": 0}}),
                encoding="utf-8",
            )
            (root / "answer_metadata.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {"answer_id": "A1", "question_id": "Q_1", "raw_text": "answer"}
                        ),
                        json.dumps(
                            {"answer_id": "A2", "question_id": "Q_1", "raw_text": ""}
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "splits" / "test.json").write_text(
                json.dumps(
                    {"question_ids": ["Q_1"], "answer_ids": ["A1", "A2"]}
                ),
                encoding="utf-8",
            )

            manifest = prepare(root, output)

            self.assertEqual(manifest["source_record_count"], 2)
            self.assertEqual(manifest["eligible_record_count"], 1)
            self.assertEqual(manifest["formal_record_count"], 1)
            self.assertEqual(manifest["excluded_record_count"], 1)
            self.assertFalse(manifest["optimized_rubric_used"])
            task = json.loads(
                (output / "0_Q_1_ShortAns.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(task["manual_label"], 3.5)
            self.assertIsNone(task["steps"][0]["label"])
            self.assertEqual(task["analysis"], "")
            error_contract = json.loads(
                (output / "error_type.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(error_contract["guideline"], "official guideline")

    def test_include_empty_preserves_blank_as_a_real_response_type(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "prepared"
            output = Path(temp) / "output"
            (root / "splits").mkdir(parents=True)
            question = {
                "question_id": "Q_1",
                "subject": "CS",
                "question_text": "question",
                "ref_text": "reference",
                "official_rubric": "rubric",
                "total_score": 5,
            }
            (root / "exam_database.json").write_text(
                json.dumps([question]), encoding="utf-8"
            )
            (root / "teacher_scores.json").write_text(
                json.dumps({"A1": {"Q_1": 0}}), encoding="utf-8"
            )
            (root / "answer_metadata.jsonl").write_text(
                json.dumps(
                    {
                        "answer_id": "A1",
                        "question_id": "Q_1",
                        "raw_text": "",
                        "isimagine": False,
                        "visual_placeholder_detected": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "splits" / "test.json").write_text(
                json.dumps({"question_ids": ["Q_1"], "answer_ids": ["A1"]}),
                encoding="utf-8",
            )

            manifest = prepare(root, output, include_empty=True)

            self.assertEqual(manifest["formal_record_count"], 1)
            self.assertEqual(manifest["excluded_record_count"], 0)
            self.assertEqual(manifest["formal_response_type_counts"]["blank"], 1)
            row = json.loads(
                (output / "0_Q_1_ShortAns.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(row["response_type"], "blank")
            self.assertEqual(row["steps"][0]["response"], "")

    def test_per_question_protocol_uses_each_questions_internal_test_partition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "prepared"
            output = Path(temp) / "output"
            (root / "splits" / "by_question").mkdir(parents=True)
            questions = []
            scores = {}
            metadata = []
            for index in range(1, 3):
                question_id = f"Q_{index}"
                questions.append(
                    {
                        "question_id": question_id,
                        "subject": "CS",
                        "question_text": f"question {index}",
                        "ref_text": f"reference {index}",
                        "official_rubric": f"rubric {index}",
                        "total_score": 5,
                    }
                )
                answer_ids = [f"A{index}_cal", f"A{index}_test"]
                for answer_id in answer_ids:
                    scores[answer_id] = {question_id: 4}
                    metadata.append(
                        {
                            "answer_id": answer_id,
                            "question_id": question_id,
                            "raw_text": answer_id,
                        }
                    )
                (root / "splits" / "by_question" / f"{question_id}.json").write_text(
                    json.dumps(
                        {
                            "question_id": question_id,
                            "calibration": [answer_ids[0]],
                            "validation": [],
                            "test": [answer_ids[1]],
                        }
                    ),
                    encoding="utf-8",
                )
            (root / "exam_database.json").write_text(
                json.dumps(questions), encoding="utf-8"
            )
            (root / "teacher_scores.json").write_text(
                json.dumps(scores), encoding="utf-8"
            )
            (root / "answer_metadata.jsonl").write_text(
                "\n".join(json.dumps(row) for row in metadata) + "\n",
                encoding="utf-8",
            )

            manifest = prepare(
                root,
                output,
                selection_protocol="per-question-test",
            )

            self.assertEqual(manifest["selection_protocol"], "per-question-test")
            self.assertEqual(manifest["source_question_count"], 2)
            self.assertEqual(manifest["source_record_count"], 2)
            self.assertEqual(manifest["formal_record_count"], 2)
            written_ids = {
                json.loads(path.read_text(encoding="utf-8"))["id"]
                for path in output.glob("*_ShortAns.jsonl")
            }
            self.assertEqual(written_ids, {"A1_test", "A2_test"})


if __name__ == "__main__":
    unittest.main()
