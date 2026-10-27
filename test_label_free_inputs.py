import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from label_free_inputs import load_unlabeled_metadata, load_unlabeled_question


class LabelFreeInputsTests(unittest.TestCase):
    def test_loads_only_metadata_without_opening_teacher_database(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "answer_metadata.jsonl"
            path.write_text(json.dumps({"answer_id": "A1", "question_id": "Q1", "raw_text": "student"}) + "\n", encoding="utf-8")
            # A teacher database must not be accessed even when it is present.
            (root / "teacher_scores.json").write_text('{"A1": 5}', encoding="utf-8")
            original_open = Path.open
            def guarded_open(file, *args, **kwargs):
                if file.name == "teacher_scores.json":
                    raise AssertionError("teacher database read")
                return original_open(file, *args, **kwargs)
            with patch.object(Path, "open", guarded_open):
                self.assertEqual(load_unlabeled_metadata(path)["A1"]["question_id"], "Q1")

    def test_rejects_teacher_labels_and_unknown_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.jsonl"
            for field in ("teacher_score", "label", "review_reason", "unknown_column"):
                path.write_text(json.dumps({"answer_id": "A1", "question_id": "Q1", field: 1}) + "\n", encoding="utf-8")
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_unlabeled_metadata(path)

    def test_rejects_duplicate_ids_and_invalid_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.jsonl"
            record = {"answer_id": "A1", "question_id": "Q1", "raw_text": "x"}
            path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_unlabeled_metadata(path)
            for invalid in ({"answer_id": "", "question_id": "Q1"}, {"answer_id": "A1", "question_id": "Q1", "raw_text": {"teacher_score": 5}}):
                path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    load_unlabeled_metadata(path)

    def test_initial_rubric_and_raw_split_do_not_access_review_or_teacher(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "rubrics/initial/Q").mkdir(parents=True)
            (root / "splits/by_question").mkdir(parents=True)
            (root / "exam_database.json").write_text(json.dumps([{"question_id":"Q_1", "rubric_group":"Q", "total_score": 2}]), encoding="utf-8")
            (root / "rubrics/initial/Q/Q_1_rubric_standard.json").write_text('[{"points": 2}]', encoding="utf-8")
            (root / "answer_metadata.jsonl").write_text(json.dumps({"answer_id":"A1", "question_id":"Q_1", "raw_text":"x"}) + "\n", encoding="utf-8")
            (root / "splits/by_question/Q_1.json").write_text(json.dumps({"question_id":"Q_1", "calibration":[], "validation":[], "test":["A1"]}), encoding="utf-8")
            (root / "teacher_scores.json").write_text("forbidden", encoding="utf-8")
            original_open = Path.open
            def guard(path, *args, **kwargs):
                if path.name == "teacher_scores.json" or "quality_control" in path.parts or "optimized" in path.parts:
                    raise AssertionError("forbidden file accessed")
                return original_open(path, *args, **kwargs)
            with patch.object(Path, "open", guard):
                result = load_unlabeled_question(root, "Q_1", "test")
            self.assertEqual(len(result["answers"]), 1)
            self.assertFalse(result["teacher_labels_loaded"])
            self.assertFalse(result["content_provenance_audited"])
            split = root / "splits/by_question/Q_1.json"
            split.write_text(json.dumps({"question_id":"Q_1", "calibration":["A1"], "validation":[], "test":["A1"]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlapping"):
                load_unlabeled_question(root, "Q_1", "test")


if __name__ == "__main__":
    unittest.main()
