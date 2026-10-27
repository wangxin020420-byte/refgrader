import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from label_free_inputs import load_unlabeled_metadata


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


if __name__ == "__main__":
    unittest.main()
