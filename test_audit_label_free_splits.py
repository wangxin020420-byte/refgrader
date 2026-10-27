import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.audit_label_free_splits import audit_splits
from scripts.prepare_csbench import split_answer_ids


class LabelFreeSplitAuditTests(unittest.TestCase):
    def test_matches_id_based_split_without_reading_teacher_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "splits/by_question").mkdir(parents=True)
            manifest = {"split_seed": "s", "calibration_ratio": 0.2,
                        "validation_ratio": 0.2, "minimum_calibration_size": 1}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            ids = [f"A{i}" for i in range(5)]
            (root / "answer_metadata.jsonl").write_text("\n".join(json.dumps({"answer_id": a, "question_id": "Q1"}) for a in ids), encoding="utf-8")
            split = split_answer_ids(ids, "s:Q1", 0.2, 0.2, 1)
            path = root / "splits/by_question/Q1.json"
            path.write_text(json.dumps({"question_id": "Q1", "split_seed": "s", **split}), encoding="utf-8")
            (root / "teacher_scores.json").write_text("forbidden", encoding="utf-8")
            original_open = Path.open
            def guard(path, *args, **kwargs):
                if path.name == "teacher_scores.json":
                    raise AssertionError("teacher file accessed")
                return original_open(path, *args, **kwargs)
            with patch.object(Path, "open", guard):
                result = audit_splits(root)
            self.assertEqual(result["matching_questions"], 1)
            self.assertEqual(result["answers"], 5)
            self.assertFalse(result["teacher_scores_opened"])
            split["test"].reverse()
            path.write_text(json.dumps({"question_id": "Q1", "split_seed": "s", **split}), encoding="utf-8")
            self.assertEqual(audit_splits(root)["mismatched_question_ids"], ["Q1"])


if __name__ == "__main__":
    unittest.main()
