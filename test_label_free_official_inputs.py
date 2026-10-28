import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.audit_label_free_official_inputs import audit


class OfficialInputAuditTests(unittest.TestCase):
    def test_detects_changed_rubric_and_multiple_images_without_teacher(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            prepared = root / "prepared"
            (source / "question").mkdir(parents=True)
            (prepared / "rubrics/initial/Q").mkdir(parents=True)
            (source / "a.png").write_bytes(b"a")
            (source / "b.png").write_bytes(b"b")
            (prepared / "reference_images/standard_answers").mkdir(parents=True)
            (prepared / "reference_images/standard_answers/Q_1.png").write_bytes(b"a")
            original = [{"question_id": "Q_1", "content": {"text": "question"}, "max_score": 5,
                "grading_rubric": [
                    {"step_id": 1, "description": "item 1", "score": 2,
                     "standard_answer_text": "教师评分口径 a", "standard_answer_image": "a.png"},
                    {"step_id": 2, "description": "item 2", "score": 3,
                     "standard_answer_text": "b", "standard_answer_image": "b.png"},
                ]}]
            (source / "question/q.json").write_text(json.dumps(original), encoding="utf-8")
            (prepared / "exam_database.json").write_text(json.dumps([
                {"question_id": "Q_1", "rubric_group": "Q", "total_score": 5, "question_text": "question"}]), encoding="utf-8")
            (prepared / "rubrics/initial/Q/Q_1_rubric_standard.json").write_text(json.dumps([
                {"id": "step_1", "item": "changed", "points": 5, "standard_answer_text": "a"}]), encoding="utf-8")
            (prepared / "teacher_scores.json").write_text("forbidden", encoding="utf-8")
            open_file = Path.open
            def guarded(path, *args, **kwargs):
                if path.name == "teacher_scores.json":
                    raise AssertionError("teacher score opened")
                return open_file(path, *args, **kwargs)
            with patch.object(Path, "open", guarded):
                result = audit(source, prepared)
            self.assertFalse(result["teacher_score_file_opened"])
            self.assertEqual(result["rubric_core_changed_count"], 1)
            self.assertEqual(result["question_text_changed_count"], 0)
            self.assertEqual(result["multiple_reference_images"], [
                {"question_id": "Q_1", "upstream_reference_image_count": 2}])
            self.assertEqual(result["reference_with_teacher_word_questions"], ["Q_1"])
            self.assertEqual(result["reference_image_mismatches"], [{"question_id": "Q_1", "unmatched_source_image_count": 1, "prepared_reference_image_count": 1}])


if __name__ == "__main__":
    unittest.main()
