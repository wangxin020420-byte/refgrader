import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from label_free_scoring import score_question_with_grader


class LabelFreeScoringTests(unittest.TestCase):
    def fixture(self, root):
        (root / "rubrics/initial/Q").mkdir(parents=True)
        (root / "splits/by_question").mkdir(parents=True)
        (root / "exam_database.json").write_text(json.dumps([{"question_id":"Q_1", "rubric_group":"Q", "total_score":10}]), encoding="utf-8")
        (root / "rubrics/initial/Q/Q_1_rubric_standard.json").write_text('[{"points":10}]', encoding="utf-8")
        records = [{"answer_id": name, "question_id":"Q_1", "raw_text":"x"} for name in ("A1", "A2", "A3")]
        (root / "answer_metadata.jsonl").write_text("\n".join(json.dumps(x) for x in records), encoding="utf-8")
        (root / "splits/by_question/Q_1.json").write_text(json.dumps({"question_id":"Q_1", "calibration":[], "validation":[], "test":["A1", "A2", "A3"]}), encoding="utf-8")

    def test_mocked_end_to_end_outputs_do_not_depend_on_teacher_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            teacher = root / "teacher_scores.json"
            teacher.write_text('{"A1":1}', encoding="utf-8")
            def grader(context, answer):
                risk = {"A1": 0.05, "A2": 0.4, "A3": 0.8}[answer["answer_id"]]
                return {"baseline_score":4, "risks":{"U_E":risk,"U_S":risk,"U_R":risk}, "valid_probe_count":3}
            loss = {"lambda1":5, "lambda2":1, "mu1":3, "mu2":7}
            original = Path.open
            def guard(path, *args, **kwargs):
                if path.name == "teacher_scores.json" or "quality_control" in path.parts or "optimized" in path.parts:
                    raise AssertionError("forbidden file read")
                return original(path, *args, **kwargs)
            with patch.object(Path, "open", guard):
                first = score_question_with_grader(root, "Q_1", "test", grader, m=0.5, loss_parameters=loss)
            teacher.write_text('{"A1":9}', encoding="utf-8")
            with patch.object(Path, "open", guard):
                second = score_question_with_grader(root, "Q_1", "test", grader, m=0.5, loss_parameters=loss)
            self.assertEqual(first, second)
            self.assertEqual([x["route"] for x in first["results"]], ["POS", "BND", "NEG"])
            self.assertEqual([x["automatic_score"] for x in first["results"]], [4, None, None])
            self.assertFalse(first["ready_for_live_scoring"])

    def test_rejects_grader_output_with_teacher_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            grader = lambda context, answer: {"baseline_score":4, "risks":{"U_E":0,"U_S":0,"U_R":0}, "valid_probe_count":3, "teacher_score":4}
            loss = {"lambda1":5, "lambda2":1, "mu1":3, "mu2":7}
            with self.assertRaisesRegex(ValueError, "exactly"):
                score_question_with_grader(root, "Q_1", "test", grader, m=0.5, loss_parameters=loss)

    def test_each_grader_receives_an_independent_rubric_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            sizes = []
            def grader(context, answer):
                sizes.append(len(context["initial_rubric"]))
                context["initial_rubric"].clear()
                return {"baseline_score":4, "risks":{"U_E":0,"U_S":0,"U_R":0}, "valid_probe_count":3}
            loss = {"lambda1":5, "lambda2":1, "mu1":3, "mu2":7}
            score_question_with_grader(root, "Q_1", "test", grader, m=0.5, loss_parameters=loss)
            self.assertEqual(sizes, [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
