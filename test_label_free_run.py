from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from label_free_run import run_questions
from test_label_free_scoring import LabelFreeScoringTests


class LabelFreeRunTests(unittest.TestCase):
    def test_resume_and_teacher_independence_with_mock_grader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            root.mkdir()
            LabelFreeScoringTests().fixture(root)
            image_dir = root / "student_images/Q_1"
            image_dir.mkdir(parents=True)
            for ident in ("A1", "A2", "A3"):
                (image_dir / (ident + ".png")).write_bytes(b"image")
            teacher = root / "teacher_scores.json"
            teacher.write_text('{"A1":1}', encoding="utf-8")
            loss = {"lambda1": 5, "lambda2": 1, "mu1": 3, "mu2": 7}
            config = {"question_ids": ["Q_1"], "split": "test", "m": 0.5, "loss_parameters": loss}
            manifest = {"run_config": config, "input_hash": "fixture"}
            calls = []
            def grader(client, context, image, **kwargs):
                calls.append(image.stem)
                return {"baseline_score": 4, "risks": {"U_E": 0, "U_S": 0, "U_R": 0},
                        "valid_probe_count": 3, "probe_scores": [4, 4, 4]}
            original = Path.open
            def guard(path, *args, **kwargs):
                if path.name == "teacher_scores.json" or "quality_control" in path.parts or "optimized" in path.parts:
                    raise AssertionError("forbidden input")
                return original(path, *args, **kwargs)
            with patch.object(Path, "open", guard):
                first = run_questions(root, ["Q_1"], "test", Path(temporary) / "run", manifest, None,
                                      m=0.5, loss_parameters=loss, grader=grader, max_new=1)
            self.assertEqual(first["completed_now"], 1)
            teacher.write_text('{"A1":9}', encoding="utf-8")
            with patch.object(Path, "open", guard):
                second = run_questions(root, ["Q_1"], "test", Path(temporary) / "run", manifest, None,
                                       m=0.5, loss_parameters=loss, grader=grader)
            self.assertEqual(second, {"completed_before": 1, "completed_now": 2})
            self.assertEqual(calls, ["A1", "A2", "A3"])
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                run_questions(root, ["Q_1"], "test", Path(temporary) / "run", {**manifest, "input_hash": "different"},
                              None, m=0.5, loss_parameters=loss, grader=grader)

    def test_config_mismatch_fails_before_creating_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Manifest"):
                run_questions(temporary, ["Q_1"], "test", Path(temporary) / "run", {"run_config": {}},
                              None, m=0.5, loss_parameters={})
            self.assertFalse((Path(temporary) / "run").exists())


if __name__ == "__main__":
    unittest.main()
