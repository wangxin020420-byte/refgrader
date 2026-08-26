import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import main_pipeline
from scripts import run_csbench
from step4_vlm_grader import grade_student_3wd_pipeline


class LabelBlindGradingTests(unittest.TestCase):
    def _write_inference_fixture(self, root: Path, teacher_score: float) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "exam_database.json").write_text(
            json.dumps([{"question_id": "Q_1"}]),
            encoding="utf-8",
        )
        (root / "answer_metadata.jsonl").write_text(
            json.dumps(
                {
                    "answer_id": "A_1",
                    "question_id": "Q_1",
                    "raw_text": "answer",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (root / "teacher_scores.json").write_text(
            json.dumps({"A_1": {"Q_1": teacher_score}}),
            encoding="utf-8",
        )

    def test_core_grading_interface_has_no_teacher_label(self):
        parameters = inspect.signature(grade_student_3wd_pipeline).parameters

        self.assertNotIn("teacher_score", parameters)
        source = inspect.getsource(grade_student_3wd_pipeline)
        self.assertNotIn('"teacher_score"', source)
        self.assertNotIn('"real_diff"', source)

    def test_formal_grade_path_does_not_lookup_or_forward_teacher_labels(self):
        process_source = inspect.getsource(main_pipeline.process_single_question)
        grade_source = inspect.getsource(run_csbench.grade)

        self.assertNotIn("get_teacher_score_from_your_database", process_source)
        self.assertNotIn("teacher_score=", process_source)
        self.assertNotIn('"--teacher-db"', grade_source)

    def test_grade_context_does_not_require_teacher_database(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = Path(temp_dir) / "prepared"
            self._write_inference_fixture(prepared, teacher_score=1.0)
            (prepared / "teacher_scores.json").unlink()

            context = run_csbench.CSBenchContext(
                str(prepared),
                "Q_1",
                require_teacher_db=False,
            )

            self.assertEqual(context.question_id, "Q_1")
            with self.assertRaises(FileNotFoundError):
                run_csbench.CSBenchContext(str(prepared), "Q_1")

    def test_teacher_label_permutation_does_not_change_grading_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first"
            second = Path(temp_dir) / "second"
            self._write_inference_fixture(first, teacher_score=0.0)
            self._write_inference_fixture(second, teacher_score=99.0)

            first_context = run_csbench.CSBenchContext(
                str(first),
                "Q_1",
                require_teacher_db=False,
            )
            second_context = run_csbench.CSBenchContext(
                str(second),
                "Q_1",
                require_teacher_db=False,
            )

            self.assertEqual(
                run_csbench._dataset_snapshot_hashes(first_context),
                run_csbench._dataset_snapshot_hashes(second_context),
            )

    def test_large_batch_slug_is_short_stable_and_order_independent(self):
        question_ids = [f"SUBJECT_{index}" for index in range(45)]
        forward = [SimpleNamespace(question_id=value) for value in question_ids]
        reverse = list(reversed(forward))

        first = run_csbench.batch_slug(forward)
        second = run_csbench.batch_slug(reverse)

        self.assertEqual(first, second)
        self.assertRegex(first, r"^q45_[0-9a-f]{16}$")
        self.assertLessEqual(len(first), 20)


if __name__ == "__main__":
    unittest.main()
