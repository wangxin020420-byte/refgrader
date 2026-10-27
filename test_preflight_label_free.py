import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.preflight_label_free import preflight
from scripts.prepare_csbench import split_answer_ids
from test_label_free_input_inventory import create_fixture


class PreflightLabelFreeTests(unittest.TestCase):
    def test_preflight_is_read_only_and_does_not_claim_live_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_fixture(root, b"\n")
            (root / "exam_database.json").write_text(json.dumps([{"question_id":"Q_1", "rubric_group":"Q", "total_score":2}]), encoding="utf-8")
            (root / "answer_metadata.jsonl").write_text(json.dumps({"answer_id":"A1", "question_id":"Q_1", "raw_text":"x"}) + "\n", encoding="utf-8")
            (root / "rubrics/initial/Q/r.json").rename(root / "rubrics/initial/Q/Q_1_rubric_standard.json")
            (root / "rubrics/initial/Q/Q_1_rubric_standard.json").write_text('[{"points":2}]', encoding="utf-8")
            (root / "splits/by_question/Q.json").unlink()
            split = split_answer_ids(["A1"], "s:Q_1", 0.1, 0.1, 1)
            (root / "splits/by_question/Q_1.json").write_text(json.dumps({"question_id":"Q_1", "split_seed":"s", **split}), encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({"split_seed":"s", "calibration_ratio":0.1, "validation_ratio":0.1, "minimum_calibration_size":1}), encoding="utf-8")
            (root / "teacher_scores.json").write_text("forbidden", encoding="utf-8")
            original = Path.open
            def guard(path, *args, **kwargs):
                if path.name == "teacher_scores.json" or "quality_control" in path.parts or "optimized" in path.parts:
                    raise AssertionError("forbidden input accessed")
                return original(path, *args, **kwargs)
            with patch.object(Path, "open", guard):
                result = preflight(root)
            self.assertTrue(result["structural_checks_passed"])
            self.assertFalse(result["ready_for_live_scoring"])
            self.assertEqual(result["live_model_calls"], 0)
            self.assertEqual(result["answer_counts"]["calibration"], 1)


if __name__ == "__main__":
    unittest.main()
