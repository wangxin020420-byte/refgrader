import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.report_reviewed_workflow import build_report


class ReportReviewedWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rid = "fixture_run"
        self.job = self.root / "results_runs/reviewed_csbench45_glm52" / self.rid
        self.job.mkdir(parents=True)
        self.state = {"run_id": self.rid, "questions": ["Q_1"], "status": "completed",
                      "stage": "completed", "error": None, "result_class": "experimental",
                      "deployment_gate_passed": False, "paths": {}, "completed": {}}
        for split in ("validation", "test"):
            directory = self.root / split
            directory.mkdir()
            checkpoint = directory / "Q_1_grading_checkpoint.json"
            self.write(checkpoint, [{"student_id": "ANS_Q_12", "3wd_route": "BND",
                                     "3wd_sequential_outcome": "auto_adjusted"}])
            self.state["paths"][split] = str(directory)
            self.state["completed"]["grade_" + split] = {"proof": {
                str(checkpoint): hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}
            self.write(directory / "completion_report.json", {
                "expected_total": 1, "checkpoint_total": 1,
                "questions": {"Q_1": {"status": "complete", "expected_count": 1, "checkpoint_count": 1}}})
            self.write(directory / "progress.json", {"questions": {"Q_1": {"completed": 0}}})
        self.state["paths"]["config"] = str(self.job / "a3wa_config.json")
        self.write(self.job / "a3wa_config.json", {"deployment_gate": {"passed": False}})
        self.write(self.root / "test/evaluation/summary.json", {
            "global": [{"score_type": "3WD", "score_key": "final", "n": 1, "MAE": 0.2, "RMSE": 0.2, "bias": -0.2}],
            "per_question": [{"score_key": "final", "QWK": None}]})
        (self.job / "review_decisions.jsonl").write_text("", encoding="utf-8")
        self.write(self.job / "workflow.json", self.state)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_stale_counters_do_not_hide_completed_checkpoint(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        report = build_report(self.root, self.rid)
        self.assertEqual(report["splits"]["test"]["checkpoint_count"], 1)
        self.assertEqual(report["splits"]["test"]["stale_progress_questions"], ["Q_1"])
        self.assertFalse(report["workflow"]["deployment_gate_passed"])
        self.assertEqual(report["test_routing"]["outcomes"], {"auto_adjusted": 1})
        self.assertIsNone(report["test_metrics"][0]["macro_qwk"])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_changed_checkpoint_is_rejected(self):
        self.write(self.root / "test/Q_1_grading_checkpoint.json", [])
        with self.assertRaisesRegex(ValueError, "proof missing or changed"):
            build_report(self.root, self.rid)

    def test_inconsistent_completion_report_is_rejected(self):
        path = self.root / "test/completion_report.json"
        payload = json.loads(path.read_text())
        payload["expected_total"] = 2
        self.write(path, payload)
        with self.assertRaisesRegex(ValueError, "inconsistent completion totals"):
            build_report(self.root, self.rid)

    def test_duplicate_answer_ids_are_rejected_even_with_matching_hash(self):
        checkpoint = self.root / "test/Q_1_grading_checkpoint.json"
        self.write(checkpoint, [{"student_id": "ANS_Q_12"}] * 2)
        self.state["completed"]["grade_test"]["proof"][str(checkpoint)] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.write(self.job / "workflow.json", self.state)
        with self.assertRaisesRegex(ValueError, "duplicate answer IDs"):
            build_report(self.root, self.rid)

    def test_run_id_path_traversal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid run ID"):
            build_report(self.root, "../elsewhere")


if __name__ == "__main__":
    unittest.main()
