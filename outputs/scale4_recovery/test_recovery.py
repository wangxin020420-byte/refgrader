import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
spec = importlib.util.spec_from_file_location("recovery", HERE / "recover_scale4.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
from rubric_semantics import prepare_rubric_semantic_contract, validate_refined_rubric


class RecoveryTests(unittest.TestCase):
    def test_invalid_total_is_not_accepted(self):
        value = [{"id": "a", "points": 10.5, "item": "yes", "answer_type": "judgement"}]
        _, errors = r.validate_candidate(value, None, 10, prepare_rubric_semantic_contract, validate_refined_rubric)
        self.assertTrue(any("total score" in e for e in errors))

    def test_invalid_shape(self):
        _, errors = r.validate_candidate({"items": []}, None, 10, prepare_rubric_semantic_contract, validate_refined_rubric)
        self.assertTrue(errors)

    def test_supplied_failed_rules_are_rejected(self):
        fixture = HERE / "failed_rule_fixtures.json"
        for q in r.load(fixture):
            _, errors = r.validate_candidate(q["initial_rubric"], q["initial_rubric"], q["total_score"], prepare_rubric_semantic_contract, validate_refined_rubric)
            self.assertTrue(any("scoring_role" in e for e in errors))

    def fixture(self, state):
        q = {"question_id": "ADS_example", "rubric_group": "group", "initial_rubric_path": "rubrics/initial/a.json", "optimized_rubric_path": "rubrics/optimized/a.json"}
        p = state / "prepared"
        r.save(p / q["initial_rubric_path"], ["old"])
        r.save(p / q["optimized_rubric_path"], ["old"])
        r.save(p / "rubrics/manifests/group/ADS_example_optimization.json", {"existing": True})
        r.save(state / "cold/ADS_example.json", {"sha256": "old"})
        r.save(state / "rubric_work/ADS_example_variance_checkpoint.json", ["stale"])
        r.save(state / "facts/variance_facts/ADS_example/a.json", {"stale": True})
        r.save(state / "rubric_work/OTHER_variance_checkpoint.json", ["preserve"])
        return q, p

    def test_archive_only_affected_caches_and_resume_transaction(self):
        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp)
            q, prepared = self.fixture(state)
            audit = state / "recovery"
            r.activate(state, prepared, q, ["new"], audit)
            self.assertEqual(r.load(prepared / q["initial_rubric_path"]), ["new"])
            self.assertEqual(r.load(audit / "before/prepared/rubrics/initial/a.json"), ["old"])
            self.assertFalse((state / "rubric_work/ADS_example_variance_checkpoint.json").exists())
            self.assertTrue((audit / "invalidated/facts/variance_facts/ADS_example/a.json").exists())
            self.assertEqual(r.load(state / "rubric_work/OTHER_variance_checkpoint.json"), ["preserve"])
            r.activate(state, prepared, q, None, audit)
            self.assertEqual(r.load(prepared / q["initial_rubric_path"]), ["new"])

    def test_downstream_and_completed_guards(self):
        for guard in ["optimization/ADS_example.json", "validation_complete.json", "metrics.json"]:
            with tempfile.TemporaryDirectory() as temp:
                state = Path(temp)
                q, p = self.fixture(state)
                r.save(state / guard, {})
                with self.assertRaises(RuntimeError):
                    r.activate(state, p, q, ["new"], state / "recovery")
                self.assertEqual(r.load(p / q["initial_rubric_path"]), ["old"])


if __name__ == "__main__":
    unittest.main()
