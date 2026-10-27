import json
import subprocess
import sys
import unittest
from pathlib import Path

from scripts.a3wa_unlabeled_analysis import analyze


class UnlabeledAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.input = {
            "memberships": [0.2, 0.6, 0.9],
            "candidates": [0.4, 0.5, 0.6],
            "loss_parameters": {"lambda1": 5, "lambda2": 1, "mu1": 3, "mu2": 7},
        }

    def test_valid_input_does_not_claim_input_provenance(self):
        result = analyze(self.input)
        self.assertFalse(result["provenance_verified"])
        self.assertEqual(result["result"]["sample_count"], 3)

    def test_rejects_unexpected_label_and_non_array_values(self):
        with self.assertRaises(ValueError):
            analyze({**self.input, "teacher_score": [1, 2, 3]})
        with self.assertRaises(ValueError):
            analyze({**self.input, "memberships": "0.6"})

    def test_stdin_cli_runs_offline_and_rejects_extra_fields(self):
        script = Path(__file__).parent / "scripts" / "a3wa_unlabeled_analysis.py"
        def run(data):
            return subprocess.run(
                [sys.executable, str(script), "--input", "-"],
                input=json.dumps(data), text=True, capture_output=True, check=False,
            )
        success = run(self.input)
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertEqual(json.loads(success.stdout)["result"]["sample_count"], 3)
        rejected = run({**self.input, "teacher_score": [1]})
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(rejected.stdout, "")


if __name__ == "__main__":
    unittest.main()
