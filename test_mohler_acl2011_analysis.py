import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmarks.analyze_mohler_acl2011 import (
    analyze,
    quadratic_weighted_kappa,
)


class MohlerACL2011AnalysisTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        rows = []
        for question_index in range(3):
            for answer_index, teacher in enumerate((0.0, 1.0, 2.0, 3.0)):
                rows.append(
                    {
                        "fold_id": f"fold_{question_index + 1:02d}",
                        "test_unit": question_index + 1,
                        "question_id": f"Q{question_index + 1}",
                        "student_id": f"S{question_index + 1}_{answer_index}",
                        "teacher_score": teacher,
                        "route": "POS",
                        "single": teacher + 1.5,
                        "avg": teacher + 1.0,
                        "selected": teacher + 1.0,
                        "3wd_core": teacher + 0.5,
                        "3wd": teacher,
                    }
                )
        predictions = root / "predictions.csv"
        with predictions.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        protocol = root / "protocol.json"
        protocol.write_text(
            json.dumps(
                {
                    "protocol_id": "fixture",
                    "answer_count": len(rows),
                    "question_count": 3,
                    "paper_reference_results": {
                        "reported_question_count": 80,
                        "average_grade_baseline_rmse": 1.097,
                        "best_pearson": 0.518,
                        "best_rmse": 0.978,
                        "best_median_per_question_rmse": 0.862,
                    },
                }
            ),
            encoding="utf-8",
        )
        baseline = root / "baseline.json"
        baseline.write_text(
            json.dumps({"global": {"mean": {"n": len(rows), "RMSE": 1.0}}}),
            encoding="utf-8",
        )
        return predictions, protocol, baseline

    def test_analysis_is_clustered_deterministic_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions, protocol, baseline = self._fixture(root)
            result = analyze(
                predictions_path=predictions,
                protocol_path=protocol,
                baseline_summary_path=baseline,
                output_dir=root / "output",
                bootstrap_iterations=200,
                seed=2011,
            )
            report = (root / "output" / "report.md").read_text(encoding="utf-8")

        mae = next(
            item
            for item in result["bootstrap"]
            if item["comparison"] == "avg_to_3wd_core" and item["metric"] == "MAE"
        )
        self.assertAlmostEqual(mae["observed_gain"], 0.5)
        self.assertGreater(mae["ci95_low"], 0.0)
        self.assertEqual(mae["cluster_unit"], "question_id")
        self.assertAlmostEqual(
            result["qwk"]["methods"]["3wd"]["sample_weighted_per_question_qwk"],
            1.0,
        )
        self.assertAlmostEqual(
            result["qwk"]["methods"]["3wd"]["sample_weighted_common_question_qwk"],
            1.0,
        )
        self.assertFalse(
            result["comparison_boundary"]["direct_paper_comparison_authorized"]
        )
        self.assertIn("Direct comparison", report)

    def test_half_point_qwk_uses_fixed_0_to_5_scale(self):
        self.assertAlmostEqual(
            quadratic_weighted_kappa([0.0, 1.0, 2.0], [0.1, 1.2, 2.1]),
            1.0,
        )
        self.assertIsNone(quadratic_weighted_kappa([2.0, 2.0], [2.0, 2.0]))

    def test_duplicate_prediction_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions, protocol, _ = self._fixture(root)
            lines = predictions.read_text(encoding="utf-8").splitlines()
            predictions.write_text("\n".join(lines + [lines[1]]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate prediction row"):
                analyze(
                    predictions_path=predictions,
                    protocol_path=protocol,
                    output_dir=root / "output",
                    bootstrap_iterations=100,
                )


if __name__ == "__main__":
    unittest.main()
