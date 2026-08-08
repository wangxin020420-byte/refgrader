import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmarks.aggregate_repeats import aggregate_repeats


class PublicBenchmarkRepeatAggregateTests(unittest.TestCase):
    def _write_run(
        self,
        root: Path,
        run_id: str,
        *,
        avg_scores: list[float],
        core_scores: list[float],
        final_scores: list[float],
        dataset_hash: str = "dataset-hash",
    ) -> None:
        run_dir = root / run_id
        evaluation = run_dir / "evaluation"
        evaluation.mkdir(parents=True)
        teacher = [2.0, 4.0]
        manifest = {
            "run_id": run_id,
            "status": "complete",
            "split": "test",
            "dataset_id": "fixture",
            "dataset_snapshot": {"prepared_content_sha256": dataset_hash},
            "dataset_manifest_sha256": "manifest-hash",
            "questions": ["Q1", "Q2"],
            "model_config": {
                "text_model": "model",
                "text_thinking": "disabled",
            },
            "a3wa_config_sha256": "a3wa-hash",
            "a3wa_deployment_class": "experimental",
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        score_rows = []
        for score_type, scores in (
            ("single", avg_scores),
            ("avg", avg_scores),
            ("selected", avg_scores),
            ("3WD-Core", core_scores),
            ("3WD", final_scores),
        ):
            errors = [abs(score - gold) for score, gold in zip(scores, teacher)]
            score_rows.append(
                {
                    "score_type": score_type,
                    "n": 2,
                    "MAE": sum(errors) / 2,
                    "RMSE": (sum(value * value for value in errors) / 2) ** 0.5,
                    "Pearson": 1.0,
                    "TAR2": 1.0,
                    "SER2": 0.0,
                    "bias": sum(
                        score - gold for score, gold in zip(scores, teacher)
                    )
                    / 2,
                }
            )
        summary = {
            "global": score_rows,
            "score_ablation": {"global": []},
        }
        (evaluation / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

        fieldnames = [
            "question",
            "student_id",
            "teacher",
            "single_first_score",
            "model_avg_score",
            "selected_baseline_score",
            "three_way_core_score",
            "final_calibrated_score",
            "single_abs_error",
            "avg_abs_error",
            "selected_abs_error",
            "core_abs_error",
            "final_abs_error",
            "route",
            "boundary_action",
            "baseline_serious_error",
            "risk_captured_by_route",
            "safe_pos",
        ]
        with (evaluation / "compare.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, gold in enumerate(teacher):
                writer.writerow(
                    {
                        "question": f"Q{index + 1}",
                        "student_id": f"S{index + 1}",
                        "teacher": gold,
                        "single_first_score": avg_scores[index],
                        "model_avg_score": avg_scores[index],
                        "selected_baseline_score": avg_scores[index],
                        "three_way_core_score": core_scores[index],
                        "final_calibrated_score": final_scores[index],
                        "single_abs_error": abs(avg_scores[index] - gold),
                        "avg_abs_error": abs(avg_scores[index] - gold),
                        "selected_abs_error": abs(avg_scores[index] - gold),
                        "core_abs_error": abs(core_scores[index] - gold),
                        "final_abs_error": abs(final_scores[index] - gold),
                        "route": "POS" if index == 0 else "BND",
                        "boundary_action": "" if index == 0 else "keep_baseline",
                        "baseline_serious_error": index == 1,
                        "risk_captured_by_route": index == 1,
                        "safe_pos": index == 0,
                    }
                )

    def test_aggregates_compatible_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            self._write_run(
                runs,
                "run1",
                avg_scores=[1.0, 2.0],
                core_scores=[1.5, 2.5],
                final_scores=[2.0, 3.0],
            )
            self._write_run(
                runs,
                "run2",
                avg_scores=[1.0, 2.5],
                core_scores=[1.5, 3.0],
                final_scores=[2.0, 3.5],
            )
            output = root / "report"

            summary = aggregate_repeats(
                runs,
                ["run1", "run2"],
                output,
                bootstrap_iterations=100,
                seed=7,
            )

            self.assertEqual(summary["run_count"], 2)
            self.assertEqual(summary["sample_count"], 2)
            self.assertGreater(
                summary["ablation"]["full_three_way"]["mean_gain"]["mean"],
                0,
            )
            self.assertEqual(summary["sample_stability"]["route"]["exact_rate"], 1)
            self.assertTrue((output / "repeat_summary.json").is_file())
            self.assertTrue((output / "run_metrics.csv").is_file())
            self.assertTrue((output / "sample_stability.csv").is_file())
            self.assertTrue((output / "question_stability.csv").is_file())
            self.assertTrue((output / "report.md").is_file())

    def test_rejects_dataset_contract_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            self._write_run(
                runs,
                "run1",
                avg_scores=[1.0, 2.0],
                core_scores=[1.5, 2.5],
                final_scores=[2.0, 3.0],
            )
            self._write_run(
                runs,
                "run2",
                avg_scores=[1.0, 2.0],
                core_scores=[1.5, 2.5],
                final_scores=[2.0, 3.0],
                dataset_hash="different-dataset",
            )

            with self.assertRaisesRegex(ValueError, "dataset_content_sha256"):
                aggregate_repeats(
                    runs,
                    ["run1", "run2"],
                    root / "report",
                    bootstrap_iterations=10,
                )

    def test_rejects_nonempty_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            self._write_run(
                runs,
                "run1",
                avg_scores=[1.0, 2.0],
                core_scores=[1.5, 2.5],
                final_scores=[2.0, 3.0],
            )
            self._write_run(
                runs,
                "run2",
                avg_scores=[1.0, 2.0],
                core_scores=[1.5, 2.5],
                final_scores=[2.0, 3.0],
            )
            output = root / "report"
            output.mkdir()
            (output / "existing.txt").write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "not empty"):
                aggregate_repeats(
                    runs,
                    ["run1", "run2"],
                    output,
                    bootstrap_iterations=10,
                )


if __name__ == "__main__":
    unittest.main()
