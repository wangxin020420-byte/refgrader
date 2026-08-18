import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_datasets.protocols.mohler_acl2011 import (
    build_mohler_acl2011_protocol,
    evaluate_prediction_rows,
    run_mohler_acl2011_baselines,
)
from scripts.benchmarks.run_mohler_acl2011 import build_parser


class MohlerACL2011ProtocolTests(unittest.TestCase):
    def _prepared_fixture(self, root: Path, *, answers_per_unit: int = 3) -> Path:
        prepared = root / "mohler"
        normalized = prepared / "normalized"
        normalized.mkdir(parents=True)
        (prepared / "manifest.json").write_text(
            json.dumps({"dataset_id": "mohler_test", "adapter": "mohler"}),
            encoding="utf-8",
        )
        questions = []
        answers = []
        for unit in range(1, 13):
            question_id = f"MOHLER_{unit}_1"
            source_question_id = f"{unit}.1"
            questions.append(
                {
                    "question_id": question_id,
                    "source_question_id": source_question_id,
                    "question_text": f"Question for unit {unit}?",
                    "reference_answer": f"Reference answer unit {unit}",
                }
            )
            for index in range(answers_per_unit):
                answers.append(
                    {
                        "answer_id": f"{question_id}_{index:03d}",
                        "question_id": question_id,
                        "source_question_id": source_question_id,
                        "raw_text": (
                            f"Reference answer unit {unit} detail {index}"
                            if index
                            else "unrelated response"
                        ),
                        "actual_score": float(index * 2),
                    }
                )
        for name, records in (("questions", questions), ("answers", answers)):
            (normalized / f"{name}.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
        return prepared

    def _build(self, prepared: Path, **kwargs):
        fake_audit = {
            "question_count": 12,
            "answer_count": 36,
            "prepared_content_sha256": "fixture",
        }
        with patch(
            "benchmark_datasets.protocols.mohler_acl2011."
            "audit_prepared_benchmark",
            return_value=fake_audit,
        ):
            return build_mohler_acl2011_protocol(prepared, **kwargs)

    def test_protocol_is_disjoint_and_tests_every_answer_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = self._prepared_fixture(Path(temporary))
            protocol = self._build(prepared)

        self.assertEqual(len(protocol["folds"]), 12)
        self.assertEqual(protocol["answer_count"], 36)
        self.assertTrue(protocol["integrity"]["partition_disjoint_per_fold"])
        self.assertTrue(protocol["integrity"]["all_answers_tested_once"])
        for fold in protocol["folds"]:
            self.assertEqual(fold["train_answer_count"], 30)
            self.assertEqual(fold["calibration_answer_count"], 3)
            self.assertEqual(fold["test_answer_count"], 3)
            units = set(fold["train_units"])
            self.assertNotIn(fold["test_unit"], units)
            self.assertNotIn(fold["calibration_unit"], units)

    def test_strict_paper_count_rejects_archive_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = self._prepared_fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "reports 80 questions"):
                self._build(prepared, require_paper_question_count=True)

    def test_unknown_exclusion_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = self._prepared_fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "Unknown Mohler"):
                self._build(prepared, excluded_question_ids=["99.9"])

    def test_prediction_metrics_include_paper_median_rmse(self):
        rows = [
            {"question_id": "Q1", "teacher_score": 0.0, "score": 1.0},
            {"question_id": "Q1", "teacher_score": 2.0, "score": 2.0},
            {"question_id": "Q2", "teacher_score": 4.0, "score": 2.0},
            {"question_id": "Q2", "teacher_score": 5.0, "score": 5.0},
        ]
        summary = evaluate_prediction_rows(rows, score_fields=["score"])
        metric = summary["global"]["score"]
        self.assertAlmostEqual(metric["RMSE"], (5.0 / 4.0) ** 0.5)
        self.assertIn("median_per_question_RMSE", metric)

    def test_baselines_cover_all_answers_without_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self._prepared_fixture(root, answers_per_unit=3)
            protocol = self._build(prepared)
            with patch(
                "benchmark_datasets.protocols.mohler_acl2011."
                "audit_prepared_benchmark",
                return_value={
                    "question_count": 12,
                    "answer_count": 36,
                    "prepared_content_sha256": "fixture",
                },
            ):
                summary = run_mohler_acl2011_baselines(
                    prepared,
                    protocol,
                    root / "output",
                )
            predictions = (root / "output" / "predictions.csv").read_text(
                encoding="utf-8"
            )
        self.assertEqual(summary["global"]["train_mean"]["n"], 36)
        self.assertEqual(predictions.count("\n"), 37)
        self.assertIn("linear_tfidf_svr_isotonic", summary["global"])

    def test_cli_refgrader_dry_run_contract_parses(self):
        args = build_parser().parse_args(
            [
                "--prepared-dir",
                "prepared",
                "refgrader",
                "--tag",
                "acl2011",
                "--variant",
                "zero_shot",
                "--a3wa-config",
                "config.json",
                "--dry-run",
            ]
        )
        self.assertEqual(args.command, "refgrader")
        self.assertEqual(args.start_fold, 1)
        self.assertEqual(args.end_fold, 12)


if __name__ == "__main__":
    unittest.main()
