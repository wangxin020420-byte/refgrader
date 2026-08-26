import json
import tempfile
import unittest
from pathlib import Path

from benchmark_datasets.adapters.sas_bench import prepare_sas_bench
from benchmark_datasets.contract import load_json, read_jsonl
from benchmark_datasets.registry import get_adapter


def _record(
    answer_id: str,
    *,
    question: str = "What is caching?",
    reference: str = "Caching stores reusable data.",
    analysis: str = "Credit correctness and completeness.",
    total: float = 5.0,
    score: float = 4.0,
) -> dict:
    return {
        "id": answer_id,
        "question": question,
        "reference": reference,
        "analysis": analysis,
        "total": total,
        "manual_label": score,
        "steps": [
            {
                "response": "Stores data for reuse.",
                "label": score,
                "errors": ["missing detail"],
            }
        ],
    }


class SASBenchAdapterTests(unittest.TestCase):
    def test_prepare_external_test_and_isolate_gold_step_annotations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "datasets"
            source.mkdir()
            records = [
                _record("A1"),
                _record("A2", score=3.0),
                _record("BAD", total=4.0, score=5.0),
            ]
            (source / "0_Physics_ShortAns.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in records),
                encoding="utf-8",
            )
            spec = root / "spec.json"
            spec.write_text(
                json.dumps(
                    {
                        "dataset_id": "sas_bench_test",
                        "expected_counts": {
                            "task_files": 1,
                            "source_records": 3,
                            "source_question_contexts": 2,
                            "formal_test_records": 2,
                            "excluded_records": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )

            output = root / "prepared"
            audit = prepare_sas_bench(source, spec, output)

            self.assertEqual(audit["answer_count"], 2)
            self.assertEqual(audit["excluded_record_count"], 1)
            self.assertEqual(
                audit["split_counts"],
                {"train": 0, "calibration": 0, "validation": 0, "test": 2},
            )
            manifest = load_json(output / "manifest.json")
            self.assertEqual(manifest["adapter"], "sas_bench")
            self.assertEqual(manifest["counts"]["source_records"], 3)
            self.assertTrue(manifest["score_label_policy"]["gold_step_annotations_isolated"])

            metadata = read_jsonl(output / "answer_metadata.jsonl")
            serialized_metadata = json.dumps(metadata, ensure_ascii=False)
            self.assertIn("Stores data for reuse.", serialized_metadata)
            self.assertNotIn("missing detail", serialized_metadata)
            self.assertNotIn('"label"', serialized_metadata)
            self.assertNotIn('"errors"', serialized_metadata)
            self.assertNotIn('"manual_label"', serialized_metadata)

            gold = read_jsonl(output / "gold_only" / "step_labels_and_errors.jsonl")
            self.assertEqual(len(gold), 2)
            self.assertEqual(gold[0]["steps"][0]["errors"], ["missing detail"])
            excluded = read_jsonl(output / "quality_control" / "excluded_records.jsonl")
            self.assertEqual(excluded[0]["answer_id"], "BAD")
            self.assertEqual(excluded[0]["reason"], "manual_label_exceeds_total")

            question = load_json(output / "exam_database.json")[0]
            split = load_json(output / question["rubric_split_path"])
            self.assertEqual(split["train"], [])
            self.assertEqual(split["calibration"], [])
            self.assertEqual(split["validation"], [])
            self.assertEqual(set(split["test"]), {"A1", "A2"})

    def test_registry_and_nonempty_output_guard(self):
        self.assertIs(get_adapter("sas_bench"), prepare_sas_bench)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "datasets"
            source.mkdir()
            (source / "task.jsonl").write_text(
                json.dumps(_record("A1")) + "\n",
                encoding="utf-8",
            )
            spec = root / "spec.json"
            spec.write_text("{}", encoding="utf-8")
            output = root / "prepared"
            output.mkdir()
            (output / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "--force"):
                prepare_sas_bench(source, spec, output)


if __name__ == "__main__":
    unittest.main()
