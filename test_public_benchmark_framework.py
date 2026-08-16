import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import main_pipeline
import step4_vlm_grader
from benchmark_datasets.adapters.asap_sas import prepare_asap_sas
from benchmark_datasets.adapters.mohler import prepare_mohler
from benchmark_datasets.contract import (
    audit_prepared_benchmark,
    load_json,
    sha256_file,
    write_json,
)
from rubric_semantics import RUBRIC_SEMANTIC_CONTRACT_VERSION
from sample_quality import SampleQualityPolicy
from scripts.benchmarks import run_benchmark


PROJECT_ROOT = Path(__file__).resolve().parent


def rubric(points):
    return [
        {
            "id": "overall_response",
            "item": "Score the response against the reference answer.",
            "points": points,
            "answer_type": "free_text",
            "role": "final",
            "score_layer": "core",
            "canonicalization": {"type": "text"},
            "evidence_source": "text",
            "standard_answer_text": "reference",
            "standard_answer_image": None,
            "source_text": "official rubric",
            "parent_official_item": "overall",
            "metadata_source": "asap_sas",
            "metadata_hard_enabled": False,
            "metadata_confidence": 1.0,
            "parent_id": "overall_response",
            "semantic_contract_version": 5,
            "parent_points": points,
            "scoring_policy": "preserve_atomic",
            "split_policy": "preserve_atomic",
            "weighting_policy": "preserve_parent",
            "full_credit_policy": "rubric_evidence_required",
            "full_credit_trigger": False,
        }
    ]


class PublicBenchmarkFrameworkTests(unittest.TestCase):
    def prepare_fixture(self, root):
        source = root / "train.tsv"
        source.write_text(
            "Id\tEssaySet\tScore1\tScore2\tEssayText\n"
            "1\t1\t0\t0\tanswer one\n"
            "2\t1\t1\t1\tanswer two\n"
            "3\t1\t2\t2\tanswer three\n"
            "4\t1\t3\t3\tanswer four\n"
            "5\t2\t0\t0\tsecond one\n"
            "6\t2\t1\t1\tsecond two\n"
            "7\t2\t2\t2\tsecond three\n"
            "8\t2\t3\t3\tsecond four\n",
            encoding="utf-8",
        )
        spec = root / "spec.json"
        spec.write_text(
            json.dumps(
                {
                    "dataset_id": "asap_sas_test",
                    "encoding": "utf-8",
                    "delimiter": "\t",
                    "columns": {
                        "answer_id": "Id",
                        "question_id": "EssaySet",
                        "answer_text": "EssayText",
                        "score_1": "Score1",
                        "score_2": "Score2",
                    },
                    "split": {
                        "seed": "test-seed",
                        "train": 0.25,
                        "calibration": 0.25,
                        "validation": 0.25,
                        "test": 0.25,
                    },
                    "questions": {
                        "1": {
                            "question_text": "Question one",
                            "reference_answer": "reference",
                            "grading_guidance": "official rubric",
                            "max_score": 3,
                            "rubric": rubric(3),
                        },
                        "2": {
                            "question_text": "Question two",
                            "reference_answer": "reference",
                            "grading_guidance": "official rubric",
                            "max_score": 3,
                            "rubric": rubric(3),
                        },
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        output = root / "prepared"
        prepare_asap_sas(source, spec, output)
        return output

    def test_prepare_and_audit_asap_sas(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = self.prepare_fixture(Path(temporary))
            audit = audit_prepared_benchmark(prepared)
            self.assertEqual(audit["question_count"], 2)
            self.assertEqual(audit["answer_count"], 8)
            self.assertEqual(sum(audit["split_counts"].values()), 8)
            self.assertEqual(
                audit["split_counts"],
                {
                    "train": 2,
                    "calibration": 2,
                    "validation": 2,
                    "test": 2,
                },
            )
            manifest = load_json(prepared / "manifest.json")
            self.assertEqual(manifest["extraction_backend"], "text_only")
            self.assertNotIn("path", manifest["source"])
            question = load_json(prepared / "exam_database.json")[0]
            self.assertFalse(Path(question["student_images_dir"]).is_absolute())

    def test_prepare_and_audit_mohler(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "ShortAnswerGrading_v2.0"
            data = source / "data"
            (data / "docs").mkdir(parents=True)
            (data / "raw").mkdir(parents=True)
            (data / "docs" / "files").write_text(
                "1.1\n#9.9\n11.1\n",
                encoding="utf-8",
            )
            (data / "raw" / "questions").write_text(
                "1.1 Assignment question?\n"
                "11.1 Exam question?\n",
                encoding="utf-8",
            )
            (data / "raw" / "answers").write_text(
                "1.1 Assignment reference.\n"
                "11.1 Exam reference.\n",
                encoding="utf-8",
            )
            normalized_scores = [0.0, 2.0, 4.0, 5.0]
            for question_id, raw_scores, official_scores in (
                ("1.1", normalized_scores, [0.0, 2.5, 4.0, 5.0]),
                ("11.1", [0.0, 4.0, 8.0, 10.0], normalized_scores),
            ):
                (data / "raw" / question_id).write_text(
                    "".join(
                        f"{question_id} response {index}<br />detail\n"
                        for index in range(1, 5)
                    ),
                    encoding="utf-8",
                )
                score_dir = data / "scores" / question_id
                score_dir.mkdir(parents=True)
                (score_dir / "ave").write_text(
                    "\n".join(str(value) for value in official_scores) + "\n",
                    encoding="utf-8",
                )
                raw_text = "\n".join(str(value) for value in raw_scores) + "\n"
                (score_dir / "me").write_text(raw_text, encoding="utf-8")
                (score_dir / "other").write_text(raw_text, encoding="utf-8")
            spec = root / "mohler.json"
            spec.write_text(
                json.dumps(
                    {
                        "dataset_id": "mohler_test",
                        "split": {
                            "seed": "mohler-test",
                            "train": 0.25,
                            "calibration": 0.25,
                            "validation": 0.25,
                            "test": 0.25,
                        },
                    }
                ),
                encoding="utf-8",
            )

            prepared = root / "prepared_mohler"
            audit = prepare_mohler(source, spec, prepared)

            self.assertEqual(audit["question_count"], 2)
            self.assertEqual(audit["answer_count"], 8)
            self.assertEqual(
                audit["split_counts"],
                {"train": 2, "calibration": 2, "validation": 2, "test": 2},
            )
            manifest = load_json(prepared / "manifest.json")
            self.assertEqual(manifest["adapter"], "mohler")
            self.assertFalse(
                manifest["rubric_policy"]["official_fine_grained_rubric"]
            )
            optimization_manifest = load_json(
                prepared
                / "rubrics"
                / "manifests"
                / "MOHLER"
                / "MOHLER_1_1_optimization.json"
            )
            self.assertEqual(
                optimization_manifest["rubric_semantic_contract_version"],
                RUBRIC_SEMANTIC_CONTRACT_VERSION,
            )
            self.assertTrue(
                optimization_manifest["semantic_policy_validated"]
            )
            self.assertEqual(
                optimization_manifest["initial_sha256"],
                sha256_file(
                    prepared
                    / "rubrics"
                    / "initial"
                    / "MOHLER"
                    / "MOHLER_1_1_rubric_standard.json"
                ),
            )
            self.assertNotIn(
                b"\r\n",
                (
                    prepared
                    / "rubrics"
                    / "initial"
                    / "MOHLER"
                    / "MOHLER_1_1_rubric_standard.json"
                ).read_bytes(),
            )
            metadata = [
                json.loads(line)
                for line in (prepared / "answer_metadata.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            exam_record = next(
                item for item in metadata if item["question_id"] == "MOHLER_11_1"
            )
            self.assertEqual(exam_record["rater_1_score"], 0.0)
            self.assertEqual(exam_record["rater_2_score"], 0.0)
            last_exam_record = next(
                item
                for item in metadata
                if item["answer_id"] == "MOHLER_11_1_004"
            )
            self.assertEqual(last_exam_record["rater_1_score"], 5.0)
            self.assertEqual(last_exam_record["rater_1_raw_score"], 10.0)
            self.assertEqual(last_exam_record["actual_score"], 5.0)

    def test_text_only_cache_is_bound_to_transcription(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "facts.json"
            with mock.patch.object(
                step4_vlm_grader,
                "map_transcription_to_facts",
                return_value={"overall_response": "student evidence"},
            ) as mapper:
                facts, evidence = step4_vlm_grader.stage1_extract_with_backend(
                    question_text="Question",
                    student_img_path=str(Path(temporary) / "missing.txt"),
                    blind_checklist=json.dumps(
                        [{"id": "overall_response", "instruction": "extract"}]
                    ),
                    rubrics_json=json.dumps(rubric(3)),
                    extraction_backend="text_only",
                    extraction_cache_path=str(cache),
                    student_transcription="student evidence",
                )
                self.assertIn("student evidence", facts)
                self.assertIsNone(evidence["image_sha256"])
                self.assertEqual(evidence["backend"], "text_only")
                mapper.reset_mock()
                step4_vlm_grader.stage1_extract_with_backend(
                    question_text="Question",
                    student_img_path=str(Path(temporary) / "missing.txt"),
                    blind_checklist="[]",
                    rubrics_json=json.dumps(rubric(3)),
                    extraction_backend="text_only",
                    extraction_cache_path=str(cache),
                    student_transcription="student evidence",
                )
                mapper.assert_not_called()

    def test_audit_hash_changes_when_rubric_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = self.prepare_fixture(Path(temporary))
            baseline = audit_prepared_benchmark(prepared)
            rubric_path = (
                prepared
                / "rubrics"
                / "optimized"
                / "ASAP_SAS"
                / "ASAP_SAS_1_rubric_standard.json"
            )
            changed = load_json(rubric_path)
            changed[0]["source_text"] = "updated official rubric"
            write_json(rubric_path, changed)
            current = audit_prepared_benchmark(prepared)
            self.assertNotEqual(
                baseline["prepared_content_sha256"],
                current["prepared_content_sha256"],
            )

    def test_text_answers_are_selected_without_image_files(self):
        previous_metadata = main_pipeline._GLOBAL_ANSWER_METADATA
        previous_policy = main_pipeline.SAMPLE_QUALITY_POLICY
        try:
            main_pipeline._GLOBAL_ANSWER_METADATA = {
                "A": {"question_id": "Q", "raw_text": "one"},
                "B": {"question_id": "Q", "raw_text": "two"},
                "C": {"question_id": "OTHER", "raw_text": "three"},
            }
            main_pipeline.SAMPLE_QUALITY_POLICY = SampleQualityPolicy.raw()
            selected = main_pipeline.select_question_text_answers(
                {"question_id": "Q"},
                None,
                "all",
            )
            self.assertEqual(selected, ["A.txt", "B.txt"])
        finally:
            main_pipeline._GLOBAL_ANSWER_METADATA = previous_metadata
            main_pipeline.SAMPLE_QUALITY_POLICY = previous_policy

    def test_public_runner_dry_run_uses_text_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            prepared = self.prepare_fixture(temporary_root)
            context = run_benchmark._prepared_context(prepared)
            with (
                mock.patch.object(
                    run_benchmark,
                    "_result_root",
                    return_value=temporary_root / "results",
                ),
                mock.patch.object(
                    run_benchmark, "_run", return_value=0
                ) as execute,
            ):
                code = run_benchmark.grade(
                    context,
                    questions=["ASAP_SAS_1"],
                    split="test",
                    run_id="dry_run",
                    force=False,
                    a3wa_config=None,
                    dry_run=True,
                    evaluate_after=False,
                )
            self.assertEqual(code, 0)
            command = execute.call_args.args[0]
            self.assertIn("text_only", command)
            self.assertIn("--answer-split", command)
            self.assertIn("test", command)

    def test_external_holdout_maps_train_partition_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            prepared = self.prepare_fixture(temporary_root)
            context = run_benchmark._prepared_context(prepared)
            result_root = temporary_root / "results"
            with (
                mock.patch.object(
                    run_benchmark,
                    "_result_root",
                    return_value=result_root,
                ),
                mock.patch.object(
                    run_benchmark, "_run", return_value=0
                ) as execute,
            ):
                code = run_benchmark.grade(
                    context,
                    questions=["ASAP_SAS_1"],
                    split="external_holdout",
                    run_id="external_holdout",
                    force=False,
                    a3wa_config=None,
                    dry_run=True,
                    evaluate_after=False,
                )
            self.assertEqual(code, 0)
            command = execute.call_args.args[0]
            self.assertEqual(
                command[command.index("--answer-split") + 1],
                "external_holdout",
            )
            manifest = load_json(
                result_root / "runs" / "external_holdout" / "run_manifest.json"
            )
            self.assertEqual(manifest["split"], "external_holdout")
            self.assertEqual(manifest["source_split"], "train")

            split_path = load_json(prepared / "exam_database.json")[0][
                "rubric_split_path"
            ]
            question = {
                "question_id": "ASAP_SAS_1",
                "rubric_split_path": str(prepared / split_path),
            }
            previous_policy = main_pipeline.SAMPLE_QUALITY_POLICY
            try:
                main_pipeline.SAMPLE_QUALITY_POLICY = SampleQualityPolicy.raw()
                selected = main_pipeline.answer_ids_for_split(
                    question,
                    "external_holdout",
                )
            finally:
                main_pipeline.SAMPLE_QUALITY_POLICY = previous_policy
            self.assertEqual(
                selected,
                set(load_json(prepared / split_path)["train"]),
            )
            checkpoint = [
                {"student_id": answer_id}
                for answer_id in sorted(selected)
            ]
            run_dir = result_root / "runs" / "external_holdout"
            write_json(
                run_dir / "ASAP_SAS_1_grading_checkpoint.json",
                checkpoint,
            )
            self.assertEqual(
                run_benchmark._run_completeness(
                    context,
                    run_dir=run_dir,
                    questions=["ASAP_SAS_1"],
                    split="external_holdout",
                    limit=None,
                ),
                {},
            )

    def test_public_run_archives_portable_inputs_and_a3wa_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            prepared = self.prepare_fixture(temporary_root)
            context = run_benchmark._prepared_context(prepared)
            config = temporary_root / "a3wa.json"
            config.write_text('{"schema_version": 1}\n', encoding="utf-8")
            result_root = temporary_root / "results"
            with (
                mock.patch.object(
                    run_benchmark,
                    "_result_root",
                    return_value=result_root,
                ),
                mock.patch.object(run_benchmark, "_run", return_value=0),
            ):
                code = run_benchmark.grade(
                    context,
                    questions=["ASAP_SAS_1"],
                    split="test",
                    run_id="portable",
                    force=False,
                    a3wa_config=str(config),
                    dry_run=False,
                    evaluate_after=False,
                )
            self.assertEqual(code, 0)
            run_dir = result_root / "runs" / "portable"
            self.assertTrue(
                (run_dir / "calibration" / "a3wa_config.json").is_file()
            )
            self.assertTrue(
                (
                    run_dir
                    / "dataset_snapshot"
                    / "splits"
                    / "ASAP_SAS_1.json"
                ).is_file()
            )
            manifest = load_json(run_dir / "run_manifest.json")
            self.assertEqual(
                manifest["a3wa_config"],
                str(Path("calibration") / "a3wa_config.json"),
            )

    def test_public_workflow_cli_dry_run_is_parser_compatible(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = self.prepare_fixture(Path(temporary))
            dataset_result_root = (
                PROJECT_ROOT
                / "results_runs"
                / "public_benchmarks"
                / "asap_sas_test"
            )
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(
                            PROJECT_ROOT
                            / "scripts"
                            / "benchmarks"
                            / "run_benchmark.py"
                        ),
                        "--prepared-dir",
                        str(prepared),
                        "workflow",
                        "--tag",
                        "cli_contract_test",
                        "--score-calibration",
                        "--limit",
                        "2",
                        "--dry-run",
                    ],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    env={
                        **os.environ,
                        "PYTHONIOENCODING": "utf-8",
                        "PYTHONUTF8": "1",
                    },
                    check=False,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    msg=(completed.stdout or "") + (completed.stderr or ""),
                )
                self.assertIn("text_only", completed.stdout)
                self.assertIn("--score-calibration", completed.stdout)
                self.assertIn("--img-limit 2", completed.stdout)
            finally:
                shutil.rmtree(dataset_result_root, ignore_errors=True)

    def test_failed_gate_is_classified_as_experimental(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "a3wa.json"
            write_json(config, {"deployment_gate": {"passed": False}})
            self.assertEqual(
                run_benchmark._a3wa_deployment_class(config),
                "experimental",
            )

    def test_workflow_continues_failed_gate_as_experimental(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            prepared = self.prepare_fixture(temporary_root)
            config = temporary_root / "a3wa.json"
            write_json(config, {"deployment_gate": {"passed": False}})
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_benchmark.py",
                        "--prepared-dir",
                        str(prepared),
                        "workflow",
                        "ASAP_SAS_1",
                        "--tag",
                        "experimental_workflow",
                    ],
                ),
                mock.patch.object(
                    run_benchmark,
                    "grade",
                    side_effect=[0, 0],
                ) as grade,
                mock.patch.object(
                    run_benchmark,
                    "calibrate",
                    return_value=(0, config),
                ),
            ):
                code = run_benchmark.main()
            self.assertEqual(code, 0)
            self.assertEqual(grade.call_count, 2)
            self.assertTrue(
                grade.call_args_list[1].kwargs["allow_experimental_a3wa"]
            )

    def test_experimental_grade_sets_runtime_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            prepared = self.prepare_fixture(temporary_root)
            context = run_benchmark._prepared_context(prepared)
            config = temporary_root / "a3wa.json"
            write_json(config, {"deployment_gate": {"passed": False}})
            result_root = temporary_root / "results"
            captured_env = {}

            def capture_run(command, *, env, dry_run):
                captured_env.update(env)
                return 0

            with (
                mock.patch.object(
                    run_benchmark,
                    "_result_root",
                    return_value=result_root,
                ),
                mock.patch.object(
                    run_benchmark,
                    "_run",
                    side_effect=capture_run,
                ),
            ):
                code = run_benchmark.grade(
                    context,
                    questions=["ASAP_SAS_1"],
                    split="test",
                    run_id="experimental",
                    force=False,
                    a3wa_config=str(config),
                    dry_run=False,
                    evaluate_after=False,
                    allow_experimental_a3wa=True,
                )
            self.assertEqual(code, 0)
            self.assertEqual(
                captured_env["REFGRADER_ALLOW_EXPERIMENTAL_A3WA"],
                "1",
            )
            manifest = load_json(
                result_root / "runs" / "experimental" / "run_manifest.json"
            )
            self.assertEqual(
                manifest["a3wa_deployment_class"],
                "experimental",
            )

    def test_strict_workflow_stops_before_failed_gate_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            prepared = self.prepare_fixture(temporary_root)
            config = temporary_root / "a3wa.json"
            write_json(config, {"deployment_gate": {"passed": False}})
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_benchmark.py",
                        "--prepared-dir",
                        str(prepared),
                        "workflow",
                        "ASAP_SAS_1",
                        "--tag",
                        "strict_workflow",
                        "--strict-deployment-gate",
                    ],
                ),
                mock.patch.object(
                    run_benchmark,
                    "grade",
                    return_value=0,
                ) as grade,
                mock.patch.object(
                    run_benchmark,
                    "calibrate",
                    return_value=(0, config),
                ),
            ):
                code = run_benchmark.main()
            self.assertEqual(code, 2)
            self.assertEqual(grade.call_count, 1)

    def test_incomplete_test_run_is_not_evaluated(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            prepared = self.prepare_fixture(temporary_root)
            context = run_benchmark._prepared_context(prepared)
            result_root = temporary_root / "results"
            with (
                mock.patch.object(
                    run_benchmark,
                    "_result_root",
                    return_value=result_root,
                ),
                mock.patch.object(run_benchmark, "_run", return_value=0),
                mock.patch.object(run_benchmark, "evaluate_run") as evaluate,
            ):
                code = run_benchmark.grade(
                    context,
                    questions=["ASAP_SAS_1"],
                    split="test",
                    run_id="incomplete",
                    force=False,
                    a3wa_config=None,
                    dry_run=False,
                    evaluate_after=True,
                )
            self.assertEqual(code, 3)
            evaluate.assert_not_called()
            manifest = load_json(
                result_root / "runs" / "incomplete" / "run_manifest.json"
            )
            self.assertEqual(manifest["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
