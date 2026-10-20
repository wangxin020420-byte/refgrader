import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import scripts.run_reviewed_full_workflow as workflow
from sample_quality import SampleQualityPolicy


class ReviewedWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prepared = self.root / "data/csbench"
        self.prepared.mkdir(parents=True)

    def dataset(self, count=2):
        questions, metadata, scores = [], [], {}
        for i in range(1, count+1):
            qid = f"CO_{i}"
            questions.append({"question_id": qid, "total_score": 10})
            ids = [f"A_{i}_{j}" for j in range(9)]
            for aid in ids:
                metadata.append({"question_id": qid, "answer_id": aid})
                scores[aid] = {qid: 5}
            workflow.write(self.prepared / f"splits/by_question/{qid}.json", {"calibration": ids[:5], "validation": ids[5:7], "test": ids[7:]})
        workflow.write(self.prepared / "exam_database.json", questions)
        workflow.write(self.prepared / "teacher_scores.json", scores)
        (self.prepared / "answer_metadata.jsonl").write_text("\n".join(json.dumps(r) for r in metadata)+"\n", encoding="utf8")

    def decision(self, q=1, j=7, kind="confirmed_noise", score=None):
        return {"question_id": f"CO_{q}", "answer_id": f"A_{q}_{j}", "decision": kind, "corrected_score": score, "reason_code": "checked", "review_note": "checked from original", "reviewer": "test", "reviewed_at": "2026-10-19T20:00:00"}

    def raw(self, *rows):
        return ("\n".join(json.dumps(r) for r in rows)+"\n").encode()

    def policy(self, *rows):
        return workflow.make_policy(self.prepared, self.raw(*rows), "test_policy", "snapshot.jsonl")

    def prepared_job(self):
        self.dataset(45)
        self.decisions = self.root / "live.jsonl"
        self.decisions.write_bytes(self.raw(self.decision()))
        self.args = SimpleNamespace(new_run=False, run_id=None, decisions=self.decisions, artifacts_repo=self.root / "artifacts", grade_attempts=2)
        base = self.root / "results_runs/reviewed_csbench45_glm52"
        for key, value in {"ROOT": self.root, "PREPARED": self.prepared, "BASE": base, "LATEST": base/"latest.json"}.items():
            p = patch.object(workflow, key, value)
            p.start()
            self.addCleanup(p.stop)
        return workflow.prepare_job(self.args)

    def test_corrected_labels_and_exclusions_share_one_policy(self):
        self.dataset()
        payload, normalized = self.policy(self.decision(), self.decision(1, 5, "corrected", 8), self.decision(2, 5, "ambiguous"))
        report = workflow.selection_report(self.prepared, payload, normalized, expected_questions=2)
        self.assertEqual(report["splits"]["test"]["effective"], 3)
        self.assertEqual(report["splits"]["validation"]["corrected"], 1)
        self.assertEqual(report["splits"]["validation"]["effective"], 4)
        path = self.root/"policy.json"
        workflow.write(path, payload)
        with patch.dict(os.environ, {"REFGRADER_SAMPLE_POLICY_MODE": "raw"}):
            policy = SampleQualityPolicy.load(self.prepared, explicit_path=path, force_raw=False)
        self.assertEqual(policy.effective_teacher_score("CO_1", "A_1_5", 5), 8)
        self.assertIsNone(policy.effective_teacher_score("CO_1", "A_1_7", 5))
        self.assertEqual(policy.effective_teacher_score("CO_2", "A_2_5", 5), 5)

    def test_duplicate_decisions_are_rejected(self):
        self.dataset()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.policy(self.decision(), self.decision())

    def test_out_of_range_correction_is_rejected(self):
        self.dataset()
        with self.assertRaisesRegex(ValueError, "between"):
            self.policy(self.decision(kind="corrected", score=11))

    def test_zero_calibration_cannot_fall_back_to_test(self):
        self.dataset()
        payload, rows = self.policy(*(self.decision(j=j) for j in range(5)))
        with self.assertRaisesRegex(ValueError, "cross-split fallback"):
            workflow.selection_report(self.prepared, payload, rows, expected_questions=2)

    def test_overlap_in_splits_is_rejected(self):
        self.dataset()
        p = self.prepared / "splits/by_question/CO_1.json"
        split = workflow.read(p)
        split["test"].append(split["calibration"][0])
        workflow.write(p, split)
        policy, rows = self.policy(self.decision())
        with self.assertRaisesRegex(ValueError, "overlapping"):
            workflow.selection_report(self.prepared, policy, rows, expected_questions=2)

    def test_live_review_changes_do_not_change_resumed_snapshot(self):
        first = self.prepared_job()
        frozen = (Path(first["job"]) / "sample_policy.json").read_bytes()
        self.decisions.write_bytes(self.raw(self.decision(), self.decision(2)))
        resumed = workflow.prepare_job(self.args)
        self.assertEqual(first["run_id"], resumed["run_id"])
        self.assertEqual(frozen, (Path(resumed["job"]) / "sample_policy.json").read_bytes())
        workflow.assert_inputs(resumed)
        self.args.new_run = True
        newer = workflow.prepare_job(self.args)
        self.assertNotEqual(first["run_id"], newer["run_id"])
        self.assertEqual(workflow.read(Path(newer["job"])/"selection.json")["reviewed_answers"], 2)
        self.assertEqual(frozen, (Path(first["job"])/"sample_policy.json").read_bytes())

    def test_raw_label_edit_blocks_resume(self):
        state = self.prepared_job()
        path = self.prepared/"teacher_scores.json"
        changed = workflow.read(path)
        changed["A_1_0"]["CO_1"] = 6
        workflow.write(path, changed)
        with self.assertRaisesRegex(RuntimeError, "output changed"):
            workflow.assert_inputs(state)

    def test_frozen_policy_edit_is_rejected(self):
        state = self.prepared_job()
        (Path(state["job"])/"sample_policy.json").write_text("{}", encoding="utf8")
        with self.assertRaisesRegex(RuntimeError, "Frozen workflow input changed"):
            workflow.assert_inputs(state)

    def job_with_recorded_code_repair(self, relative='scripts/run_csbench.py'):
        source = self.root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('# original code\n', encoding='utf-8')
        state = self.prepared_job()
        job = Path(state['job'])
        manifest_before = (job/'input_sha256.json').read_bytes()
        before = workflow.sha(source)
        backup = job/'code_repairs/original.py'
        backup.parent.mkdir(parents=True)
        backup.write_bytes(source.read_bytes())
        source.write_text('# repaired code\n', encoding='utf-8')
        repair = {
            'schema_version': 1, 'run_id': state['run_id'],
            'original_input_manifest_sha256': state['input_manifest_sha256'],
            'files': [{'path': str(source.resolve()), 'before_sha256': before,
                       'after_sha256': workflow.sha(source),
                       'before_copy': 'code_repairs/original.py'}],
        }
        workflow.write(job/'resume_code_repair.json', repair)
        state['resume_code_repair_sha256'] = workflow.sha(job/'resume_code_repair.json')
        return state, source, manifest_before

    def test_recorded_code_repair_preserves_original_input_manifest(self):
        state, source, before = self.job_with_recorded_code_repair()
        workflow.assert_inputs(state)
        self.assertEqual((Path(state['job'])/'input_sha256.json').read_bytes(), before)

    def test_unrecorded_followup_code_edit_is_rejected(self):
        state, source, _ = self.job_with_recorded_code_repair()
        source.write_text('# unrelated change\n', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'code repair is invalid'):
            workflow.assert_inputs(state)

    def test_code_repair_does_not_allow_changed_teacher_labels(self):
        state, _, _ = self.job_with_recorded_code_repair()
        workflow.write(self.prepared/'teacher_scores.json', {'changed': 1})
        with self.assertRaisesRegex(RuntimeError, 'output changed'):
            workflow.assert_inputs(state)

    def test_changed_repair_record_is_rejected(self):
        state, _, _ = self.job_with_recorded_code_repair()
        workflow.write(Path(state['job'])/'resume_code_repair.json', {})
        with self.assertRaisesRegex(RuntimeError, 'record is missing or changed'):
            workflow.assert_inputs(state)

    def test_repair_scope_rejects_other_code_modules(self):
        state, _, _ = self.job_with_recorded_code_repair('model_runtime.py')
        with self.assertRaisesRegex(RuntimeError, 'code repair is invalid'):
            workflow.assert_inputs(state)

    def test_publish_path_normalization_preserves_completed_rubric_proof(self):
        state = self.prepared_job()
        path = self.prepared/'rubrics/manifests/CO/CO_1_optimization.json'
        path.parent.mkdir(parents=True)
        payload = {'question_id':'CO_1', 'score':5,
                   'optimized_rubric':str(self.prepared/'rubrics/optimized/CO/CO_1.json')}
        original = json.dumps(payload, ensure_ascii=False, indent=4).replace('\n', '\r\n').encode('utf-8')
        import hashlib
        expected = hashlib.sha256(original).hexdigest()
        payload['optimized_rubric'] = '${PREPARED_CSBENCH_ROOT}/rubrics/optimized/CO/CO_1.json'
        payload['path_format'] = 'refgrader_placeholders_v1'
        workflow.write(path, payload)
        workflow.verify_stage_proof(state, 'rubric_CO_1', {str(path):expected})
        payload['score'] = 6
        workflow.write(path, payload)
        with self.assertRaisesRegex(RuntimeError, 'Completed stage output changed'):
            workflow.verify_stage_proof(state, 'rubric_CO_1', {str(path):expected})

    def test_retry_and_resume_do_not_repeat_completed_stages(self):
        state = self.prepared_job()
        proof = self.root / "score.json"
        workflow.write(proof, {"score": 5})
        item = {"name": "grade_test", "command": ["python", "grade", "--run-id", "fixed"], "attempts": 2, "proof": [str(proof)]}
        run = Mock(side_effect=[SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)])
        sleep = Mock()
        workflow.execute_stage(state, item, {}, run=run, sleep=sleep)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args, run.call_args_list[1].args)
        sleep.assert_called_once_with(60)
        workflow.execute_stage(state, item, {}, run=run, sleep=sleep)
        self.assertEqual(run.call_count, 2)
        workflow.write(proof, {"score": 6})
        with self.assertRaisesRegex(RuntimeError, "Completed stage output changed"):
            workflow.execute_stage(state, item, {}, run=run, sleep=sleep)
        self.assertEqual(run.call_count, 2)

    def test_failed_analytical_gate_keeps_diagnostic_test_enabled(self):
        state = self.prepared_job()
        config = Path(state["job"])/"a3wa_config.json"
        workflow.write(config, {"deployment_gate": {"passed": False}, "score_calibration": {"enabled": False}})
        run = Mock(return_value=SimpleNamespace(returncode=0))
        item = {"name": "calibrate", "command": ["python", "calibrate"], "attempts": 1, "proof": [str(config)]}
        workflow.execute_stage(state, item, {}, run=run)
        self.assertEqual(state["result_class"], "experimental")
        stages, _ = workflow.build_stages(state)
        test = next(s for s in stages if s["name"] == "grade_test")
        self.assertIn("--allow-experimental-a3wa", test["command"])
        test["proof"] = []
        workflow.execute_stage(state, test, {}, run=run)
        self.assertEqual(run.call_count, 2)

    def test_all_commands_match_existing_parsers_and_private_run_paths(self):
        state = self.prepared_job()
        stages, paths = workflow.build_stages(state)
        workflow.validate_plan(stages)
        rubrics = [s for s in stages if s["name"].startswith("rubric_")]
        self.assertEqual(len(rubrics), 45)
        import main_pipeline
        for stage in rubrics:
            command = stage["command"]
            with patch.object(sys, "argv", command[2:]):
                args = main_pipeline.parse_args()
            self.assertTrue(args.resume_optimization)
            self.assertFalse(args.force_rerun)
            self.assertTrue(Path(args.results_dir).is_relative_to(Path(state["job"])/"rubric_work"))
        val = next(s for s in stages if s["name"]=="grade_validation")
        test = next(s for s in stages if s["name"]=="grade_test")
        self.assertIn(state["run_id"]+"_validation", val["command"])
        self.assertIn(state["run_id"]+"_test", test["command"])
        self.assertNotIn("--force", val["command"])
        self.assertNotIn("--force", test["command"])
        self.assertTrue(paths["test"].endswith(state["run_id"]+"_test"))
        self.assertFalse(any("--push" in s["command"] or "--push-artifacts" in s["command"] for s in stages))

    def test_worker_environment_cannot_inherit_raw_mode_or_old_calibration(self):
        with patch.dict(os.environ, {"REFGRADER_SAMPLE_POLICY_MODE": "raw", "A3WA_CALIBRATION_CONFIG": "old.json", "REFGRADER_TEXT_TEMPERATURE_OVERRIDE": "0", "REFGRADER_TEXT_MODEL_PROVIDER": "deepseek", "REFGRADER_POST_SUCCESS_CMD": "old-hook"}):
            env = workflow.frozen_environment(self.root/"fixed.json")
        self.assertEqual(env["REFGRADER_SAMPLE_POLICY_MODE"], "active")
        self.assertEqual(env["A3WA_CALIBRATION_CONFIG"], "")
        self.assertEqual(env["REFGRADER_TEXT_MODEL_PROVIDER"], "glm5")
        self.assertNotIn("REFGRADER_TEXT_TEMPERATURE_OVERRIDE", env)
        self.assertNotIn("REFGRADER_POST_SUCCESS_CMD", env)

    def test_process_lock_prevents_duplicate_workflows(self):
        base = self.root/"locks"
        code = "from pathlib import Path; import sys; from scripts import run_reviewed_full_workflow as w; w.BASE=Path(sys.argv[1]);\ntry:\n with w.workflow_lock(): pass\nexcept RuntimeError:\n raise SystemExit(23)\n"
        with patch.object(workflow, "BASE", base):
            with workflow.workflow_lock():
                busy = subprocess.run([sys.executable, "-c", code, str(base)], cwd=workflow.ROOT, capture_output=True)
            free = subprocess.run([sys.executable, "-c", code, str(base)], cwd=workflow.ROOT, capture_output=True)
        self.assertEqual(busy.returncode, 23, busy.stderr)
        self.assertEqual(free.returncode, 0, free.stderr)


if __name__ == "__main__":
    unittest.main()
