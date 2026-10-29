"""Offline end-to-end test for candidate inputs and resumable grading."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from label_free_model import BASE_URL, create_client
from label_free_checkpoint import Checkpoint
from label_free_run import run_questions
from scripts.verify_label_free_question_snapshot import verify
from test_label_free_scoring import LabelFreeScoringTests


class CandidateEndToEndTests(unittest.TestCase):
    def test_candidate_multireference_resume_never_reads_teacher_or_old_rubric(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            prepared.mkdir()
            LabelFreeScoringTests().fixture(prepared)
            students = prepared / "student_images/Q_1"
            students.mkdir(parents=True)
            for answer_id in ("A1", "A2", "A3"):
                (students / (answer_id + ".png")).write_bytes(b"student-image")
            (prepared / "teacher_scores.json").write_text("teacher-private", encoding="utf-8")
            snapshot_root = root / "candidate"
            images = snapshot_root / "images"
            images.mkdir(parents=True)
            refs = []
            for name in ("a.png", "b.png"):
                content = name.encode("ascii")
                (images / name).write_bytes(content)
                refs.append({"path": "images/" + name, "sha256": hashlib.sha256(content).hexdigest()})
            candidate = {"schema": "label_free_question_candidate_v1", "source_revision": "a" * 40,
                "approved_for_formal_experiment": False, "reference_with_teacher_word_questions": [],
                "questions": [{"question_id": "Q_1", "question_text": "question", "max_score": 10,
                    "question_images": [], "rubric": [
                        {"id": "s1", "description": "first", "points": 4, "reference_text": "one", "reference_image": refs[0]},
                        {"id": "s2", "description": "second", "points": 6, "reference_text": "two", "reference_image": refs[1]},
                    ]}]}
            (snapshot_root / "snapshot.json").write_text(json.dumps(candidate), encoding="utf-8")
            candidate_hash = verify(snapshot_root)["snapshot_sha256_normalized"]
            loss = {"lambda1": 1, "lambda2": 1, "mu1": 1, "mu2": 1}
            config = {"question_ids": ["Q_1"], "split": "test", "m": 0.5, "loss_parameters": loss,
                      "candidate_snapshot_sha256": candidate_hash}
            manifest = {"run_config": config, "data_hash": "fixture"}
            calls = []
            def respond(request):
                calls.append(request)
                body = json.loads(request.content)
                content = body["messages"][0]["content"]
                if body["model"] == "glm-4.6v" and "reference-answer image" in content[0]["text"]:
                    payload = {"reference_text": "reference"}
                elif body["model"] == "glm-4.6v":
                    payload = {"items": [{"id": ident, "status": "present", "evidence": "visible"} for ident in ("s1", "s2")]}
                else:
                    payload = {"items": [{"id": "s1", "score": 4, "category": "match"},
                                         {"id": "s2", "score": 6, "category": "match"}]}
                return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 0,
                    "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps(payload)}}]})
            original_open = Path.open
            def guard(path, *args, **kwargs):
                if path.name in {"teacher_scores.json", "exam_database.json"} or "rubrics" in path.parts or "quality_control" in path.parts:
                    raise AssertionError("forbidden input read")
                return original_open(path, *args, **kwargs)
            run_dir = root / "run"
            candidate_manifest = {**manifest, "run_config": {**config, "risk_policy": "candidate_max_v1"}}
            candidate_run = root / "candidate_run"
            with patch.object(Path, "open", guard), create_client("fake", http_client=httpx.Client(
                    transport=httpx.MockTransport(respond), trust_env=False)) as client:
                first = run_questions(prepared, ["Q_1"], "test", run_dir, manifest, client,
                    m=0.5, loss_parameters=loss, max_new=1, candidate_snapshot=snapshot_root)
                second = run_questions(prepared, ["Q_1"], "test", run_dir, manifest, client,
                    m=0.5, loss_parameters=loss, max_new=1, candidate_snapshot=snapshot_root)
                alternative = run_questions(prepared, ["Q_1"], "test", candidate_run, candidate_manifest, client,
                    m=0.5, loss_parameters=loss, max_new=1, candidate_snapshot=snapshot_root,
                    risk_policy="candidate_max_v1")
            self.assertEqual(first, {"completed_before": 0, "completed_now": 1})
            self.assertEqual(second, {"completed_before": 1, "completed_now": 1})
            self.assertEqual(alternative, {"completed_before": 0, "completed_now": 1})
            self.assertEqual(len(calls), 18)
            self.assertTrue(all(str(call.url) == BASE_URL + "/chat/completions" for call in calls))
            self.assertTrue(all(b"teacher-private" not in call.content for call in calls))
            self.assertEqual(len(list((run_dir / "answers").glob("*.json"))), 2)
            decision = Checkpoint(candidate_run, candidate_manifest).load("Q_1_A1")["decision"]
            self.assertEqual(decision["policy"], "candidate_max_bnd_defer_v1")
            self.assertEqual(decision["route"], "POS")


if __name__ == "__main__":
    unittest.main()
