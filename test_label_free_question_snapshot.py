import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import httpx

from scripts.build_label_free_question_snapshot import build
from scripts.verify_label_free_question_snapshot import verify
from label_free_model import create_client, grade_one
from label_free_question_snapshot import load_candidate_question


class CandidateSnapshotTests(unittest.TestCase):
    def test_freezes_original_items_and_all_images_without_teacher_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            (root / "question").mkdir()
            (root / "images").mkdir()
            (root / "images/a.png").write_bytes(b"image-a")
            (root / "images/b.png").write_bytes(b"image-b")
            (root / "teacher_scores.json").write_text("private", encoding="utf-8")
            payload = [{"question_id": "Q_1", "max_score": 5,
                "content": {"text": "question", "image_paths": []},
                "grading_rubric": [
                    {"step_id": 1, "description": "item1", "score": 2, "standard_answer_image": "images/a.png"},
                    {"step_id": 2, "description": "item2", "score": 3, "standard_answer_image": "images/b.png", "standard_answer_text": "教师评分口径"},
                ]}, {"question_id": "Q_2", "max_score": 5,
                "content": {"text": "question two", "image_paths": []},
                "grading_rubric": [
                    {"step_id": 1, "description": "one", "score": 2, "standard_answer_image": "images/a.png"},
                    {"step_id": 2, "description": "two", "score": 3, "standard_answer_image": "images/b.png"},
                ]}]
            (root / "question/q.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "add", "--", "question", "images"], cwd=root, check=True)
            subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                            "commit", "-qm", "fixture"], cwd=root, check=True)
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            destination = Path(tmp) / "snapshot"
            original = Path.open
            def guarded(path, *args, **kwargs):
                if path.name == "teacher_scores.json":
                    raise AssertionError("teacher score opened")
                return original(path, *args, **kwargs)
            with patch.object(Path, "open", guarded):
                result = build(root, destination, revision)
            self.assertEqual(result["images"], 2)
            self.assertEqual(result["flagged_questions"], ["Q_1"])
            snapshot = json.loads((destination / "snapshot.json").read_text(encoding="utf-8"))
            self.assertFalse(snapshot["approved_for_formal_experiment"])
            self.assertEqual([x["points"] for x in snapshot["questions"][0]["rubric"]], [2, 3])
            self.assertEqual(len(list((destination / "images").iterdir())), 2)
            self.assertEqual(verify(destination)["images"], 2)
            with self.assertRaisesRegex(ValueError, "unresolved teacher"):
                load_candidate_question(destination, "Q_1")
            candidate = load_candidate_question(destination, "Q_2")
            self.assertEqual([x["points"] for x in candidate["context"]["initial_rubric"]], [2, 3])
            self.assertEqual({tuple(x["item_ids"]) for x in candidate["reference_images"]}, {("step_1",), ("step_2",)})
            self.assertFalse(candidate["approved_for_formal_experiment"])
            calls = []
            def respond(request):
                calls.append(str(request.url))
                body = json.loads(request.content)
                content = body["messages"][0]["content"]
                if body["model"] == "glm-4.6v" and "reference-answer image" in content[0]["text"]:
                    data = {"reference_text": "visible reference"}
                elif body["model"] == "glm-4.6v":
                    data = {"items": [{"id": key, "status": "present", "evidence": "answer"} for key in ("step_1", "step_2")]}
                else:
                    data = {"items": [{"id": "step_1", "score": 2, "category": "match"},
                                      {"id": "step_2", "score": 3, "category": "match"}]}
                return httpx.Response(200, json={"id": "mock", "object": "chat.completion", "created": 0,
                    "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps(data)}}]})
            with create_client("fake-key", http_client=httpx.Client(transport=httpx.MockTransport(respond), trust_env=False)) as client:
                graded = grade_one(client, candidate["context"], root / "images/a.png",
                                   reference_images=candidate["reference_images"])
            self.assertEqual(graded["probe_scores"], [5, 5, 5])
            self.assertEqual(len(graded["reference_image_transcriptions"]), 2)
            self.assertEqual(len(calls), 6)
            self.assertTrue(all(url == "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions" for url in calls))
            with self.assertRaises(FileExistsError):
                build(root, destination, revision)
            (root / "images/a.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "uncommitted"):
                build(root, Path(tmp) / "another", revision)
            self.assertFalse((Path(tmp) / "another").exists())
            next((destination / "images").iterdir()).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify(destination)


if __name__ == "__main__":
    unittest.main()
