import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import httpx

from label_free_model import BASE_URL, _content, create_client, grade_one


class LabelFreeModelTests(unittest.TestCase):
    def setUp(self):
        self.context = {
            "question_text": "Q", "reference_text": "R", "total_score": 2,
            "initial_rubric": [{"id": "s1", "item": "A", "points": 2,
                                "standard_answer_text": "42", "teacher_score": 999}],
        }
        self.tmp = tempfile.TemporaryDirectory()
        self.image = Path(self.tmp.name) / "student.png"
        self.image.write_bytes(b"fixture")
        self.requests = []

    def tearDown(self):
        self.tmp.cleanup()

    def client(self, bad=False):
        def reply(request):
            self.requests.append(request)
            body = json.loads(request.content)
            first = body["messages"][0]["content"]
            if body["model"] == "glm-4.6v" and "reference-answer image" in first[0]["text"]:
                payload = {"reference_text": "42"}
            elif body["model"] == "glm-4.6v":
                payload = {"items": [{"id": "s1", "status": "present", "evidence": "42"}]}
            else:
                payload = {"items": [{"id": "s1", "score": 3 if bad else 2, "category": "match"}]}
            return httpx.Response(200, json={"id": "mock", "object": "chat.completion", "created": 0,
                "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(payload)}}]})
        return create_client("fake-key", http_client=httpx.Client(transport=httpx.MockTransport(reply), trust_env=False))

    def test_mocked_probes_use_only_coding_plan_and_no_teacher_field(self):
        with self.client() as client:
            result = grade_one(client, self.context, self.image)
        self.assertEqual(len(self.requests), 4)
        self.assertTrue(all(str(r.url) == BASE_URL + "/chat/completions" for r in self.requests))
        self.assertTrue(all(b"teacher_score" not in r.content for r in self.requests))
        self.assertEqual(result["probe_scores"], [2, 2, 2])
        self.assertEqual(result["risks"], {"U_E": 0, "U_S": 0, "U_R": 0})

    def test_out_of_range_probe_fails(self):
        with self.client(bad=True) as client:
            with self.assertRaisesRegex(ValueError, "outside rubric"):
                grade_one(client, self.context, self.image)

    def test_reference_image_transcribed_before_scoring(self):
        with self.client() as client:
            result = grade_one(client, self.context, self.image, reference_image=self.image)
        self.assertEqual(len(self.requests), 5)
        self.assertEqual(result["reference_image_transcription"], "42")
        self.assertEqual(result["probe_scores"], [2, 2, 2])

    def test_multiple_reference_images_keep_item_mapping(self):
        second = Path(self.tmp.name) / "second.png"
        second.write_bytes(b"fixture2")
        mappings = [{"path": self.image, "item_ids": ["s1"]},
                    {"path": second, "item_ids": ["s1"]}]
        with self.client() as client:
            result = grade_one(client, self.context, self.image, reference_images=mappings)
        self.assertEqual(len(self.requests), 6)
        self.assertEqual(result["reference_image_transcriptions"], [
            {"item_ids": ["s1"], "reference_text": "42"},
            {"item_ids": ["s1"], "reference_text": "42"}])
        scoring_body = json.loads(self.requests[-1].content)
        self.assertIn('"item_ids": ["s1"]', scoring_body["messages"][0]["content"])

    def test_invalid_image_mapping_fails_before_calls(self):
        with self.client() as client:
            with self.assertRaisesRegex(ValueError, "item IDs"):
                grade_one(client, self.context, self.image,
                          reference_images=[{"path": self.image, "item_ids": ["unknown"]}])
        self.assertEqual(len(self.requests), 0)

    def test_json_code_fence_is_accepted_without_repair(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content='```json\n{"items": []}\n```'))])
        self.assertEqual(_content(response, stage="student_image"), {"items": []})

    def test_invalid_json_reports_stage_and_metadata_without_answer_content(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="length", message=SimpleNamespace(content='private student text {'))])
        with self.assertRaises(ValueError) as error:
            _content(response, stage="text_probe_2")
        message = str(error.exception)
        self.assertIn("text_probe_2", message)
        self.assertIn("finish_reason=length", message)
        self.assertIn("chars=", message)
        self.assertNotIn("private student text", message)

    def test_invalid_nontext_content_is_not_silently_accepted(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=None))])
        with self.assertRaisesRegex(ValueError, "content_type=NoneType"):
            _content(response, stage="student_image")

    def test_invalid_json_retries_only_failed_model_request(self):
        calls = {"glm-4.6v": 0, "glm-5.2": 0}
        def reply(request):
            body = json.loads(request.content)
            model = body["model"]
            calls[model] += 1
            if calls[model] == 1:
                content = "private malformed {"
            elif model == "glm-4.6v":
                content = json.dumps({"items": [{"id": "s1", "status": "present", "evidence": "42"}]})
            else:
                content = json.dumps({"items": [{"id": "s1", "score": 2, "category": "match"}]})
            return httpx.Response(200, json={"id": "mock", "object": "chat.completion", "created": 0,
                "model": model, "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": content}}]})
        with create_client("fake", http_client=httpx.Client(transport=httpx.MockTransport(reply), trust_env=False)) as client:
            result = grade_one(client, self.context, self.image)
        self.assertEqual(calls, {"glm-4.6v": 2, "glm-5.2": 4})
        self.assertEqual(result["probe_scores"], [2, 2, 2])

    def test_invalid_json_exhaustion_is_bounded_and_redacted(self):
        calls = []
        def reply(request):
            calls.append(request)
            return httpx.Response(200, json={"id": "mock", "object": "chat.completion", "created": 0,
                "model": "glm-4.6v", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "private student text {"}}]})
        with create_client("fake", http_client=httpx.Client(transport=httpx.MockTransport(reply), trust_env=False)) as client:
            with self.assertRaises(ValueError) as error:
                grade_one(client, self.context, self.image)
        self.assertEqual(len(calls), 3)
        self.assertIn("JSON attempts exhausted=3", str(error.exception))
        self.assertNotIn("private student text", str(error.exception))
        self.assertTrue(all(str(r.url) == BASE_URL + "/chat/completions" for r in calls))


if __name__ == "__main__":
    unittest.main()
