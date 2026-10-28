import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import httpx

from label_free_model import BASE_URL, create_client, grade_one


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


if __name__ == "__main__":
    unittest.main()
