import json
from pathlib import Path
import tempfile
import unittest

from label_free_checkpoint import Checkpoint


class LabelFreeCheckpointTests(unittest.TestCase):
    def test_resume_retains_completed_samples_without_regrading(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = {"input_sha256": "abc", "policy": {"m": 0.5}}
            checkpoint = Checkpoint(temporary, manifest)
            checkpoint.save("A1", {"route": "POS", "score": 3})
            resumed = Checkpoint(temporary, manifest)
            self.assertEqual(resumed.load("A1"), {"route": "POS", "score": 3})
            self.assertIsNone(resumed.load("A2"))
            resumed.save("A1", {"route": "POS", "score": 3})
            with self.assertRaisesRegex(ValueError, "overwrite"):
                resumed.save("A1", {"route": "POS", "score": 4})

    def test_manifest_change_refuses_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            Checkpoint(temporary, {"input_sha256": "before"})
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                Checkpoint(temporary, {"input_sha256": "after"})

    def test_tampering_and_invalid_ids_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Checkpoint(temporary, {"input_sha256": "abc"})
            checkpoint.save("A1", {"score": 3})
            path = Path(temporary) / "answers" / "A1.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["result"]["score"] = 4
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity"):
                checkpoint.load("A1")
            with self.assertRaisesRegex(ValueError, "identifier"):
                checkpoint.load("../other")
            with self.assertRaises(ValueError):
                checkpoint.save("A2", {"score": float("nan")})

    def test_answer_files_without_manifest_are_not_adopted(self):
        with tempfile.TemporaryDirectory() as temporary:
            answers = Path(temporary) / "answers"
            answers.mkdir()
            (answers / "A1.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "without manifest"):
                Checkpoint(temporary, {"input_sha256": "abc"})


if __name__ == "__main__":
    unittest.main()
