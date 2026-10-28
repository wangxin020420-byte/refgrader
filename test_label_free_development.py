import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_label_free_development as development


class LabelFreeDevelopmentTests(unittest.TestCase):
    def test_check_is_offline_and_never_marks_formal_ready(self):
        manifest = {"scope": "development_only_not_formal"}
        with patch.object(development, "_inputs", return_value=manifest), \
             patch.object(development, "create_client", side_effect=AssertionError("model called")), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            code = development.main(["--check", "--run-id", "test1"])
        self.assertEqual(code, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["live_calls"], 0)
        self.assertFalse(report["ready_for_formal_experiment"])

    def test_test_split_is_not_accepted(self):
        with patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as result:
                development.main(["--check", "--run-id", "test1", "--split", "test"])
        self.assertEqual(result.exception.code, 2)

    def test_single_writer_lock_is_released(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with development._single_writer(directory):
                with self.assertRaisesRegex(ValueError, "writer"):
                    with development._single_writer(directory):
                        pass
            with development._single_writer(directory):
                self.assertTrue((directory / "writer.lock").is_file())


if __name__ == "__main__":
    unittest.main()
