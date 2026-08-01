import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import step4_vlm_grader as grader


class A3WARuntimeConfigTests(unittest.TestCase):
    def setUp(self):
        self.original_path = grader.A3WA_CALIBRATION_CONFIG_PATH
        self.original_config = grader._A3WA_RUNTIME_CONFIG

    def tearDown(self):
        grader.A3WA_CALIBRATION_CONFIG_PATH = self.original_path
        grader._A3WA_RUNTIME_CONFIG = self.original_config

    def test_failed_gate_is_rejected_consistently_across_threads(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config_path = Path(temporary_dir) / "a3wa.json"
            config_path.write_text(
                json.dumps({"deployment_gate": {"passed": False}}),
                encoding="utf-8",
            )
            grader.A3WA_CALIBRATION_CONFIG_PATH = str(config_path)
            grader._A3WA_RUNTIME_CONFIG = None

            first_load_started = threading.Event()
            release_first_load = threading.Event()
            original_load = grader.json.load

            def delayed_load(handle):
                first_load_started.set()
                self.assertTrue(release_first_load.wait(timeout=2))
                return original_load(handle)

            def load_config():
                try:
                    grader.load_a3wa_runtime_config()
                except Exception as exc:
                    return exc
                return None

            with mock.patch.dict(
                os.environ,
                {"REFGRADER_ALLOW_EXPERIMENTAL_A3WA": ""},
            ), mock.patch.object(grader.json, "load", side_effect=delayed_load):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    first = executor.submit(load_config)
                    self.assertTrue(first_load_started.wait(timeout=2))
                    second = executor.submit(load_config)
                    self.assertFalse(second.done())
                    release_first_load.set()
                    errors = [first.result(timeout=2), second.result(timeout=2)]

            self.assertTrue(all(isinstance(error, RuntimeError) for error in errors))
            self.assertIsNone(grader._A3WA_RUNTIME_CONFIG)

    def test_experimental_override_is_cached_after_validation(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config_path = Path(temporary_dir) / "a3wa.json"
            payload = {
                "deployment_gate": {"passed": False},
                "marker": "experimental",
            }
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            grader.A3WA_CALIBRATION_CONFIG_PATH = str(config_path)
            grader._A3WA_RUNTIME_CONFIG = None

            with mock.patch.dict(
                os.environ,
                {"REFGRADER_ALLOW_EXPERIMENTAL_A3WA": "1"},
            ):
                loaded = grader.load_a3wa_runtime_config()

            self.assertEqual(loaded, payload)
            self.assertIs(grader._A3WA_RUNTIME_CONFIG, loaded)


if __name__ == "__main__":
    unittest.main()
