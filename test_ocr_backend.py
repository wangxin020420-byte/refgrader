import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ocr.backend import ensure_paddle_ocr_cache
from ocr.runtime_env import (
    PADDLEX_CACHE_ENV,
    PADDLEX_SOURCE_CHECK_ENV,
    PADDLEX_SOURCE_ENV,
    build_paddlex_environment,
    resolve_paddlex_cache_home,
)


class OCRBackendTests(unittest.TestCase):
    def _paths(self, root):
        python_path = root / "python.exe"
        worker_path = root / "worker.py"
        python_path.touch()
        worker_path.touch()
        return python_path, worker_path

    def test_force_flag_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            python_path, worker_path = self._paths(root)
            completed = subprocess.CompletedProcess([], 0, stdout="ok", stderr="")
            env = {
                "REFGRADER_OCR_PYTHON": str(python_path),
                "REFGRADER_OCR_WORKER": str(worker_path),
                PADDLEX_CACHE_ENV: str(root / "model-cache"),
            }
            with patch.dict("os.environ", env, clear=True), patch(
                "ocr.backend.subprocess.run", return_value=completed
            ) as run:
                ensure_paddle_ocr_cache("image.jpg", "cache", force=False)
                self.assertNotIn("--force", run.call_args.args[0])
                ensure_paddle_ocr_cache("image.jpg", "cache", force=True)
                self.assertIn("--force", run.call_args.args[0])

    def test_worker_failure_includes_captured_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            python_path, worker_path = self._paths(root)
            failure = subprocess.CalledProcessError(
                1,
                [str(python_path), str(worker_path)],
                output="worker stdout",
                stderr="worker stderr",
            )
            env = {
                "REFGRADER_OCR_PYTHON": str(python_path),
                "REFGRADER_OCR_WORKER": str(worker_path),
                PADDLEX_CACHE_ENV: str(root / "model-cache"),
            }
            with patch.dict("os.environ", env, clear=True), patch(
                "ocr.backend.subprocess.run", side_effect=failure
            ):
                with self.assertRaisesRegex(RuntimeError, "worker stdout") as raised:
                    ensure_paddle_ocr_cache("image.jpg", "cache")
            self.assertIn("worker stderr", str(raised.exception))
            self.assertIn("exit code 1", str(raised.exception))

    def test_explicit_model_cache_is_passed_to_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            python_path, worker_path = self._paths(root)
            cache_path = root / "model-cache"
            completed = subprocess.CompletedProcess([], 0, stdout="ok", stderr="")
            env = {
                "REFGRADER_OCR_PYTHON": str(python_path),
                "REFGRADER_OCR_WORKER": str(worker_path),
                PADDLEX_CACHE_ENV: str(cache_path),
            }
            with patch.dict(os.environ, env, clear=True), patch(
                "ocr.backend.subprocess.run", return_value=completed
            ) as run:
                ensure_paddle_ocr_cache("image.jpg", "cache")

            worker_env = run.call_args.kwargs["env"]
            self.assertEqual(Path(worker_env[PADDLEX_CACHE_ENV]), cache_path)
            self.assertEqual(worker_env[PADDLEX_SOURCE_ENV], "modelscope")
            self.assertEqual(worker_env[PADDLEX_SOURCE_CHECK_ENV], "True")
            self.assertTrue(cache_path.is_dir())

    def test_explicit_model_cache_has_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            configured = Path(tmp) / "shared-cache"
            resolved = resolve_paddlex_cache_home(
                Path(tmp) / "checkout",
                {PADDLEX_CACHE_ENV: str(configured)},
            )
            self.assertEqual(resolved, configured.absolute())

    def test_environment_builder_preserves_explicit_source_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache"
            env, _ = build_paddlex_environment(
                tmp,
                {
                    PADDLEX_CACHE_ENV: str(cache_path),
                    PADDLEX_SOURCE_ENV: "aistudio",
                    PADDLEX_SOURCE_CHECK_ENV: "False",
                },
            )
            self.assertEqual(env[PADDLEX_SOURCE_ENV], "aistudio")
            self.assertEqual(env[PADDLEX_SOURCE_CHECK_ENV], "False")


if __name__ == "__main__":
    unittest.main()
