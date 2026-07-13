import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ocr.backend import ensure_paddle_ocr_cache


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
            }
            with patch.dict("os.environ", env), patch(
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
            }
            with patch.dict("os.environ", env), patch(
                "ocr.backend.subprocess.run", side_effect=failure
            ):
                with self.assertRaisesRegex(RuntimeError, "worker stdout") as raised:
                    ensure_paddle_ocr_cache("image.jpg", "cache")
            self.assertIn("worker stderr", str(raised.exception))
            self.assertIn("exit code 1", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
