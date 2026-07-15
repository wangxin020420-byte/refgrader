"""Runtime helpers for the optional PaddleOCR extraction backend.

The formal grading environment invokes the isolated ``.venv-ocr`` worker as a
subprocess, so importing this module never imports PaddleOCR itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .runtime_env import build_paddlex_environment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKER = PROJECT_ROOT / "ocr" / "paddle_ocr_worker.py"


def default_ocr_python() -> Path:
    """Return the platform-specific Python path inside .venv-ocr."""
    if os.name == "nt":
        return PROJECT_ROOT / ".venv-ocr" / "Scripts" / "python.exe"
    return PROJECT_ROOT / ".venv-ocr" / "bin" / "python"


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def ocr_json_path(
    output_dir: str | os.PathLike[str],
    image_path: str | os.PathLike[str],
) -> Path:
    return Path(output_dir) / f"{Path(image_path).stem}.json"


def cache_matches_image(
    cache_path: str | os.PathLike[str],
    image_path: str | os.PathLike[str],
) -> bool:
    cached = load_json(cache_path)
    if not cached:
        return False
    return cached.get("image", {}).get("sha256") == sha256_file(image_path)


def ensure_paddle_ocr_cache(
    input_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    force: bool = False,
    min_confidence: float = 0.5,
    timeout: int = 3600,
) -> subprocess.CompletedProcess[str]:
    python_path = Path(
        os.getenv("REFGRADER_OCR_PYTHON", str(default_ocr_python()))
    )
    worker_path = Path(os.getenv("REFGRADER_OCR_WORKER", str(DEFAULT_WORKER)))
    ocr_device = os.getenv("REFGRADER_OCR_DEVICE", "cpu")
    if not python_path.exists():
        raise FileNotFoundError(
            f"Isolated OCR Python not found: {python_path}. "
            "Run scripts/setup_paddle_ocr.ps1 on Windows or "
            "scripts/setup_paddle_ocr.sh on Linux."
        )
    if not worker_path.exists():
        raise FileNotFoundError(f"PaddleOCR worker not found: {worker_path}")

    command = [
        str(python_path),
        str(worker_path),
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--device",
        ocr_device,
        "--min-confidence",
        str(min_confidence),
    ]
    if force:
        command.append("--force")

    worker_environment, _ = build_paddlex_environment(PROJECT_ROOT)

    try:
        return subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=worker_environment,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        diagnostics = "\n".join(
            part
            for part in (
                f"stdout:\n{stdout}" if stdout else "",
                f"stderr:\n{stderr}" if stderr else "",
            )
            if part
        )
        raise RuntimeError(
            f"PaddleOCR worker failed for {input_path} with exit code "
            f"{exc.returncode}."
            + (f"\n{diagnostics}" if diagnostics else "")
        ) from exc
