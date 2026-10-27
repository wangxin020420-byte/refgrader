"""Single-process checkpoint primitive for simulated label-free runs.

The caller must supply complete, independently audited input/config fingerprints.
This module does not establish label independence or run a grading model.
Concurrent writers to the same run directory are not supported.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Invalid checkpoint identifier")
    return value


def _write_atomic(path, payload):
    path = Path(path)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as f:
            temp_path = Path(f.name)
            f.write(_canonical(payload))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


class Checkpoint:
    """Store one immutable manifest and immutable per-answer results.

    One process only; use a distinct run directory for every logical run.
    The manifest detects changes in exactly the fields supplied by the caller.
    """

    def __init__(self, directory, manifest):
        if not isinstance(manifest, dict) or not manifest:
            raise ValueError("A nonempty manifest is required")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.result_dir = self.directory / "answers"
        self.result_dir.mkdir(exist_ok=True)
        self.manifest_path = self.directory / "manifest.json"
        self.manifest_digest = _digest(manifest)
        expected = {"schema": "label_free_checkpoint_v1", "manifest": manifest,
                    "sha256": self.manifest_digest}
        if self.manifest_path.exists():
            stored = self._read(self.manifest_path)
            if stored != expected:
                raise ValueError("Checkpoint manifest mismatch; do not resume this run")
        else:
            if any(self.result_dir.iterdir()):
                raise ValueError("Existing answer files without manifest")
            _write_atomic(self.manifest_path, expected)

    @staticmethod
    def _read(path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid or incomplete checkpoint file") from exc

    def load(self, answer_id):
        answer_id = _identifier(answer_id)
        path = self.result_dir / (answer_id + ".json")
        if not path.exists():
            return None
        record = self._read(path)
        if not isinstance(record, dict) or set(record) != {"answer_id", "manifest_sha256", "result", "sha256"}:
            raise ValueError("Invalid answer checkpoint")
        body = {key: record[key] for key in ("answer_id", "manifest_sha256", "result")}
        if record["answer_id"] != answer_id or record["manifest_sha256"] != self.manifest_digest or record["sha256"] != _digest(body):
            raise ValueError("Answer checkpoint integrity mismatch")
        return record["result"]

    def save(self, answer_id, result):
        answer_id = _identifier(answer_id)
        if not isinstance(result, dict):
            raise ValueError("Result must be an object")
        body = {"answer_id": answer_id, "manifest_sha256": self.manifest_digest, "result": result}
        record = {**body, "sha256": _digest(body)}
        path = self.result_dir / (answer_id + ".json")
        if path.exists():
            if self.load(answer_id) != result:
                raise ValueError("Refusing to overwrite an existing answer result")
            return
        _write_atomic(path, record)
