from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_file_hash_matches(expected: str | None, path: str | Path) -> bool:
    """Match UTF-8 text while tolerating Git LF/CRLF checkout conversion."""
    if not expected:
        return False
    data = Path(path).read_bytes()
    expected_normalized = str(expected).strip().lower()
    candidates = {hashlib.sha256(data).hexdigest()}
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return expected_normalized in candidates

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for rendered in (normalized, normalized.replace("\n", "\r\n")):
        candidates.add(hashlib.sha256(rendered.encode("utf-8")).hexdigest())
    return expected_normalized in candidates
