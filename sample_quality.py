"""Shared sample-quality policy for CSBench data selection and labels.

The raw prepared dataset remains immutable.  An optional active policy removes
confirmed unusable samples and overlays adjudicated teacher scores at read
time.  When no policy exists, every public helper preserves the legacy
behaviour.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


POLICY_SCHEMA_VERSION = 1
DEFAULT_POLICY_RELATIVE_PATH = (
    Path("quality_control") / "policies" / "active_sample_policy.json"
)
POLICY_ENV = "REFGRADER_SAMPLE_POLICY"
POLICY_MODE_ENV = "REFGRADER_SAMPLE_POLICY_MODE"


def _normalize_question_id(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_answer_id(value: Any) -> str:
    return str(value or "").strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_policy_path(prepared_root: str | Path) -> Path:
    return Path(prepared_root).expanduser().resolve() / DEFAULT_POLICY_RELATIVE_PATH


def resolve_policy_path(
    prepared_root: str | Path,
    explicit_path: str | Path | None = None,
) -> Path | None:
    configured = explicit_path or os.getenv(POLICY_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    candidate = default_policy_path(prepared_root)
    return candidate if candidate.is_file() else None


def raw_mode_requested() -> bool:
    return os.getenv(POLICY_MODE_ENV, "active").strip().lower() == "raw"


def _excluded_ids(payload: Any) -> set[str]:
    if isinstance(payload, list):
        return {
            answer_id
            for value in payload
            if (answer_id := _normalize_answer_id(value))
        }
    if isinstance(payload, dict):
        return {
            answer_id
            for value in payload
            if (answer_id := _normalize_answer_id(value))
        }
    return set()


@dataclass(frozen=True)
class SampleQualityPolicy:
    mode: str = "raw"
    policy_id: str = "raw"
    path: Path | None = None
    sha256: str | None = None
    excluded: dict[str, frozenset[str]] = field(default_factory=dict)
    corrected_scores: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def raw(cls) -> "SampleQualityPolicy":
        return cls()

    @classmethod
    def load(
        cls,
        prepared_root: str | Path,
        *,
        explicit_path: str | Path | None = None,
        force_raw: bool | None = None,
    ) -> "SampleQualityPolicy":
        if force_raw is True or (force_raw is None and raw_mode_requested()):
            return cls.raw()
        path = resolve_policy_path(prepared_root, explicit_path)
        if path is None:
            return cls.raw()
        if not path.is_file():
            raise FileNotFoundError(f"Sample-quality policy not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError(f"Sample-quality policy must be a JSON object: {path}")
        if payload.get("schema_version") != POLICY_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported sample-quality policy schema version: "
                f"{payload.get('schema_version')}"
            )

        policy_id = str(payload.get("policy_id") or "").strip()
        if not policy_id:
            raise ValueError(f"Sample-quality policy has no policy_id: {path}")

        excluded: dict[str, frozenset[str]] = {}
        for question_id, values in (payload.get("excluded") or {}).items():
            qid = _normalize_question_id(question_id)
            ids = _excluded_ids(values)
            if qid and ids:
                excluded[qid] = frozenset(ids)

        corrected: dict[str, dict[str, float]] = {}
        for question_id, values in (
            payload.get("corrected_scores") or {}
        ).items():
            qid = _normalize_question_id(question_id)
            if not qid or not isinstance(values, dict):
                continue
            scores: dict[str, float] = {}
            for answer_id, score in values.items():
                aid = _normalize_answer_id(answer_id)
                if not aid:
                    continue
                try:
                    scores[aid] = float(score)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid corrected score for {qid}/{aid}: {score!r}"
                    ) from exc
            if scores:
                corrected[qid] = scores

        overlaps = []
        for qid, scores in corrected.items():
            for answer_id in set(scores) & set(excluded.get(qid, ())):
                overlaps.append(f"{qid}/{answer_id}")
        if overlaps:
            raise ValueError(
                "Samples cannot be both excluded and corrected: "
                + ", ".join(sorted(overlaps))
            )
        for filename, expected_hash in (
            payload.get("dataset_sha256") or {}
        ).items():
            source = Path(prepared_root).expanduser().resolve() / filename
            if not source.is_file():
                raise FileNotFoundError(
                    f"Policy dataset file not found: {source}"
                )
            if _sha256(source) != expected_hash:
                raise ValueError(
                    "Sample-quality policy was compiled for a different "
                    f"dataset file: {source}. Re-audit and recompile it."
                )

        return cls(
            mode="active",
            policy_id=policy_id,
            path=path,
            sha256=_sha256(path),
            excluded=excluded,
            corrected_scores=corrected,
        )

    def is_excluded(self, question_id: Any, answer_id: Any) -> bool:
        qid = _normalize_question_id(question_id)
        aid = _normalize_answer_id(answer_id)
        return aid in self.excluded.get(qid, ())

    def filter_ids(
        self,
        question_id: Any,
        answer_ids: Iterable[Any],
    ) -> set[str]:
        qid = _normalize_question_id(question_id)
        excluded = self.excluded.get(qid, ())
        return {
            aid
            for value in answer_ids
            if (aid := _normalize_answer_id(value)) and aid not in excluded
        }

    def effective_teacher_score(
        self,
        question_id: Any,
        answer_id: Any,
        original_score: Any,
        *,
        include_excluded: bool = False,
    ) -> float | None:
        qid = _normalize_question_id(question_id)
        aid = _normalize_answer_id(answer_id)
        if not include_excluded and aid in self.excluded.get(qid, ()):
            return None
        if aid in self.corrected_scores.get(qid, {}):
            return self.corrected_scores[qid][aid]
        if original_score is None:
            return None
        return float(original_score)

    def descriptor(self) -> dict[str, Any]:
        excluded_by_question = {
            qid: len(values) for qid, values in sorted(self.excluded.items())
        }
        corrected_by_question = {
            qid: len(values)
            for qid, values in sorted(self.corrected_scores.items())
        }
        return {
            "mode": self.mode,
            "policy_id": self.policy_id,
            "sha256": self.sha256,
            "excluded_count": sum(excluded_by_question.values()),
            "corrected_count": sum(corrected_by_question.values()),
            "excluded_by_question": excluded_by_question,
            "corrected_by_question": corrected_by_question,
        }


def load_policy_for_data_path(
    data_path: str | Path,
    *,
    explicit_path: str | Path | None = None,
    force_raw: bool | None = None,
) -> SampleQualityPolicy:
    """Load a policy from a file located directly under the prepared root."""
    prepared_root = Path(data_path).expanduser().resolve().parent
    return SampleQualityPolicy.load(
        prepared_root,
        explicit_path=explicit_path,
        force_raw=force_raw,
    )
