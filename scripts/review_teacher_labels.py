"""Local browser UI for reviewing teacher-label audit candidates.

The server is intentionally dependency-free and binds to localhost by
default. Raw dataset files are read-only. Human decisions are written to a
tracked JSONL file and become effective only after an explicit policy
activation.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import sys
import threading
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_FILE = PROJECT_ROOT / "assets" / "teacher_label_review" / "index.html"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compile_sample_quality_policy import (  # noqa: E402
    ALLOWED_DECISIONS,
    atomic_write_json,
    build_policy,
    load_decisions,
    load_max_scores,
    load_valid_answers,
    normalized_decision,
    sha256_file,
)
from sample_quality import default_policy_path  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.is_file():
        return records
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"JSONL line {line_number} must be an object: {path}"
                )
            records.append(row)
    return records


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("answer_id")): row
        for row in read_jsonl(path)
        if row.get("answer_id")
    }


def load_screening(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (
            str(row.get("question_id") or "").strip().upper(),
            str(row.get("answer_id") or "").strip(),
        ): dict(row)
        for row in rows
        if row.get("question_id") and row.get("answer_id")
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_questions(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"exam_database.json must contain a list: {path}")
    return {
        str(row.get("question_id") or "").strip().upper(): row
        for row in payload
        if isinstance(row, dict) and row.get("question_id")
    }


def rubric_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("items", "rubric", "grading_rubric"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def relative_display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def optional_number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def priority_rank(value: str) -> int:
    normalized = str(value or "").strip().upper()
    if normalized.startswith("P") and normalized[1:].isdigit():
        return int(normalized[1:])
    return 99


class TeacherLabelReviewStore:
    """Load candidates and persist one review decision per answer."""

    def __init__(
        self,
        *,
        prepared_dir: Path,
        report_dir: Path,
        decisions_path: Path,
        screening_path: Path | None = None,
        screening_only: bool = False,
    ) -> None:
        self.prepared_dir = prepared_dir.expanduser().resolve()
        self.report_dir = report_dir.expanduser().resolve()
        self.decisions_path = decisions_path.expanduser().resolve()
        self.screening_path = (
            screening_path.expanduser().resolve()
            if screening_path
            else None
        )
        self.screening_only = screening_only
        self.run_id = self.report_dir.name
        self.lock = threading.RLock()

        if not self.report_dir.is_dir():
            raise FileNotFoundError(f"Candidate report not found: {report_dir}")
        if not (self.prepared_dir / "answer_metadata.jsonl").is_file():
            raise FileNotFoundError(
                f"answer_metadata.jsonl not found under {self.prepared_dir}"
            )

        self.metadata = load_metadata(
            self.prepared_dir / "answer_metadata.jsonl"
        )
        self.questions = load_questions(
            self.prepared_dir / "exam_database.json"
        )
        self.max_scores = load_max_scores(
            self.prepared_dir / "exam_database.json"
        )
        self.valid_answers = load_valid_answers(
            self.prepared_dir / "answer_metadata.jsonl"
        )
        self.screening = load_screening(self.screening_path) if (
            self.screening_path
        ) else {}
        self.question_contexts = self._load_question_contexts()
        self.candidates = self._load_candidates()
        self.candidate_map = {
            (row["question_id"], row["answer_id"]): row
            for row in self.candidates
        }
        self.decisions = self._load_normalized_decisions()

    def _resolve_dataset_path(self, raw_path: Any) -> Path | None:
        value = str(raw_path or "").strip()
        if not value:
            return None
        value = value.replace(
            "${REFGRADER_ROOT}",
            str(PROJECT_ROOT),
        ).replace(
            "${PREPARED_CSBENCH_ROOT}",
            str(self.prepared_dir),
        )
        path = Path(value)
        if path.is_absolute():
            return path.resolve()
        prepared_candidate = (self.prepared_dir / path).resolve()
        project_candidate = (PROJECT_ROOT / path).resolve()
        if prepared_candidate.exists():
            return prepared_candidate
        return project_candidate

    def _active_rubric_paths(self) -> dict[str, Path]:
        active_path = (
            self.prepared_dir / "rubrics" / "active_rubric_set.json"
        )
        if not active_path.is_file():
            return {}
        payload = read_json(active_path)
        questions = payload.get("questions") if isinstance(payload, dict) else {}
        if not isinstance(questions, dict):
            return {}
        paths = {}
        for question_id, row in questions.items():
            if not isinstance(row, dict):
                continue
            path = self._resolve_dataset_path(row.get("optimized_rubric"))
            if path is not None:
                paths[str(question_id).upper()] = path
        return paths

    def _load_question_contexts(self) -> dict[str, dict[str, Any]]:
        active_paths = self._active_rubric_paths()
        contexts = {}
        for question_id, question in self.questions.items():
            active_rubric_path = active_paths.get(question_id)
            rubric_source = "active_rubric_set"
            if active_rubric_path is None:
                active_rubric_path = self._resolve_dataset_path(
                    question.get("optimized_rubric_path")
                )
                rubric_source = "exam_database_fallback"

            items = []
            rubric_error = ""
            if active_rubric_path and active_rubric_path.is_file():
                try:
                    for item in rubric_items(read_json(active_rubric_path)):
                        items.append(
                            {
                                "id": str(item.get("id") or ""),
                                "item": str(
                                    item.get("item")
                                    or item.get("description")
                                    or item.get("source_text")
                                    or ""
                                ),
                                "points": optional_number(
                                    item.get("points")
                                    if item.get("points") is not None
                                    else item.get("score")
                                ),
                                "standard_answer_text": str(
                                    item.get("standard_answer_text")
                                    or item.get("full_credit_anchor")
                                    or ""
                                ),
                                "role": str(item.get("role") or ""),
                                "score_layer": str(
                                    item.get("score_layer") or ""
                                ),
                                "task_semantics": str(
                                    item.get("task_semantics") or ""
                                ),
                                "scoring_policy": str(
                                    item.get("scoring_policy") or ""
                                ),
                                "parent_id": str(
                                    item.get("parent_id") or ""
                                ),
                                "dependency_mode": str(
                                    item.get("dependency_mode") or ""
                                ),
                            }
                        )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    rubric_error = str(exc)
            else:
                rubric_error = "当前生效的优化评分准则文件不存在。"

            question_image = self._resolve_dataset_path(
                question.get("question_image")
            )
            allowed_reference_root = (
                self.prepared_dir / "reference_images"
            ).resolve()
            question_image_available = bool(
                question_image
                and question_image.is_file()
                and question_image.is_relative_to(allowed_reference_root)
            )
            contexts[question_id] = {
                "question_id": question_id,
                "question_text": str(question.get("question_text") or ""),
                "reference_answer": str(question.get("ref_text") or ""),
                "official_rubric": str(
                    question.get("official_rubric") or ""
                ),
                "total_score": optional_number(
                    question.get("total_score")
                ),
                "question_image_available": question_image_available,
                "active_rubric": {
                    "source": rubric_source,
                    "path": (
                        relative_display_path(active_rubric_path)
                        if active_rubric_path
                        else ""
                    ),
                    "items": items,
                    "error": rubric_error,
                },
            }
        return contexts

    def _load_candidates(self) -> list[dict[str, Any]]:
        files = sorted(
            self.report_dir.rglob("teacher_label_candidates.csv")
        )
        if not files:
            raise FileNotFoundError(
                f"No teacher_label_candidates.csv under {self.report_dir}"
            )

        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for path in files:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            source_name = path.parent.name
            for raw in rows:
                question_id = str(
                    raw.get("question_id") or ""
                ).strip().upper()
                answer_id = str(raw.get("answer_id") or "").strip()
                if not question_id or not answer_id:
                    continue
                key = (question_id, answer_id)
                current = merged.setdefault(
                    key,
                    {
                        "question_id": question_id,
                        "answer_id": answer_id,
                        "candidate_sources": [],
                        "candidate_types": [],
                        "candidate_reasons": [],
                        "confounds": [],
                    },
                )
                if source_name not in current["candidate_sources"]:
                    current["candidate_sources"].append(source_name)
                for source_field, target_field in (
                    ("candidate_type", "candidate_types"),
                    ("candidate_reasons", "candidate_reasons"),
                    ("confounds", "confounds"),
                ):
                    for value in str(raw.get(source_field) or "").split("|"):
                        value = value.strip()
                        if value and value not in current[target_field]:
                            current[target_field].append(value)

                incoming_priority = str(
                    raw.get("review_priority") or ""
                ).strip().upper()
                existing_priority = str(
                    current.get("review_priority") or ""
                ).strip().upper()
                if priority_rank(incoming_priority) < priority_rank(
                    existing_priority
                ):
                    current["review_priority"] = incoming_priority

                for name in (
                    "teacher_score",
                    "reference_score",
                    "teacher_minus_reference",
                    "absolute_difference",
                    "candidate_threshold",
                    "question_median_residual",
                    "max_score",
                    "model_avg_score",
                    "selected_baseline_score",
                    "three_way_core_score",
                    "final_calibrated_score",
                    "std_dev",
                    "U_E",
                    "U_S",
                    "U_R",
                ):
                    number = optional_number(raw.get(name))
                    if number is not None:
                        current[name] = number
                for name in (
                    "reference_score_key",
                    "route",
                    "extraction_quality",
                    "raw_text",
                    "student_image",
                ):
                    value = str(raw.get(name) or "").strip()
                    if value:
                        current[name] = value

        candidates = []
        for key, row in merged.items():
            if self.screening_only and key not in self.screening:
                continue
            metadata = self.metadata.get(row["answer_id"]) or {}
            if not row.get("raw_text"):
                row["raw_text"] = str(metadata.get("raw_text") or "")
            if not row.get("student_image"):
                row["student_image"] = str(
                    metadata.get("student_image")
                    or metadata.get("source_image")
                    or ""
                )
            row["max_score"] = (
                optional_number(row.get("max_score"))
                or self.max_scores.get(row["question_id"])
            )
            screen = self.screening.get(key) or {}
            row["initial_class"] = str(
                screen.get("initial_class") or ""
            )
            row["screening_confidence"] = str(
                screen.get("confidence") or ""
            )
            row["initial_finding"] = str(
                screen.get("initial_finding") or ""
            )
            row["recommended_action"] = str(
                screen.get("recommended_action") or ""
            )
            row["image_available"] = bool(self.resolve_image_path(row))
            candidates.append(row)

        return sorted(
            candidates,
            key=lambda row: (
                priority_rank(str(row.get("review_priority") or "")),
                -float(row.get("absolute_difference") or 0.0),
                row["question_id"],
                row["answer_id"],
            ),
        )

    def _load_normalized_decisions(
        self,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        if not self.decisions_path.is_file():
            return {}
        normalized = {}
        for index, row in enumerate(
            load_decisions(self.decisions_path),
            1,
        ):
            decision = normalized_decision(
                row,
                line_number=index,
                valid_answers=self.valid_answers,
                max_scores=self.max_scores,
            )
            key = (decision["question_id"], decision["answer_id"])
            if key in normalized:
                raise ValueError(
                    "Duplicate decision in review file: " + "/".join(key)
                )
            normalized[key] = decision
        return normalized

    def resolve_image_path(
        self,
        candidate: dict[str, Any],
    ) -> Path | None:
        raw_path = str(candidate.get("student_image") or "").strip()
        if not raw_path:
            return None
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        allowed_root = (self.prepared_dir / "student_images").resolve()
        if not path.is_relative_to(allowed_root) or not path.is_file():
            return None
        return path

    def resolve_question_image_path(self, question_id: str) -> Path | None:
        question = self.questions.get(str(question_id).strip().upper())
        if not question:
            return None
        path = self._resolve_dataset_path(question.get("question_image"))
        allowed_root = (self.prepared_dir / "reference_images").resolve()
        if (
            path is None
            or not path.is_relative_to(allowed_root)
            or not path.is_file()
        ):
            return None
        return path

    def decision_rows(self) -> list[dict[str, Any]]:
        order = {
            (row["question_id"], row["answer_id"]): index
            for index, row in enumerate(self.candidates)
        }
        return [
            row
            for _, row in sorted(
                self.decisions.items(),
                key=lambda item: (
                    order.get(item[0], 10**9),
                    item[0],
                ),
            )
        ]

    def _atomic_write_decisions(self) -> None:
        self.decisions_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.decisions_path.with_name(
            f".{self.decisions_path.name}.tmp-{os.getpid()}"
        )
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in self.decision_rows():
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
        os.replace(temporary, self.decisions_path)

    def save_decision(self, payload: dict[str, Any]) -> dict[str, Any]:
        question_id = str(
            payload.get("question_id") or ""
        ).strip().upper()
        answer_id = str(payload.get("answer_id") or "").strip()
        key = (question_id, answer_id)
        if key not in self.candidate_map:
            raise ValueError(
                f"Answer is not in this candidate report: {question_id}/{answer_id}"
            )
        row = {
            "question_id": question_id,
            "answer_id": answer_id,
            "decision": str(payload.get("decision") or "").strip(),
            "corrected_score": payload.get("corrected_score"),
            "reason_code": str(payload.get("reason_code") or "").strip(),
            "review_note": str(payload.get("review_note") or "").strip(),
            "reviewer": str(payload.get("reviewer") or "").strip(),
            "reviewed_at": datetime.now().isoformat(timespec="seconds"),
        }
        decision = normalized_decision(
            row,
            line_number=1,
            valid_answers=self.valid_answers,
            max_scores=self.max_scores,
        )
        with self.lock:
            # Reload before writing so a manually edited decision file is not
            # silently overwritten while the UI is open.
            self.decisions = self._load_normalized_decisions()
            self.decisions[key] = decision
            self._atomic_write_decisions()
        return decision

    def delete_decision(
        self,
        question_id: str,
        answer_id: str,
    ) -> bool:
        key = (question_id.strip().upper(), answer_id.strip())
        with self.lock:
            self.decisions = self._load_normalized_decisions()
            removed = self.decisions.pop(key, None) is not None
            if removed:
                self._atomic_write_decisions()
        return removed

    def activate_policy(self, policy_id: str | None = None) -> dict[str, Any]:
        with self.lock:
            self.decisions = self._load_normalized_decisions()
            if not self.decisions:
                raise ValueError("No reviewed decisions are available.")
            resolved_policy_id = (
                str(policy_id or "").strip()
                or f"{self.run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            source = (
                "${PREPARED_CSBENCH_ROOT}/"
                + self.decisions_path.relative_to(
                    self.prepared_dir
                ).as_posix()
                if self.decisions_path.is_relative_to(self.prepared_dir)
                else str(self.decisions_path)
            )
            policy = build_policy(
                self.decision_rows(),
                policy_id=resolved_policy_id,
                source=source,
            )
            policy["dataset_sha256"] = {
                "teacher_scores.json": sha256_file(
                    self.prepared_dir / "teacher_scores.json"
                ),
                "answer_metadata.jsonl": sha256_file(
                    self.prepared_dir / "answer_metadata.jsonl"
                ),
            }
            output = default_policy_path(self.prepared_dir)
            atomic_write_json(output, policy)
        return {
            "policy_id": resolved_policy_id,
            "path": str(output),
            "excluded": sum(
                len(value) for value in policy["excluded"].values()
            ),
            "corrected": sum(
                len(value)
                for value in policy["corrected_scores"].values()
            ),
            "ambiguous": sum(
                len(value)
                for value in policy[
                    "ambiguous_pending_review"
                ].values()
            ),
        }

    def state(self) -> dict[str, Any]:
        with self.lock:
            self.decisions = self._load_normalized_decisions()
            candidates = []
            for row in self.candidates:
                key = (row["question_id"], row["answer_id"])
                candidate = dict(row)
                candidate["decision"] = self.decisions.get(key)
                candidate["image_url"] = (
                    "/api/image?"
                    f"question_id={row['question_id']}&"
                    f"answer_id={row['answer_id']}"
                    if row.get("image_available")
                    else None
                )
                candidates.append(candidate)
            reviewed = len(
                {
                    key
                    for key in self.decisions
                    if key in self.candidate_map
                }
            )
            policy_path = default_policy_path(self.prepared_dir)
            return {
                "report_run": self.run_id,
                "report_dir": str(self.report_dir),
                "decisions_path": str(self.decisions_path),
                "policy_path": str(policy_path),
                "policy_active": policy_path.is_file(),
                "screening_only": self.screening_only,
                "allowed_decisions": sorted(ALLOWED_DECISIONS),
                "counts": {
                    "total": len(self.candidates),
                    "reviewed": reviewed,
                    "pending": len(self.candidates) - reviewed,
                },
                "question_contexts": self.question_contexts,
                "candidates": candidates,
            }


class ReviewRequestHandler(BaseHTTPRequestHandler):
    server_version = "RefGraderReview/1.0"

    @property
    def store(self) -> TeacherLabelReviewStore:
        return self.server.store  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        print(
            f"[review-ui] {self.address_string()} "
            f"{format % args}",
            flush=True,
        )

    def send_json(
        self,
        payload: Any,
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(
        self,
        message: str,
        *,
        status: HTTPStatus = HTTPStatus.BAD_REQUEST,
    ) -> None:
        self.send_json({"ok": False, "error": message}, status=status)

    def read_json_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length") or "0"
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length.") from exc
        if length <= 0 or length > 1024 * 1024:
            raise ValueError("Request body must be between 1 byte and 1 MiB.")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON request body must be an object.")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            if not UI_FILE.is_file():
                self.send_error_json(
                    f"UI file not found: {UI_FILE}",
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            body = UI_FILE.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/state":
            try:
                self.send_json({"ok": True, **self.store.state()})
            except Exception as exc:
                self.send_error_json(
                    str(exc),
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return
        if parsed.path == "/api/image":
            query = parse_qs(parsed.query)
            key = (
                str((query.get("question_id") or [""])[0]).upper(),
                str((query.get("answer_id") or [""])[0]),
            )
            candidate = self.store.candidate_map.get(key)
            image = (
                self.store.resolve_image_path(candidate)
                if candidate
                else None
            )
            if image is None:
                self.send_error_json(
                    "Image not found.",
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            body = image.read_bytes()
            media_type = mimetypes.guess_type(image.name)[0] or (
                "application/octet-stream"
            )
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", media_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "private, max-age=300")
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/question-image":
            query = parse_qs(parsed.query)
            question_id = str(
                (query.get("question_id") or [""])[0]
            ).upper()
            image = self.store.resolve_question_image_path(question_id)
            if image is None:
                self.send_error_json(
                    "Question image not found.",
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            body = image.read_bytes()
            media_type = mimetypes.guess_type(image.name)[0] or (
                "application/octet-stream"
            )
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", media_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "private, max-age=300")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error_json("Not found.", status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self.read_json_body()
            if parsed.path == "/api/decision":
                decision = self.store.save_decision(payload)
                self.send_json(
                    {
                        "ok": True,
                        "decision": decision,
                        "counts": self.store.state()["counts"],
                    }
                )
                return
            if parsed.path == "/api/policy/activate":
                result = self.store.activate_policy(payload.get("policy_id"))
                self.send_json({"ok": True, **result})
                return
            self.send_error_json("Not found.", status=HTTPStatus.NOT_FOUND)
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_error_json(str(exc))
        except Exception as exc:
            self.send_error_json(
                str(exc),
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/decision":
            self.send_error_json("Not found.", status=HTTPStatus.NOT_FOUND)
            return
        query = parse_qs(parsed.query)
        question_id = str((query.get("question_id") or [""])[0])
        answer_id = str((query.get("answer_id") or [""])[0])
        removed = self.store.delete_decision(question_id, answer_id)
        self.send_json(
            {
                "ok": True,
                "removed": removed,
                "counts": self.store.state()["counts"],
            }
        )


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        store: TeacherLabelReviewStore,
    ) -> None:
        super().__init__(address, ReviewRequestHandler)
        self.store = store


def find_report_dir(
    prepared_dir: Path,
    report_run: str | None,
    report_dir: Path | None,
) -> Path:
    if report_dir:
        return report_dir.expanduser().resolve()
    reports_root = prepared_dir / "quality_control" / "reports"
    if report_run:
        return (reports_root / report_run).resolve()
    candidates = sorted(
        (
            path
            for path in reports_root.iterdir()
            if path.is_dir()
            and any(path.rglob("teacher_label_candidates.csv"))
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No candidate report directories found under {reports_root}"
        )
    return candidates[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the local teacher-label review UI."
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("data/csbench"),
    )
    parser.add_argument(
        "--report-run",
        help="Report directory name under quality_control/reports.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="Explicit candidate report directory.",
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        help="Explicit decisions JSONL path.",
    )
    parser.add_argument(
        "--screening",
        type=Path,
        help="Optional preliminary-screening CSV.",
    )
    parser.add_argument(
        "--all-candidates",
        action="store_true",
        help=(
            "Include every generated candidate even when a preliminary "
            "screening CSV exists."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the default browser automatically.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    prepared_dir = args.prepared_dir.expanduser().resolve()
    report_dir = find_report_dir(
        prepared_dir,
        args.report_run,
        args.report_dir,
    )
    run_id = report_dir.name
    reviews_dir = prepared_dir / "quality_control" / "reviews"
    decisions_path = (
        args.decisions.expanduser().resolve()
        if args.decisions
        else reviews_dir / f"{run_id}_decisions.jsonl"
    )
    default_screening = reviews_dir / f"{run_id}_initial_screening.csv"
    screening_path = (
        args.screening.expanduser().resolve()
        if args.screening
        else (default_screening if default_screening.is_file() else None)
    )
    store = TeacherLabelReviewStore(
        prepared_dir=prepared_dir,
        report_dir=report_dir,
        decisions_path=decisions_path,
        screening_path=screening_path,
        screening_only=bool(screening_path and not args.all_candidates),
    )
    server = ReviewServer((args.host, args.port), store)
    host, port = server.server_address[:2]
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{browser_host}:{port}"
    state = store.state()
    print("=" * 68)
    print("RefGrader teacher-label review UI")
    print(f"Candidates: {state['counts']['total']}")
    print(f"Reviewed:   {state['counts']['reviewed']}")
    print(f"Pending:    {state['counts']['pending']}")
    print(f"Decisions:  {state['decisions_path']}")
    print(f"Policy:     {state['policy_path']}")
    print(f"Open:       {url}")
    print("Press Ctrl+C to stop the UI. Saved decisions remain on disk.")
    print("=" * 68, flush=True)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReview UI stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
