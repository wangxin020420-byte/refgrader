"""Compile reviewed teacher-label decisions into an active read-time policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sample_quality import POLICY_SCHEMA_VERSION, default_policy_path


ALLOWED_DECISIONS = {
    "confirmed_noise",
    "corrected",
    "retained_hard_case",
    "ambiguous",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_decisions(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [
                dict(row)
                for row in csv.DictReader(handle)
                if (row.get("human_decision") or row.get("decision"))
            ]
    records = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"Decision line {line_number} must be a JSON object."
                )
            records.append(row)
    return records


def load_valid_answers(path: Path) -> dict[str, str]:
    answers = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            answer_id = str(row.get("answer_id") or "").strip()
            question_id = str(row.get("question_id") or "").strip().upper()
            if answer_id and question_id:
                answers[answer_id] = question_id
    return answers


def load_max_scores(path: Path) -> dict[str, float]:
    payload = read_json(path)
    return {
        str(row["question_id"]).upper(): float(row["total_score"])
        for row in payload
        if isinstance(row, dict)
        and row.get("question_id")
        and row.get("total_score") is not None
    }


def normalized_decision(
    row: dict[str, Any],
    *,
    line_number: int,
    valid_answers: dict[str, str],
    max_scores: dict[str, float],
) -> dict[str, Any]:
    answer_id = str(row.get("answer_id") or "").strip()
    question_id = str(row.get("question_id") or "").strip().upper()
    decision = str(
        row.get("decision") or row.get("human_decision") or ""
    ).strip().lower()
    if not answer_id or answer_id not in valid_answers:
        raise ValueError(
            f"Unknown answer_id on decision line {line_number}: {answer_id!r}"
        )
    actual_question = valid_answers[answer_id]
    if question_id != actual_question:
        raise ValueError(
            f"Question mismatch on decision line {line_number}: "
            f"{answer_id} belongs to {actual_question}, not {question_id}"
        )
    if decision not in ALLOWED_DECISIONS:
        raise ValueError(
            f"Invalid decision on line {line_number}: {decision!r}. "
            f"Expected one of {sorted(ALLOWED_DECISIONS)}"
        )

    corrected_score = None
    if decision == "corrected":
        try:
            corrected_score = float(row.get("corrected_score"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Corrected decision on line {line_number} needs a score."
            ) from exc
        if not math.isfinite(corrected_score):
            raise ValueError(
                f"Non-finite corrected score on line {line_number}."
            )
        maximum = max_scores.get(question_id)
        if maximum is None or not 0 <= corrected_score <= maximum:
            raise ValueError(
                f"Corrected score for {question_id}/{answer_id} must be "
                f"between 0 and {maximum}."
            )

    return {
        "question_id": question_id,
        "answer_id": answer_id,
        "decision": decision,
        "corrected_score": corrected_score,
        "reason_code": str(row.get("reason_code") or "").strip(),
        "review_note": str(row.get("review_note") or "").strip(),
        "reviewer": str(row.get("reviewer") or "").strip(),
        "reviewed_at": str(row.get("reviewed_at") or "").strip(),
    }


def build_policy(
    decisions: list[dict[str, Any]],
    *,
    policy_id: str,
    source: str | Path,
) -> dict[str, Any]:
    excluded: dict[str, dict[str, Any]] = {}
    corrected: dict[str, dict[str, float]] = {}
    retained: dict[str, list[str]] = {}
    ambiguous: dict[str, list[str]] = {}
    seen = set()
    for row in decisions:
        identity = (row["question_id"], row["answer_id"])
        if identity in seen:
            raise ValueError(
                "Duplicate reviewed decision: " + "/".join(identity)
            )
        seen.add(identity)
        question_id, answer_id = identity
        decision = row["decision"]
        if decision == "confirmed_noise":
            excluded.setdefault(question_id, {})[answer_id] = {
                "reason_code": row["reason_code"],
                "review_note": row["review_note"],
                "reviewer": row["reviewer"],
                "reviewed_at": row["reviewed_at"],
            }
        elif decision == "corrected":
            corrected.setdefault(question_id, {})[answer_id] = row[
                "corrected_score"
            ]
        elif decision == "retained_hard_case":
            retained.setdefault(question_id, []).append(answer_id)
        elif decision == "ambiguous":
            ambiguous.setdefault(question_id, []).append(answer_id)

    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "policy_id": policy_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_decisions": str(source),
        "selection_rule": (
            "Only confirmed_noise samples are excluded. Corrected labels "
            "overlay raw teacher scores. Ambiguous and retained hard cases "
            "remain in the dataset."
        ),
        "excluded": excluded,
        "corrected_scores": corrected,
        "retained_hard_cases": {
            key: sorted(value) for key, value in sorted(retained.items())
        },
        "ambiguous_pending_review": {
            key: sorted(value) for key, value in sorted(ambiguous.items())
        },
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile human decisions into a sample-quality policy."
    )
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/csbench"))
    parser.add_argument("--policy-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Write to the tracked active policy path.",
    )
    args = parser.parse_args()

    prepared = args.prepared_dir.expanduser().resolve()
    decisions_path = args.decisions.expanduser().resolve()
    raw_decisions = load_decisions(decisions_path)
    valid_answers = load_valid_answers(prepared / "answer_metadata.jsonl")
    max_scores = load_max_scores(prepared / "exam_database.json")
    decisions = [
        normalized_decision(
            row,
            line_number=index,
            valid_answers=valid_answers,
            max_scores=max_scores,
        )
        for index, row in enumerate(raw_decisions, 1)
    ]
    policy_id = (
        args.policy_id
        or f"teacher_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if args.activate and args.output:
        raise ValueError("Use either --activate or --output, not both.")
    output = (
        default_policy_path(prepared)
        if args.activate
        else (
            args.output.expanduser().resolve()
            if args.output
            else decisions_path.with_name(f"{policy_id}.policy.json")
        )
    )
    policy = build_policy(
        decisions,
        policy_id=policy_id,
        source=(
            "${PREPARED_CSBENCH_ROOT}/"
            + decisions_path.relative_to(prepared).as_posix()
            if decisions_path.is_relative_to(prepared)
            else decisions_path
        ),
    )
    policy["dataset_sha256"] = {
        "teacher_scores.json": sha256_file(prepared / "teacher_scores.json"),
        "answer_metadata.jsonl": sha256_file(
            prepared / "answer_metadata.jsonl"
        ),
    }
    atomic_write_json(output, policy)
    print(f"Policy written: {output}")
    print(
        "Excluded: "
        f"{sum(len(value) for value in policy['excluded'].values())}; "
        "corrected: "
        f"{sum(len(value) for value in policy['corrected_scores'].values())}; "
        "ambiguous retained: "
        f"{sum(len(value) for value in policy['ambiguous_pending_review'].values())}"
    )
    if not args.activate:
        print("Policy is not active. Review it, then rerun with --activate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
