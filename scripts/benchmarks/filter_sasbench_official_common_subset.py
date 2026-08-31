"""Filter official SAS-Bench prediction files to RefGrader's evaluated IDs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path


PREDICTION_PATTERN = re.compile(r"^\d+_.+_prediction\.jsonl$", re.IGNORECASE)


def common_ids(compare_path: Path) -> set[str]:
    with compare_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "student_id" not in (reader.fieldnames or []):
            raise ValueError("compare.csv is missing student_id")
        values = [row["student_id"] for row in reader]
    if len(values) != len(set(values)):
        raise ValueError("compare.csv contains duplicate student_id values")
    return set(values)


def filter_predictions(
    prediction_dir: Path,
    output_dir: Path,
    ids: set[str],
) -> tuple[int, int]:
    found = set()
    task_count = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(prediction_dir.glob("*_prediction.jsonl")):
        if not PREDICTION_PATTERN.fullmatch(source.name):
            continue
        task_count += 1
        with source.open("r", encoding="utf-8-sig") as input_handle, (
            output_dir / source.name
        ).open("w", encoding="utf-8") as output_handle:
            for line in input_handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                answer_id = str(record.get("id", ""))
                if answer_id in ids:
                    if answer_id in found:
                        raise ValueError(f"Duplicate answer in predictions: {answer_id}")
                    found.add(answer_id)
                    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    missing = ids - found
    if missing:
        raise ValueError(f"Official predictions missing common IDs: {sorted(missing)[:10]}")
    return task_count, len(found)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--compare-csv", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--protocol-base-dir", type=Path, required=True)
    parser.add_argument("--save-type-name", required=True)
    parser.add_argument("--expected-records", type=int)
    args = parser.parse_args()

    ids = common_ids(args.compare_csv.resolve())
    if args.expected_records is not None and len(ids) != args.expected_records:
        raise ValueError(
            f"Common ID count mismatch: expected={args.expected_records}, actual={len(ids)}"
        )
    protocol_base = args.protocol_base_dir.resolve()
    output_dir = Path(f"{protocol_base}_{args.save_type_name}_Scored")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Common-subset output is not empty: {output_dir}. "
            "Use a new save-type-name."
        )
    task_count, record_count = filter_predictions(
        args.prediction_dir.resolve(), output_dir, ids
    )
    protocol_base.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        args.source_dir.resolve() / "error_type.jsonl",
        protocol_base / "error_type.jsonl",
    )
    manifest = {
        "source_prediction_dir": str(args.prediction_dir.resolve()),
        "record_count": record_count,
        "task_count": task_count,
        "filter": "student_id in RefGrader compare.csv",
    }
    (output_dir / "common_subset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Common subset: {record_count} records across {task_count} tasks")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
