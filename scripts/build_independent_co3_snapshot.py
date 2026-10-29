"""Create a separate, exploratory CO_3 snapshot without target teacher scores.

The original question and item structure remain fixed. This is a derived
reference, not a certified official solution. A new source digest is mandatory.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_label_free_question_snapshot import verify

SOURCE_HASH = "ff720e1ed920888212c9cb4cd3a42be86ba0f143860e0c03a03fc0f0d4781f27"
DERIVED_REFERENCE = (
    "题目要求判断8位运算结果的OF标志。若2y和末步加法均采用8位补码运算，"
    "54H+54H=A8H，该步OF=1；A8H取负得58H；36H+58H=8EH，末步OF=1。"
    "若将x−2y视作不截断的精确整数表达式，则54−2×84=−114，处于8位有符号数范围内。"
    "精确整数结果可表示与特定8位逐步运算的OF标志是不同判断；应记录作答所采用的解释，"
    "不应把两者混为同一个溢出结论。"
)


def build(source, destination):
    source = Path(source).resolve(strict=True)
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError("Destination already exists; refusing overwrite")
    if not destination.parent.is_dir():
        raise ValueError("Destination parent must exist")
    report = verify(source)
    if report["snapshot_sha256_normalized"] != SOURCE_HASH:
        raise ValueError("Unexpected source snapshot; refusing to derive reference")
    data = json.loads((source / "snapshot.json").read_text(encoding="utf-8-sig"))
    if data.get("reference_with_teacher_word_questions") != ["CO_3"]:
        raise ValueError("Unexpected flagged question set")
    matches = [q for q in data["questions"] if q["question_id"] == "CO_3"]
    if len(matches) != 1:
        raise ValueError("CO_3 must exist exactly once")
    q = matches[0]
    if not all(x in q["question_text"] for x in ("36H", "54H", "D5H")):
        raise ValueError("CO_3 question operands differ")
    items = [item for item in q["rubric"] if item["id"] == "step_1"]
    if len(items) != 1 or items[0]["points"] != 5.0:
        raise ValueError("CO_3 item structure differs")
    items[0]["reference_text"] = DERIVED_REFERENCE
    data["reference_with_teacher_word_questions"] = []
    data["derived_reference_provenance"] = {
        "CO_3_step_1": {"basis": "independent arithmetic from question operands",
                         "source_snapshot_sha256": SOURCE_HASH,
                         "teacher_scores_used": False,
                         "official_reference_verified": False,
                         "interpretation": "8-bit operation OF versus exact mathematical range"}}
    data["source_provenance_audited"] = False
    data["approved_for_formal_experiment"] = False
    with tempfile.TemporaryDirectory(prefix=".co3-derived-", dir=destination.parent) as tmp:
        temporary = Path(tmp) / "snapshot"
        temporary.mkdir()
        shutil.copytree(source / "images", temporary / "images")
        (temporary / "snapshot.json").write_text(
            json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
        check = verify(temporary)
        if check["flagged_questions"] or check["questions"] != report["questions"] or check["images"] != report["images"]:
            raise ValueError("Derived snapshot verification failed")
        if destination.exists():
            raise FileExistsError("Destination appeared during construction")
        temporary.rename(destination)
    return verify(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build(args.source, args.destination), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Cannot derive snapshot: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
