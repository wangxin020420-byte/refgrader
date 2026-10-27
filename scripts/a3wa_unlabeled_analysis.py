"""Offline A3WA information-loss analysis; this is not a grading workflow.

The input is a JSON object with exactly memberships, candidates and
loss_parameters. Memberships must be collected independently of teacher
scores; this tool cannot verify the provenance of numeric inputs.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from label_free_a3wa import select_m_by_unlabeled_information_loss


def analyze(payload):
    expected = {"memberships", "candidates", "loss_parameters"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("Input must contain exactly memberships, candidates and loss_parameters")
    if not isinstance(payload["memberships"], list) or not isinstance(payload["candidates"], list):
        raise ValueError("memberships and candidates must be JSON arrays")
    result = select_m_by_unlabeled_information_loss(
        payload["memberships"],
        candidates=payload["candidates"],
        loss_parameters=payload["loss_parameters"],
    )
    return {"schema": "a3wa_unlabeled_analysis_v1", "provenance_verified": False, "result": result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to a JSON input; use - for stdin")
    args = parser.parse_args(argv)
    try:
        raw = sys.stdin.buffer.read() if args.input == "-" else Path(args.input).read_bytes()
        payload = json.loads(raw.decode("utf-8-sig"))
        result = analyze(payload)
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 2
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
