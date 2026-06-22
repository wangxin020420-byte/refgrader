"""Check the focused Q2 interrupt-sequence relations in an extraction cache."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def compact(value: object) -> str:
    return re.sub(r"\s+", "", str(value)).replace("；", ";").replace("，", ",")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cache",
        nargs="?",
        default="ocr_cache/facts/Q2/E01914115_Q2.json",
    )
    args = parser.parse_args()
    data = json.loads(Path(args.cache).read_text(encoding="utf-8"))
    joined = compact(
        " ".join(
            [
                str(data.get("observed_execution_path", "")),
                *[str(value) for value in data.get("diagram_facts", {}).values()],
            ]
        )
    )
    required = {
        "first_A": "→A" in joined or joined.startswith("A"),
        "C_interrupted_by_D": "C→D" in joined,
        "D_returns_to_C": "D→C" in joined,
        "E_before_B": "E→B" in joined,
        "returns_to_user": "B→用户程序" in joined,
    }
    print(json.dumps(required, ensure_ascii=False, indent=2))
    return 0 if all(required.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
