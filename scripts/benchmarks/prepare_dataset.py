from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_datasets.registry import get_adapter


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize a public scoring dataset into RefGrader format."
    )
    parser.add_argument("dataset", choices=["asap_sas", "mohler"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    adapter = get_adapter(args.dataset)
    audit = adapter(
        args.source,
        args.spec,
        args.output_dir,
        force=args.force,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
