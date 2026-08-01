from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_datasets.contract import audit_prepared_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a prepared public benchmark snapshot."
    )
    parser.add_argument("--prepared-dir", required=True)
    args = parser.parse_args()
    audit = audit_prepared_benchmark(args.prepared_dir)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
