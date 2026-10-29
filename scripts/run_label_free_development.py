"""Development-only label-free scoring entry point, not a formal test runner.

Use --check for an offline preflight. Runs are foreground. Test split requires
an explicit exploratory flag and is never a formal confirmation. Never provide
teacher scores to this script. Source-data provenance is not yet established.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from label_free_model import BASE_URL, TEXT_MODEL, VISION_MODEL, create_client
from label_free_checkpoint import Checkpoint
from label_free_run import run_questions
from label_free_question_snapshot import load_candidate_question
from scripts.verify_label_free_question_snapshot import verify
from scripts.preflight_label_free import preflight

PREPARED = ROOT / "data/csbench"
RUN_ROOT = ROOT / "results_runs/label_free_development"
LOSS = {"lambda1": 1, "lambda2": 1, "mu1": 1, "mu2": 1}
M = 0.5


def _source_hash():
    files = subprocess.check_output(["git", "ls-files", "--", "*.py", "prompts/*"], cwd=ROOT, text=True, encoding="utf-8").splitlines()
    dirty = subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", "*.py", "prompts/*"], cwd=ROOT, text=True, encoding="utf-8").splitlines()
    if dirty:
        raise ValueError("Tracked Python/prompt files have uncommitted changes; commit them before running")
    digest = hashlib.sha256()
    for name in sorted(files):
        path = ROOT / name
        if not path.is_file():
            raise ValueError("Tracked source file is missing: " + name)
        content = path.read_bytes().replace(b"\r\n", b"\n")
        digest.update(name.encode("utf-8") + b"\0" + hashlib.sha256(content).digest())
    return digest.hexdigest()


@contextlib.contextmanager
def _single_writer(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "writer.lock"
    with path.open("a+b") as handle:
        try:
            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Another writer holds this run") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _inputs(question_ids, split, candidate_snapshot=None, risk_policy="baseline"):
    report = preflight(PREPARED)
    database = json.loads((PREPARED / "exam_database.json").read_text(encoding="utf-8-sig"))
    available = {item["question_id"] for item in database}
    if not set(question_ids) <= available:
        raise ValueError("Unknown question ID")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    split_manifest = (PREPARED / "manifest.json").read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    run_config = {"question_ids": question_ids, "split": split, "m": M, "loss_parameters": LOSS}
    if risk_policy != "baseline":
        run_config["risk_policy"] = risk_policy
    candidate_report = None
    if candidate_snapshot is not None:
        candidate_root = Path(candidate_snapshot).resolve(strict=True)
        if not candidate_root.is_relative_to(ROOT.resolve()):
            raise ValueError("Candidate snapshot must be inside the repository")
        candidate_report = verify(candidate_root)
        for question_id in question_ids:
            load_candidate_question(candidate_root, question_id)
        run_config["candidate_snapshot_sha256"] = candidate_report["snapshot_sha256_normalized"]
    result = {
        "run_config": run_config,
        "input_groups": report["input_groups"],
        "split_manifest_sha256": hashlib.sha256(split_manifest.encode("utf-8")).hexdigest(),
        "model": {"text": TEXT_MODEL, "vision": VISION_MODEL, "base_url": BASE_URL,
                  "thinking": "disabled", "text_temperature": 0.35, "vision_temperature": 0},
        "source": {"git_commit": revision, "source_sha256": _source_hash(),
                   "python": platform.python_version(), "openai": importlib.metadata.version("openai"),
                   "httpx": importlib.metadata.version("httpx")},
        "scope": "exploratory_test_not_formal" if split == "test" else "development_only_not_formal", "teacher_scores_used": False,
        "limitations": report["checks_pending"],
    }
    if candidate_report is not None:
        result["candidate_snapshot"] = candidate_report
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--status", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--questions", nargs="+", default=["CO_1"])
    parser.add_argument("--split", choices=["calibration", "validation", "test"], default="calibration")
    parser.add_argument("--exploratory-test", action="store_true",
                        help="Explicitly allow exploratory use of the historical test split; not a formal holdout")
    parser.add_argument("--max-new", type=int, default=1, help="Maximum new answers in this invocation")
    parser.add_argument("--candidate-snapshot", help="Unapproved candidate snapshot for development only")
    parser.add_argument("--risk-policy", choices=["baseline", "candidate_max_v1"], default="baseline")
    args = parser.parse_args(argv)
    if args.split == "test" and not args.exploratory_test:
        parser.error("test split requires --exploratory-test; this does not establish a formal holdout")
    if args.exploratory_test and args.split != "test":
        parser.error("--exploratory-test is only valid with --split test")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("Invalid run ID")
    if len(set(args.questions)) != len(args.questions) or args.max_new < 1:
        parser.error("Question IDs must be unique and max-new must be positive")
    run_dir = RUN_ROOT / args.run_id
    try:
        if args.status:
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.is_file():
                raise ValueError("Run has not been initialized")
            envelope = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = envelope.get("manifest")
            checkpoint = Checkpoint(run_dir, manifest)
            answer_dir = run_dir / "answers"
            files = list(answer_dir.glob("*.json")) if answer_dir.exists() else []
            for path in files:
                checkpoint.load(path.stem)
            try:
                with _single_writer(run_dir):
                    running = False
            except ValueError:
                running = True
            print(json.dumps({"run_id": args.run_id, "running": running, "verified_answer_files": len(files),
                              "scope": manifest.get("scope"), "run_config": manifest.get("run_config")}, indent=2))
            return 0
        if args.risk_policy == "baseline":
            manifest = _inputs(args.questions, args.split, args.candidate_snapshot) if args.candidate_snapshot else _inputs(args.questions, args.split)
        else:
            manifest = _inputs(args.questions, args.split, args.candidate_snapshot, args.risk_policy)
        if args.check:
            print(json.dumps({"offline_check": "passed", "run_id": args.run_id,
                              "manifest": manifest, "live_calls": 0,
                              "ready_for_formal_experiment": False}, ensure_ascii=False, indent=2))
            return 0
        key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
        if not key:
            raise ValueError("Coding Plan key is missing from this process environment")
        with _single_writer(run_dir), create_client(key) as client:
            counts = run_questions(PREPARED, args.questions, args.split, run_dir, manifest, client,
                                   m=M, loss_parameters=LOSS, max_new=args.max_new,
                                   candidate_snapshot=args.candidate_snapshot, risk_policy=args.risk_policy)
        print(json.dumps({"run_id": args.run_id, "scope": manifest["scope"], **counts}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(f"Development run failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
