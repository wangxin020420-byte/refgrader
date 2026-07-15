"""Run PaddleOCR as an isolated visual-transcription worker.

This module does not grade answers and does not map OCR text to rubric items.
It writes auditable OCR evidence: text, confidence, coordinates, model version,
image hash, and conservative blank/nonblank diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Configure PaddleX before importing Paddle/PaddleOCR. The worker is also
# invoked directly, so it cannot rely only on the parent pipeline environment.
if __package__:
    from .runtime_env import configure_paddlex_process
else:
    from runtime_env import configure_paddlex_process


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PADDLEX_CACHE_HOME = configure_paddlex_process(PROJECT_ROOT)

import numpy as np
import paddle
import paddleocr
from PIL import Image, ImageOps
from paddleocr import PaddleOCR


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract auditable OCR evidence from one image or a directory."
    )
    parser.add_argument("--input", required=True, help="Input image or image directory.")
    parser.add_argument(
        "--output-dir",
        default="ocr_cache/manual_test",
        help="OCR JSON output directory.",
    )
    parser.add_argument("--lang", default="ch", help="PaddleOCR language. Default: ch.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device, for example cpu or gpu:0.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence used by summaries and blank diagnostics.",
    )
    parser.add_argument(
        "--enable-mkldnn",
        action="store_true",
        help=(
            "Enable CPU oneDNN/MKL-DNN. Disabled by default on Windows because "
            "PaddlePaddle 3.3.x can fail on some executors."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively find images when input is a directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite OCR JSON even when the image hash is unchanged.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def result_to_dict(result: Any) -> dict[str, Any]:
    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported PaddleOCR result type: {type(payload)!r}")
    payload = to_builtin(payload)
    inner = payload.get("res")
    return inner if isinstance(inner, dict) else payload


def collect_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_existing_hash(output_path: Path) -> str | None:
    if not output_path.exists():
        return None
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
        return data.get("image", {}).get("sha256")
    except (OSError, json.JSONDecodeError):
        return None


def build_pipeline(args: argparse.Namespace) -> PaddleOCR:
    return PaddleOCR(
        lang=args.lang,
        device=args.device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=args.enable_mkldnn,
    )


def compute_visual_blank_diagnostics(image_path: Path) -> dict[str, Any]:
    """Return conservative image-only blank/nonblank evidence.

    Crops can contain printed question text, axes, or score marks. Therefore a
    blank answer is confirmed only when the whole crop has almost no dark ink.
    Ambiguous cases remain ``uncertain``.
    """
    image = Image.open(image_path)
    gray = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
    if gray.size == 0:
        return {
            "status": "uncertain",
            "reason": "empty_image_array",
            "dark_pixel_ratio": None,
            "very_dark_pixel_ratio": None,
            "lower_region_dark_pixel_ratio": None,
        }

    dark_ratio = float(np.mean(gray < 210))
    very_dark_ratio = float(np.mean(gray < 120))
    lower = gray[int(gray.shape[0] * 0.18) :, :]
    lower_dark_ratio = float(np.mean(lower < 210)) if lower.size else dark_ratio

    if dark_ratio <= 0.0025 and very_dark_ratio <= 0.0008:
        status = "confirmed_blank"
        reason = "almost_no_visible_ink"
    elif lower_dark_ratio >= 0.030 or dark_ratio >= 0.035:
        status = "confirmed_nonblank"
        reason = "substantial_visible_ink"
    else:
        status = "uncertain"
        reason = "printed_content_or_faint_marks_cannot_be_separated_safely"

    return {
        "status": status,
        "reason": reason,
        "dark_pixel_ratio": round(dark_ratio, 6),
        "very_dark_pixel_ratio": round(very_dark_ratio, 6),
        "lower_region_dark_pixel_ratio": round(lower_dark_ratio, 6),
        "thresholds": {
            "dark_pixel": 210,
            "very_dark_pixel": 120,
            "lower_region_start_ratio": 0.18,
        },
    }


def run_one(
    pipeline: PaddleOCR,
    image_path: Path,
    output_path: Path,
    min_confidence: float,
) -> dict[str, Any]:
    pages = []
    tokens = []
    for page_index, result in enumerate(pipeline.predict(str(image_path))):
        data = result_to_dict(result)
        texts = list(data.get("rec_texts") or [])
        scores = list(data.get("rec_scores") or [])
        boxes = list(data.get("rec_boxes") or [])
        polygons = list(data.get("dt_polys") or [])
        page_tokens = []
        for index, text in enumerate(texts):
            score = float(scores[index]) if index < len(scores) else None
            token = {
                "id": f"p{page_index + 1}_t{index + 1}",
                "text": str(text),
                "confidence": score,
                "box": boxes[index] if index < len(boxes) else None,
                "polygon": polygons[index] if index < len(polygons) else None,
                "page": page_index,
            }
            page_tokens.append(token)
            tokens.append(token)
        pages.append({"page": page_index, "tokens": page_tokens, "raw_result": data})

    accepted_tokens = [
        token
        for token in tokens
        if token["confidence"] is None or token["confidence"] >= min_confidence
    ]
    confidences = [
        token["confidence"]
        for token in accepted_tokens
        if token["confidence"] is not None
    ]
    visual_blank = compute_visual_blank_diagnostics(image_path)
    if visual_blank["status"] == "confirmed_blank" and accepted_tokens:
        visual_blank = {
            **visual_blank,
            "status": "uncertain",
            "reason": "ocr_detected_text_despite_low_visual_ink",
        }
    elif visual_blank["status"] == "uncertain" and len(accepted_tokens) >= 4:
        visual_blank = {
            **visual_blank,
            "status": "confirmed_nonblank",
            "reason": "multiple_confident_ocr_tokens",
        }

    payload = {
        "schema_version": 2,
        "created_at": datetime.now().isoformat(),
        "image": {
            "path": str(image_path.resolve()),
            "name": image_path.name,
            "sha256": sha256_file(image_path),
            "size_bytes": image_path.stat().st_size,
        },
        "engine": {
            "name": "PaddleOCR",
            "paddle_version": paddle.__version__,
            "paddleocr_version": paddleocr.__version__,
            "device": paddle.get_device(),
            "mkldnn_enabled": bool(
                paddle.get_flags(["FLAGS_use_mkldnn"]).get("FLAGS_use_mkldnn", False)
            ),
        },
        "summary": {
            "token_count": len(tokens),
            "accepted_token_count": len(accepted_tokens),
            "mean_confidence": (
                round(sum(confidences) / len(confidences), 6) if confidences else None
            ),
            "min_confidence": min_confidence,
            "joined_text": "\n".join(token["text"] for token in accepted_tokens),
            "ocr_empty_candidate": len(accepted_tokens) == 0,
            "blank_authenticity": visual_blank,
        },
        "tokens": tokens,
        "pages": pages,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    images = collect_images(input_path, args.recursive)
    if not images:
        print(f"No images found: {input_path}", file=sys.stderr)
        return 2

    completed = 0
    skipped = 0
    failed = 0
    pending = []
    for index, image_path in enumerate(images, start=1):
        output_path = output_dir / f"{image_path.stem}.json"
        image_hash = sha256_file(image_path)
        if not args.force and load_existing_hash(output_path) == image_hash:
            print(f"[{index}/{len(images)}] cache hit: {image_path.name}")
            skipped += 1
            continue
        pending.append((index, image_path, output_path))

    if not pending:
        print(f"summary: completed=0, skipped={skipped}, failed=0")
        return 0

    print(
        f"Initialize PaddleOCR | device={args.device} | lang={args.lang} | "
        f"pending={len(pending)}/{len(images)} | "
        f"model_cache={PADDLEX_CACHE_HOME}"
    )
    pipeline = build_pipeline(args)

    for index, image_path, output_path in pending:
        try:
            payload = run_one(pipeline, image_path, output_path, args.min_confidence)
            summary = payload["summary"]
            print(
                f"[{index}/{len(images)}] completed: {image_path.name} | "
                f"tokens={summary['token_count']} | "
                f"mean_conf={summary['mean_confidence']} | output={output_path}"
            )
            completed += 1
        except Exception as exc:
            print(f"[{index}/{len(images)}] failed: {image_path.name} | {exc}")
            failed += 1

    print(f"summary: completed={completed}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
