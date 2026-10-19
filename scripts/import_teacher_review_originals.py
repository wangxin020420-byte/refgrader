"""Copy uncleaned CSBench images into the read-only review image collection."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--prepared-dir', type=Path, default=Path('data/csbench'))
    args = parser.parse_args()
    source = args.source_root.resolve()
    prepared = args.prepared_dir.resolve()
    metadata = [json.loads(line) for line in (prepared / 'answer_metadata.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    answers = {}
    for filename in {row['source_file'] for row in metadata}:
        path = source / 'answer' / filename
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if line.strip():
                row = json.loads(line)
                key = (row['question_id'], row['answer_id'])
                if key in answers:
                    raise ValueError(f'Duplicate source answer: {key}')
                answers[key] = row
    copied = missing = 0
    for row in metadata:
        answer = answers.get((row['question_id'], row['answer_id'])) or {}
        images = (answer.get('student_input') or {}).get('image_paths') or []
        if len(images) != 1:
            missing += 1
            continue
        image = (source / images[0]).resolve()
        if not image.is_relative_to(source / 'images') or not image.is_file() or image.is_relative_to(source / 'images' / 'cleaned'):
            missing += 1
            continue
        destination = prepared / 'original_student_images' / row['question_id'] / (row['answer_id'] + image.suffix.lower())
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != image.read_bytes():
                raise ValueError(f'Existing original differs: {destination}')
        else:
            shutil.copy2(image, destination)
        copied += 1
    print(f'Original images available: {copied}; missing/unsupported: {missing}')


if __name__ == '__main__':
    main()
