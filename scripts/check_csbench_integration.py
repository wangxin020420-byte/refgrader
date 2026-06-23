"""Offline integration checks for the prepared CSBench compatibility view."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main_pipeline
import step4_vlm_grader as grader
from ocr.backend import sha256_file


def load_metadata(path: Path) -> dict[str, dict]:
    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[record["answer_id"]] = record
    return records


def checklist_for(rubric: list[dict]) -> str:
    return json.dumps(
        [
            {"id": item["id"], "instruction": item["item"]}
            for item in rubric
        ],
        ensure_ascii=False,
    )


def load_question_split(question: dict) -> dict:
    return json.loads(
        Path(question["rubric_split_path"]).read_text(encoding="utf-8")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", default="data/csbench")
    args = parser.parse_args()
    root = Path(args.prepared_dir).resolve()

    exam = json.loads((root / "exam_database.json").read_text(encoding="utf-8"))
    metadata = load_metadata(root / "answer_metadata.jsonl")
    teacher_scores = json.loads(
        (root / "teacher_scores.json").read_text(encoding="utf-8")
    )
    assert len(exam) == 43
    assert len(metadata) == len(teacher_scores) == 3326
    assert "ANS_CO_01" in metadata
    assert teacher_scores["ANS_CO_01"]["CO_1"] == 5.0

    questions_by_id = {
        question["question_id"]: question for question in exam
    }
    for question in exam:
        source_path = Path(question["source_rubric_path"])
        initial_path = Path(question["initial_rubric_path"])
        optimized_path = Path(question["optimized_rubric_path"])
        assert source_path.is_file()
        assert initial_path.is_file()
        assert source_path.is_relative_to(root / "rubrics" / "source")
        assert initial_path.is_relative_to(root / "rubrics" / "initial")
        assert optimized_path.is_relative_to(root / "rubrics" / "optimized")

        rubric = json.loads(initial_path.read_text(encoding="utf-8"))
        rubric_total = sum(float(item.get("points", 0)) for item in rubric)
        assert abs(rubric_total - float(question["total_score"])) < 1e-6

        split = load_question_split(question)
        calibration = set(split["calibration"])
        validation = set(split["validation"])
        test = set(split["test"])
        assert not calibration & validation
        assert not calibration & test
        assert not validation & test
        assert calibration | validation | test == {
            record["answer_id"]
            for record in metadata.values()
            if record["question_id"] == question["question_id"]
        }

    main_pipeline.TEACHER_DB_PATH = str(root / "teacher_scores.json")
    main_pipeline.ANSWER_METADATA_PATH = str(root / "answer_metadata.jsonl")
    main_pipeline.INITIAL_RUBRIC_DIR = str(root / "rubrics" / "initial")
    main_pipeline.RUBRIC_DIR = str(root / "rubrics" / "optimized")
    main_pipeline.ALLOW_INITIAL_RUBRIC = True
    main_pipeline._GLOBAL_SCORES_DB = None
    main_pipeline._GLOBAL_ANSWER_METADATA = None
    assert main_pipeline.get_teacher_score_from_your_database(
        "ANS_CO_01", "CO_1"
    ) == 5.0
    selected = main_pipeline.selected_image_files(
        ["ANS_CO_01.jpg", "ANS_CO_02.jpg"], ["ANS_CO_02"]
    )
    assert selected == ["ANS_CO_02.jpg"]
    co2_question = questions_by_id["CO_2"]
    co2_split = load_question_split(co2_question)
    co2_images = sorted(
        path.name
        for path in Path(co2_question["student_images_dir"]).iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    selected_test_images = main_pipeline.select_question_images(
        co2_images, 5, co2_question, "test"
    )
    assert len(selected_test_images) == 5
    assert {
        Path(filename).stem for filename in selected_test_images
    }.issubset(set(co2_split["test"]))

    original_call = grader.call_glm5_text
    original_diagram = grader.parse_diagram_relations_with_glm4v
    try:
        # Text-only CSBench sample: raw_text is sufficient and OCR is not required.
        text_meta = metadata["ANS_CO_01"]
        text_question = questions_by_id["CO_1"]
        text_rubric = json.loads(
            Path(text_question["initial_rubric_path"]).read_text(
                encoding="utf-8"
            )
        )
        assert main_pipeline.rubric_path_for(
            "CO_1", text_question
        ) == text_question["initial_rubric_path"]
        grader.call_glm5_text = lambda *args, **kwargs: json.dumps(
            {"step_1": "37H"}, ensure_ascii=False
        )
        text_facts, text_evidence = grader.stage1_extract_with_backend(
            question_text="CO_1",
            student_img_path=text_meta["student_image"],
            blind_checklist=checklist_for(text_rubric),
            rubrics_json=json.dumps(text_rubric, ensure_ascii=False),
            extraction_backend="csbench_hybrid",
            student_transcription=text_meta["raw_text"],
            answer_metadata=text_meta,
            force_extraction=True,
        )
        assert json.loads(text_facts)["step_1"] == "37H"
        assert text_evidence["diagram_parser_used"] is False

        # Visual-placeholder sample: preserve text facts and merge diagram facts.
        visual_meta = next(
            record
            for record in metadata.values()
            if record["question_id"] == "CO_2"
            and record["visual_placeholder_detected"]
        )
        visual_rubric = json.loads(
            Path(
                questions_by_id["CO_2"]["initial_rubric_path"]
            ).read_text(encoding="utf-8")
        )
        mapped = {
            item["id"]: (
                "需要查看图像"
                if "diagram" in item["evidence_source"]
                else f"transcribed-{item['id']}"
            )
            for item in visual_rubric
        }
        grader.call_glm5_text = lambda *args, **kwargs: json.dumps(
            mapped, ensure_ascii=False
        )
        grader.parse_diagram_relations_with_glm4v = (
            lambda *args, **kwargs: (
                {"step_7": "用户程序→A→C→D→C→E→B→用户程序"},
                "用户程序→A→C→D→C→E→B→用户程序",
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            ocr_path = Path(temp_dir) / "ocr.json"
            ocr_path.write_text(
                json.dumps(
                    {
                        "image": {
                            "sha256": sha256_file(visual_meta["student_image"])
                        },
                        "engine": {"name": "test"},
                        "summary": {},
                        "tokens": [{"text": "A", "confidence": 0.9, "box": [0, 0, 1, 1]}],
                    }
                ),
                encoding="utf-8",
            )
            visual_facts, visual_evidence = grader.stage1_extract_with_backend(
                question_text="CO_2",
                student_img_path=visual_meta["student_image"],
                blind_checklist=checklist_for(visual_rubric),
                rubrics_json=json.dumps(visual_rubric, ensure_ascii=False),
                extraction_backend="csbench_hybrid",
                ocr_json_path=str(ocr_path),
                student_transcription=visual_meta["raw_text"],
                answer_metadata=visual_meta,
                force_extraction=True,
            )
        visual_facts = json.loads(visual_facts)
        assert visual_facts["step_1"] == "transcribed-step_1"
        assert "C→D→C" in visual_facts["step_7"]
        assert visual_evidence["visual_placeholder_detected"] is True
        assert visual_evidence["diagram_parser_used"] is True
    finally:
        grader.call_glm5_text = original_call
        grader.parse_diagram_relations_with_glm4v = original_diagram

    print(
        json.dumps(
            {
                "prepared_questions": len(exam),
                "prepared_answers": len(metadata),
                "three_layer_rubric_layout": "PASS",
                "rubric_total_score_validation": "PASS",
                "calibration_validation_test_isolation": "PASS",
                "test_split_runtime_filter": "PASS",
                "complete_answer_id_lookup": "PASS",
                "text_transcription_route": "PASS",
                "visual_placeholder_route": "PASS",
                "diagram_merge": "PASS",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
