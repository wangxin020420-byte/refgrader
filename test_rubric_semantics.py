import json
import unittest
from pathlib import Path

from calibration_utils import (
    compute_extraction_quality_counts,
    compute_extraction_risk_features,
)
from canonicalizers import build_canonical_grading_context
from rubric_semantics import (
    apply_hierarchical_scoring_policy,
    has_deterministic_hierarchical_full_credit,
    prepare_rubric_semantic_contract,
    project_rubric_for_risk,
    validate_refined_rubric,
)
from step4_vlm_grader import _category_points_ratio


class RubricSemanticContractTests(unittest.TestCase):
    def test_co_inputs_keep_official_granularity_without_locked_splits(self):
        root = Path(__file__).resolve().parent
        expected_counts = {
            "CO_1": 3,
            "CO_2": 7,
            "CO_3": 2,
            "CO_4": 7,
            "CO_5": 3,
            "CO_6": 6,
            "CO_7": 2,
        }
        for question_id, expected_count in expected_counts.items():
            source = json.loads((root / (
                f"data/csbench/rubrics/source/CO/{question_id}.json"
            )).read_text(encoding="utf-8"))["grading_rubric"]
            initial = json.loads((root / (
                "data/csbench/rubrics/initial/CO/"
                f"{question_id}_rubric_standard.json"
            )).read_text(encoding="utf-8"))
            with self.subTest(question_id=question_id):
                self.assertEqual(len(source), expected_count)
                self.assertEqual(len(initial), expected_count)
                self.assertAlmostEqual(
                    sum(float(item["score"]) for item in source),
                    sum(float(item["points"]) for item in initial),
                )
                self.assertFalse(any(
                    "decomposition_locked" in item
                    or "minimum_scoring_children" in item
                    for item in source + initial
                ))

    def test_full_credit_format_equivalence_has_no_undercredit_risk(self):
        rubric = [{"id": "step_1", "points": 2.0}]
        full = [{
            "details": [{
                "id": "step_1",
                "score_given": 2.0,
                "error_category": "FORMAT_MINOR",
            }]
        }]
        discounted = [{
            "details": [{
                "id": "step_1",
                "score_given": 1.4,
                "error_category": "FORMAT_MINOR",
            }]
        }]
        self.assertEqual(
            _category_points_ratio(full, rubric, 2.0)["format_minor_points_ratio"],
            0.0,
        )
        self.assertAlmostEqual(
            _category_points_ratio(
                discounted, rubric, 2.0
            )["format_minor_points_ratio"],
            0.3,
        )

    def test_explicit_strict_atomic_result_cannot_be_split(self):
        original = prepare_rubric_semantic_contract([
            {
                "id": "step_1",
                "item": "写出唯一结果",
                "points": 5,
                "role": "final",
                "standard_answer_text": "37H",
                "scoring_policy": "strict_atomic",
            }
        ])
        refined = [
            {
                "id": "step_1_process",
                "parent_id": "step_1",
                "points": 3,
                "standard_answer_text": "34H",
            },
            {
                "id": "step_1_result",
                "parent_id": "step_1",
                "points": 2,
                "standard_answer_text": "37H",
            },
        ]
        valid, errors = validate_refined_rubric(original, refined, 5)
        self.assertFalse(valid)
        self.assertTrue(any("must remain one scoring item" in error for error in errors))

    @staticmethod
    def hierarchical_rubric():
        common = {
            "parent_id": "step_1",
            "parent_points": 5.0,
            "scoring_policy": "final_sufficient_partial_credit",
            "weighting_policy": "preserve_parent",
            "full_credit_policy": "final_answer_sufficient",
            "full_credit_anchor": "37H",
            "fallback_cap": 2.0,
        }
        return prepare_rubric_semantic_contract([
            {
                **common,
                "id": "step_1_address",
                "item": "正确解析地址字段",
                "points": 1.0,
                "standard_answer_text": "37H",
                "full_credit_trigger": False,
            },
            {
                **common,
                "id": "step_1_effective_address",
                "item": "正确得到有效地址",
                "points": 1.0,
                "standard_answer_text": "34H",
                "full_credit_trigger": False,
            },
            {
                **common,
                "id": "step_1_final",
                "item": "正确得到最终操作数",
                "points": 3.0,
                "standard_answer_text": "37H",
                "full_credit_trigger": True,
            },
        ])

    def test_hierarchical_rubric_preserves_parent_and_fallback_points(self):
        rubric = self.hierarchical_rubric()
        valid, errors = validate_refined_rubric(rubric, rubric, 5.0)
        self.assertTrue(valid, errors)

    def test_correct_final_answer_grants_full_parent_credit(self):
        rubric = self.hierarchical_rubric()
        result = {
            "total_score": 3.0,
            "details": [
                {"id": "step_1_address", "score_given": 0, "error_category": "BLANK"},
                {"id": "step_1_effective_address", "score_given": 0, "error_category": "BLANK"},
                {"id": "step_1_final", "score_given": 3, "error_category": "MATCH"},
            ],
        }
        updated = apply_hierarchical_scoring_policy(result, rubric, {})
        self.assertEqual(updated["total_score"], 5.0)
        self.assertEqual(
            updated["hierarchical_scoring"][0]["mode"],
            "final_answer_full_credit",
        )
        self.assertEqual(updated["details"][0]["error_category"], "BLANK")
        self.assertEqual(
            updated["details"][0]["credit_requirement"],
            "waived_by_final_answer",
        )

    def test_wrong_final_uses_process_fallback_only(self):
        rubric = self.hierarchical_rubric()
        result = {
            "total_score": 5.0,
            "details": [
                {"id": "step_1_address", "score_given": 1, "error_category": "MATCH"},
                {"id": "step_1_effective_address", "score_given": 1, "error_category": "MATCH"},
                {"id": "step_1_final", "score_given": 3, "error_category": "SEMANTIC_FATAL"},
            ],
        }
        updated = apply_hierarchical_scoring_policy(result, rubric, {})
        self.assertEqual(updated["total_score"], 2.0)
        self.assertEqual(updated["details"][2]["score_given"], 0.0)
        self.assertEqual(
            updated["hierarchical_scoring"][0]["mode"],
            "partial_process_fallback",
        )

    def test_canonical_mismatch_cannot_be_overridden_by_model_match(self):
        rubric = self.hierarchical_rubric()
        result = {
            "total_score": 5.0,
            "details": [
                {"id": "step_1_address", "score_given": 1, "error_category": "MATCH"},
                {"id": "step_1_effective_address", "score_given": 0, "error_category": "BLANK"},
                {"id": "step_1_final", "score_given": 3, "error_category": "MATCH"},
            ],
        }
        canonical_context = {
            "items": [
                {"id": "step_1_final", "comparison": {"match": False}}
            ]
        }
        updated = apply_hierarchical_scoring_policy(
            result,
            rubric,
            canonical_context,
        )
        self.assertEqual(updated["total_score"], 1.0)
        audit = updated["hierarchical_scoring"][0]
        self.assertEqual(audit["mode"], "partial_process_fallback")
        self.assertEqual(audit["trigger_match_source"], "canonicalizer")

    def test_canonical_blank_cannot_be_overridden_by_model_match(self):
        rubric = self.hierarchical_rubric()
        result = {
            "total_score": 3.0,
            "details": [
                {"id": "step_1_address", "score_given": 0, "error_category": "BLANK"},
                {"id": "step_1_effective_address", "score_given": 0, "error_category": "BLANK"},
                {"id": "step_1_final", "score_given": 3, "error_category": "MATCH"},
            ],
        }
        canonical_context = {
            "items": [
                {
                    "id": "step_1_final",
                    "comparison": {"status": "student_blank", "match": None},
                }
            ]
        }
        updated = apply_hierarchical_scoring_policy(
            result,
            rubric,
            canonical_context,
        )
        self.assertEqual(updated["total_score"], 0.0)
        self.assertEqual(
            updated["hierarchical_scoring"][0]["trigger_match_source"],
            "canonicalizer",
        )

    def test_embedded_co1_rubric_satisfies_hierarchical_contract(self):
        path = Path(__file__).resolve().parent / (
            "data/csbench/rubrics/initial/CO/CO_1_rubric_standard.json"
        )
        rubric = json.loads(path.read_text(encoding="utf-8"))
        valid, errors = validate_refined_rubric(rubric, rubric, 5.0)
        self.assertTrue(valid, errors)
        self.assertEqual(sum(item["points"] for item in rubric), 5.0)
        self.assertEqual(
            sum(item["points"] for item in rubric if not item["full_credit_trigger"]),
            3.5,
        )

    def test_real_co1_final_answer_drives_full_contract_end_to_end(self):
        path = Path(__file__).resolve().parent / (
            "data/csbench/rubrics/initial/CO/CO_1_rubric_standard.json"
        )
        rubric = json.loads(path.read_text(encoding="utf-8"))
        facts = {
            "step_1_address": "未书写",
            "step_1_effective_address": "未书写",
            "step_1_final": "37H",
        }
        canonical_context = build_canonical_grading_context(facts, rubric)
        raw_probe = {
            "total_score": 0.0,
            "details": [
                {"id": "step_1_address", "score_given": 0, "error_category": "BLANK"},
                {"id": "step_1_effective_address", "score_given": 0, "error_category": "BLANK"},
                {"id": "step_1_final", "score_given": 0, "error_category": "SEMANTIC_FATAL"},
            ],
        }
        probes = [
            apply_hierarchical_scoring_policy(
                json.loads(json.dumps(raw_probe)),
                rubric,
                canonical_context,
            )
            for _ in range(3)
        ]
        self.assertTrue(all(probe["total_score"] == 5.0 for probe in probes))
        self.assertTrue(has_deterministic_hierarchical_full_credit(rubric, probes))
        projected = project_rubric_for_risk(rubric, probes)
        self.assertEqual([item["id"] for item in projected], ["step_1_final"])
        self.assertEqual(projected[0]["points"], 5.0)

    def test_risk_projection_waives_optional_process_after_majority_full_credit(self):
        rubric = self.hierarchical_rubric()
        full = {
            "hierarchical_scoring": [
                {"parent_id": "step_1", "mode": "final_answer_full_credit"}
            ]
        }
        fallback = {
            "hierarchical_scoring": [
                {"parent_id": "step_1", "mode": "partial_process_fallback"}
            ]
        }
        projected = project_rubric_for_risk(rubric, [full, full, fallback])
        self.assertEqual(len(projected), 1)
        self.assertEqual(projected[0]["id"], "step_1_final")
        self.assertEqual(projected[0]["points"], 5.0)
        facts = {
            "step_1_address": "未书写",
            "step_1_effective_address": "未书写",
            "step_1_final": "37H",
        }
        counts = compute_extraction_quality_counts(facts, projected)
        risks = compute_extraction_risk_features(counts)
        self.assertEqual(counts["total_items"], 1)
        self.assertEqual(risks["blank_rate"], 0.0)
        self.assertEqual(risks["extraction_risk"], 0.0)

    def test_risk_projection_keeps_process_without_majority_full_credit(self):
        rubric = self.hierarchical_rubric()
        full = {
            "hierarchical_scoring": [
                {"parent_id": "step_1", "mode": "final_answer_full_credit"}
            ]
        }
        fallback = {
            "hierarchical_scoring": [
                {"parent_id": "step_1", "mode": "partial_process_fallback"}
            ]
        }
        projected = project_rubric_for_risk(rubric, [full, fallback, fallback])
        self.assertEqual(len(projected), 3)
        projected = project_rubric_for_risk(rubric, [full])
        self.assertEqual(len(projected), 3)

    def test_deterministic_full_credit_lock_requires_canonical_majority(self):
        rubric = self.hierarchical_rubric()
        canonical_full = {
            "hierarchical_scoring": [{
                "parent_id": "step_1",
                "mode": "final_answer_full_credit",
                "trigger_match_source": "canonicalizer",
            }]
        }
        semantic_full = {
            "hierarchical_scoring": [{
                "parent_id": "step_1",
                "mode": "final_answer_full_credit",
                "trigger_match_source": "semantic_grader",
            }]
        }
        fallback = {
            "hierarchical_scoring": [{
                "parent_id": "step_1",
                "mode": "partial_process_fallback",
                "trigger_match_source": "canonicalizer",
            }]
        }
        self.assertTrue(has_deterministic_hierarchical_full_credit(
            rubric,
            [canonical_full, canonical_full, fallback],
        ))
        self.assertFalse(has_deterministic_hierarchical_full_credit(
            rubric,
            [canonical_full, semantic_full, fallback],
        ))
        self.assertFalse(has_deterministic_hierarchical_full_credit(
            rubric,
            [canonical_full],
        ))

    def test_full_credit_lock_requires_complete_hierarchical_coverage(self):
        rubric = self.hierarchical_rubric() + prepare_rubric_semantic_contract([{
            "id": "independent_item",
            "points": 1.0,
            "standard_answer_text": "x",
            "scoring_policy": "strict_atomic",
        }])
        canonical_full = {
            "hierarchical_scoring": [{
                "parent_id": "step_1",
                "mode": "final_answer_full_credit",
                "trigger_match_source": "canonicalizer",
            }]
        }
        self.assertFalse(has_deterministic_hierarchical_full_credit(
            rubric,
            [canonical_full, canonical_full, canonical_full],
        ))

    def test_all_embedded_initial_rubrics_match_question_contracts(self):
        root = Path(__file__).resolve().parent
        questions = json.loads(
            (root / "data/csbench/exam_database.json").read_text(encoding="utf-8")
        )
        for question in questions:
            question_id = question["question_id"]
            rubric_path = root / question["initial_rubric_path"]
            rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
            total_score = float(question["total_score"])
            with self.subTest(question_id=question_id):
                self.assertAlmostEqual(
                    sum(float(item.get("points", 0)) for item in rubric),
                    total_score,
                )
                valid, errors = validate_refined_rubric(
                    rubric,
                    rubric,
                    total_score,
                )
                self.assertTrue(valid, errors)

    def test_compound_item_may_split_with_parent_score_conservation(self):
        original = prepare_rubric_semantic_contract([
            {
                "id": "step_1",
                "item": "正确计算补码及判断溢出",
                "points": 5,
                "standard_answer_text": "8EH，发生溢出",
            }
        ])
        refined = [
            {"id": f"step_1_atom_{index}", "parent_id": "step_1", "points": 1}
            for index in range(1, 6)
        ]
        valid, errors = validate_refined_rubric(original, refined, 5)
        self.assertTrue(valid, errors)

    def test_compound_item_rejects_subjective_unequal_weights(self):
        original = prepare_rubric_semantic_contract([
            {"id": "step_1", "item": "判断命中并说明理由", "points": 5}
        ])
        refined = [
            {"id": "step_1_result", "parent_id": "step_1", "points": 2},
            {"id": "step_1_reason", "parent_id": "step_1", "points": 3},
        ]
        valid, errors = validate_refined_rubric(original, refined, 5)
        self.assertFalse(valid)
        self.assertTrue(any("equal atomic weights" in error for error in errors))

    def test_atomic_item_cannot_add_a_process_requirement(self):
        original = prepare_rubric_semantic_contract([
            {
                "id": "step_1",
                "item": "正确计算出M1的运行时间",
                "points": 5,
                "standard_answer_text": "50 μs",
            }
        ])
        refined = [
            {
                "id": "step_1",
                "parent_id": "step_1",
                "item": "写出公式并正确计算出M1的运行时间",
                "points": 5,
                "standard_answer_text": "50 μs",
            }
        ]
        valid, errors = validate_refined_rubric(original, refined, 5)
        self.assertFalse(valid)
        self.assertTrue(any("changed its scoring meaning" in error for error in errors))

    def test_points_cannot_move_between_parents(self):
        original = prepare_rubric_semantic_contract([
            {"id": "a", "item": "过程及结果", "points": 4},
            {"id": "b", "item": "判断并说明理由", "points": 6},
        ])
        refined = [
            {"id": "a1", "parent_id": "a", "points": 3},
            {"id": "b1", "parent_id": "b", "points": 7},
        ]
        valid, errors = validate_refined_rubric(original, refined, 10)
        self.assertFalse(valid)
        self.assertTrue(any("parent a score changed" in error for error in errors))
        self.assertTrue(any("parent b score changed" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
