import json
import unittest
from pathlib import Path

from calibration_utils import (
    compute_extraction_quality_counts,
    compute_extraction_risk_features,
    infer_rubric_task_profile,
)
from canonicalizers import build_canonical_grading_context
from rubric_semantics import (
    HIGH_VALUE_SPLIT_THRESHOLD,
    apply_hierarchical_scoring_policy,
    apply_role_weighted_scoring_policy,
    high_value_split_targets,
    has_deterministic_hierarchical_full_credit,
    prepare_rubric_semantic_contract,
    assess_candidate_replay,
    project_refined_candidate_to_contract,
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

    def test_all_embedded_initial_rubrics_prepare_high_value_contracts(self):
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
                prepared = prepare_rubric_semantic_contract(rubric)
                for item in prepared:
                    parent_points = float(item.get("parent_points", 0) or 0)
                    if parent_points < HIGH_VALUE_SPLIT_THRESHOLD:
                        continue
                    if item["scoring_policy"] == "additive_split":
                        if not str(item.get("standard_answer_text", "")).strip():
                            self.assertFalse(item["decomposition_required"])
                            self.assertEqual(
                                item["decomposition_exemption"],
                                "insufficient_machine_readable_reference_anchor",
                            )
                        else:
                            self.assertTrue(item["decomposition_required"])
                            self.assertGreaterEqual(
                                item["minimum_scoring_children"], 2
                            )
                    elif item["scoring_policy"] == "strict_atomic":
                        self.assertFalse(item["decomposition_required"])
                        self.assertEqual(
                            item["decomposition_exemption"],
                            "strict_atomic_single_outcome",
                        )
                    elif item["scoring_policy"] == "role_weighted_additive":
                        self.assertTrue(item["decomposition_required"])
                        self.assertEqual(item["weighting_policy"], "role_constrained")
                        self.assertLessEqual(item["maximum_final_ratio"], 0.35)
                        self.assertGreaterEqual(item["minimum_process_ratio"], 0.65)

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
            {
                "id": "step_1_result",
                "parent_id": "step_1",
                "item": "正确计算补码",
                "points": 2.5,
                "standard_answer_text": "8EH",
            },
            {
                "id": "step_1_overflow",
                "parent_id": "step_1",
                "item": "正确判断溢出",
                "points": 2.5,
                "standard_answer_text": "发生溢出",
            },
        ]
        valid, errors = validate_refined_rubric(original, refined, 5)
        self.assertTrue(valid, errors)

    def test_high_value_process_item_cannot_remain_unsplit(self):
        original = prepare_rubric_semantic_contract([{
            "id": "step_1",
            "item": "判断命中并说明理由",
            "points": 5,
            "standard_answer_text": "未命中，因为标记不匹配",
        }])
        valid, errors = validate_refined_rubric(original, original, 5)
        self.assertFalse(valid)
        self.assertEqual(original[0]["scoring_policy"], "role_weighted_additive")
        self.assertTrue(any("role-weighted parent" in error for error in errors))
        self.assertEqual(
            high_value_split_targets(original)[0]["parent_id"],
            "step_1",
        )

    def test_high_value_strict_atomic_item_has_auditable_exemption(self):
        original = prepare_rubric_semantic_contract([{
            "id": "step_1",
            "item": "写出唯一最终结果",
            "points": 5,
            "standard_answer_text": "37H",
            "scoring_policy": "strict_atomic",
        }])
        valid, errors = validate_refined_rubric(original, original, 5)
        self.assertTrue(valid, errors)
        self.assertEqual(
            original[0]["decomposition_exemption"],
            "strict_atomic_single_outcome",
        )
        self.assertEqual(high_value_split_targets(original), [])

    def test_image_only_parent_is_not_split_by_text_optimizer(self):
        original = prepare_rubric_semantic_contract([{
            "id": "step_1",
            "item": "画出处理顺序图",
            "points": 8,
            "standard_answer_text": "",
            "standard_answer_image": "reference.png",
            "evidence_source": "diagram",
            "split_policy": "allow_semantic_split",
        }])
        self.assertFalse(original[0]["decomposition_required"])
        self.assertEqual(
            original[0]["decomposition_exemption"],
            "insufficient_machine_readable_reference_anchor",
        )
        self.assertEqual(high_value_split_targets(original), [])
        valid, errors = validate_refined_rubric(original, original, 8)
        self.assertTrue(valid, errors)

    def test_candidate_projection_restores_atomic_and_unanchored_parents(self):
        original = prepare_rubric_semantic_contract([
            {
                "id": "step_1",
                "item": "写出屏蔽字",
                "points": 2,
                "standard_answer_text": "A:11111",
                "split_policy": "preserve_atomic",
            },
            {
                "id": "step_2",
                "item": "画出处理顺序图",
                "points": 8,
                "standard_answer_text": "",
                "standard_answer_image": "reference.png",
                "split_policy": "allow_semantic_split",
            },
        ])
        candidate = [
            {
                "id": "step_1",
                "parent_id": "step_1",
                "item": "写出屏蔽字",
                "points": 2,
                "standard_answer_text": "A:00000",
            },
            {
                "id": "step_2_part_1",
                "parent_id": "step_2",
                "item": "第一段",
                "points": 8,
                "standard_answer_text": "invented-1",
            },
            {
                "id": "step_2_part_2",
                "parent_id": "step_2",
                "item": "第二段",
                "points": 8,
                "standard_answer_text": "invented-2",
            },
        ]
        projected = project_refined_candidate_to_contract(original, candidate)
        self.assertEqual(
            [(item["id"], item["standard_answer_text"], item["points"]) for item in projected],
            [("step_1", "A:11111", 2), ("step_2", "", 8)],
        )
        valid, errors = validate_refined_rubric(original, projected, 10)
        self.assertTrue(valid, errors)

    def test_completed_high_value_split_is_not_targeted_again(self):
        split = prepare_rubric_semantic_contract([
            {
                "id": "step_1_result",
                "parent_id": "step_1",
                "parent_points": 5,
                "item": "给出正确结论",
                "points": 2.5,
                "standard_answer_text": "未命中",
                "scoring_policy": "additive_split",
            },
            {
                "id": "step_1_reason",
                "parent_id": "step_1",
                "parent_points": 5,
                "item": "给出关键理由",
                "points": 2.5,
                "standard_answer_text": "标记不匹配",
                "scoring_policy": "additive_split",
            },
        ])
        self.assertEqual(high_value_split_targets(split), [])

    def test_high_value_threshold_is_inclusive(self):
        at_threshold = prepare_rubric_semantic_contract([{
            "id": "step_1",
            "item": "给出结论并说明理由",
            "points": HIGH_VALUE_SPLIT_THRESHOLD,
            "standard_answer_text": "命中，因为标记一致",
        }])
        below_threshold = prepare_rubric_semantic_contract([{
            "id": "step_1",
            "item": "给出结论并说明理由",
            "points": HIGH_VALUE_SPLIT_THRESHOLD - 0.1,
            "standard_answer_text": "命中，因为标记一致",
        }])
        self.assertEqual(len(high_value_split_targets(at_threshold)), 1)
        self.assertEqual(high_value_split_targets(below_threshold), [])

    def test_process_item_rejects_untyped_unequal_children(self):
        original = prepare_rubric_semantic_contract([
            {"id": "step_1", "item": "判断命中并说明理由", "points": 5}
        ])
        refined = [
            {"id": "step_1_result", "parent_id": "step_1", "points": 2},
            {"id": "step_1_reason", "parent_id": "step_1", "points": 3},
        ]
        valid, errors = validate_refined_rubric(original, refined, 5)
        self.assertFalse(valid)
        self.assertTrue(any("invalid scoring_role" in error for error in errors))

    def test_atomic_item_cannot_add_a_process_requirement(self):
        original = prepare_rubric_semantic_contract([
            {
                "id": "step_1",
                "item": "正确计算出M1的运行时间",
                "points": 5,
                "standard_answer_text": "50 μs",
                "scoring_policy": "strict_atomic",
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

    def test_complex_process_parent_accepts_role_constrained_weights(self):
        original = prepare_rubric_semantic_contract([{
            "id": "step_1",
            "item": "完成地址映射，比较标记并说明是否命中",
            "points": 5,
            "standard_answer_text": "映射到组0，标记不一致，因此未命中",
            "task_semantics": "process_dominant",
            "process_complexity": "complex",
        }])
        refined = [
            {
                "id": "step_1_support", "parent_id": "step_1",
                "item": "正确映射到组0", "points": 1.5,
                "standard_answer_text": "组0", "scoring_role": "support_process",
            },
            {
                "id": "step_1_core", "parent_id": "step_1",
                "item": "正确比较标记", "points": 2.5,
                "standard_answer_text": "标记不一致", "scoring_role": "core_process",
            },
            {
                "id": "step_1_final", "parent_id": "step_1",
                "item": "得出未命中结论", "points": 1.0,
                "standard_answer_text": "未命中", "scoring_role": "final",
            },
        ]
        valid, errors = validate_refined_rubric(original, refined, 5)
        self.assertTrue(valid, errors)

    def test_complex_process_parent_rejects_half_weight_bare_conclusion(self):
        original = prepare_rubric_semantic_contract([{
            "id": "step_1",
            "item": "完成地址映射，比较标记并说明是否命中",
            "points": 5,
            "standard_answer_text": "映射到组0，标记不一致，因此未命中",
            "task_semantics": "process_dominant",
            "process_complexity": "complex",
        }])
        refined = [
            {
                "id": "step_1_core", "parent_id": "step_1",
                "item": "正确比较标记", "points": 2.5,
                "standard_answer_text": "标记不一致", "scoring_role": "core_process",
            },
            {
                "id": "step_1_final", "parent_id": "step_1",
                "item": "得出未命中结论", "points": 2.5,
                "standard_answer_text": "未命中", "scoring_role": "final",
            },
        ]
        valid, errors = validate_refined_rubric(original, refined, 5)
        self.assertFalse(valid)
        self.assertTrue(any("too much weight" in error for error in errors))
        self.assertTrue(any("support-process" in error for error in errors))

    def test_role_weighted_final_is_independent_by_default(self):
        rubric = prepare_rubric_semantic_contract([
            {
                "id": "support", "parent_id": "p", "parent_points": 5,
                "item": "映射", "points": 1.5, "standard_answer_text": "组0",
                "task_semantics": "process_dominant", "process_complexity": "complex",
                "scoring_policy": "role_weighted_additive",
                "scoring_role": "support_process",
            },
            {
                "id": "core", "parent_id": "p", "parent_points": 5,
                "item": "比较", "points": 2.5, "standard_answer_text": "不一致",
                "task_semantics": "process_dominant", "process_complexity": "complex",
                "scoring_policy": "role_weighted_additive",
                "scoring_role": "core_process",
            },
            {
                "id": "final", "parent_id": "p", "parent_points": 5,
                "item": "结论", "points": 1, "standard_answer_text": "未命中",
                "task_semantics": "process_dominant", "process_complexity": "complex",
                "scoring_policy": "role_weighted_additive",
                "scoring_role": "final",
            },
        ])
        result = {
            "total_score": 3,
            "details": [
                {"id": "support", "score_given": 0},
                {"id": "core", "score_given": 0},
                {"id": "final", "score_given": 3},
            ],
        }
        updated = apply_role_weighted_scoring_policy(result, rubric)
        self.assertEqual(updated["total_score"], 1.0)
        self.assertEqual(updated["details"][2]["score_given"], 1.0)

    def test_explicit_evidence_dependency_blocks_bare_conclusion(self):
        rubric = prepare_rubric_semantic_contract([
            {
                "id": "core", "parent_id": "p", "parent_points": 5,
                "item": "证明", "points": 4, "standard_answer_text": "关键证明",
                "task_semantics": "process_dominant", "process_complexity": "short",
                "scoring_policy": "role_weighted_additive",
                "scoring_role": "core_process", "dependency_mode": "evidence_required",
            },
            {
                "id": "final", "parent_id": "p", "parent_points": 5,
                "item": "结论", "points": 1, "standard_answer_text": "成立",
                "task_semantics": "process_dominant", "process_complexity": "short",
                "scoring_policy": "role_weighted_additive",
                "scoring_role": "final", "dependency_mode": "evidence_required",
            },
        ])
        result = {
            "total_score": 1,
            "details": [
                {"id": "core", "score_given": 0},
                {"id": "final", "score_given": 1},
            ],
        }
        updated = apply_role_weighted_scoring_policy(result, rubric)
        self.assertEqual(updated["total_score"], 0.0)
        self.assertEqual(updated["details"][1]["dependency_status"], "blocked_without_core_evidence")

    def test_process_complexity_does_not_imply_result_sufficiency(self):
        rubric = [
            {
                "id": "core", "points": 4, "answer_type": "formula",
                "role": "intermediate", "scoring_role": "core_process",
                "task_semantics": "process_dominant",
            },
            {
                "id": "final", "points": 1, "answer_type": "judgement",
                "role": "final", "scoring_role": "final",
                "task_semantics": "process_dominant",
            },
        ]
        profile = infer_rubric_task_profile(rubric, 5)
        self.assertFalse(profile["final_answer_weight_high"])
        self.assertEqual(profile["result_sufficiency_ratio"], 0.0)

    def test_explicit_result_sufficiency_remains_high_weight(self):
        rubric = [{
            "id": "final", "points": 5, "answer_type": "base_number",
            "role": "final", "task_semantics": "result_sufficient",
            "scoring_policy": "final_sufficient_partial_credit",
        }]
        profile = infer_rubric_task_profile(rubric, 5)
        self.assertTrue(profile["final_answer_weight_high"])
        self.assertEqual(profile["result_sufficiency_ratio"], 1.0)

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

    def test_split_child_cannot_invent_binary_answer_literal(self):
        original = prepare_rubric_semantic_contract([{
            "id": "step_1",
            "item": "完成地址映射，比较标记并说明是否命中",
            "points": 5,
            "standard_answer_text": "地址2D07FFH的标记为0101 1010B，因此未命中",
            "task_semantics": "process_dominant",
            "process_complexity": "complex",
        }])
        refined = [
            {
                "id": "step_1_support", "parent_id": "step_1",
                "item": "提取标记", "points": 1.5,
                "standard_answer_text": "标记为00101101B",
                "scoring_role": "support_process",
            },
            {
                "id": "step_1_core", "parent_id": "step_1",
                "item": "比较标记", "points": 2.5,
                "standard_answer_text": "01011010B与Cache标记不一致",
                "scoring_role": "core_process",
            },
            {
                "id": "step_1_final", "parent_id": "step_1",
                "item": "给出结论", "points": 1.0,
                "standard_answer_text": "未命中",
                "scoring_role": "final",
            },
        ]
        valid, errors = validate_refined_rubric(original, refined, 5)
        self.assertFalse(valid)
        self.assertTrue(any("unsupported answer literals" in error for error in errors))

    def test_split_child_accepts_regrouped_parent_binary_literal(self):
        original = prepare_rubric_semantic_contract([{
            "id": "step_1",
            "item": "完成地址映射，比较标记并说明是否命中",
            "points": 5,
            "standard_answer_text": "地址2D07FFH的标记为0101 1010B，因此未命中",
            "task_semantics": "process_dominant",
            "process_complexity": "complex",
        }])
        refined = [
            {
                "id": "step_1_support", "parent_id": "step_1",
                "item": "提取标记", "points": 1.5,
                "standard_answer_text": "标记为01011010B",
                "scoring_role": "support_process",
            },
            {
                "id": "step_1_core", "parent_id": "step_1",
                "item": "比较标记", "points": 2.5,
                "standard_answer_text": "0101 1010B与Cache标记不一致",
                "scoring_role": "core_process",
            },
            {
                "id": "step_1_final", "parent_id": "step_1",
                "item": "给出结论", "points": 1.0,
                "standard_answer_text": "未命中",
                "scoring_role": "final",
            },
        ]
        valid, errors = validate_refined_rubric(original, refined, 5)
        self.assertTrue(valid, errors)

    def test_split_final_child_cannot_reverse_parent_conclusion(self):
        original = prepare_rubric_semantic_contract([{
            "id": "step_1",
            "item": "完成映射并说明是否命中",
            "points": 5,
            "standard_answer_text": "标记不一致，因此未命中",
            "task_semantics": "process_dominant",
            "process_complexity": "complex",
        }])
        refined = [
            {
                "id": "step_1_support", "parent_id": "step_1",
                "item": "映射", "points": 1.5,
                "standard_answer_text": "完成映射",
                "scoring_role": "support_process",
            },
            {
                "id": "step_1_core", "parent_id": "step_1",
                "item": "比较", "points": 2.5,
                "standard_answer_text": "标记不一致",
                "scoring_role": "core_process",
            },
            {
                "id": "step_1_final", "parent_id": "step_1",
                "item": "结论", "points": 1.0,
                "standard_answer_text": "命中",
                "scoring_role": "final",
            },
        ]
        valid, errors = validate_refined_rubric(original, refined, 5)
        self.assertFalse(valid)
        self.assertTrue(any("changed parent conclusion" in error for error in errors))

    def test_candidate_replay_requires_teacher_score_noninferiority(self):
        accepted = assess_candidate_replay([
            {"baseline_score": 5, "candidate_score": 5.1, "teacher_score": 5},
            {"baseline_score": 3, "candidate_score": 3.1, "teacher_score": 3},
        ], 10)
        rejected = assess_candidate_replay([
            {"baseline_score": 5, "candidate_score": 8, "teacher_score": 5},
            {"baseline_score": 3, "candidate_score": 6, "teacher_score": 3},
        ], 10)
        self.assertTrue(accepted["accepted"])
        self.assertFalse(rejected["accepted"])
        self.assertEqual(rejected["reason"], "severe_sample_regression")

    def test_rejected_paired_replay_can_formally_select_unchanged_baseline(self):
        from rubric_semantics import (
            SEMANTIC_MODE_CALIBRATED_BASELINE,
            candidate_replay_supports_baseline_selection,
            manifest_allows_unchanged_baseline,
        )

        report = assess_candidate_replay([
            {"baseline_score": 5, "candidate_score": 2, "teacher_score": 6},
            {"baseline_score": 8, "candidate_score": 8, "teacher_score": 9},
        ], 10)
        self.assertTrue(candidate_replay_supports_baseline_selection(report))
        manifest = {
            "semantic_validation_mode": SEMANTIC_MODE_CALIBRATED_BASELINE,
            "selected_variant": "baseline",
            "decomposition_deferred": True,
            "candidate_replay": report,
        }
        self.assertTrue(manifest_allows_unchanged_baseline(manifest))
        manifest["decomposition_deferred"] = False
        self.assertFalse(manifest_allows_unchanged_baseline(manifest))

        diagnostic_manifest = {
            "semantic_validation_mode": "noninferiority_baseline_fallback",
            "semantic_policy_validated": True,
            "selected_variant": "baseline",
            "decomposition_deferred": True,
            "fallback_reason": "candidate_noninferiority_rejected",
        }
        self.assertFalse(
            manifest_allows_unchanged_baseline(diagnostic_manifest)
        )
        self.assertTrue(
            manifest_allows_unchanged_baseline(
                diagnostic_manifest,
                allow_diagnostic_fallback=True,
            )
        )

    def test_noninferiority_fallback_allows_only_unchanged_coarse_baseline(self):
        baseline = [{
            "id": "step_1",
            "parent_id": "step_1",
            "parent_points": 5,
            "item": "compute the result and determine overflow",
            "points": 5,
            "standard_answer_text": "8EH; overflow",
            "scoring_policy": "additive_split",
            "split_policy": "allow_semantic_split",
        }]

        strict_valid, strict_errors = validate_refined_rubric(
            baseline, baseline, 5
        )
        fallback_valid, fallback_errors = validate_refined_rubric(
            baseline,
            baseline,
            5,
            allow_unchanged_baseline=True,
        )

        changed = [dict(baseline[0], standard_answer_text="B7H; no overflow")]
        changed_valid, changed_errors = validate_refined_rubric(
            baseline,
            changed,
            5,
            allow_unchanged_baseline=True,
        )

        self.assertFalse(strict_valid)
        self.assertTrue(any("at least 2 scoring items" in e for e in strict_errors))
        self.assertTrue(fallback_valid, fallback_errors)
        self.assertFalse(changed_valid)
        self.assertTrue(changed_errors)


if __name__ == "__main__":
    unittest.main()
