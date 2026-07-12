import unittest

from rubric_semantics import (
    prepare_rubric_semantic_contract,
    validate_refined_rubric,
)


class RubricSemanticContractTests(unittest.TestCase):
    def test_co1_atomic_result_cannot_be_split(self):
        original = prepare_rubric_semantic_contract([
            {
                "id": "step_1",
                "item": "正确计算并得出操作数内容",
                "points": 5,
                "role": "final",
                "standard_answer_text": "37H",
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
