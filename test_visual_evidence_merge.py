import unittest

from step4_vlm_grader import (
    CSBENCH_EXTRACTION_SCHEMA_VERSION,
    _diagram_checklist,
    _is_concrete_visual_fact,
    _merge_visual_fallback_facts,
    _table_checklist,
    _value_needs_visual_fallback,
    detect_visual_placeholder,
)


class VisualEvidenceMergeTests(unittest.TestCase):
    def setUp(self):
        self.checklist = [
            {"id": "step_1", "instruction": "extract process text"},
            {"id": "step_2", "instruction": "extract mask values"},
            {"id": "step_3", "instruction": "extract execution path"},
        ]
        self.rubric = [
            {
                "id": "step_1",
                "item": "简述CPU响应中断的基本过程",
                "answer_type": "table_entry",
                "canonicalization": "table",
                "evidence_source": "text_and_ocr",
            },
            {
                "id": "step_2",
                "item": "设计满足指定处理顺序的屏蔽字",
                "answer_type": "bit_vector",
                "canonicalization": "bit_vector",
                "evidence_source": "transcription",
            },
            {
                "id": "step_3",
                "item": "画出CPU执行用户程序和处理中断的轨迹图",
                "answer_type": "table_entry",
                "canonicalization": "table",
                "evidence_source": "ocr_table",
            },
        ]

    def test_visual_reference_phrases_trigger_and_are_replaceable(self):
        values = (
            "（真值表如表所示）",
            "（中断执行时序图如图所示）",
            "答案见表格",
        )
        for value in values:
            with self.subTest(value=value):
                self.assertTrue(detect_visual_placeholder(value))
                self.assertTrue(_value_needs_visual_fallback(value))
                self.assertFalse(_is_concrete_visual_fact(value))

    def test_concrete_visual_facts_replace_only_placeholders(self):
        base = {
            "step_1": "关中断，保存断点",
            "step_2": "（真值表如表所示）",
            "step_3": "（中断执行时序图如图所示）",
        }
        fallback = {
            "step_1": "不同的过程文本",
            "step_2": "C=1011; B=1001; A=0000; D=0001",
            "step_3": "用户程序→C→B→D→A",
        }

        merged, recovered, conflicts = _merge_visual_fallback_facts(
            base,
            fallback,
            ["step_1", "step_2", "step_3"],
        )

        self.assertEqual(merged["step_1"], base["step_1"])
        self.assertEqual(merged["step_2"], fallback["step_2"])
        self.assertEqual(merged["step_3"], fallback["step_3"])
        self.assertEqual(recovered, ["step_2", "step_3"])
        self.assertEqual(conflicts, ["step_1"])

    def test_semantic_routing_overrides_stale_table_metadata(self):
        table_items = _table_checklist(self.checklist, self.rubric)
        diagram_items = _diagram_checklist(self.checklist, self.rubric)

        self.assertEqual([item["id"] for item in table_items], ["step_2"])
        self.assertEqual([item["id"] for item in diagram_items], ["step_3"])

    def test_visual_cache_contract_is_invalidated(self):
        self.assertEqual(CSBENCH_EXTRACTION_SCHEMA_VERSION, 4)


if __name__ == "__main__":
    unittest.main()
