import json
import unittest

from canonicalizers import build_canonical_grading_context


class BaseNumberCanonicalizerTests(unittest.TestCase):
    @staticmethod
    def compare(student, standard, implicit_bases=None):
        item = {
            "id": "value",
            "item": "base number",
            "points": 1,
            "answer_type": "base_number",
            "canonicalization": {
                "type": "base_number",
                "implicit_bases": implicit_bases or [],
            },
            "standard_answer_text": standard,
        }
        context = build_canonical_grading_context(
            json.dumps({"value": student}, ensure_ascii=False),
            json.dumps([item], ensure_ascii=False),
        )
        return context["items"][0]["comparison"]

    def test_explicit_hex_matches(self):
        self.assertTrue(self.compare("37H", "37H")["match"])

    def test_binary_and_hex_are_equivalent(self):
        self.assertTrue(self.compare("110111B", "37H")["match"])
        self.assertTrue(self.compare("110111₂", "37H")["match"])

    def test_bare_number_requires_explicit_rubric_policy(self):
        self.assertIsNone(self.compare("37", "37H")["match"])
        self.assertTrue(self.compare("37", "37H", [16])["match"])

    def test_multiple_distinct_values_require_semantic_resolution(self):
        comparison = self.compare("从34H取出37H", "37H")
        self.assertEqual(comparison["status"], "ambiguous")
        self.assertIsNone(comparison["match"])

    def test_multiple_equivalent_representations_remain_deterministic(self):
        comparison = self.compare("110111B = 37H", "37H")
        self.assertEqual(comparison["status"], "match")
        self.assertTrue(comparison["match"])


class StructuredFieldsCanonicalizerTests(unittest.TestCase):
    @staticmethod
    def compare(student):
        item = {
            "id": "address_format",
            "item": "写出主存地址字段结构",
            "points": 2,
            "answer_type": "structured_fields",
            "canonicalization": {
                "type": "structured_fields",
                "ordered": True,
                "fields": [
                    {
                        "name": "tag",
                        "aliases": ["字块标记", "标记字段"],
                        "required": True,
                    },
                    {
                        "name": "set",
                        "aliases": ["Cache组号", "组号"],
                        "required": True,
                    },
                    {
                        "name": "offset",
                        "aliases": ["块内偏移", "偏移"],
                        "required": True,
                    },
                ],
            },
            "standard_answer_text": "字块标记 8 位 | Cache组号 4 位 | 块内偏移 11 位",
        }
        context = build_canonical_grading_context(
            json.dumps({"address_format": student}, ensure_ascii=False),
            json.dumps([item], ensure_ascii=False),
        )
        return context["items"][0]["comparison"]

    def test_aliases_and_all_fields_match(self):
        result = self.compare("标记字段 8位 | 组号 4位 | 偏移 11位")
        self.assertTrue(result["match"])
        self.assertEqual(result["field_match_ratio"], 1.0)

    def test_one_matching_field_does_not_match_whole_structure(self):
        result = self.compare("主存块号 12位 | Cache组号 4位 | 块内偏移 11位")
        self.assertFalse(result["match"])
        self.assertEqual(result["missing_fields"], ["tag"])

    def test_wrong_field_order_is_not_exact_match(self):
        result = self.compare("块内偏移 11位 | Cache组号 4位 | 字块标记 8位")
        self.assertFalse(result["match"])
        self.assertFalse(result["order_match"])


if __name__ == "__main__":
    unittest.main()
