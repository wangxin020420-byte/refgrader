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


if __name__ == "__main__":
    unittest.main()
