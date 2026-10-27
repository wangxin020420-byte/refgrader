import unittest

from label_free_boundary import assess_boundary_evidence


class BoundaryEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.rubric = [{"id":"s1", "points":2, "answer_type":"base_number", "standard_answer_text":"42H"}]

    def test_exact_source_span_and_canonical_match_do_not_automatically_change_score(self):
        result = assess_boundary_evidence(answer_text="answer: 42H", rubric=self.rubric, claim={"item_id":"s1", "start":8, "end":11})
        self.assertTrue(result["canonical_match"])
        self.assertFalse(result["criterion_relevance_verified"])
        self.assertFalse(result["score_change_authorized"])
        self.assertTrue(result["requires_human_review"])
        self.assertNotIn("42H", str(result))

    def test_canonical_mismatch_and_unknown_also_defer(self):
        for text, expected in (("43H", False), ("?", None)):
            with self.subTest(text=text):
                result = assess_boundary_evidence(answer_text=text, rubric=self.rubric, claim={"item_id":"s1", "start":0, "end":len(text)})
                self.assertIs(result["canonical_match"], expected)
                self.assertTrue(result["requires_human_review"])

    def test_rejects_invalid_spans_unknown_items_and_teacher_fields(self):
        for claim in (
            {"item_id":"s1", "start":0, "end":5},
            {"item_id":"other", "start":0, "end":3},
            {"item_id":"s1", "start":False, "end":3},
            {"item_id":"s1", "start":0, "end":3, "teacher_score":2},
        ):
            with self.subTest(claim=claim), self.assertRaises(ValueError):
                assess_boundary_evidence(answer_text="42H", rubric=self.rubric, claim=claim)


if __name__ == "__main__":
    unittest.main()
