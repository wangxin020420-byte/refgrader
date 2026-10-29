import unittest

from label_free_field_comparison import compare_partitioned_fields


class FieldComparisonTests(unittest.TestCase):
    def setUp(self):
        self.student = {"group_total": 13, "subfields": None,
                        "following_fields": [3, 8], "source_ref": "student:image"}
        self.reference = {"group_total": 13, "subfields": [3, 4, 4, 2],
                          "following_fields": [3, 8], "source_ref": "reference:image"}

    def test_total_match_does_not_infer_unwritten_subgroups(self):
        result = compare_partitioned_fields(student=self.student, reference=self.reference)
        self.assertTrue(result["group_total_match"])
        self.assertTrue(result["following_fields_match"])
        self.assertIsNone(result["subfields_match"])
        self.assertIsNone(result["student_subfields_sum_matches_total"])
        self.assertTrue(result["reference_subfields_sum_matches_total"])
        self.assertFalse(result["score_change_authorized"])
        self.assertTrue(result["requires_human_review"])

    def test_order_and_internal_consistency_are_distinct(self):
        student = {**self.student, "subfields": [4, 3, 4, 2], "following_fields": [8, 3]}
        result = compare_partitioned_fields(student=student, reference=self.reference)
        self.assertTrue(result["student_subfields_sum_matches_total"])
        self.assertFalse(result["subfields_match"])
        self.assertFalse(result["following_fields_match"])

    def test_rejects_unexpected_labels_and_invalid_fields(self):
        for student in ({**self.student, "teacher_score": 2}, {**self.student, "source_ref": ""},
                        {**self.student, "following_fields": [True]}, {**self.student, "subfields": []}):
            with self.subTest(student=student), self.assertRaises(ValueError):
                compare_partitioned_fields(student=student, reference=self.reference)


if __name__ == "__main__":
    unittest.main()
