import unittest

from step4_vlm_grader import normalize_stage2_score_contract


class Stage2ScoreContractTests(unittest.TestCase):
    def setUp(self):
        self.rubric = [
            {"id": "step_1", "points": 2.0},
            {"id": "step_2", "points": 3.0},
        ]

    def test_existing_total_score_is_preserved(self):
        result = {
            "total_score": 4.0,
            "details": [{"id": "step_1", "score_given": 1.0}],
        }

        normalized = normalize_stage2_score_contract(result, self.rubric)

        self.assertIs(normalized, result)
        self.assertEqual(normalized["total_score"], 4.0)
        self.assertNotIn("total_score_reconstructed_from_details", normalized)

    def test_complete_item_ledger_reconstructs_missing_total(self):
        result = {
            "details": [
                {"id": "step_1", "score_given": 1.5},
                {"id": "step_2", "score_given": 2.0},
            ]
        }

        normalized = normalize_stage2_score_contract(result, self.rubric)

        self.assertEqual(normalized["total_score"], 3.5)
        self.assertTrue(normalized["total_score_reconstructed_from_details"])

    def test_incomplete_or_invalid_item_ledger_is_rejected(self):
        invalid_results = [
            {"details": [{"id": "step_1", "score_given": 1.0}]},
            {
                "details": [
                    {"id": "step_1", "score_given": 1.0},
                    {"id": "step_1", "score_given": 1.0},
                    {"id": "step_2", "score_given": 1.0},
                ]
            },
            {
                "details": [
                    {"id": "step_1", "score_given": 1.0},
                    {"id": "step_2", "score_given": 1.0},
                    {"id": "invented", "score_given": 0.0},
                ]
            },
            {
                "details": [
                    {"id": "step_1", "score_given": 2.5},
                    {"id": "step_2", "score_given": 1.0},
                ]
            },
            {
                "details": [
                    {"id": "step_1", "score_given": "not-a-number"},
                    {"id": "step_2", "score_given": 1.0},
                ]
            },
        ]

        for result in invalid_results:
            with self.subTest(result=result):
                self.assertIsNone(
                    normalize_stage2_score_contract(result, self.rubric)
                )

if __name__ == "__main__":
    unittest.main()
