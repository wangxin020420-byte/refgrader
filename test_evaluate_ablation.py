import unittest

from evaluate import get_score_value, summarize_score_transition


class EvaluateAblationTests(unittest.TestCase):
    def test_missing_core_is_not_inferred_from_final_score(self):
        record = {"final_calibrated_score": 4.0}
        self.assertIsNone(get_score_value(record, "three_way_core_score"))

    def test_transition_separates_core_and_residual_gain(self):
        records = [
            {"teacher": 5.0, "avg": 3.0, "core": 4.0, "final": 5.0},
            {"teacher": 2.0, "avg": 2.0, "core": 2.0, "final": 1.0},
        ]
        core = summarize_score_transition(records, "avg", "core")
        residual = summarize_score_transition(records, "core", "final")

        self.assertAlmostEqual(core["baseline_mae"], 1.0)
        self.assertAlmostEqual(core["candidate_mae"], 0.5)
        self.assertAlmostEqual(core["mean_gain"], 0.5)
        self.assertEqual((core["improved"], core["unchanged"], core["worsened"]), (1, 1, 0))

        self.assertAlmostEqual(residual["baseline_mae"], 0.5)
        self.assertAlmostEqual(residual["candidate_mae"], 0.5)
        self.assertAlmostEqual(residual["mean_gain"], 0.0)
        self.assertEqual(
            (residual["improved"], residual["unchanged"], residual["worsened"]),
            (1, 0, 1),
        )


if __name__ == "__main__":
    unittest.main()
