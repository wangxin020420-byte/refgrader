import unittest

from label_free_action import decide_action


class LabelFreeActionTests(unittest.TestCase):
    def setUp(self):
        self.loss = {"lambda1": 5, "lambda2": 1, "mu1": 3, "mu2": 7}

    def run_case(self, risk, score=4):
        return decide_action(
            baseline_score=score, max_score=10,
            risks={"U_E": risk, "U_S": risk, "U_R": risk},
            valid_probe_count=3, m=0.5, loss_parameters=self.loss,
        )

    def test_positive_region_is_automatically_accepted(self):
        result = self.run_case(0.05)
        self.assertEqual(result["route"], "POS")
        self.assertEqual(result["automatic_score"], 4)
        self.assertFalse(result["requires_human_review"])

    def test_boundary_and_negative_keep_provisional_scores_without_auto_final(self):
        for risk, route in ((0.4, "BND"), (0.8, "NEG")):
            with self.subTest(route=route):
                result = self.run_case(risk, score=4)
                self.assertEqual(result["route"], route)
                self.assertEqual(result["provisional_score"], 4)
                self.assertIsNone(result["automatic_score"])
                self.assertTrue(result["requires_human_review"])

    def test_rejects_incomplete_scores_and_invalid_input_instead_of_auto_accepting(self):
        for score in (-1, 11, float("nan")):
            with self.subTest(score=score), self.assertRaises(ValueError):
                self.run_case(0.1, score=score)
        with self.assertRaises(ValueError):
            decide_action(baseline_score=4, max_score=10, risks={"U_E":0, "U_S":0, "U_R":0}, valid_probe_count=1, m=0.5, loss_parameters=self.loss)

    def test_forbids_unexpected_teacher_score_argument(self):
        with self.assertRaises(TypeError):
            decide_action(baseline_score=4, max_score=10, risks={"U_E":0, "U_S":0, "U_R":0}, valid_probe_count=3, m=0.5, loss_parameters=self.loss, teacher_score=4)


if __name__ == "__main__":
    unittest.main()
