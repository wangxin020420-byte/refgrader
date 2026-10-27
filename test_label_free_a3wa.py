import unittest

from label_free_a3wa import (
    equal_weight_membership,
    route_membership,
    select_m_by_unlabeled_information_loss,
)


class LabelFreeA3WATests(unittest.TestCase):
    def setUp(self):
        self.loss = {"lambda1": 5, "lambda2": 1, "mu1": 3, "mu2": 7}

    def test_equal_risks_and_threshold_routes(self):
        mu = equal_weight_membership({"U_E": 0.25, "U_S": 0.5, "U_R": 0.375}, valid_probe_count=3)
        self.assertAlmostEqual(mu, 0.625)
        result = route_membership(mu, m=0.5, loss_parameters=self.loss)
        self.assertAlmostEqual(result["alpha"], 5.5 / 6)
        self.assertAlmostEqual(result["beta"], 0.35)
        self.assertEqual(result["route"], "BND")
        self.assertEqual(route_membership(result["alpha"], m=0.5, loss_parameters=self.loss)["route"], "POS")
        self.assertEqual(route_membership(result["beta"], m=0.5, loss_parameters=self.loss)["route"], "NEG")

    def test_missing_probes_are_not_treated_as_stable(self):
        for count in (0, 1):
            with self.assertRaises(ValueError):
                equal_weight_membership({"U_E": 0, "U_S": 0, "U_R": 0}, valid_probe_count=count)

    def test_reject_invalid_signals_and_threshold_inputs(self):
        for value in (float("nan"), float("inf"), -0.01, 1.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                equal_weight_membership({"U_E": value, "U_S": 0, "U_R": 0}, valid_probe_count=3)
        with self.assertRaises(ValueError):
            route_membership(0.5, m=0, loss_parameters=self.loss)
        with self.assertRaises(ValueError):
            route_membership(0.5, m=0.5, loss_parameters={**self.loss, "mu1": 0})

    def test_m_selection_is_deterministic_and_uses_only_supplied_memberships(self):
        first = select_m_by_unlabeled_information_loss([0.2, 0.6, 0.9], candidates=[0.6, 0.4, 0.5], loss_parameters=self.loss)
        second = select_m_by_unlabeled_information_loss([0.2, 0.6, 0.9], candidates=[0.5, 0.4, 0.6], loss_parameters=self.loss)
        self.assertEqual(first, second)
        self.assertEqual(first["sample_count"], 3)
        self.assertEqual(first["selected"]["information_loss"], min(x["information_loss"] for x in first["candidates"]))

    def test_empty_development_data_and_unknown_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            select_m_by_unlabeled_information_loss([], candidates=[0.5], loss_parameters=self.loss)
        with self.assertRaises(ValueError):
            equal_weight_membership({"U_E": 0, "U_S": 0, "U_R": 0, "teacher_score": 2}, valid_probe_count=3)


if __name__ == "__main__":
    unittest.main()
