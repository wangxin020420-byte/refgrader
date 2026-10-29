import unittest

from label_free_risk_candidates import candidate_memberships, evidence_availability_risk, item_score_range_risk
from label_free_a3wa import route_membership


class CandidateRiskTests(unittest.TestCase):
    def test_verified_blank_differs_from_unresolved_missing(self):
        limits = {"a": 3, "b": 7}
        blank = {item: {"state": "verified_absent", "source": "human_image_check",
                        "description": "answer region inspected"} for item in limits}
        self.assertEqual(evidence_availability_risk(blank, limits), 0.0)
        unresolved = {item: {"state": "unlocated", "source": "extractor",
                             "description": "location not confirmed"} for item in limits}
        self.assertEqual(evidence_availability_risk(unresolved, limits), 1.0)
        unresolved["a"] = blank["a"]
        self.assertAlmostEqual(evidence_availability_risk(unresolved, limits), 0.7)

    def test_raw_missing_and_unproven_status_are_rejected(self):
        for entry in ({"state": "missing", "source": "extractor", "description": "none"},
                      {"state": "verified_absent", "source": "", "description": "none"},
                      {"state": "verified_absent", "source": "human", "description": ""}):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                evidence_availability_risk({"a": entry}, {"a": 1})

    def test_item_disagreement_can_cancel_in_total(self):
        probes = [{"a": {"score": 3}, "b": {"score": 0}},
                  {"a": {"score": 0}, "b": {"score": 3}}]
        self.assertEqual(item_score_range_risk(probes, {"a": 3, "b": 3}), 1.0)

    def test_observed_case_and_routes(self):
        scores = [[1, 1, 3, 4, 0, 3], [2, 1, 2, 2, 0, 2], [1, 2, 3, 4, 1, 5]]
        limits = {str(i): x for i, x in enumerate([4, 4, 4, 4, 4, 5])}
        probes = [{str(i): {"score": s} for i, s in enumerate(row)} for row in scores]
        risk = item_score_range_risk(probes, limits)
        self.assertAlmostEqual(risk, 0.36)
        memberships = candidate_memberships({"U_E": 0, "U_S": risk, "U_R": .12}, valid_probe_count=3)
        self.assertAlmostEqual(memberships["noncompensatory_max"], .64)
        loss = dict(lambda1=1, lambda2=1, mu1=1, mu2=1)
        self.assertEqual(route_membership(memberships["noncompensatory_max"], m=.5, loss_parameters=loss)["route"], "BND")

    def test_reject_incomplete_or_invalid_inputs(self):
        for probes in ([{"a": {"score": 0}}], [{"a": {"score": 0}}, {}],
                       [{"a": {"score": 0}}, {"a": {"score": 2}}],
                       [{"a": {"score": 0}}, {"a": {"score": float('nan')}}]):
            with self.subTest(probes=probes), self.assertRaises(ValueError):
                item_score_range_risk(probes, {"a": 1})
        with self.assertRaises(ValueError):
            candidate_memberships({"U_E": 0, "U_S": 0, "U_R": 0, "teacher_score": 1}, valid_probe_count=3)
        with self.assertRaises(ValueError):
            candidate_memberships({"U_E": 0, "U_S": 0, "U_R": 0}, valid_probe_count=1)


if __name__ == "__main__":
    unittest.main()
