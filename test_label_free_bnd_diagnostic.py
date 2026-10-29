import unittest

from label_free_bnd_diagnostic import diagnose_bnd


class BNDDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.rubric = [{"id": "s1", "points": 2, "answer_type": "base_number", "standard_answer_text": "42H"},
                       {"id": "s2", "points": 3, "answer_type": "base_number", "standard_answer_text": "10H"}]
        self.decision = {"route": "BND", "provisional_score": 1, "automatic_score": None}

    def test_matching_excerpt_does_not_authorize_or_narrow(self):
        obs = {"facts": {"s1": {"status": "present", "evidence": "42H"},
                         "s2": {"status": "missing", "evidence": ""}}}
        result = diagnose_bnd(decision=self.decision, observation=obs, rubric=self.rubric)
        self.assertTrue(result["checks"][0]["canonical_match"])
        self.assertFalse(result["checks"][0]["criterion_relevance_verified"])
        self.assertEqual(result["projection"]["score_interval"], {"lower": 0, "upper": 5})
        self.assertEqual(result["projection"]["unresolved_items"], ["s1", "s2"])
        self.assertEqual(result["projection"]["proposed_score"], 1)
        self.assertFalse(result["score_change_authorized"])
        self.assertTrue(result["requires_human_review"])

    def test_non_bnd_and_wrong_item_set_rejected(self):
        with self.assertRaises(ValueError):
            diagnose_bnd(decision={"route": "POS"}, observation={}, rubric=self.rubric)
        with self.assertRaises(ValueError):
            diagnose_bnd(decision=self.decision, observation={"facts": {}}, rubric=self.rubric)

    def test_prioritizes_item_level_disagreement_without_narrowing_constraints(self):
        obs = {"facts": {"s1": {"status": "present", "evidence": "42H"},
                         "s2": {"status": "present", "evidence": "10H"}},
               "probes": [{"s1": {"score": 1}, "s2": {"score": 3}},
                          {"s1": {"score": 2}, "s2": {"score": 0}},
                          {"s1": {"score": 1}, "s2": {"score": 3}}]}
        result = diagnose_bnd(decision=self.decision, observation=obs, rubric=self.rubric)
        self.assertEqual([x["item_id"] for x in result["review_priority"]], ["s2", "s1"])
        self.assertEqual(result["review_priority"][0]["item_normalized_range"], 1)
        self.assertFalse(result["priority_is_error_probability"])
        self.assertEqual(result["projection"]["score_interval"], {"lower": 0, "upper": 5})
        self.assertFalse(result["score_change_authorized"])


if __name__ == "__main__":
    unittest.main()
