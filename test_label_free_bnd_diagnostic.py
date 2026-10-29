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


if __name__ == "__main__":
    unittest.main()
