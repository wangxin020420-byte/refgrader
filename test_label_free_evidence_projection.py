import unittest

from label_free_evidence_projection import propose_evidence_projection
from label_free_action import decide_action


class EvidenceProjectionTests(unittest.TestCase):
    def test_bnd_proposal_never_overrides_the_defer_action(self):
        decision = decide_action(baseline_score=5, max_score=10,
            risks={"U_E": 0, "U_S": .4, "U_R": .1}, valid_probe_count=3,
            m=.5, loss_parameters={"lambda1": 1, "lambda2": 1, "mu1": 1, "mu2": 1},
            fusion="noncompensatory_max")
        proposal = propose_evidence_projection(core_score=decision["provisional_score"], item_limits={"q": 10},
            constraints={"q": {"state": "supported", "lower": 6, "upper": 8,
                               "rubric_ref": "rule:q", "evidence_ref": "image:q"}})
        self.assertEqual(decision["route"], "BND")
        self.assertIsNone(decision["automatic_score"])
        self.assertEqual(proposal["proposed_score"], 6)
        self.assertFalse(proposal["score_change_authorized"])
    def test_projection_direction_and_idempotence(self):
        c = {"q": {"state": "supported", "lower": 6, "upper": 8,
                   "rubric_ref": "rubric:q", "evidence_ref": "image:region1"}}
        for score, expected in ((5, 6), (7, 7), (9, 8)):
            with self.subTest(score=score):
                result = propose_evidence_projection(core_score=score, item_limits={"q": 10}, constraints=c)
                self.assertEqual(result["proposed_score"], expected)
                self.assertFalse(result["score_change_authorized"])
                self.assertTrue(result["requires_human_review"])
                again = propose_evidence_projection(core_score=expected, item_limits={"q": 10}, constraints=c)
                self.assertEqual(again["proposed_delta"], 0)

    def test_unknown_remains_wide_and_mixed_constraints_keep_uncertainty(self):
        c = {"a": {"state": "supported", "lower": 2, "upper": 3,
                   "rubric_ref": "rubric:a", "evidence_ref": "image:a"},
             "b": {"state": "unknown"}}
        result = propose_evidence_projection(core_score=0, item_limits={"a": 4, "b": 6}, constraints=c)
        self.assertEqual(result["score_interval"], {"lower": 2, "upper": 9})
        self.assertEqual(result["unresolved_items"], ["b"])
        self.assertFalse(result["score_change_authorized"])
        unknown = propose_evidence_projection(core_score=4, item_limits={"a": 10}, constraints={"a": {"state": "unknown"}})
        self.assertEqual(unknown["proposed_score"], 4)

    def test_invalid_or_teacher_fields_cannot_produce_a_proposal(self):
        template = {"state": "supported", "lower": 0, "upper": 2,
                    "rubric_ref": "rubric:a", "evidence_ref": "image:a"}
        for record in ({**template, "lower": 3}, {**template, "upper": 5},
                       {**template, "evidence_ref": ""}, {**template, "teacher_score": 2},
                       {"state": "unknown", "teacher_score": 2}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                propose_evidence_projection(core_score=1, item_limits={"a": 4}, constraints={"a": record})


if __name__ == "__main__":
    unittest.main()
