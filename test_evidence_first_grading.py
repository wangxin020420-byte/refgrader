import copy
import unittest
from unittest.mock import patch

from calibration_utils import (
    close_evidence_verification_contract,
    compute_evidence_verifier_directional_risks,
    normalize_evidence_verification_result,
)
from scripts.run_csbench import build_parser, grading_contract_from_args


class EvidenceFirstGradingTests(unittest.TestCase):
    def setUp(self):
        self.rubric = [
            {
                "id": "step_1",
                "points": 2.0,
                "type": "formula",
                "description": "核验公式",
            },
            {
                "id": "step_2",
                "points": 3.0,
                "type": "judgement",
                "description": "核验结论",
            },
        ]
        self.facts = {
            "step_1": "2 + 2 = 4",
            "step_2": "命中",
            "_extraction_quality": "high",
        }

    def test_verification_parser_preserves_invalid_references(self):
        raw = {
            "items": [
                {
                    "id": "step_1",
                    "evidence_ids": [
                        "step_1",
                        "missing",
                        "_extraction_quality",
                    ],
                    "evidence_support": 1.0,
                    "contradiction": False,
                    "confidence": 0.9,
                },
                {
                    "id": "step_2",
                    "evidence_ids": ["step_2"],
                    "evidence_support": 0.5,
                    "contradiction": False,
                    "confidence": 0.7,
                },
                {"id": "unknown", "evidence_ids": []},
            ]
        }

        normalized = normalize_evidence_verification_result(
            raw,
            self.facts,
            self.rubric,
        )

        self.assertEqual(normalized["schema_version"], 2)
        self.assertNotIn("total_score", normalized)
        self.assertEqual(
            normalized["items"][0]["requested_evidence_ids"],
            ["step_1", "missing", "_extraction_quality"],
        )
        self.assertEqual(
            normalized["items"][0]["invalid_evidence_ids"],
            ["missing", "_extraction_quality"],
        )
        audit = normalized["audit"]
        self.assertEqual(audit["invalid_evidence_reference_count"], 2)
        self.assertEqual(audit["unknown_item_ids"], ["unknown"])
        self.assertTrue(audit["requires_repair"])

    def test_directional_risks_use_immutable_scores(self):
        probes = [
            {
                "total_score": 2.5,
                "details": [
                    {"id": "step_1", "score_given": 1.0},
                    {"id": "step_2", "score_given": 1.5},
                ],
            }
        ]
        original = copy.deepcopy(probes)
        verification = normalize_evidence_verification_result(
            {
                "items": [
                    {
                        "id": "step_1",
                        "evidence_ids": ["step_1"],
                        "evidence_support": 1.0,
                        "contradiction": False,
                        "confidence": 0.9,
                    },
                    {
                        "id": "step_2",
                        "evidence_ids": [],
                        "evidence_support": 0.0,
                        "contradiction": False,
                        "confidence": 0.8,
                    },
                ]
            },
            self.facts,
            self.rubric,
        )

        risks = compute_evidence_verifier_directional_risks(
            self.rubric,
            probes,
            verification,
        )

        self.assertEqual(probes, original)
        self.assertAlmostEqual(risks["R_under"], 0.2)
        self.assertAlmostEqual(risks["R_over"], 0.3)
        self.assertEqual(risks["source"], "independent_post_scoring_verifier")
        self.assertEqual(risks["scoring_effect"], "none")
        self.assertEqual(risks["route_effect"], "none")

    def test_deterministic_closure_is_fail_closed_and_auditable(self):
        normalized = normalize_evidence_verification_result(
            {
                "items": [
                    {
                        "id": "step_1",
                        "evidence_ids": ["invented"],
                        "evidence_support": 1.0,
                        "contradiction": True,
                        "confidence": 0.9,
                    },
                    {
                        "id": "step_2",
                        "evidence_ids": ["step_2"],
                        "evidence_support": 0.5,
                        "contradiction": False,
                        "confidence": 0.8,
                    },
                ]
            },
            self.facts,
            self.rubric,
        )

        closed = close_evidence_verification_contract(normalized)

        degraded, valid = closed["items"]
        self.assertEqual(degraded["evidence_ids"], [])
        self.assertEqual(degraded["evidence_support"], 0.0)
        self.assertFalse(degraded["contradiction"])
        self.assertEqual(degraded["confidence"], 0.0)
        self.assertTrue(degraded["evidence_contract_complete"])
        self.assertFalse(degraded["risk_eligible"])
        self.assertEqual(valid["evidence_ids"], ["step_2"])
        self.assertTrue(valid["risk_eligible"])
        audit = closed["audit"]
        self.assertTrue(audit["preclosure_requires_repair"])
        self.assertEqual(audit["deterministic_degraded_item_ids"], ["step_1"])
        self.assertEqual(audit["protocol_contract_completeness"], 1.0)
        self.assertEqual(audit["semantic_evidence_coverage"], 0.5)
        self.assertFalse(audit["requires_repair"])

    def test_fail_closed_item_does_not_create_directional_risk(self):
        normalized = normalize_evidence_verification_result(
            {
                "items": [
                    {
                        "id": "step_1",
                        "evidence_ids": ["invented"],
                        "evidence_support": 1.0,
                        "contradiction": False,
                        "confidence": 1.0,
                    },
                    {
                        "id": "step_2",
                        "evidence_ids": ["step_2"],
                        "evidence_support": 0.5,
                        "contradiction": False,
                        "confidence": 0.8,
                    },
                ]
            },
            self.facts,
            self.rubric,
        )
        closed = close_evidence_verification_contract(normalized)
        risks = compute_evidence_verifier_directional_risks(
            self.rubric,
            [{
                "details": [
                    {"id": "step_1", "score_given": 2.0},
                    {"id": "step_2", "score_given": 1.5},
                ]
            }],
            closed,
        )

        first = risks["item_diagnostics"][0]
        self.assertFalse(first["risk_eligible"])
        self.assertEqual(first["undercredit_gap"], 0.0)
        self.assertEqual(first["overcredit_gap"], 0.0)
        self.assertEqual(risks["risk_eligible_points_ratio"], 0.6)
        self.assertAlmostEqual(risks["R_over"], 0.0)

    def test_unknown_duplicate_and_missing_items_are_closed_generically(self):
        normalized = normalize_evidence_verification_result(
            {
                "items": [
                    {
                        "id": "step_1",
                        "evidence_ids": ["step_1"],
                        "evidence_support": 1.0,
                        "contradiction": False,
                        "confidence": 0.9,
                    },
                    {
                        "id": "step_1",
                        "evidence_ids": ["step_2"],
                        "evidence_support": 0.0,
                        "contradiction": True,
                        "confidence": 0.2,
                    },
                    {
                        "id": "unknown",
                        "evidence_ids": ["step_2"],
                        "evidence_support": 1.0,
                        "contradiction": False,
                        "confidence": 1.0,
                    },
                ]
            },
            self.facts,
            self.rubric,
        )

        closed = close_evidence_verification_contract(normalized)

        self.assertEqual([item["id"] for item in closed["items"]], [
            "step_1",
            "step_2",
        ])
        self.assertTrue(closed["items"][0]["risk_eligible"])
        self.assertFalse(closed["items"][1]["risk_eligible"])
        audit = closed["audit"]
        self.assertEqual(
            audit["deterministic_discarded_unknown_item_ids"],
            ["unknown"],
        )
        self.assertEqual(
            audit["deterministic_discarded_duplicate_item_ids"],
            ["step_1"],
        )
        self.assertEqual(audit["deterministic_degraded_item_ids"], ["step_2"])
        self.assertTrue(audit["deterministic_closure_applied"])

    def test_cli_defaults_off_and_records_v2_contract(self):
        parser = build_parser()
        default_args = parser.parse_args(["grade", "CO_1"])
        enabled_args = parser.parse_args(
            ["grade", "CO_1", "--evidence-first-grading"]
        )

        self.assertFalse(default_args.evidence_first_grading)
        self.assertTrue(enabled_args.evidence_first_grading)
        self.assertEqual(
            grading_contract_from_args(enabled_args),
            {
                "evidence_first_grading": True,
                "evidence_first_schema_version": 2,
                "evidence_verifier": "independent_post_scoring",
                "directional_credit_risk": "diagnostic_only",
                "scoring_effect": "none",
                "route_effect": "none",
            },
        )

    def test_stage2_scoring_prompt_is_identical_when_diagnostic_is_enabled(self):
        from step4_vlm_grader import stage2_logic_grading

        captured = []

        def fake_call(messages, **_kwargs):
            captured.append(messages[0]["content"])
            return "{}"

        with patch("step4_vlm_grader.call_text_model", side_effect=fake_call):
            stage2_logic_grading("{}", "[]")
            stage2_logic_grading(
                "{}",
                "[]",
                evidence_first_grading=True,
            )

        self.assertEqual(captured[0], captured[1])
        self.assertNotIn("Evidence-First Experimental Contract", captured[1])

    def test_verifier_prompt_is_score_blind_and_uses_explicit_ledger(self):
        from step4_vlm_grader import build_evidence_verification_prompt

        prompt = build_evidence_verification_prompt(self.facts, self.rubric)

        self.assertIn('"evidence_id": "step_1"', prompt)
        self.assertNotIn('"evidence_id": "_extraction_quality"', prompt)
        self.assertNotIn("teacher_score", prompt)
        self.assertNotIn("score_given", prompt)
        self.assertNotIn("total_score", prompt)

    def test_repair_prompt_uses_whitelist_without_replaying_raw_response(self):
        from step4_vlm_grader import build_evidence_verification_prompt

        prompt = build_evidence_verification_prompt(
            self.facts,
            self.rubric,
            repair_context={
                "audit": {
                    "invalid_evidence_ids": ["step_1_child"],
                    "requires_repair": True,
                },
                "previous_response": "DO_NOT_REPLAY_THIS_RESPONSE",
            },
        )

        self.assertIn('evidence_ids \u552f\u4e00\u767d\u540d\u5355', prompt)
        self.assertIn('["step_1", "step_2"]', prompt)
        self.assertIn('step_1_child', prompt)
        self.assertNotIn("DO_NOT_REPLAY_THIS_RESPONSE", prompt)

    def test_invalid_first_pass_gets_one_repair_call(self):
        from step4_vlm_grader import run_independent_evidence_verification

        invalid = (
            "first raw",
            {
                "items": [
                    {
                        "id": "step_1",
                        "evidence_ids": ["invented"],
                        "evidence_support": 1.0,
                        "contradiction": False,
                        "confidence": 0.9,
                    }
                ]
            },
        )
        repaired = (
            "repaired raw",
            {
                "items": [
                    {
                        "id": "step_1",
                        "evidence_ids": ["step_1"],
                        "evidence_support": 1.0,
                        "contradiction": False,
                        "confidence": 0.9,
                    },
                    {
                        "id": "step_2",
                        "evidence_ids": ["step_2"],
                        "evidence_support": 1.0,
                        "contradiction": False,
                        "confidence": 0.9,
                    },
                ]
            },
        )

        with patch(
            "step4_vlm_grader.stage2_evidence_verification",
            side_effect=[invalid, repaired],
        ) as verifier:
            result = run_independent_evidence_verification(
                self.facts,
                self.rubric,
            )

        self.assertEqual(verifier.call_count, 2)
        self.assertTrue(result["audit"]["repair_attempted"])
        self.assertTrue(result["audit"]["repair_succeeded"])
        self.assertEqual(
            result["audit"]["first_pass"][
                "invalid_evidence_reference_count"
            ],
            1,
        )
        self.assertEqual(
            result["first_pass_items"][0]["invalid_evidence_ids"],
            ["invented"],
        )

    def test_failed_model_repair_is_closed_without_third_call(self):
        from step4_vlm_grader import run_independent_evidence_verification

        invalid = (
            "invalid raw",
            {
                "items": [
                    {
                        "id": "step_1",
                        "evidence_ids": ["invented"],
                        "evidence_support": 1.0,
                        "contradiction": True,
                        "confidence": 1.0,
                    },
                    {
                        "id": "step_2",
                        "evidence_ids": ["step_2"],
                        "evidence_support": 1.0,
                        "contradiction": False,
                        "confidence": 0.9,
                    },
                ]
            },
        )
        with patch(
            "step4_vlm_grader.stage2_evidence_verification",
            side_effect=[invalid, invalid],
        ) as verifier:
            result = run_independent_evidence_verification(
                self.facts,
                self.rubric,
            )

        self.assertEqual(verifier.call_count, 2)
        self.assertTrue(result["audit"]["repair_attempted"])
        self.assertFalse(result["audit"]["repair_succeeded"])
        self.assertTrue(result["audit"]["deterministic_closure_applied"])
        self.assertEqual(result["audit"]["protocol_contract_completeness"], 1.0)
        self.assertFalse(result["items"][0]["risk_eligible"])
        self.assertEqual(result["items"][0]["evidence_ids"], [])


if __name__ == "__main__":
    unittest.main()
