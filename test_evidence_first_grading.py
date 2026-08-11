import unittest
from unittest.mock import patch

from calibration_utils import (
    compute_evidence_first_directional_risks,
    normalize_evidence_first_grading_result,
)
from scripts.run_csbench import build_parser, grading_contract_from_args


class EvidenceFirstGradingTests(unittest.TestCase):
    def setUp(self):
        self.rubric = [
            {"id": "step_1", "points": 2.0, "type": "formula"},
            {"id": "step_2", "points": 3.0, "type": "judgement"},
        ]
        self.facts = {
            "step_1": "2 + 2 = 4",
            "step_2": "命中",
            "_extraction_quality": "high",
        }

    def test_normalizer_validates_evidence_and_item_sum(self):
        result = {
            "total_score": 99,
            "details": [
                {
                    "id": "step_1",
                    "score_given": 3,
                    "error_category": "MATCH",
                    "evidence_ids": [
                        "step_1",
                        "missing",
                        "_extraction_quality",
                    ],
                    "evidence_support": 1,
                    "contradiction": "false",
                    "confidence": 0.9,
                },
                {
                    "id": "step_2",
                    "score_given": 1.5,
                    "error_category": "PARTIAL_MATCH",
                    "evidence_ids": ["step_2"],
                    "evidence_support": 0.5,
                    "contradiction": False,
                    "confidence": 0.7,
                },
                {"id": "unknown", "score_given": 5},
            ],
        }

        normalized = normalize_evidence_first_grading_result(
            result,
            self.facts,
            self.rubric,
        )

        self.assertEqual(normalized["total_score"], 3.5)
        self.assertEqual(len(normalized["details"]), 2)
        self.assertEqual(normalized["details"][0]["score_given"], 2.0)
        self.assertFalse(normalized["details"][0]["contradiction"])
        self.assertEqual(
            normalized["details"][0]["evidence_ids"], ["step_1"]
        )
        audit = normalized["evidence_first_audit"]
        self.assertEqual(audit["invalid_evidence_reference_count"], 2)
        self.assertEqual(audit["unknown_detail_ids"], ["unknown"])
        self.assertTrue(audit["score_sum_adjusted"])

    def test_directional_risks_are_diagnostic_only(self):
        probes = [
            {
                "details": [
                    {
                        "id": "step_1",
                        "score_given": 1.0,
                        "evidence_support": 1.0,
                        "evidence_valid": True,
                        "contradiction": False,
                        "evidence_contract_complete": True,
                    },
                    {
                        "id": "step_2",
                        "score_given": 1.5,
                        "evidence_support": 0.0,
                        "evidence_valid": False,
                        "contradiction": False,
                        "evidence_contract_complete": True,
                    },
                ]
            }
        ]

        risks = compute_evidence_first_directional_risks(
            self.rubric,
            probes,
        )

        self.assertAlmostEqual(risks["R_under"], 0.2)
        self.assertAlmostEqual(risks["R_over"], 0.3)
        self.assertEqual(risks["status"], "diagnostic_only")
        self.assertEqual(risks["route_effect"], "none")

    def test_cli_defaults_off_and_records_explicit_contract(self):
        parser = build_parser()
        default_args = parser.parse_args(["grade", "CO_1"])
        enabled_args = parser.parse_args(
            ["grade", "CO_1", "--evidence-first-grading"]
        )
        run_args = parser.parse_args(
            ["run", "CO_1", "--evidence-first-grading", "--dry-run"]
        )

        self.assertFalse(default_args.evidence_first_grading)
        self.assertTrue(enabled_args.evidence_first_grading)
        self.assertTrue(run_args.evidence_first_grading)
        self.assertEqual(
            grading_contract_from_args(default_args)[
                "directional_credit_risk"
            ],
            "disabled",
        )
        self.assertEqual(
            grading_contract_from_args(enabled_args),
            {
                "evidence_first_grading": True,
                "evidence_first_schema_version": 1,
                "directional_credit_risk": "diagnostic_only",
                "route_effect": "none",
            },
        )

    def test_prompt_contract_is_added_only_when_enabled(self):
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

        marker = "# Evidence-First Experimental Contract"
        self.assertNotIn(marker, captured[0])
        self.assertIn(marker, captured[1])


if __name__ == "__main__":
    unittest.main()
