import unittest

from calibration_utils import (
    apply_route_score_calibration,
    apply_structured_boundary_action_policy,
    build_a3wa_decision,
    calibrated_a3wa_membership,
    conformal_score_interval,
)
from scripts.calibrate_a3wa import (
    _residual_entry,
    build_score_calibration,
    fit_monotonic_membership,
    fit_score_uncertainty,
    leave_one_question_out_validation,
)
from scripts.replay_calibration import infer_question_id


class A3WATheoryTests(unittest.TestCase):
    def test_replay_preserves_csbench_question_id(self):
        self.assertEqual(
            infer_question_id("CO_1_grading_checkpoint.json"),
            "CO_1",
        )

    def decision(self, post_calibration=None, **overrides):
        arguments = {
            "model_scores": [5.0, 5.0, 5.0],
            "avg_model_score": 5.0,
            "std_dev": 0.0,
            "max_score": 10.0,
            "blank_rate": 0.0,
            "low_quality_rate": 0.0,
            "perception_failure_rate": 0.0,
            "extraction_quality": "high",
            "fatal_points_ratio": 0.0,
            "post_calibration": post_calibration or {},
        }
        arguments.update(overrides)
        return build_a3wa_decision(**arguments)

    def test_membership_is_monotonic_in_each_risk(self):
        model = {
            "type": "monotonic_logistic",
            "intercept": 2.0,
            "coefficients": {"U_E": 2.0, "U_S": 3.0, "U_R": 4.0},
        }
        low = calibrated_a3wa_membership(0.1, 0.1, 0.1, model=model)["mu"]
        high_e = calibrated_a3wa_membership(0.5, 0.1, 0.1, model=model)["mu"]
        high_s = calibrated_a3wa_membership(0.1, 0.5, 0.1, model=model)["mu"]
        high_r = calibrated_a3wa_membership(0.1, 0.1, 0.5, model=model)["mu"]
        self.assertGreater(low, high_e)
        self.assertGreater(low, high_s)
        self.assertGreater(low, high_r)

    def test_primary_weights_are_not_bypassed(self):
        post = {"primary_risks": {"U_E": 1.0, "U_S": 0.0, "U_R": 0.0, "risk": 1.0 / 3.0}}
        decision = self.decision(
            post,
            weights={"extract": 0.0, "score": 1.0, "semantic": 0.0, "blank": 0.0, "overcredit": 0.0},
        )
        self.assertEqual(decision["risk"], 0.0)
        self.assertEqual(decision["membership_source"], "weighted_primary_risks")

    def test_conformal_interval_expands_with_score_spread(self):
        config = {
            "enabled": True,
            "coverage": 0.9,
            "nonconformity_quantile": 2.0,
            "scale_floor": 0.05,
            "safe_tolerance_ratio": 0.10,
        }
        narrow = conformal_score_interval(5.0, 10.0, 0.05, config)
        wide = conformal_score_interval(5.0, 10.0, 0.20, config)
        self.assertGreater(wide["half_width"], narrow["half_width"])
        self.assertGreaterEqual(wide["stability_risk"], narrow["stability_risk"])

    def test_high_membership_is_not_rewritten_by_review_heuristics(self):
        post = {
            "primary_risks": {"U_E": 0.0, "U_S": 0.0, "U_R": 0.0},
            "lenient_undercredit_signal": 1.0,
            "stable_undercredit_review": True,
        }
        model = {
            "type": "monotonic_logistic",
            "intercept": 6.0,
            "coefficients": {"U_E": 1.0, "U_S": 1.0, "U_R": 1.0},
        }
        decision = self.decision(post, membership_model=model)
        self.assertEqual(decision["route"], "POS")
        self.assertIn("possible_undercredit", decision["review_signals"])

    def test_structured_boundary_evidence_caps_adjustment(self):
        evidence = {
            "confidence": 0.9,
            "missed_credit_items": [{
                "id": "step_1",
                "points": 3.0,
                "evidence": "explicit correct formula",
                "score_layer": "core",
                "evidence_status": "explicit",
                "reason_type": "process_credit",
            }],
            "over_credit_items": [],
        }
        result = apply_structured_boundary_action_policy(
            avg_model_score=5.0,
            candidate_score=8.0,
            max_score=10.0,
            post_calibration={"rubric_item_points": {"step_1": 2.0}},
            agent_evidence=evidence,
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["final_score"], 7.0)
        self.assertEqual(result["sequential_outcome"], "auto_adjusted")

    def test_weak_generic_boundary_evidence_is_rejected(self):
        evidence = {
            "confidence": 0.9,
            "missed_credit_items": [{
                "id": "step_1",
                "points": 2.0,
                "evidence": "has process",
                "score_layer": "support",
                "evidence_status": "weak_generic",
                "reason_type": "process_credit",
            }],
            "over_credit_items": [],
        }
        result = apply_structured_boundary_action_policy(
            avg_model_score=5.0,
            candidate_score=7.0,
            max_score=10.0,
            post_calibration={"rubric_item_points": {"step_1": 2.0}},
            agent_evidence=evidence,
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["final_score"], 5.0)

    def test_small_residual_cells_do_not_apply_correction(self):
        records = [
            {"teacher": 4.0, "trial_score": 3.0},
            {"teacher": 5.0, "trial_score": 4.0},
        ]
        entry = _residual_entry(records, shrinkage_k=8.0, max_correction=2.0)
        self.assertFalse(entry["sign_stable"])
        self.assertEqual(entry["correction"], 0.0)

    def test_cross_question_residual_is_blocked_on_local_direction_conflict(self):
        records = [
            {
                "qid": "CO_1", "trial_route": "BND", "trial_score": 5.0,
                "teacher": 7.0, "max_score": 10.0,
            }
            for _ in range(20)
        ]
        records.extend([
            {
                "qid": "CO_4", "trial_route": "BND", "trial_score": 5.0,
                "teacher": 4.0, "max_score": 10.0,
            }
            for _ in range(3)
        ])
        score_calibration = build_score_calibration(
            records,
            min_cell_count=20,
            shrinkage_k=8.0,
            max_correction_ratio=0.12,
            max_correction_points=2.0,
        )
        result = apply_route_score_calibration(
            score=5.0,
            max_score=10.0,
            question_id="CO_4",
            route="BND",
            config={"score_calibration": score_calibration},
        )
        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "cross_question_direction_conflict")
        self.assertEqual(result["local_diagnostic_n"], 3)

    def test_calibration_produces_cross_question_diagnostics(self):
        records = []
        for qid, teacher, avg, risks in (
            ("CO_1", 5.0, 5.0, (0.0, 0.0, 0.0)),
            ("CO_1", 0.0, 4.0, (0.2, 1.0, 0.7)),
            ("CO_2", 10.0, 9.5, (0.0, 0.2, 0.1)),
            ("CO_2", 2.0, 8.0, (0.4, 1.0, 0.8)),
        ):
            records.append({
                "qid": qid,
                "student_id": f"{qid}_{len(records)}",
                "teacher": teacher,
                "avg": avg,
                "model_avg": avg,
                "final": avg,
                "raw_candidate": avg,
                "max_score": 10.0,
                "model_scores": [avg, avg, avg],
                "std_dev": 0.0,
                "blank_rate": 0.0,
                "low_quality_rate": 0.0,
                "perception_failure_rate": 0.0,
                "structure_missing_rate": 0.0,
                "extraction_risk": risks[0],
                "extraction_quality": "high",
                "fatal_points_ratio": 0.0,
                "high_blank_high_score": False,
                "post_calibration": {
                    "primary_risks": {"U_E": risks[0], "U_S": risks[1], "U_R": risks[2]},
                    "rubric_item_points": {},
                },
                "U_E": risks[0],
                "U_S": risks[1],
                "U_R": risks[2],
                "agent_evidence": {},
                "old_route": "",
                "old_mu": 0.0,
            })
        uncertainty = fit_score_uncertainty(records)
        self.assertTrue(uncertainty["enabled"])
        membership = fit_monotonic_membership(records, iterations=50)
        self.assertEqual(membership["type"], "monotonic_logistic")
        cv = leave_one_question_out_validation(records, {
            "conformal_coverage": 0.9,
            "conformal_scale_floor": 0.05,
            "safe_error_ratio": 0.10,
            "safe_error_points": 0.50,
            "bnd_max": 0.75,
            "neg_max": 0.75,
            "bnd_cost": 0.02,
            "neg_cost": 0.10,
            "unsafe_pos_cost": 1.0,
            "max_unsafe_pos_rate": 0.75,
            "boundary_policy": {},
        })
        self.assertTrue(cv["available"])
        self.assertEqual(len(cv["folds"]), 2)


if __name__ == "__main__":
    unittest.main()
