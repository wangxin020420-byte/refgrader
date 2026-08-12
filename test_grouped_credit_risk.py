import unittest

from scripts.diagnose_bidirectional_credit_risk import (
    grouped_question_oof,
)


def row(question_id, student_id, risk, unsafe):
    return {
        "question_id": question_id,
        "student_id": student_id,
        "unsafe": unsafe,
        "U_E": 0.0,
        "U_S": risk,
        "U_R": risk,
        "U_R_undercredit_existing": risk,
        "U_R_allocation_undercredit": risk,
        "U_R_allocation_overcredit": risk,
        "U_R_allocation_disagreement": risk,
        "U_R_deterministic_undercredit": risk,
        "U_R_deterministic_overcredit": risk,
    }


class GroupedCreditRiskTests(unittest.TestCase):
    def test_leave_one_question_out_produces_exact_oof_coverage(self):
        rows = [
            row("Q1", "A", 0.1, False),
            row("Q1", "B", 0.9, True),
            row("Q2", "C", 0.1, False),
            row("Q2", "D", 0.9, True),
            row("Q3", "E", 0.1, False),
            row("Q3", "F", 0.9, True),
        ]
        report, folds, oof_rows = grouped_question_oof(
            rows, max_bnd_ratio=0.60, max_unsafe_pos_rate=0.10
        )

        self.assertEqual(report["n"], len(rows))
        self.assertEqual(report["fold_count"], 3)
        self.assertEqual(len(oof_rows), len(rows))
        self.assertEqual(
            {fold["heldout_question"] for fold in folds},
            {"Q1", "Q2", "Q3"},
        )
        self.assertTrue(report["passed"])

    def test_grouped_oof_rejects_single_question_data(self):
        with self.assertRaisesRegex(ValueError, "at least two questions"):
            grouped_question_oof([row("Q1", "A", 0.1, False)])


if __name__ == "__main__":
    unittest.main()
