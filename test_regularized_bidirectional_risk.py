import unittest

from scripts.diagnose_regularized_bidirectional_risk import (
    nested_grouped_oof,
)


def row(question_id, student_id, risk, unsafe_direction=None):
    under = unsafe_direction == "under"
    over = unsafe_direction == "over"
    return {
        "question_id": question_id,
        "student_id": student_id,
        "unsafe": under or over,
        "undercredit_unsafe": under,
        "overcredit_unsafe": over,
        "U_E": risk,
        "U_S": risk,
        "U_R": risk,
        "U_R_undercredit_existing": risk if under else 0.0,
        "U_R_allocation_undercredit": risk if under else 0.0,
        "U_R_allocation_overcredit": risk if over else 0.0,
        "U_R_allocation_disagreement": risk,
        "U_R_deterministic_undercredit": risk if under else 0.0,
        "U_R_deterministic_overcredit": risk if over else 0.0,
        "missing_judgement_risk": 0.0,
    }


class RegularizedBidirectionalRiskTests(unittest.TestCase):
    def test_nested_oof_covers_every_record_once(self):
        rows = []
        for question_index in range(4):
            question = f"Q{question_index}"
            rows.extend([
                row(question, f"{question}-safe-1", 0.05),
                row(question, f"{question}-safe-2", 0.10),
                row(question, f"{question}-under", 0.90, "under"),
                row(question, f"{question}-over", 0.85, "over"),
            ])

        report, folds, oof_rows = nested_grouped_oof(rows)

        self.assertEqual(report["n"], len(rows))
        self.assertEqual(report["fold_count"], 4)
        self.assertEqual(len(oof_rows), len(rows))
        self.assertEqual(
            len({row["student_id"] for row in oof_rows}), len(rows)
        )
        self.assertEqual(
            {fold["heldout_question"] for fold in folds},
            {"Q0", "Q1", "Q2", "Q3"},
        )
        self.assertTrue(report["passed"])

    def test_nested_oof_requires_three_question_groups(self):
        rows = [
            row("Q1", "A", 0.1),
            row("Q2", "B", 0.9, "under"),
        ]
        with self.assertRaisesRegex(ValueError, "at least three questions"):
            nested_grouped_oof(rows)


if __name__ == "__main__":
    unittest.main()
